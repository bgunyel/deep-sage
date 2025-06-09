import asyncio
import time
from typing import Any
from pydantic import BaseModel
from langchain_core.callbacks import get_usage_metadata_callback

from ai_common import WebSearch, format_sources, TavilySearchCategory, LlmServers, get_llm
from ..enums import Node


SUMMARIZER_INSTRUCTIONS = """
You are a world class researcher who is working on a report about a specific topic.

<goal>
Generate a very high quality informative summary of the given context in accordance with the topic.
</goal>

The topic you are working on:
<topic>
{topic}
</topic>

The context to use in generating the informative summary:
<context>
{context}
</context>

Prepare your summary according to the topic. 
Include all necessary information related with the topic in your summary.
"""


class WebSearchNode:
    def __init__(self,
                 web_search_api_key: str,
                 search_category: TavilySearchCategory,
                 number_of_days_back: int,
                 max_results_per_query: int,
                 max_tokens_per_source: int,
                 llm_server: LlmServers,
                 model_params: dict[str, Any]):
        self.web_search = WebSearch(api_key=web_search_api_key,
                                    search_category=search_category,
                                    number_of_days_back=number_of_days_back,
                                    include_raw_content=True,
                                    max_tokens_per_source = max_tokens_per_source,
                                    max_results_per_query = max_results_per_query,
        )
        self.max_tokens_per_source = max_tokens_per_source

        self.model_name = model_params['language_model']
        model_params['model_name'] = self.model_name
        self.base_llm = get_llm(llm_server=llm_server, model_params=model_params)

    def summarize_source(self, topic: str, source_dict: dict[str, Any]) -> (str, str, dict[str, Any]):
        max_length = 102400  # 100K
        raw_content = source_dict['raw_content'][:max_length] if source_dict['raw_content'] is not None else source_dict['content']
        instructions = SUMMARIZER_INSTRUCTIONS.format(topic=topic, context=raw_content)

        with get_usage_metadata_callback() as cb:
            # This call should be ainvoke()
            summary = self.base_llm.invoke(instructions,
                                           max_completion_tokens=32768,
                                           temperature=0,
                                           top_p=0.95)
            token_usage = {
                'input_tokens': cb.usage_metadata[self.model_name]['input_tokens'],
                'output_tokens': cb.usage_metadata[self.model_name]['output_tokens'],
            }

        return raw_content, summary.content, token_usage


    def run(self, state: BaseModel) -> BaseModel:
        """
        Execute web searches and compile formatted results for research purposes.
        
        This method performs web searches using the provided search queries from the state,
        retrieves unique sources, formats them with content truncation, and updates the
        state with both formatted source strings and raw source data for further processing.
        
        Args:
            state (BaseModel): The current flow state containing search queries.
                             Must have a 'search_queries' attribute with query objects
                             that have 'search_query' attributes.
        
        Returns:
            BaseModel: The updated state with the following new attributes:
                      - source_str: Formatted string of all search results
                      - unique_sources: Raw source data from web searches
                      - steps: Updated with WEB_SEARCH node tracking
        
        Note:
            The method uses the configured search parameters (category, days back,
            max tokens per source) to control search behavior and result formatting.
        """
        if not hasattr(state, 'search_queries'):
            raise AttributeError("State must have a 'search_queries' attribute")
        
        if not state.search_queries:
            raise ValueError("State must contain at least one search query")
        
        for i, query in enumerate(state.search_queries):
            if not hasattr(query, 'search_query'):
                raise AttributeError(f"Query at index {i} must have a 'search_query' attribute")
        
        unique_sources = self.web_search.search(search_queries=[query.search_query for query in state.search_queries])

        t1 = time.time()

        # TODO: This loop shall be async
        for k, v in unique_sources.items():
            raw_content, summary, token_usage = self.summarize_source(topic=state.topic, source_dict=v)
            state.token_usage[self.model_name]['input_tokens'] += token_usage['input_tokens']
            state.token_usage[self.model_name]['output_tokens'] += token_usage['output_tokens']
            unique_sources[k]['content'] = summary
        """
        tasks = [self.summarize_source(topic=state.topic, source_dict=v) for v in unique_sources.values()]
        out = await asyncio.gather(*tasks)
        """

        t2 = time.time()
        print(f'Elapsed time: {t2 - t1} seconds')

        source_str = format_sources(unique_sources=unique_sources,
                                    max_tokens_per_source=self.max_tokens_per_source,
                                    include_raw_content=False)
        state.steps.append(Node.WEB_SEARCH.value)
        state.source_str = source_str
        state.unique_sources = unique_sources
        return state
