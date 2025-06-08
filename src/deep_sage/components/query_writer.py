import datetime
from typing import Any
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import get_usage_metadata_callback
from pydantic import BaseModel

from ai_common import LlmServers, Queries, get_llm
from ..enums import Node
from ..configuration import Configuration


QUERY_WRITER_INSTRUCTIONS = """
Your goal is to generate targeted web search queries that will gather comprehensive information for writing a summary about a topic.
You will generate exactly {number_of_queries} queries.

<topic>
{topic}
</topic>

Today's date is:
<today>
{today}
</today>

When generating the search queries:
1. Make sure to cover different aspects of the topic.
2. Make sure that your queries account for the most current information available as of today.

Your queries should be:
- Specific enough to avoid generic or irrelevant results.
- Targeted to gather specific information about the topic.
- Diverse enough to cover all aspects of the summary plan.

It is very important that you generate exactly {number_of_queries} queries.
Generate targeted web search queries that will gather specific information about the given topic.
"""


class QueryWriter:
    def __init__(self, llm_server: LlmServers, model_params: dict[str, Any]):
        self.model_name = model_params['language_model']

        model_params['model_name'] = self.model_name
        base_llm = get_llm(llm_server=llm_server, model_params=model_params)
        self.structured_llm = base_llm.with_structured_output(Queries)


    def run(self, state: BaseModel, config: RunnableConfig) -> BaseModel:
        """
        Generate targeted web search queries for comprehensive topic research.
        
        This method creates a specified number of diverse, specific search queries
        designed to gather comprehensive information about a given topic. The queries
        are generated using an LLM with structured output to ensure they cover
        different aspects of the topic and account for current information.
        
        Args:
            state (BaseModel): The current flow state containing the research topic
                             and other relevant information. Must have a 'topic' attribute.
            config (RunnableConfig): The runnable configuration containing parameters
                                   like the number of queries to generate.
        
        Returns:
            BaseModel: The updated state with search_queries populated with a list
                      of generated search query strings.
        
        Note:
            The method appends the QUERY_WRITER node to the state's steps tracking
            and uses today's date to ensure queries capture current information.
        """
        if not hasattr(state, 'topic'):
            raise AttributeError("State must have a 'topic' attribute")
        
        configurable = Configuration.from_runnable_config(config=config)
        state.steps.append(Node.QUERY_WRITER.value)

        instructions = QUERY_WRITER_INSTRUCTIONS.format(topic=state.topic,
                                                        today=datetime.date.today().isoformat(),
                                                        number_of_queries=configurable.number_of_queries)
        with get_usage_metadata_callback() as cb:
            results = self.structured_llm.invoke(instructions)
            state.token_usage[self.model_name]['input_tokens'] += cb.usage_metadata[self.model_name]['input_tokens']
            state.token_usage[self.model_name]['output_tokens'] += cb.usage_metadata[self.model_name]['output_tokens']
        state.search_queries = results.queries
        return state
