from ai_common import WebSearch, format_sources, TavilySearchCategory
from pydantic import BaseModel

from ..enums import Node


class WebSearchNode:
    def __init__(self,
                 web_search_api_key: str,
                 search_category: TavilySearchCategory,
                 number_of_days_back: int,
                 max_tokens_per_source: int = 5000):
        self.web_search = WebSearch(api_key=web_search_api_key,
                                    search_category=search_category,
                                    number_of_days_back=number_of_days_back,
                                    include_raw_content=True)
        self.max_tokens_per_source = max_tokens_per_source

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
        source_str = format_sources(unique_sources=unique_sources,
                                    max_tokens_per_source=self.max_tokens_per_source,
                                    include_raw_content=True)
        state.steps.append(Node.WEB_SEARCH.value)
        state.source_str = source_str
        state.unique_sources = unique_sources
        return state
