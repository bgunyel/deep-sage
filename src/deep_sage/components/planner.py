import json
from typing import Any
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

from ai_common import LlmServers, get_llm, TavilySearchCategory, strip_thinking_tokens
from ..enums import Node
from ..configuration import Configuration
from ..state import Sections
from .query_writer import QueryWriter
from .web_search_node import WebSearchNode


PLANNER_INSTRUCTIONS = """
You are an expert writer planning the outline of sections of a report about a given topic.

<goal>
Generate a list of sections for the report.
</goal> 

The topic of the report is:
<topic>
{topic}
</topic>

The report should follow this organization:
<report organization> 
{report_organization}
</report organization>

The context to use in planning the sections of the report:
<context> 
{context}
</context>

<task>
Generate a list of sections for the report. Your plan should be tight and focused with NO overlapping sections or unnecessary filler. 

For example, a good report structure might look like:
1/ introduction
2/ overview of topic A
3/ overview of topic B
4/ comparison between A and B
5/ conclusion
</task>

<requirements>
Each section should have the following fields:

- Name - Name for this section of the report.
- Description - Brief overview of the main topics and concepts to be covered in this section.
- Research - Whether to perform web research for this section of the report (binary score 'yes' or 'no').
- Content - The content of the section, which you will leave blank for now.

Guidelines:
- Ensure each section has a distinct purpose with no content overlap
- Include examples and related details within main topic sections, not as separate sections
- Combine related concepts rather than separating them
- CRITICAL: Every section MUST be directly relevant to the main topic
- Avoid tangential or loosely related sections that don't directly address the core topic
- Consider which sections require web research. 
- Introduction and conclusion will not require research because they will distill information from other parts of the report.
- Main body sections MUST have Research=True. 
</requirements>

<format>
Return the sections of the report as a JSON object:

{{
    sections: [
            {{
                "name": "string",
                "description": "string",
                "research": "string",
                "content": "string",
            }}
    ]
}}
</format>

Now, generate the sections of the report. 
Before submitting, review your structure to ensure it has no redundant sections and follows a logical flow.
"""


class Planner:
    def __init__(self,
                 llm_server: LlmServers,
                 model_params: dict[str, Any],
                 web_search_api_key: str,
                 search_category: TavilySearchCategory,
                 number_of_days_back: int,
                 max_tokens_per_source: int = 5000):
        self.query_writer = QueryWriter(llm_server=llm_server, model_params=model_params)
        self.web_search_node = WebSearchNode(web_search_api_key=web_search_api_key,
                                             search_category=search_category,
                                             number_of_days_back=number_of_days_back,
                                             max_tokens_per_source=max_tokens_per_source)

        model_params['model_name'] = model_params['reasoning_model']
        self.base_llm = get_llm(llm_server=llm_server, model_params=model_params)
        # self.structured_llm = base_llm.with_structured_output(Sections)
        # self.structured_llm = base_llm | JsonOutputParser()

    def run(self, state: BaseModel, config: RunnableConfig) -> BaseModel:
        """
        Generate a structured research plan by creating sections for a comprehensive report.
        
        This method orchestrates the complete planning workflow: generates targeted search queries,
        performs web research, and creates a detailed section outline for the report. It uses
        the gathered context to inform section planning and ensures each section has a clear
        purpose with appropriate research requirements.
        
        Args:
            state (BaseModel): The current flow state containing the research topic.
                             Must have a 'topic' attribute.
            config (RunnableConfig): The runnable configuration containing parameters
                                   like report structure and number of queries.
        
        Returns:
            BaseModel: The updated state with the following new attributes:
                      - search_queries: Generated search queries for the topic
                      - source_str: Formatted string of research results
                      - unique_sources: Raw source data from web searches
                      - sections: List of structured section dictionaries with name,
                                description, research flag, and content fields
                      - steps: Updated with PLANNER node tracking
        
        Note:
            The method combines query generation, web search, and LLM-based planning
            to create a comprehensive research outline. It parses JSON output from
            the reasoning model to extract structured section information.
        """

        state = self.query_writer.run(state=state, config=config)
        state = self.web_search_node.run(state=state)

        configurable = Configuration.from_runnable_config(config=config)
        state.steps.append(Node.PLANNER.value)

        instructions = PLANNER_INSTRUCTIONS.format(topic=state.topic,
                                                   report_organization=configurable.report_structure,
                                                   context=state.source_str)

        results = self.base_llm.invoke(input=instructions)
        idx = results.content.rfind('</think>') # last occurrence of '</think>'
        json_str = results.content[idx+len('</think>'):]
        json_dict = json.loads(json_str)
        state.sections = json_dict['sections']
        return state
