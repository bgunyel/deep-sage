import asyncio
import json
from typing import Any, Final

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from pydantic import BaseModel

from ai_common import get_config_from_runnable
from ai_common.components import QueryWriter, WebSearchNode
from ..enums import Node
from ..state import Section

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

Generate the sections of the report. 
Before submitting, review your structure to ensure it has no redundant sections and follows a logical flow.
Your response must include a 'sections' field containing a list of sections.
Each section must have: name, description, research, and content fields.
"""


class Planner:
    def __init__(self,
                 llm_config: dict[str, Any],
                 web_search_api_key: str,
                 configuration_module_prefix: str):
        self.configuration_module_prefix: Final = configuration_module_prefix
        self.event_loop = asyncio.get_event_loop()

        self.query_writer = QueryWriter(model_params = llm_config['language_model'],
                                        configuration_module_prefix = self.configuration_module_prefix)
        self.web_search_node = WebSearchNode(web_search_api_key = web_search_api_key,
                                             model_params = llm_config['language_model'],
                                             configuration_module_prefix = self.configuration_module_prefix)

        model_params = llm_config['reasoning_model']
        self.model_name = model_params['model']
        self.base_llm = init_chat_model(
            model=model_params['model'],
            model_provider=model_params['model_provider'],
            api_key=model_params['api_key'],
            **model_params['model_args']
        )


    def run(self, state: BaseModel, config: RunnableConfig) -> BaseModel:
        event_loop = asyncio.get_event_loop()
        state = event_loop.run_until_complete(self.run_async(state=state, config=config))
        return state

    async def run_async(self, state: BaseModel, config: RunnableConfig) -> BaseModel:
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
        state = self.web_search_node.run(state=state, config=config)

        configurable = get_config_from_runnable(
            configuration_module_prefix = self.configuration_module_prefix,
            config = config
        )
        state.steps.append(Node.PLANNER.value)

        instructions = PLANNER_INSTRUCTIONS.format(topic=state.topic,
                                                   report_organization=configurable.report_structure,
                                                   context=state.source_str)

        with get_usage_metadata_callback() as cb:
            results = self.base_llm.invoke(instructions, response_format = {"type": "json_object"})
            state.token_usage[self.model_name]['input_tokens'] += cb.usage_metadata[self.model_name]['input_tokens']
            state.token_usage[self.model_name]['output_tokens'] += cb.usage_metadata[self.model_name]['output_tokens']
        json_dict = json.loads(results.content)
        state.sections = [Section(**s) for s in json_dict['sections']]
        return state
