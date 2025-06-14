from uuid import uuid4
from typing import Any, Final
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables import RunnableConfig

from ai_common import LlmServers, GraphBase
from summary_writer import SummaryWriter

from .configuration import Configuration
from .enums import Node
from .state import ReportState, section_template
from .components.planner import Planner


class Researcher(GraphBase):
    def __init__(self, llm_server: LlmServers, llm_config: dict[str, Any], web_search_api_key: str):
        self.models = list({llm_config['language_model'], llm_config['reasoning_model']})
        self.configuration_module_prefix: Final = 'src.deep_sage.configuration'

        self.planner = Planner(
            llm_server = llm_server,
            model_params = llm_config,
            web_search_api_key = web_search_api_key,
            configuration_module_prefix = self.configuration_module_prefix,
        )
        self.section_writer = SummaryWriter(
            llm_server=llm_server,
            llm_config=llm_config,
            web_search_api_key=web_search_api_key
        )
        self.graph = self.build_graph()

    def run(self, topic: str, config: RunnableConfig) -> dict[str, Any]:
        in_state = ReportState(
            content='',
            iteration=0,
            sections=[],
            search_queries=[],
            source_str='',
            steps=[],
            token_usage={m: {'input_tokens': 0, 'output_tokens': 0} for m in self.models},
            topic=topic,
            unique_sources={},
        )
        out_state = self.graph.invoke(in_state, config)
        out_dict = {
            'content': out_state['content'],
            'unique_sources': out_state['unique_sources'],
            'token_usage': out_state['token_usage'],
        }
        return out_dict

    def get_response(self, input_dict: dict[str, Any], verbose: bool = False) -> str:
        config = {"configurable": {"thread_id": str(uuid4())}}

        in_state = ReportState(
            content = '',
            iteration = 0,
            sections = [],
            search_queries = [],
            source_str = '',
            steps = [],
            token_usage = {m:{'input_tokens': 0, 'output_tokens': 0} for m in self.models},
            topic = input_dict['topic'],
            unique_sources = {},
        )
        out_state = self.graph.invoke(in_state, config)
        return out_state

    def section_writing_node(self, state: ReportState) -> ReportState:
        # input_dict = {'topic': ,}

        section = state.sections[2]
        topic = section_template.format(topic=state.topic, section_name=section.name, section_topic=section.description)

        config = {
            "configurable": {
                "thread_id": str(uuid4()),
            }
        }

        dummy = -32
        return state


    def build_graph(self):
        workflow = StateGraph(ReportState, config_schema=Configuration)

        ## Nodes
        workflow.add_node(node=Node.PLANNER.value, action=self.planner.run)
        workflow.add_node(node=Node.WRITER.value, action=self.section_writing_node)

        ## Edges
        workflow.add_edge(start_key=START, end_key=Node.PLANNER.value)
        workflow.add_edge(start_key=Node.PLANNER.value, end_key=Node.WRITER.value)
        workflow.add_edge(start_key=Node.WRITER.value, end_key=END)

        compiled_graph = workflow.compile()
        return compiled_graph

