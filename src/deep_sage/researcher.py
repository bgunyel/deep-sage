import asyncio
from uuid import uuid4
from typing import Any, Final
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from ai_common import GraphBase

from .configuration import Configuration
from .enums import Node
from .state import ReportState
from .components import Planner, SectionsWriter, FinalWriter, Finalizer


class Researcher(GraphBase):
    def __init__(self, llm_config: dict[str, Any], web_search_api_key: str):
        self.memory_saver = MemorySaver()
        self.models = list({llm_config['language_model']['model'], llm_config['reasoning_model']['model']})
        self.configuration_module_prefix: Final = 'src.deep_sage.configuration'

        self.planner = Planner(
            llm_config = llm_config,
            web_search_api_key = web_search_api_key,
            configuration_module_prefix = self.configuration_module_prefix,
        )
        self.sections_writer = SectionsWriter(
            llm_config=llm_config,
            web_search_api_key=web_search_api_key,
            configuration_module_prefix=self.configuration_module_prefix,
        )
        self.final_writer = FinalWriter(
            model_params = llm_config['language_model'],
            configuration_module_prefix=self.configuration_module_prefix,
        )
        self.finalizer = Finalizer()

        self.graph = self.build_graph()

    def run(self, topic: str, config: RunnableConfig) -> dict[str, Any]:
        in_state = ReportState(
            content='',
            iteration=0,
            report_title='',
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
        out_dict = self.run(topic=input_dict['topic'], config=config)
        return out_dict['content']

    def build_graph(self):
        workflow = StateGraph(ReportState, config_schema=Configuration)

        ## Nodes
        workflow.add_node(node=Node.PLANNER.value, action=self.planner.run)
        workflow.add_node(node=Node.SECTIONS_WRITER.value, action=self.sections_writer.run)
        workflow.add_node(node=Node.FINAL_WRITER.value, action=self.final_writer.run)
        workflow.add_node(node=Node.FINALIZER.value, action=self.finalizer.run)

        ## Edges
        workflow.add_edge(start_key=START, end_key=Node.PLANNER.value)
        workflow.add_edge(start_key=Node.PLANNER.value, end_key=Node.SECTIONS_WRITER.value)
        workflow.add_edge(start_key=Node.SECTIONS_WRITER.value, end_key=Node.FINAL_WRITER.value)
        workflow.add_edge(start_key=Node.FINAL_WRITER.value, end_key=Node.FINALIZER.value)
        workflow.add_edge(start_key=Node.FINALIZER.value, end_key=END)

        ## Compile Graph
        compiled_graph = workflow.compile(checkpointer=self.memory_saver)
        return compiled_graph

