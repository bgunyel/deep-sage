import asyncio
from uuid import uuid4
from typing import Any, Final

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from pydantic import BaseModel

from ai_common import get_config_from_runnable
from ai_common.components import QueryWriter, WebSearchNode
from summary_writer import SummaryWriter
from ..enums import Node
from ..state import section_template


class SectionsWriter:
    def __init__(self,
                 llm_config: dict[str, Any],
                 web_search_api_key: str,
                 configuration_module_prefix: str):
        self.configuration_module_prefix: Final = configuration_module_prefix
        self.section_writer = SummaryWriter(
            llm_config=llm_config,
            web_search_api_key=web_search_api_key
        )

    def run(self, state: BaseModel, config: RunnableConfig) -> BaseModel:

        configurable = get_config_from_runnable(
            configuration_module_prefix=self.configuration_module_prefix,
            config=config
        )

        event_loop = asyncio.get_event_loop()
        out_list = event_loop.run_until_complete(self.run_async(state=state, config=configurable.sections_config))

        state.steps.append(Node.SECTIONS_WRITER)
        return state

    async def run_async(self, state: BaseModel, config: RunnableConfig) -> BaseModel:


        tasks = [
            self.section_writer.run(
                topic=section_template.format(
                    topic=state.topic, section_title=section.name, section_description=section.description
                ),
                config=config
            ) for section in state.sections if section.research
        ]

        out_list = await asyncio.gather(*tasks)
        return out_list
