import asyncio
from typing import Any, Final

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
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

        event_loop = asyncio.new_event_loop()
        tasks = [
            self.section_writer.run(
                topic=section_template.format(
                    topic=state.topic, section_title=section.name, section_description=section.description
                ),
                config=config
            ) for section in state.sections if section.research
        ]

        # out_list = await asyncio.gather(*tasks)
        out_list = event_loop.run_until_complete(asyncio.gather(*tasks))
        event_loop.close()
        state.steps.append(Node.SECTIONS_WRITER)

        models_list = [*state.token_usage.keys()]
        research_idx = [idx for (idx, section) in enumerate(state.sections) if section.research]

        for (idx, s) in enumerate(out_list):
            state.sections[research_idx[idx]].content = s['content']
            state.sections[research_idx[idx]].unique_sources = s['unique_sources']

            for m in models_list:
                state.token_usage[m]['input_tokens'] += s['token_usage'][m]['input_tokens']
                state.token_usage[m]['output_tokens'] += s['token_usage'][m]['output_tokens']

        return state
