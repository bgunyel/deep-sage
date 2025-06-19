import asyncio
from typing import Any, Final

from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import get_usage_metadata_callback
from langchain.chat_models import init_chat_model
from pydantic import BaseModel
from ai_common import strip_thinking_tokens, get_config_from_runnable

from ..enums import Node

WRITING_INSTRUCTIONS = """
You are an expert writer working on writing a section that synthesizes information from the rest of the report about a given topic.

<Goal>
Write a high quality section to complement and synthesize the information from other report sections. 
</Goal>

The topic you are writing about:
<topic>
{topic}
</topic>

The name/title of the section you are going to write:
<section name>
{section_name}
</section>>

The description of the section you are going to write:
<section description>
{section_description}
</section description>

The already written content of the report:
<context>
{context}
</context>

<Requirements>
1. For Introduction:
- Write in a clear language.
- Focus on the core motivation for the report.
- Provide a detailed overview of the key points covered in the main body sections (mention key examples, case studies, or findings, etc).
- Use a clear and logical story flow to introduce the report.
- Do not include any structural elements (no lists or tables).
- No sources section needed.

2. For Conclusion/Summary:
- Synthesize and tie together the key themes, findings, and insights from the main body sections.
- Reference specific examples, case studies, or data points covered in the report.
- Ensure a coherent flow with the rest of the report.
- Use structural elements only if they help distill the information given in the report.
- If you use a focused table comparing items present in the report, make sure it obeys the Markdown table syntax.
- If you use a list, make sure it obeys the Markdown list syntax:
      - Use `*` or `-` for unordered lists.
      - Use `1.`, `2.`, `3.`, etc for ordered lists.
      - Ensure proper indentation and spacing.
- End with specific next steps or implications based on the report content.
- No sources section needed.
</Requirements>

<Formatting>
- Start directly with the section writing, without preamble or titles. Do not use XML tags in the output.  
</Formatting>

<Task>
Think carefully about the provided context first. Then write the section.
</Task>
"""

REPORT_TITLE_INSTRUCTIONS = """
You are an expert writer working on writing a title for the report about a given topic.

<Goal>
Write a high quality report title for the given report context. 
</Goal>

The topic of the:
<topic>
{topic}
</topic>

The already written content of the report:
<context>
{context}
</context>

<Formatting>
- Write only the report title, nothing else.
- The title should be a single sentence or phrase.
- Do not include quotation marks, colons, or formatting tags.
- Do not repeat or restate the topic verbatim.
- Make the title concise, engaging, and professional.  
</Formatting>

<Task>
Think carefully about the provided context first. Then write the report title.
</Task>
"""


class FinalWriter:
    def __init__(self, model_params: dict[str, Any], configuration_module_prefix: str):
        self.model_name = model_params['model']
        self.configuration_module_prefix: Final = configuration_module_prefix
        self.writer_llm = init_chat_model(
            model=model_params['model'],
            model_provider=model_params['model_provider'],
            api_key=model_params['api_key'],
            **model_params['model_args']
        )

    def run(self, state: BaseModel, config: RunnableConfig) -> BaseModel:

        # Report context written previously (for the sections requiring research)
        context = ''.join(
            [f'## {section.name}\n\n{section.content}\n\n' for section in state.sections if section.research]
        )

        # TODO: This loop will be async'ed
        # TODO: Token usage will be added to state
        """
        for idx, section in enumerate(state.sections):
            if not section.research:
                out_dict = self.write_section(topic=state.topic,
                                              section_name=section.name,
                                              section_description=section.description,
                                              context=context)
                section.content = out_dict['content']
        """
        event_loop = asyncio.get_event_loop()
        tasks = [
            self.write_section(
                topic = state.topic,
                section_name = section.name,
                section_description = section.description,
                context = context
            ) for section in state.sections if not section.research
        ]
        out_list = event_loop.run_until_complete(asyncio.gather(*tasks))

        non_research_idx = [idx for (idx, section) in enumerate(state.sections) if not section.research]
        for (idx, s) in enumerate(out_list):
            state.sections[non_research_idx[idx]].content = s['content']
            state.token_usage[self.model_name]['input_tokens'] += s['token_usage'][self.model_name]['input_tokens']
            state.token_usage[self.model_name]['output_tokens'] += s['token_usage'][self.model_name]['output_tokens']

        # All the report context including the final sections written above
        # Theoretically, the final (non-research) sections can be anywhere in the report (Planner decides)
        # Hence, we compute the context from scratch, instead of using the above generated context.
        context = ''.join([f'## {section.name}\n\n{section.content}\n\n' for section in state.sections])
        out_dict = self.write_report_title(topic=state.topic, context=context)

        state.token_usage[self.model_name]['input_tokens'] += out_dict['token_usage'][self.model_name]['input_tokens']
        state.token_usage[self.model_name]['output_tokens'] += out_dict['token_usage'][self.model_name]['output_tokens']
        state.report_title = out_dict['title']
        state.steps.append(Node.FINAL_WRITER.value)

        return state


    async def write_section(self, topic: str, section_name: str, section_description: str, context: str) -> dict[str, Any]:

        with get_usage_metadata_callback() as cb:
            instructions = WRITING_INSTRUCTIONS.format(
                topic=topic,
                section_name=section_name,
                section_description=section_description,
                context=context,
            )
            results = await self.writer_llm.ainvoke(instructions)
            out_dict = {
                'content': results.content,
                'token_usage': cb.usage_metadata
            }
        return out_dict

    def write_report_title(self, topic: str, context: str) -> dict[str, Any]:

        with get_usage_metadata_callback() as cb:
            instructions = REPORT_TITLE_INSTRUCTIONS.format(
                topic=topic,
                context=context,
            )
            results = self.writer_llm.invoke(instructions)
            out_dict = {
                'title': results.content,
                'token_usage': cb.usage_metadata
            }
        return out_dict
