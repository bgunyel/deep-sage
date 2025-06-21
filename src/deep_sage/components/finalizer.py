from pydantic import BaseModel

from ..enums import Node

class Finalizer:
    def __init__(self):
        pass

    def run(self, state: BaseModel) -> BaseModel:

        # Collect all unique sources from research sections and merge efficiently
        sources_list = [section.unique_sources for section in state.sections if section.research]
        unique_sources = {k: v for d in sources_list for k, v in d.items()}
        state.unique_sources = unique_sources

        # Report Context
        context = f'# {state.report_title}'
        context += f'\n\n\n{state.sections[0].content}'
        context += ''.join([f'\n\n## {section.name}\n\n{section.content}' for idx, section in enumerate(state.sections) if idx > 0])
        context += f'\n\n## Citations\n\n'
        context += '\n'.join([f"{idx}.\t{v['title']}: [{k.replace('_', '\_')}]({k})" for idx, (k, v) in enumerate(unique_sources.items(), start=1)])
        state.content = context

        state.steps.append(Node.FINALIZER)
        return state
