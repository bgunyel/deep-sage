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
        context += ''.join([f'\n\n## {section.name}\n\n{section.content}' for section in state.sections])
        context += f'\n\n## Citations\n\n'
        context += ''.join([f'* {v["title"]}: {k}\n' for k, v in unique_sources.items()])
        state.content = context

        state.steps.append(Node.FINALIZER.value)
        return state
