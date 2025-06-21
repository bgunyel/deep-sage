from typing import ClassVar
from ai_common import NodeBase
from pydantic import ConfigDict


class Node(NodeBase):
    model_config = ConfigDict(frozen=True)

    # Class attributes
    FINAL_WRITER: ClassVar[str] = 'final_writer'
    FINALIZER: ClassVar[str] = 'finalizer'
    PLANNER: ClassVar[str] = 'planner'
    SECTIONS_WRITER: ClassVar[str] = 'sections_writer'
