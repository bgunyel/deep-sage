from typing import Any, List

from pydantic import BaseModel, Field
from ai_common import SearchQuery


section_template = """
Overall topic: {topic}
Section title: {section_title}
Section description: {section_description}
"""


class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    research: bool = Field(
        description="Whether to perform web research for this section of the report."
    )
    content: str = Field(
        description="The content of the section."
    )
    unique_sources: dict[str, Any] = Field(
        description="Unique sources for this section."
    )

class Sections(BaseModel):
    sections: List[Section] = Field(description="Sections of the report.")

class ReportState(BaseModel):
    """
    Represents the state of the research report.

    Attributes:
        topic: research topic
        search_queries: list of search queries
        source_str: String of formatted source content from web search
        content: Content generated from sources
        steps: steps followed during graph run

    """
    content: str
    iteration: int = 0
    report_title: str
    sections: list[Section]
    search_queries: list[SearchQuery]
    source_str: str
    steps: list[str]
    token_usage: dict
    topic: str
    unique_sources: dict[str, Any]
