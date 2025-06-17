from enum import Enum


class Node(Enum):
    # In alphabetical order
    PLANNER = 'planner'
    QUERY_WRITER = 'query_writer'
    RESET = 'reset'
    REVIEWER = 'reviewer'
    SECTIONS_WRITER = 'sections_writer'
    WEB_SEARCH = 'web_search'
    WRITER = 'writer'

