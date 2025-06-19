from enum import Enum


class Node(Enum):
    # In alphabetical order
    FINAL_WRITER = 'final_writer'
    FINALIZER = 'finalizer'
    PLANNER = 'planner'
    QUERY_WRITER = 'query_writer'
    RESET = 'reset'
    REVIEWER = 'reviewer'
    SECTIONS_WRITER = 'sections_writer'
    WEB_SEARCH = 'web_search'
    WRITER = 'writer'

