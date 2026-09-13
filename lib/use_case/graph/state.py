"""
State schema for the TotChef batch workflow graph.
"""
from typing import TypedDict


class TotChefState(TypedDict, total=False):
    """
    State passed between nodes of the TotChef LangGraph workflow.

    Attributes:
        kindergarten_menu: The generated kindergarten weekly menu.
        home_menu: The generated home weekly menu.
        merged_menu: The kindergarten and home menus merged into one table.
        shopping_list: The shopping list generated from the merged menu.
        markdown_path: The path to the generated markdown file.
    """
    kindergarten_menu: str
    home_menu: str
    merged_menu: str
    shopping_list: str
    markdown_path: str
