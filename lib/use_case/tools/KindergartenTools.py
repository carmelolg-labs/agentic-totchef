"""
Kindergarten Tools module for retrieving kindergarten menu information.
"""
from typing import List

from langchain_core.tools import BaseTool, tool

from lib.use_case.integration.http.KindergartenHttpService import KindergartenHttpService


def get_tools() -> List[BaseTool]:
    """
    Returns the list of available tools for kindergarten menu retrieval
    :return: list of tools
    """
    return [get_kindergarten_menu]


@tool
def get_kindergarten_menu(week: int = 1) -> dict:
    """
    Get the kindergarten menu for a specified week
    Args:
      week: The week number (default 1)
    Returns:
        The menu of the kindergarten for the specified week
    """
    service = KindergartenHttpService()
    menu = service.get_current_menu()

    if not menu:
        return {'result': "Not found"}

    return menu["week"].get(str(week), {})
