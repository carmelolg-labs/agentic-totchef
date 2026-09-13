"""
Home Kitchen Tools Module for retrieving home kitchen recipes.
"""
from typing import List

from langchain_core.tools import BaseTool, tool

from lib.core.service.KnowledgeService import get_best_matching_chunk
from lib.use_case.integration.http.HomeKitchenHttpService import HomeKitchenHttpService


def get_tools() -> List[BaseTool]:
    """
    Returns the list of available tools for home kitchen recipes retrieval
    :return: list of tools
    """
    return [get_home_kitchen_recipes, get_home_kitchen_recipes_by_category]


@tool
def get_home_kitchen_recipes() -> dict:
    """
    Get home kitchen recipes divided by categories (carbohydrate, protein, vegetables)
    Returns:
        The home kitchen recipes in JSON format where the keys are carbohydrate, protein, vegetables
    """
    service = HomeKitchenHttpService()
    recipes = service.get_available_recipes()

    if not recipes:
        return {'result': "Not found"}

    return recipes


@tool
def get_home_kitchen_recipes_by_category(category: str = None) -> dict:
    """
    Get home kitchen recipes based on category (for instance carbohydrate, protein, vegetables)
    Args:
        category (str): The category of recipes to retrieve
    Returns:
        The array list with the recipes based on the category
    """
    service = HomeKitchenHttpService()
    recipes = service.get_available_recipes()

    if not recipes:
        return {'result': "Not found"}

    best_category = get_best_matching_chunk(category, recipes.keys())

    if not best_category or best_category["similarity"] < 0.7:
        return {'result': "Not found"}
    else:
        return recipes.get(best_category["match"], {'result': "Not found"})