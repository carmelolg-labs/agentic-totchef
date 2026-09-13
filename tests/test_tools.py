"""
Unit tests for lib.use_case.tools.HomeKitchenTools and lib.use_case.tools.KindergartenTools.

Both modules now expose LangChain @tool-decorated StructuredTool objects
(via get_tools()) instead of plain functions in a dict, so tools are invoked
with `.invoke({...})` rather than called directly with keyword arguments.
"""

from unittest.mock import patch
from langchain_core.tools import BaseTool
import lib.use_case.tools.HomeKitchenTools as hkt
import lib.use_case.tools.KindergartenTools as kgt


class TestHomeKitchenTools:
    def test_get_tools_returns_list_of_tool_objects(self):
        tools = hkt.get_tools()
        names = {t.name for t in tools}
        assert names == {"get_home_kitchen_recipes", "get_home_kitchen_recipes_by_category"}
        assert all(isinstance(t, BaseTool) for t in tools)

    def test_get_home_kitchen_recipes_returns_data(self):
        recipes = {"carbohydrates": ["pasta"]}
        with patch("lib.use_case.tools.HomeKitchenTools.HomeKitchenHttpService") as mock_svc_cls:
            mock_svc = mock_svc_cls.return_value
            mock_svc.get_available_recipes.return_value = recipes
            result = hkt.get_home_kitchen_recipes.invoke({})
        assert result == recipes

    def test_get_home_kitchen_recipes_returns_not_found_when_empty(self):
        with patch("lib.use_case.tools.HomeKitchenTools.HomeKitchenHttpService") as mock_svc_cls:
            mock_svc = mock_svc_cls.return_value
            mock_svc.get_available_recipes.return_value = None
            result = hkt.get_home_kitchen_recipes.invoke({})
        assert result == {"result": "Not found"}

    def test_get_home_kitchen_recipes_by_category_returns_category(self):
        recipes = {"carbohydrates": ["pasta"], "proteins": ["chicken"]}
        best_match = {"match": "carbohydrates", "similarity": 0.95}
        with patch("lib.use_case.tools.HomeKitchenTools.HomeKitchenHttpService") as mock_svc_cls:
            mock_svc = mock_svc_cls.return_value
            mock_svc.get_available_recipes.return_value = recipes
            with patch("lib.use_case.tools.HomeKitchenTools.get_best_matching_chunk", return_value=best_match):
                result = hkt.get_home_kitchen_recipes_by_category.invoke({"category": "carb"})
        assert result == ["pasta"]

    def test_get_home_kitchen_recipes_by_category_low_similarity(self):
        recipes = {"carbohydrates": ["pasta"]}
        low_match = {"match": "carbohydrates", "similarity": 0.5}
        with patch("lib.use_case.tools.HomeKitchenTools.HomeKitchenHttpService") as mock_svc_cls:
            mock_svc = mock_svc_cls.return_value
            mock_svc.get_available_recipes.return_value = recipes
            with patch("lib.use_case.tools.HomeKitchenTools.get_best_matching_chunk", return_value=low_match):
                result = hkt.get_home_kitchen_recipes_by_category.invoke({"category": "xyz"})
        assert result == {"result": "Not found"}

    def test_get_home_kitchen_recipes_by_category_no_recipes(self):
        with patch("lib.use_case.tools.HomeKitchenTools.HomeKitchenHttpService") as mock_svc_cls:
            mock_svc = mock_svc_cls.return_value
            mock_svc.get_available_recipes.return_value = None
            result = hkt.get_home_kitchen_recipes_by_category.invoke({"category": "carb"})
        assert result == {"result": "Not found"}

    def test_get_home_kitchen_recipes_by_category_no_best_match(self):
        recipes = {"carbohydrates": ["pasta"]}
        with patch("lib.use_case.tools.HomeKitchenTools.HomeKitchenHttpService") as mock_svc_cls:
            mock_svc = mock_svc_cls.return_value
            mock_svc.get_available_recipes.return_value = recipes
            with patch("lib.use_case.tools.HomeKitchenTools.get_best_matching_chunk", return_value=None):
                result = hkt.get_home_kitchen_recipes_by_category.invoke({"category": "xyz"})
        assert result == {"result": "Not found"}


class TestKindergartenTools:
    def test_get_tools_returns_list_of_tool_objects(self):
        tools = kgt.get_tools()
        names = {t.name for t in tools}
        assert names == {"get_kindergarten_menu"}
        assert all(isinstance(t, BaseTool) for t in tools)

    def test_get_kindergarten_menu_returns_week_data(self):
        menu = {"week": {"1": {"monday": "pasta"}}}
        with patch("lib.use_case.tools.KindergartenTools.KindergartenHttpService") as mock_svc_cls:
            mock_svc = mock_svc_cls.return_value
            mock_svc.get_current_menu.return_value = menu
            result = kgt.get_kindergarten_menu.invoke({"week": 1})
        assert result == {"monday": "pasta"}

    def test_get_kindergarten_menu_returns_not_found_when_empty(self):
        with patch("lib.use_case.tools.KindergartenTools.KindergartenHttpService") as mock_svc_cls:
            mock_svc = mock_svc_cls.return_value
            mock_svc.get_current_menu.return_value = None
            result = kgt.get_kindergarten_menu.invoke({"week": 1})
        assert result == {"result": "Not found"}
