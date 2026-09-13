"""
Unit tests for lib.use_case.integration.http.HomeKitchenHttpService
and lib.use_case.integration.http.KindergartenHttpService.
"""

import json
from datetime import date
from unittest.mock import patch, mock_open

from lib.use_case.integration.http.HomeKitchenHttpService import HomeKitchenHttpService
from lib.use_case.integration.http.KindergartenHttpService import KindergartenHttpService


class TestHomeKitchenHttpService:
    def test_get_available_recipes_returns_data(self):
        recipes = {"carbohydrates": ["pasta"], "proteins": ["chicken"]}
        svc = HomeKitchenHttpService()
        with patch.object(svc, "get", return_value=recipes) as mock_get:
            result = svc.get_available_recipes()
        assert result == recipes

    def test_get_available_recipes_calls_get_with_correct_fallback(self):
        svc = HomeKitchenHttpService()
        with patch.object(svc, "get", return_value=None) as mock_get:
            result = svc.get_available_recipes()
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["fallback_path"] == "static/home-kitchen-recipes.json"


class TestKindergartenHttpService:
    def test_get_winter_menu_returns_data(self):
        menu = {"week": {"1": {"monday": "pasta"}}}
        svc = KindergartenHttpService()
        with patch.object(svc, "get", return_value=menu):
            result = svc.get_winter_menu()
        assert result == menu

    def test_get_summer_menu_returns_data(self):
        menu = {"week": {"1": {"monday": "salad"}}}
        svc = KindergartenHttpService()
        with patch.object(svc, "get", return_value=menu):
            result = svc.get_summer_menu()
        assert result == menu

    def test_get_summer_menu_falls_back_to_winter_when_no_data(self, caplog):
        winter_menu = {"week": {"1": {"monday": "pasta"}}}
        svc = KindergartenHttpService()
        with caplog.at_level("WARNING"):
            with patch.object(svc, "get", return_value=None), \
                 patch.object(svc, "get_winter_menu", return_value=winter_menu) as mock_winter:
                result = svc.get_summer_menu()
        mock_winter.assert_called_once()
        assert result == winter_menu
        assert "falling back to the winter menu" in caplog.text

    def test_get_current_menu_winter_when_november(self):
        svc = KindergartenHttpService()
        menu = {"week": {}}
        with patch("lib.use_case.integration.http.KindergartenHttpService.date") as mock_date:
            mock_date.today.return_value.month = 11
            with patch.object(svc, "get_winter_menu", return_value=menu) as mock_winter:
                result = svc.get_current_menu()
        mock_winter.assert_called_once()
        assert result == menu

    def test_get_current_menu_summer_when_june(self):
        svc = KindergartenHttpService()
        menu = {"week": {}}
        with patch("lib.use_case.integration.http.KindergartenHttpService.date") as mock_date:
            mock_date.today.return_value.month = 6
            with patch.object(svc, "get_summer_menu", return_value=menu) as mock_summer:
                result = svc.get_current_menu()
        mock_summer.assert_called_once()

    def test_get_current_menu_winter_when_january(self):
        svc = KindergartenHttpService()
        with patch("lib.use_case.integration.http.KindergartenHttpService.date") as mock_date:
            mock_date.today.return_value.month = 1
            with patch.object(svc, "get_winter_menu", return_value={}) as mock_winter:
                svc.get_current_menu()
        mock_winter.assert_called_once()
