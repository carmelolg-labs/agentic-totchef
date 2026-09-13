"""
Unit tests for lib.core.integration.http.GenericHttpService.

Verifies that get() works as a regular instance method after removing @staticmethod.
"""

import json
import pytest
from unittest.mock import patch, MagicMock, mock_open
from lib.core.integration.http.GenericHttpService import GenericHttpService


class TestGenericHttpService:
    def setup_method(self):
        self.service = GenericHttpService()

    def test_get_returns_api_response_when_available(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": "value"}
        mock_response.raise_for_status.return_value = None

        with patch("lib.core.integration.http.GenericHttpService.requests.get", return_value=mock_response):
            result = self.service.get(
                api_host="http://api.example.com",
                api_path="endpoint",
                fallback_path="fallback.json"
            )

        assert result == {"data": "value"}

    def test_get_falls_back_to_file_when_api_fails(self, caplog):
        import requests as req
        fallback_data = {"fallback": True}

        with caplog.at_level("WARNING"):
            with patch("lib.core.integration.http.GenericHttpService.requests.get", side_effect=req.exceptions.RequestException("connection error")):
                with patch("builtins.open", mock_open(read_data=json.dumps(fallback_data))):
                    result = self.service.get(
                        api_host="http://api.example.com",
                        api_path="endpoint",
                        fallback_path="fallback.json"
                    )

        assert result == fallback_data
        assert "falling back to local file" in caplog.text

    def test_get_falls_back_to_file_when_no_api_host(self):
        fallback_data = [{"item": 1}]

        with patch("builtins.open", mock_open(read_data=json.dumps(fallback_data))):
            result = self.service.get(
                api_host=None,
                api_path=None,
                fallback_path="fallback.json"
            )

        assert result == fallback_data

    def test_get_returns_none_when_fallback_file_missing(self, caplog):
        import requests as req
        with caplog.at_level("ERROR"):
            with patch("lib.core.integration.http.GenericHttpService.requests.get", side_effect=req.exceptions.RequestException("connection error")):
                with patch("builtins.open", side_effect=FileNotFoundError):
                    result = self.service.get(
                        api_host="http://api.example.com",
                        api_path="endpoint",
                        fallback_path="missing.json"
                    )

        assert result is None
        assert "Fallback file not found" in caplog.text

    def test_get_returns_none_when_fallback_json_invalid(self, caplog):
        with caplog.at_level("ERROR"):
            with patch("builtins.open", mock_open(read_data="not valid json")):
                result = self.service.get(
                    api_host=None,
                    api_path=None,
                    fallback_path="broken.json"
                )

        assert result is None
        assert "invalid JSON" in caplog.text

    def test_get_is_callable_on_subclass_instance(self):
        """Subclasses that call self.get(...) must still work."""

        class ConcreteService(GenericHttpService):
            def fetch(self):
                return self.get(api_host=None, api_path=None, fallback_path="fallback.json")

        fallback_data = {"key": "val"}
        svc = ConcreteService()
        with patch("builtins.open", mock_open(read_data=json.dumps(fallback_data))):
            result = svc.fetch()

        assert result == fallback_data
