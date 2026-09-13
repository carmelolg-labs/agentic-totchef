"""
Unit tests for lib.commons.EnvironmentVariables.
"""

import os
from unittest.mock import patch
from lib.commons.EnvironmentVariables import (
    get_embedding_model,
    get_language_model,
    get_thinking_mode,
    get_kindergarten_api_host,
    get_home_kitchen_api_host,
    get_kindergarten_api_path,
    get_home_kitchen_api_path,
    get_llm_provider,
    get_ollama_host,
    is_thinking_mode_enabled,
)


class TestEnvironmentVariables:
    def test_get_embedding_model_returns_env_value(self):
        with patch.dict(os.environ, {"EMBEDDING_MODEL": "test-embed"}):
            assert get_embedding_model() == "test-embed"

    def test_get_embedding_model_returns_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_embedding_model("default-embed") == "default-embed"

    def test_get_language_model_returns_env_value(self):
        with patch.dict(os.environ, {"LANGUAGE_MODEL": "test-lm"}):
            assert get_language_model() == "test-lm"

    def test_get_language_model_returns_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_language_model("default-lm") == "default-lm"

    def test_get_thinking_mode_returns_env_value(self):
        with patch.dict(os.environ, {"THINKING_MODE": "True"}):
            assert get_thinking_mode() == "True"

    def test_get_thinking_mode_returns_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_thinking_mode("False") == "False"

    def test_get_kindergarten_api_host_returns_env_value(self):
        with patch.dict(os.environ, {"KINDERGARTEN_API_HOST": "http://kg-host"}):
            assert get_kindergarten_api_host() == "http://kg-host"

    def test_get_kindergarten_api_host_returns_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_kindergarten_api_host("default-kg") == "default-kg"

    def test_get_home_kitchen_api_host_returns_env_value(self):
        with patch.dict(os.environ, {"HOME_KITCHEN_API_HOST": "http://hk-host"}):
            assert get_home_kitchen_api_host() == "http://hk-host"

    def test_get_home_kitchen_api_host_returns_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_home_kitchen_api_host("default-hk") == "default-hk"

    def test_get_kindergarten_api_path_returns_env_value(self):
        with patch.dict(os.environ, {"KINDERGARTEN_API_PATH": "/kg/path"}):
            assert get_kindergarten_api_path() == "/kg/path"

    def test_get_kindergarten_api_path_returns_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_kindergarten_api_path("/default") == "/default"

    def test_get_home_kitchen_api_path_returns_env_value(self):
        with patch.dict(os.environ, {"HOME_KITCHEN_API_PATH": "/hk/path"}):
            assert get_home_kitchen_api_path() == "/hk/path"

    def test_get_home_kitchen_api_path_returns_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_home_kitchen_api_path("/default") == "/default"

    def test_get_llm_provider_returns_env_value(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "custom-provider"}):
            assert get_llm_provider() == "custom-provider"

    def test_get_llm_provider_returns_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_llm_provider("ollama") == "ollama"

    def test_get_ollama_host_returns_env_value(self):
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://ollama-host:11434"}):
            assert get_ollama_host() == "http://ollama-host:11434"

    def test_get_ollama_host_returns_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_ollama_host("http://localhost:11434") == "http://localhost:11434"

    def test_is_thinking_mode_enabled_true(self):
        with patch.dict(os.environ, {"THINKING_MODE": "True"}):
            assert is_thinking_mode_enabled() is True

    def test_is_thinking_mode_enabled_false_string(self):
        with patch.dict(os.environ, {"THINKING_MODE": "False"}):
            assert is_thinking_mode_enabled() is False

    def test_is_thinking_mode_enabled_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            assert is_thinking_mode_enabled() is False

    def test_is_thinking_mode_enabled_case_insensitive(self):
        with patch.dict(os.environ, {"THINKING_MODE": "TRUE"}):
            assert is_thinking_mode_enabled() is True
