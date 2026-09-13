"""
Unit tests for lib.use_case.graph.nodes.

Each node's module-level model/agent object is mocked so no network call to
a real Ollama server ever happens.
"""

from unittest.mock import MagicMock, patch

import lib.use_case.graph.nodes as nodes


def _agent_result(content: str) -> dict:
    message = MagicMock()
    message.content = content
    return {"messages": [message]}


class TestGenerateKindergartenMenu:
    def test_returns_agent_response_content(self):
        with patch.object(nodes, "_kindergarten_agent") as mock_agent:
            mock_agent.invoke.return_value = _agent_result("kg menu")
            result = nodes.generate_kindergarten_menu({})
        assert result == {"kindergarten_menu": "kg menu"}
        mock_agent.invoke.assert_called_once()


class TestGenerateHomeMenu:
    def test_returns_agent_response_content(self):
        with patch.object(nodes, "_home_agent") as mock_agent:
            mock_agent.invoke.return_value = _agent_result("home menu")
            result = nodes.generate_home_menu({})
        assert result == {"home_menu": "home menu"}
        mock_agent.invoke.assert_called_once()


class TestMergeMenus:
    def test_merges_kindergarten_and_home_menus(self):
        with patch.object(nodes, "_plain_model") as mock_model:
            mock_response = MagicMock()
            mock_response.content = "merged menu"
            mock_model.invoke.return_value = mock_response

            result = nodes.merge_menus({"kindergarten_menu": "kg", "home_menu": "home"})

        assert result == {"merged_menu": "merged menu"}
        mock_model.invoke.assert_called_once()


class TestGenerateShoppingList:
    def test_returns_shopping_list_content(self):
        with patch.object(nodes, "_plain_model") as mock_model:
            mock_response = MagicMock()
            mock_response.content = "shopping list"
            mock_model.invoke.return_value = mock_response

            result = nodes.generate_shopping_list({"merged_menu": "merged"})

        assert result == {"shopping_list": "shopping list"}
        call_args = mock_model.invoke.call_args[0][0]
        assert call_args[0]["role"] == "system"
        assert call_args[1]["role"] == "user"


class TestWriteMarkdown:
    def test_writes_file_and_returns_path(self, tmp_path):
        fake_project_root = tmp_path
        (fake_project_root / "static").mkdir()

        with patch("lib.use_case.graph.nodes.pathlib.Path") as mock_path_cls:
            mock_path_cls.return_value.resolve.return_value.parents.__getitem__.return_value = fake_project_root

            result = nodes.write_markdown({
                "merged_menu": "full menu",
                "shopping_list": "shopping list",
            })

        markdown_path = result["markdown_path"]
        assert markdown_path.startswith(str(fake_project_root / "static" / "menu_shopping_"))
        content = open(markdown_path, encoding="utf-8").read()
        assert "full menu" in content
        assert "shopping list" in content

    def test_handles_missing_menu_and_shopping_list(self, tmp_path):
        fake_project_root = tmp_path
        (fake_project_root / "static").mkdir()

        with patch("lib.use_case.graph.nodes.pathlib.Path") as mock_path_cls:
            mock_path_cls.return_value.resolve.return_value.parents.__getitem__.return_value = fake_project_root

            result = nodes.write_markdown({})

        content = open(result["markdown_path"], encoding="utf-8").read()
        assert "(no menu provided)" in content
        assert "(no shopping list provided)" in content

    def test_returns_none_markdown_path_on_write_failure(self, caplog):
        fake_static_dir = MagicMock()
        fake_static_dir.mkdir.side_effect = OSError("disk full")
        fake_project_root = MagicMock()
        fake_project_root.__truediv__.return_value = fake_static_dir

        with patch("lib.use_case.graph.nodes.pathlib.Path") as mock_path_cls, \
             caplog.at_level("ERROR"):
            mock_path_cls.return_value.resolve.return_value.parents.__getitem__.return_value = fake_project_root

            result = nodes.write_markdown({"merged_menu": "menu", "shopping_list": "list"})

        assert result == {"markdown_path": None}
        assert "Failed to write markdown file" in caplog.text
