"""
Unit tests for lib.use_case.runner.TotChef (run and log_element functions).
"""

from unittest.mock import patch, MagicMock

import lib.use_case.runner.TotChef as tot_chef


class TestTotChefRunner:
    def test_log_element_prints(self, capsys):
        tot_chef.log_element("Hello World")
        captured = capsys.readouterr()
        assert "Hello World" in captured.out
        assert "---" in captured.out

    def test_run_invokes_graph_and_logs_markdown_path(self, capsys):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "kindergarten_menu": "kg menu",
            "home_menu": "home menu",
            "merged_menu": "merged menu",
            "shopping_list": "shopping list",
            "markdown_path": "/static/menu.md",
        }

        with patch("lib.use_case.runner.TotChef.build_graph", return_value=mock_graph):
            tot_chef.run()

        mock_graph.invoke.assert_called_once_with({})
        captured = capsys.readouterr()
        assert "Markdown created: /static/menu.md" in captured.out

    def test_run_handles_missing_markdown_path(self, capsys):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {}

        with patch("lib.use_case.runner.TotChef.build_graph", return_value=mock_graph):
            tot_chef.run()  # Should not raise

        captured = capsys.readouterr()
        assert "Failed to create markdown file" in captured.out
