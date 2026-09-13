"""
Unit tests for lib.use_case.graph.graph (StateGraph wiring), verifying
topology only — no node is actually executed here (see test_graph_nodes.py).
"""

from lib.use_case.graph.graph import build_graph


class TestBuildGraph:
    def test_compiles_without_error(self):
        graph = build_graph()
        assert graph is not None

    def test_expected_nodes_present(self):
        graph = build_graph()
        node_names = set(graph.get_graph().nodes.keys())
        assert {
            "kindergarten_menu",
            "home_menu",
            "merge_menus",
            "shopping_list",
            "write_markdown",
        }.issubset(node_names)

    def test_kindergarten_and_home_menu_fan_out_from_start(self):
        graph = build_graph()
        edges = {(e.source, e.target) for e in graph.get_graph().edges}
        assert ("__start__", "kindergarten_menu") in edges
        assert ("__start__", "home_menu") in edges

    def test_merge_menus_fans_in_from_both_branches(self):
        graph = build_graph()
        edges = {(e.source, e.target) for e in graph.get_graph().edges}
        assert ("kindergarten_menu", "merge_menus") in edges
        assert ("home_menu", "merge_menus") in edges

    def test_sequential_tail_wiring(self):
        graph = build_graph()
        edges = {(e.source, e.target) for e in graph.get_graph().edges}
        assert ("merge_menus", "shopping_list") in edges
        assert ("shopping_list", "write_markdown") in edges
        assert ("write_markdown", "__end__") in edges
