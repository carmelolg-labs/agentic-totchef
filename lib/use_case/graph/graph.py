"""
LangGraph wiring for the TotChef batch workflow.

Kindergarten menu and home menu are generated in parallel (both fan out from
START); merge_menus fans them back in (waits for both, since LangGraph only
fires a node once every incoming edge's predecessor has completed) before
the sequential shopping-list and markdown steps.
"""
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from lib.use_case.graph.nodes import (
    generate_kindergarten_menu,
    generate_home_menu,
    merge_menus,
    generate_shopping_list,
    write_markdown,
)
from lib.use_case.graph.state import TotChefState


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(TotChefState)

    graph.add_node("kindergarten_menu", generate_kindergarten_menu)
    graph.add_node("home_menu", generate_home_menu)
    graph.add_node("merge_menus", merge_menus)
    graph.add_node("shopping_list", generate_shopping_list)
    graph.add_node("write_markdown", write_markdown)

    graph.add_edge(START, "kindergarten_menu")
    graph.add_edge(START, "home_menu")
    graph.add_edge("kindergarten_menu", "merge_menus")
    graph.add_edge("home_menu", "merge_menus")
    graph.add_edge("merge_menus", "shopping_list")
    graph.add_edge("shopping_list", "write_markdown")
    graph.add_edge("write_markdown", END)

    return graph.compile()
