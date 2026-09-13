"""
TotChef runner using a LangGraph StateGraph for the batch workflow.
"""
from lib.use_case.graph.graph import build_graph


def run():
    """
    Run the TotChef workflow to generate menus and shopping lists.

    Builds and invokes the LangGraph workflow: kindergarten menu and home menu
    are generated in parallel, then merged, then used to generate a shopping
    list, then written to a markdown file under static/.
    """
    graph = build_graph()
    final_state = graph.invoke({})

    markdown_path = final_state.get("markdown_path")
    if markdown_path:
        log_element(f"Markdown created: {markdown_path}")
    else:
        log_element("Failed to create markdown file")


def log_element(string: str):
    """
    Log an element with separators.

    Prints the given string followed by a blank line and a separator line.

    Args:
        string (str): The string to log.
    """
    print(string)
    print()
    print('-----------------------------------')
    print()
