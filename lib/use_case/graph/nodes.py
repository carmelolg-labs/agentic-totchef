"""
Node functions for the TotChef batch workflow graph.

Each function takes the current TotChefState and returns a partial dict of
the state keys it updates; LangGraph merges the returned dict into state.
"""
import logging
import pathlib
from datetime import datetime, timezone

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from lib.commons.EnvironmentVariables import get_language_model, get_ollama_host, is_thinking_mode_enabled
from lib.use_case.graph.state import TotChefState
from lib.use_case.prompts.HomeMenuPrompt import GenerateHomeMenuPrompt
from lib.use_case.prompts.KindergartenMenuPrompt import GetKindergartenMenuPrompt
from lib.use_case.prompts.MergeMenuPrompt import MergeMenuPrompt
from lib.use_case.prompts.ShoppingListPrompt import ShoppingListPrompt
from lib.use_case.tools import HomeKitchenTools, KindergartenTools

logger = logging.getLogger(__name__)

kindergarten_menu_prompt = GetKindergartenMenuPrompt()
home_menu_prompt = GenerateHomeMenuPrompt()
merge_menu_prompt = MergeMenuPrompt()
shopping_list_prompt = ShoppingListPrompt()

_language_model = get_language_model()
_ollama_host = get_ollama_host()

_kindergarten_agent = create_agent(
    model=ChatOllama(model=_language_model, base_url=_ollama_host, reasoning=is_thinking_mode_enabled()),
    tools=KindergartenTools.get_tools(),
    system_prompt=kindergarten_menu_prompt.get_system_prompt(),
)
_home_agent = create_agent(
    model=ChatOllama(model=_language_model, base_url=_ollama_host, reasoning=is_thinking_mode_enabled()),
    tools=HomeKitchenTools.get_tools(),
    system_prompt=home_menu_prompt.get_system_prompt(),
)
# Merge/shopping-list steps never need reasoning mode (mirrors the old disable_think=True).
_plain_model = ChatOllama(model=_language_model, base_url=_ollama_host, reasoning=False)


def generate_kindergarten_menu(state: TotChefState) -> dict:
    print("Generating Kindergarten Menu for Week 1...")
    result = _kindergarten_agent.invoke({
        "messages": [{"role": "user", "content": kindergarten_menu_prompt.get_user_prompt(1)}]
    })
    return {"kindergarten_menu": result["messages"][-1].content}


def generate_home_menu(state: TotChefState) -> dict:
    print("Generating Home Menu for Week 1...")
    result = _home_agent.invoke({
        "messages": [{"role": "user", "content": home_menu_prompt.get_user_prompt()}]
    })
    return {"home_menu": result["messages"][-1].content}


def merge_menus(state: TotChefState) -> dict:
    print("Merging menus...")
    prompt = merge_menu_prompt.get_user_prompt(state["kindergarten_menu"], state["home_menu"])
    response = _plain_model.invoke(prompt)
    return {"merged_menu": response.content}


def generate_shopping_list(state: TotChefState) -> dict:
    print("Generating Shopping List...")
    response = _plain_model.invoke([
        {"role": "system", "content": shopping_list_prompt.get_system_prompt()},
        {"role": "user", "content": shopping_list_prompt.get_user_prompt(state["merged_menu"])},
    ])
    return {"shopping_list": response.content}


def write_markdown(state: TotChefState) -> dict:
    """Write a markdown file with the full menu and shopping list under static/.

    Returns {"markdown_path": None} on failure (instead of raising) so a disk
    error degrades gracefully rather than crashing the whole graph run and
    discarding the already-generated menu/shopping list.
    """
    try:
        project_root = pathlib.Path(__file__).resolve().parents[3]
        static_dir = project_root / "static"
        static_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        file_path = static_dir / f"menu_shopping_{timestamp}.md"

        content_lines = [
            "# Weekly Menu and Shopping List\n",
            "## Menu\n",
            state.get("merged_menu") or "(no menu provided)",
            "## Shopping List\n",
            state.get("shopping_list") or "(no shopping list provided)",
            "\n",
        ]
        file_path.write_text("\n".join(content_lines), encoding="utf-8")

        return {"markdown_path": str(file_path)}
    except OSError as exc:
        logger.error("Failed to write markdown file: %s", exc)
        return {"markdown_path": None}
