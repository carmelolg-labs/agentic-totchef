"""
TotChefChatbot Module

This module provides the TotChefChatbot class, which implements an interactive chatbot
for culinary and nutritional assistance. It supports both command-line and GUI interfaces,
integrating tools from KindergartenTools and HomeKitchenTools.
"""

import uuid

from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from html_sanitizer import Sanitizer
from nicegui import ui

from lib.commons.EnvironmentVariables import get_language_model, get_ollama_host, is_thinking_mode_enabled
from lib.use_case.tools import KindergartenTools, HomeKitchenTools


class TotChefChatbot:
    """
    TotChef Chatbot class that provides an interactive chat interface
    for culinary and nutritional assistance using functions from
    KindergartenTools and HomeKitchenTools.
    """
    def __init__(self):
        """
        Initialize the TotChefChatbot instance.

        Builds a LangGraph ReAct-style agent (via create_agent) over the combined
        KindergartenTools/HomeKitchenTools tools, with an in-memory checkpointer
        so conversation history is preserved across turns within a thread_id.
        """
        model = ChatOllama(
            model=get_language_model(),
            base_url=get_ollama_host(),
            reasoning=is_thinking_mode_enabled(),
        )
        tools = KindergartenTools.get_tools() + HomeKitchenTools.get_tools()
        self.agent = create_agent(model=model, tools=tools, checkpointer=InMemorySaver())

    def run(self):
        """
        Run an interactive chatbot session for TotChef in the command line.

        The chatbot utilizes functions from KindergartenTools and HomeKitchenTools
        to assist users with culinary and nutritional queries. Type 'exit' or 'quit'
        to end the session. The whole CLI session shares one thread_id, so the agent
        keeps conversation memory across turns.
        """
        print("Welcome to TotChef Chat! Type 'exit' or 'quit' to end the chat.")
        thread_id = str(uuid.uuid4())

        # Start chat
        while True:
            user_prompt = input('\nUser > ')
            if user_prompt.lower() in ['exit', 'quit']:
                break
            else:
                print('Assistant >', end=' ')
                for chunk_text in self._chat(user_prompt, thread_id):
                    print(chunk_text, end='', flush=True)

    def _chat(self, user_prompt: str, thread_id: str):
        """
        Chat with the TotChef agent using the provided user prompt, streaming
        only the assistant's text deltas (tool-call/tool-result chunks are
        filtered out).

        Args:
            user_prompt (str): The prompt or question from the user.
            thread_id (str): Conversation thread id, scoping memory to one
                session (CLI run, or one NiceGUI client).

        Yields:
            str: Incremental text chunks of the assistant's reply.
        """
        config = {"configurable": {"thread_id": thread_id}}
        for chunk, _metadata in self.agent.stream(
            {"messages": [{"role": "user", "content": user_prompt}]},
            config=config,
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                yield chunk.content

    def _root(self):
        """
        Define the GUI layout and behavior for the TotChef chatbot.

        This method sets up the chat interface, including message display and input handling.
        It is used internally by the gui method.
        """
        # Add custom CSS for nutritionist theme
        ui.add_head_html('''
            <style>
                body {
                    background: linear-gradient(135deg, #f5f7fa 0%, #e8f5e9 100%);
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                }
                .q-page {
                    background: transparent !important;
                }
                .q-footer {
                    background: linear-gradient(to right, #ffffff 0%, #f1f8f4 100%) !important;
                    border-top: 2px solid #a5d6a7 !important;
                    box-shadow: 0 -4px 6px rgba(0, 0, 0, 0.05);
                }
                .chat-container {
                    background: rgba(255, 255, 255, 0.95);
                    border-radius: 16px;
                    padding: 20px;
                    box-shadow: 0 4px 20px\ rgba(0, 0, 0, 0.08);
                    margin-top: 20px;
                }
                .q-message-sent {
                    //background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%) !important;
                    //color: white !important;
                    border: 1px solid #e0e0e0;
                    border-radius: 18px 18px 4px 18px !important;
                    padding: 12px 16px !important;
                }
                .q-message-received {
                    //background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%) !important;
                    border: 1px solid #e0e0e0;
                    border-radius: 18px 18px 18px 4px !important;
                    color: #2e7d32 !important;
                    padding: 12px 16px !important;
                }
                .header-container {
                    background: linear-gradient(135deg, #66bb6a 0%, #43a047 100%);
                    color: white;
                    padding: 24px;
                    border-radius: 0 0 20px 20px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                    margin-bottom: 10px;
                }
                .header-title {
                    font-size: 28px;
                    font-weight: 600;
                    margin: 0;
                    text-align: center;
                    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }
                .header-subtitle {
                    font-size: 14px;
                    text-align: center;
                    margin-top: 8px;
                    opacity: 0.95;
                }
            </style>
        ''')

        # Add professional nutritionist header
        with ui.element('div').classes('header-container'):
            ui.html('<h1 class="header-title">🥗 TotChef Nutritionist</h1>', sanitize=False)
            ui.html('<p class="header-subtitle">Your Personal AI Nutrition & Culinary Expert</p>', sanitize=False)

        async def send() -> None:
            question = text.value
            text.value = ''
            with message_container:
                ui.chat_message(text=question, name='👤 You', sent=True)
                response_message = ui.chat_message(name='🍎 Nutritionist', sent=False)
                spinner = ui.spinner(type='dots', size='lg', color='green')

            await ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')
            response = ''
            thread_id = ui.context.client.id
            for chunk_text in self._chat(question, thread_id):
                response += chunk_text
                with response_message.clear():
                    ui.html(response, sanitize=Sanitizer().sanitize)
                    await ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')
            message_container.remove(spinner)

        message_container = ui.column().classes('w-full max-w-2xl mx-auto flex-grow items-stretch chat-container')

        with (ui.footer(), ui.column().classes('w-full max-w-3xl mx-auto my-6')):
            with ui.row().classes('w-full no-wrap items-center'):
                placeholder = 'Ask me about nutrition, recipes, or healthy eating...'
                text = ui.input(placeholder=placeholder).props('rounded outlined input-class=mx-3').classes('w-full self-center').style('color: #2e7d32').on('keydown.enter', send)
                button = ui.button(text="Send", color="green-7", on_click=send).props('rounded').classes('self-center px-6')

            ui.markdown('Made with 🔥 by [carmelolg](https://carmelolg.github.io)') \
                .classes('text-xs self-center mr-12 m-[-1em] text-green-800') \
                .classes('[&_a]:text-inherit [&_a]:no-underline [&_a]:font-medium')

    def gui(self):
        """
        Launch the GUI for the TotChef chatbot.

        This method initializes the user interface and starts the event loop.
        """
        ui.run(self._root, title='Agentic TotChef', favicon='', show_welcome_message=True, reconnect_timeout=600)
