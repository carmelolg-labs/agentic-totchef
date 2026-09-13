"""
Unit tests for lib.use_case.runner.TotChefChatbot.

The chatbot is now built on langchain.agents.create_agent (LangGraph) with an
InMemorySaver checkpointer for cross-turn memory, so tests mock `self.agent`
(and its `.stream()`), rather than the old raw-`ollama`-client `chat_func`.
"""

import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from langchain_core.messages import AIMessageChunk, ToolMessageChunk

from lib.use_case.runner.TotChefChatbot import TotChefChatbot


class TestTotChefChatbotInit:
    def test_init_builds_agent_with_combined_tools(self):
        with patch("lib.use_case.runner.TotChefChatbot.create_agent") as mock_create_agent:
            TotChefChatbot()
        _, kwargs = mock_create_agent.call_args
        tool_names = {t.name for t in kwargs["tools"]}
        assert tool_names == {
            "get_kindergarten_menu",
            "get_home_kitchen_recipes",
            "get_home_kitchen_recipes_by_category",
        }
        assert kwargs["checkpointer"] is not None


class TestChat:
    def test_yields_only_non_empty_ai_message_chunk_content(self):
        chatbot = TotChefChatbot()

        ai_chunk_1 = AIMessageChunk(content="Hello")
        ai_chunk_empty = AIMessageChunk(content="")
        tool_chunk = ToolMessageChunk(content="tool result", tool_call_id="1")
        ai_chunk_2 = AIMessageChunk(content=" world")

        chatbot.agent = MagicMock()
        chatbot.agent.stream.return_value = iter([
            (ai_chunk_1, {}),
            (tool_chunk, {}),
            (ai_chunk_empty, {}),
            (ai_chunk_2, {}),
        ])

        result = list(chatbot._chat("hi", thread_id="t1"))

        assert result == ["Hello", " world"]
        call_args, call_kwargs = chatbot.agent.stream.call_args
        assert call_args[0] == {"messages": [{"role": "user", "content": "hi"}]}
        assert call_kwargs["config"] == {"configurable": {"thread_id": "t1"}}
        assert call_kwargs["stream_mode"] == "messages"


class TestRun:
    def test_run_exits_on_quit(self):
        chatbot = TotChefChatbot()
        with patch("builtins.input", side_effect=["quit"]), \
             patch("builtins.print"):
            chatbot.run()  # should not block

    def test_run_exits_on_exit(self):
        chatbot = TotChefChatbot()
        with patch("builtins.input", side_effect=["exit"]), \
             patch("builtins.print"):
            chatbot.run()

    def test_run_handles_user_message(self):
        chatbot = TotChefChatbot()
        with patch.object(chatbot, "_chat", return_value=iter(["hi", "!"])) as mock_chat, \
             patch("builtins.input", side_effect=["hello", "exit"]), \
             patch("builtins.print"):
            chatbot.run()
        mock_chat.assert_called_once()
        args, _ = mock_chat.call_args
        assert args[0] == "hello"


class TestGui:
    def test_gui_calls_ui_run(self):
        chatbot = TotChefChatbot()
        with patch("lib.use_case.runner.TotChefChatbot.ui") as mock_ui:
            chatbot.gui()
            mock_ui.run.assert_called_once()

    def test_root_executes_without_error(self):
        """
        Calls _root() with NiceGUI ui fully mocked so no server is started.
        Also retrieves and invokes the `send` coroutine to cover async lines.
        """
        chatbot = TotChefChatbot()

        # --- Build a mock ecosystem for NiceGUI ---
        # Most UI builders need to support chaining (.classes(), .props(), .style(), .on())
        # and context manager protocol.

        def chainable_mock():
            m = MagicMock()
            m.__enter__ = MagicMock(return_value=m)
            m.__exit__ = MagicMock(return_value=False)
            m.classes.return_value = m
            m.props.return_value = m
            m.style.return_value = m
            m.on.return_value = m
            m.clear.return_value = m
            return m

        text_mock = chainable_mock()
        text_mock.value = "my question"

        send_holder = {}

        def capture_on(event, callback):
            if event == "keydown.enter":
                send_holder["send"] = callback
            return text_mock

        text_mock.on = capture_on

        column_mock = chainable_mock()
        footer_mock = chainable_mock()
        row_mock = chainable_mock()
        element_mock = chainable_mock()
        button_mock = chainable_mock()
        chat_msg_mock = chainable_mock()
        spinner_mock = chainable_mock()
        html_mock = chainable_mock()
        markdown_mock = chainable_mock()

        with patch("lib.use_case.runner.TotChefChatbot.ui") as mock_ui, \
             patch.object(chatbot, "_chat", return_value=iter(["response text"])) as mock_chat:
            mock_ui.add_head_html = MagicMock()
            mock_ui.element.return_value = element_mock
            mock_ui.html.return_value = html_mock
            mock_ui.column.return_value = column_mock
            mock_ui.footer.return_value = footer_mock
            mock_ui.row.return_value = row_mock
            mock_ui.input.return_value = text_mock
            mock_ui.button.return_value = button_mock
            mock_ui.chat_message.return_value = chat_msg_mock
            mock_ui.spinner.return_value = spinner_mock
            mock_ui.markdown.return_value = markdown_mock
            mock_ui.run_javascript = AsyncMock()
            mock_ui.context.client.id = "client-123"

            chatbot._root()

            # Now exercise the captured async `send` callback
            assert "send" in send_holder, "send callback was not captured"
            send_fn = send_holder["send"]

            mock_chat.return_value = iter(["response text"])
            asyncio.run(send_fn())

        args, _ = mock_chat.call_args
        assert args[1] == "client-123"
