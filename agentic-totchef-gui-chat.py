"""
Agentic TotChef GUI Chat Entry Point

This script serves as the main entry point for the Agentic TotChef GUI chatbot application.
It launches an interactive graphical user interface chatbot for culinary and nutritional assistance.
"""

from lib.use_case.runner.TotChefChatbot import TotChefChatbot

if __name__ in {"__main__", "__mp_main__"}:
    chatbot = TotChefChatbot()
    chatbot.gui()