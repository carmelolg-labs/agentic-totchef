"""
This module provides utility functions to generate system prompts for an expert assistant.
"""
from lib.use_case.prompts.FilePromptManager import FilePromptManager

class GenerateHomeMenuPrompt(FilePromptManager):
    def __init__(self):
        super().__init__(
            system_prompt_path='lib/use_case/prompts/templates/home_menu_system.prompt',
            user_prompt_path='lib/use_case/prompts/templates/home_menu_user.prompt'
        )

    def get_user_prompt(self):
        return self.user_prompt_template.format()
