from typing import Union

from langchain_core.prompts import PromptTemplate

from lib.use_case.prompts.PromptManager import PromptManager


class FilePromptManager(PromptManager):
    def __init__(self, system_prompt_path: str, user_prompt_path: str):
        self.system_prompt_template = self._load_prompt(system_prompt_path)
        self.user_prompt_template = self._load_prompt(user_prompt_path)

    def _load_prompt(self, file_path: str) -> Union[str, PromptTemplate]:
        if file_path is None or len(file_path) == 0:
            return ""
        return PromptTemplate.from_file(file_path)

    def get_system_prompt(self, **kwargs):
        return self.system_prompt_template.format(**kwargs)

    def get_user_prompt(self, **kwargs):
        return self.user_prompt_template.format(**kwargs)
