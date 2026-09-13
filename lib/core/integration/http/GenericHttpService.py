"""
Generic service module for interacting with APIs.

This module exposes `GenericHttpService` as a reusable base class.
Concrete services should inherit from it and use the `get` method defined
on the class.
"""
from typing import Any
import requests
import json
import logging

logger = logging.getLogger(__name__)


class GenericHttpService:
    """Base HTTP service providing a reusable GET implementation.

    Subclasses can reuse the concrete `get` implementation below.
    """

    def get(self, api_host: str, api_path: str, fallback_path: str) -> Any | None:
        """
        Makes a GET request to a specified API and falls back to a local file on failure.

        Args:
            api_host (str): The base URL of the API.
            api_path (str): The endpoint to make the request to.
            fallback_path (str): The path to the local JSON file to use as a fallback.

        Returns:
            dict: The JSON response from the API or the content of the fallback file.
        """
        if api_host and api_path:
            try:
                response = requests.get(f"{api_host}/{api_path}")
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "GET %s/%s failed (%s); falling back to local file %s",
                    api_host, api_path, exc, fallback_path,
                )

        try:
            with open(fallback_path) as json_file:
                return json.load(json_file)
        except FileNotFoundError:
            logger.error("Fallback file not found: %s", fallback_path)
            return None
        except json.JSONDecodeError as exc:
            logger.error("Fallback file %s contains invalid JSON: %s", fallback_path, exc)
            return None
