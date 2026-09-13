"""
Service module to interact with the Kindergarten API.

This module exposes `KindergartenHttpService` which inherits from
`lib.core.integration.http.GenericHttpService.GenericHttpService`.
Module-level helper functions were removed; callers should instantiate
`KindergartenHttpService` and call instance methods (e.g. `.get_current_menu()`).
"""
from typing import Any
from datetime import date
import logging
from lib.commons.EnvironmentVariables import get_kindergarten_api_host, get_kindergarten_api_path
from lib.core.integration.http.GenericHttpService import GenericHttpService

logger = logging.getLogger(__name__)


class KindergartenHttpService(GenericHttpService):
    """Service class to interact with the Kindergarten API."""

    def get_current_menu(self) -> Any | None:
        """Return the appropriate menu depending on the current month."""
        month = date.today().month
        if month >= 11 or month <= 4:
            return self.get_winter_menu()
        return self.get_summer_menu()

    def get_winter_menu(self) -> Any | None:
        """Retrieve the winter menu from the Kindergarten API or fallback file."""
        return self.get(
            api_host=get_kindergarten_api_host(),
            api_path=get_kindergarten_api_path(),
            fallback_path='static/kindergarten-winter-menu.json'
        )

    def get_summer_menu(self) -> Any | None:
        """Retrieve the summer menu from the Kindergarten API or fallback file.

        No real summer-menu dataset exists yet (static/kindergarten-summer-menu.json
        is intentionally absent). Falls back to the winter menu as a documented
        placeholder rather than silently returning None for six months a year.
        """
        menu = self.get(
            api_host=get_kindergarten_api_host(),
            api_path=get_kindergarten_api_path(),
            fallback_path='static/kindergarten-summer-menu.json'
        )
        if menu is None:
            logger.warning(
                "Summer kindergarten menu unavailable (no API configured/reachable and "
                "static/kindergarten-summer-menu.json does not exist yet); "
                "falling back to the winter menu as a placeholder."
            )
            menu = self.get_winter_menu()
        return menu