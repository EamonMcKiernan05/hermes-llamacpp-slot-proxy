from dataclasses import dataclass
from typing import Any

from slot_proxy.config import settings


@dataclass
class SessionSignal:
    is_new_session: bool
    reason: str
    current_message_count: int = 0


class NewSessionDetector:
    """Detect when Hermes starts a logically new session.

    Heuristic: if the current request's first message has role "system" AND
    the message count is less than half the previous count, it's a /new.
    Also catches the reset-after-N-messages pattern: if the previous request
    had >max_observed_messages and the current one has ≤3.
    """

    def __init__(self) -> None:
        self._previous_message_count = 0
        self._previous_first_role: str | None = None
        self._consecutive_no_match = 0

    def inspect_chat_request(self, body: dict[str, Any]) -> SessionSignal:
        messages = body.get("messages") or []
        current_count = len(messages)
        first_role = messages[0].get("role") if messages else None

        is_new = self._heuristic_match(current_count, first_role)
        reason = self._build_reason(current_count, first_role, is_new)

        self._previous_message_count = current_count
        self._previous_first_role = first_role
        if not is_new:
            self._consecutive_no_match += 1
        else:
            self._consecutive_no_match = 0

        return SessionSignal(
            is_new_session=is_new,
            reason=reason,
            current_message_count=current_count,
        )

    def _heuristic_match(self, current_count: int, first_role: str | None) -> bool:
        if settings.detect_mode != "auto":
            return False

        # Rule 1: both start with system, current count dropped substantially
        if self._previous_first_role == "system" and first_role == "system":
            if current_count <= self._previous_message_count * 0.5:
                return True

        # Rule 2: previous was large, current is tiny (resume from 1-3)
        if self._previous_message_count > settings.max_observed_messages:
            if current_count <= 3:
                return True

        return False

    def _build_reason(
        self, current_count: int, first_role: str | None, matched: bool
    ) -> str:
        if not matched:
            return (
                f"no-match prev={self._previous_message_count}"
                f" now={current_count} first={first_role}"
            )
        return (
            f"matched prev={self._previous_message_count} now={current_count}"
            f" first={first_role}"
        )
