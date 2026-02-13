from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import keyboard

from autopilot_bot.utils import monotonic_ms


@dataclass
class ActionDecision:
    action: str
    confidence: float


class InputController:
    def __init__(self, action_keys: Dict[str, str], hold_ms: int = 70, anti_spam_ms: int = 120) -> None:
        self.action_keys = action_keys
        self.hold_ms = hold_ms
        self.anti_spam_ms = anti_spam_ms
        self.last_sent_ms = 0
        self.last_key: Optional[str] = None

    def can_send(self) -> bool:
        return (monotonic_ms() - self.last_sent_ms) >= self.anti_spam_ms

    def send_action(self, action: str) -> bool:
        key = self.action_keys.get(action, "")
        if not key:
            return False
        if not self.can_send():
            return False
        keyboard.press(key)
        keyboard.release(key)
        self.last_sent_ms = monotonic_ms()
        self.last_key = key
        return True
