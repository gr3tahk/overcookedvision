from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from overcooked_ai_py.mdp.actions import Action


LOW_LEVEL_ACTIONS = ["up", "down", "left", "right", "stay", "interact"]
ACTION_TO_OVERCOOKED = {
    "up": (0, -1),
    "down": (0, 1),
    "right": (1, 0),
    "left": (-1, 0),
    "stay": Action.STAY,
    "interact": Action.INTERACT,
}


@dataclass
class AgentDecision:
    player_id: int
    player_name: str
    action: str
    message: str = ""
    plan: str = ""
    raw_response: str = ""
    prompt: str = ""
    valid: bool = True
    invalid_reason: str | None = None

    def to_trace(self) -> dict[str, Any]:
        return {
            "playerId": self.player_id,
            "playerName": self.player_name,
            "goal": "low_level_action",
            "goalLabel": self.action,
            "previousGoal": "low_level_action",
            "previousGoalLabel": self.action,
            "changed": False,
            "action": self.action,
            "message": self.message,
            "plan": self.plan,
            "rawResponse": self.raw_response,
            "prompt": self.prompt,
            "valid": self.valid,
            "invalidReason": self.invalid_reason,
        }


@dataclass
class AgentObservation:
    state: Any
    mdp: Any
    tick: int
    score: int
    player_id: int
    task: dict[str, Any]
    teammate_message: str = ""
    current_plan: str = ""
    phase_hint: str = ""
    action_feedback: str = "No previous action yet."
    no_op_warning: str = ""
    recent_events: list[dict[str, Any]] = field(default_factory=list)


class BenchmarkAgent:
    def __init__(self, player_id: int, player_name: str):
        self.player_id = player_id
        self.player_name = player_name
        self.current_goal = "low_level_action"
        self.last_decision: AgentDecision | None = None

    def act(self, observation: AgentObservation):
        raise NotImplementedError


def parse_agent_response(text: str) -> tuple[str, str, str, bool, str | None]:
    raw = text.strip()
    payload = None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = None

    if isinstance(payload, dict):
        action = str(payload.get("action", "")).strip().lower()
        message = str(payload.get("message", "")).strip()
        plan = str(payload.get("plan", "")).strip()
    else:
        lowered = raw.lower()
        action = next((candidate for candidate in LOW_LEVEL_ACTIONS if re.search(rf"\b{candidate}\b", lowered)), "")
        message = ""
        plan = ""

    if action not in LOW_LEVEL_ACTIONS:
        return "stay", message, plan, False, f"Could not parse valid action from response: {raw[:160]}"
    return action, message, plan, True, None
