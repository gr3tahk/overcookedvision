from __future__ import annotations

import json

from overcooked_benchmark.agents.base import ACTION_TO_OVERCOOKED, LOW_LEVEL_ACTIONS, AgentDecision, AgentObservation, BenchmarkAgent


class ScriptedAgent(BenchmarkAgent):
    """Deterministic agent for local smoke tests of the benchmark harness."""

    def __init__(self, player_id: int, player_name: str, actions: list[str] | None = None):
        super().__init__(player_id, player_name)
        self.actions = actions or ["stay"]
        self.index = 0

    def act(self, observation: AgentObservation):
        action = self.actions[min(self.index, len(self.actions) - 1)]
        self.index += 1
        if action not in LOW_LEVEL_ACTIONS:
            action = "stay"
        self.last_decision = AgentDecision(
            player_id=self.player_id,
            player_name=self.player_name,
            action=action,
            message="",
            plan="scripted smoke test path",
            raw_response=json.dumps({"action": action, "message": "", "plan": "scripted smoke test path"}),
            valid=True,
        )
        return ACTION_TO_OVERCOOKED[action]
