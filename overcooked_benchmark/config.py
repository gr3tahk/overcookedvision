from __future__ import annotations

from pathlib import Path

from overcooked_ai_py.mdp.actions import Action


LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit_o_1order",
]
DEFAULT_MAX_TICKS = 400
DEFAULT_TRIALS = 3
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
DEFAULT_VISION_MODEL = "gpt-4o"
DEFAULT_TRACE_OUTPUT = Path("replay-ui/public/traces/latest.json")

PLAYER_NAMES = ["Alice", "Bob"]
ACTION_LABELS = {
    (0, -1): "up",
    (0, 1): "down",
    (1, 0): "right",
    (-1, 0): "left",
    Action.STAY: "stay",
    Action.INTERACT: "interact",
}
EVENT_MESSAGES = {
    "onion_pickup": "picked up an onion",
    "onion_drop": "set down an onion",
    "potting_onion": "added an onion to the pot",
    "dish_pickup": "grabbed a plate",
    "dish_drop": "set down a plate",
    "soup_pickup": "picked up soup",
    "soup_delivery": "served a soup",
    "soup_drop": "set down soup",
    "tomato_pickup": "picked up a tomato",
    "tomato_drop": "set down a tomato",
    "potting_tomato": "added a tomato to the pot",
}
TERRAIN_THEME = {
    " ": "path",
    "X": "counter",
    "O": "onion",
    "D": "dish",
    "P": "pot",
    "S": "serve",
}
