from __future__ import annotations

from typing import Any


def _state_to_dict(state):
    return state if isinstance(state, dict) else state.to_dict()


def _held_name(player: dict[str, Any]) -> str:
    held = player.get("held_object")
    return held.get("name", "nothing") if held else "nothing"


def _pot_ingredient_count(pots: dict[str, Any]) -> int:
    count = 0
    for pot in pots.values() if isinstance(pots, dict) else pots:
        ingredients = pot.get("ingredients", [])
        count += len(ingredients)
    return count


def _event_action(event_name: str) -> str | None:
    event_actions = {
        "onion_pickup": "pickup_onion",
        "potting_onion": "place_onion_in_pot",
        "dish_pickup": "pickup_dish",
        "soup_pickup": "pickup_soup",
        "soup_delivery": "deliver_soup",
    }
    return event_actions.get(event_name)


def classify_player_action(
    before_state,
    after_state,
    player_id: int,
    event_infos: dict[str, Any] | None = None,
    action=None,
    *_,
) -> str | None:
    """Map an Overcooked transition to a symbolic action for TES/ITES scoring."""
    event_infos = event_infos or {}
    for event_name, flags in event_infos.items():
        if player_id < len(flags) and flags[player_id]:
            symbolic_action = _event_action(event_name)
            if symbolic_action:
                return symbolic_action

    before = _state_to_dict(before_state)
    after = _state_to_dict(after_state)
    before_player = before["players"][player_id]
    after_player = after["players"][player_id]
    before_held = _held_name(before_player)
    after_held = _held_name(after_player)

    if before_held == "nothing" and after_held == "onion":
        return "pickup_onion"
    if before_held == "nothing" and after_held == "dish":
        return "pickup_dish"
    if before_held == "dish" and after_held in {"soup", "soup in plate"}:
        return "pickup_soup"

    before_pots = before.get("pots", {})
    after_pots = after.get("pots", {})
    if before_held == "onion" and after_held == "nothing":
        if _pot_ingredient_count(after_pots) > _pot_ingredient_count(before_pots):
            return "place_onion_in_pot"

    return None
