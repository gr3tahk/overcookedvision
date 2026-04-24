from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from overcooked_benchmark.config import ACTION_LABELS, EVENT_MESSAGES, PLAYER_NAMES, TERRAIN_THEME


def serialize_position(pos) -> list[int]:
    return [int(pos[0]), int(pos[1])]


def serialize_object(obj: dict[str, Any] | None):
    if not obj:
        return None
    payload = {"name": obj["name"], "position": serialize_position(obj["position"])}
    if "ingredients" in obj:
        payload["ingredients"] = list(obj["ingredients"])
    if "is_cooking" in obj:
        payload["is_cooking"] = bool(obj["is_cooking"])
    if "is_ready" in obj:
        payload["is_ready"] = bool(obj["is_ready"])
    return payload


def serialize_action(action) -> str:
    return ACTION_LABELS.get(action, str(action))


def build_layout_snapshot(mdp):
    terrain = []
    for y in range(mdp.height):
        row = []
        for x in range(mdp.width):
            tile = mdp.get_terrain_type_at_pos((x, y))
            row.append(
                {
                    "terrain": tile,
                    "theme": TERRAIN_THEME.get(tile, "counter"),
                    "position": [x, y],
                }
            )
        terrain.append(row)

    stations = []
    for station_type, positions in [
        ("pot", mdp.get_pot_locations()),
        ("onion", mdp.get_onion_dispenser_locations()),
        ("dish", mdp.get_dish_dispenser_locations()),
        ("serve", mdp.get_serving_locations()),
    ]:
        for pos in positions:
            stations.append({"type": station_type, "position": serialize_position(pos)})

    return {"width": mdp.width, "height": mdp.height, "terrain": terrain, "stations": stations}


def build_pot_snapshots(state, mdp):
    pots = []
    for pot_pos in mdp.get_pot_locations():
        pot = {"position": serialize_position(pot_pos), "stage": "empty", "ingredients": [], "ingredient_count": 0}
        if state.has_object(pot_pos):
            soup = state.get_object(pot_pos)
            ingredient_names = [getattr(ingredient, "name", ingredient) for ingredient in soup.ingredients]
            pot["ingredients"] = ingredient_names
            pot["ingredient_count"] = len(ingredient_names)
            if soup.is_ready:
                pot["stage"] = "ready"
            elif soup.is_cooking:
                pot["stage"] = "cooking"
            elif len(ingredient_names) >= 3:
                pot["stage"] = "full"
            else:
                pot["stage"] = "filling"
        pots.append(pot)
    return pots


def build_counter_objects(state):
    return [serialize_object(obj) for obj in state.to_dict()["objects"]]


def build_frame(state, mdp, agents, tick: int, score: int):
    state_dict = state.to_dict()
    players = []
    for player_id, player in enumerate(state_dict["players"]):
        agent = agents[player_id]
        players.append(
            {
                "id": player_id,
                "name": PLAYER_NAMES[player_id],
                "position": serialize_position(player["position"]),
                "orientation": serialize_position(player["orientation"]),
                "held_object": serialize_object(player["held_object"]),
                "goal": agent.current_goal,
                "goal_label": agent.current_goal,
            }
        )

    return {
        "tick": tick,
        "score": score,
        "players": players,
        "pots": build_pot_snapshots(state, mdp),
        "counter_objects": build_counter_objects(state),
    }


def build_event_entries(event_infos):
    events = []
    for event_name, flags in event_infos.items():
        for player_id, happened in enumerate(flags):
            if happened:
                events.append(
                    {
                        "type": event_name,
                        "playerId": player_id,
                        "playerName": PLAYER_NAMES[player_id],
                        "message": EVENT_MESSAGES.get(event_name, event_name.replace("_", " ")),
                    }
                )
    return events


def build_headline(events, score_delta):
    if score_delta > 0:
        return f"Soup served! Team score +{score_delta}"
    if events:
        first = events[0]
        return f"{first['playerName']} {first['message']}"
    return "The team keeps moving through the kitchen."


def save_trajectory(trajectory, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(trajectory, handle, indent=2)
