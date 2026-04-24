from __future__ import annotations

from overcooked_benchmark.agents.base import AgentObservation, LOW_LEVEL_ACTIONS


def describe_map(mdp) -> str:
    legend = "X=counter/wall, space=floor, O=onion dispenser, D=dish dispenser, P=pot, S=serve"
    rows = []
    for y in range(mdp.height):
        row = "".join(mdp.get_terrain_type_at_pos((x, y)) for x in range(mdp.width))
        rows.append(f"y={y}: {row}")
    return legend + "\n" + "\n".join(rows)


def describe_facing_tile(state, mdp, player_id: int) -> str:
    player = state.to_dict()["players"][player_id]
    x, y = player["position"]
    dx, dy = player["orientation"]
    facing = (x + dx, y + dy)
    fx, fy = facing
    if fx < 0 or fy < 0 or fx >= mdp.width or fy >= mdp.height:
        return f"facing {facing}: out_of_bounds"
    tile = mdp.get_terrain_type_at_pos(facing)
    label = {
        " ": "floor",
        "X": "counter/wall",
        "O": "onion dispenser",
        "D": "dish dispenser",
        "P": "pot",
        "S": "serve",
    }.get(tile, tile)
    if state.has_object(facing):
        obj = state.get_object(facing)
        label += f" holding_station_object={getattr(obj, 'name', obj)}"
    return f"facing {facing}: {label}"


def describe_state(state, mdp, player_id: int) -> str:
    state_dict = state.to_dict()
    lines = []
    for index, player in enumerate(state_dict["players"]):
        held = player["held_object"]["name"] if player["held_object"] else "nothing"
        role = "You" if index == player_id else f"Player {index}"
        lines.append(
            f"{role}: position={tuple(player['position'])}, facing={tuple(player['orientation'])}, holding={held}"
        )

    pot_lines = []
    for pot_pos in mdp.get_pot_locations():
        if state.has_object(pot_pos):
            soup = state.get_object(pot_pos)
            ingredients = [getattr(ingredient, "name", ingredient) for ingredient in soup.ingredients]
            if soup.is_ready:
                stage = "ready"
            elif soup.is_cooking:
                stage = "cooking"
            elif len(ingredients) >= 3:
                stage = "full_needs_interact"
            else:
                stage = "filling"
            pot_lines.append(f"pot@{pot_pos}: {stage}, ingredients={ingredients}")
        else:
            pot_lines.append(f"pot@{pot_pos}: empty")

    objects = [f"{obj['name']}@{tuple(obj['position'])}" for obj in state_dict["objects"]]
    lines.extend(
        [
            "map:\n" + describe_map(mdp),
            "your facing tile: " + describe_facing_tile(state, mdp, player_id),
            f"onion dispensers: {list(mdp.get_onion_dispenser_locations())}",
            f"dish dispensers: {list(mdp.get_dish_dispenser_locations())}",
            f"serving locations: {list(mdp.get_serving_locations())}",
            "pots: " + "; ".join(pot_lines),
            "loose counter objects: " + (", ".join(objects) if objects else "none"),
        ]
    )
    return "\n".join(lines)


def build_action_prompt(observation: AgentObservation, include_text_state: bool) -> str:
    task = observation.task
    player_name = "Alice" if observation.player_id == 0 else "Bob"
    other_name = "Bob" if observation.player_id == 0 else "Alice"
    state_block = describe_state(observation.state, observation.mdp, observation.player_id) if include_text_state else ""
    text_state_instruction = (
        f"\nCurrent symbolic state:\n{state_block}\n"
        if include_text_state
        else "\nUse the board image as your state observation. Do not assume hidden state beyond the image.\n"
    )
    teammate = observation.teammate_message or "[none]"
    return f"""You are {player_name} (player {observation.player_id}) in a two-player Overcooked benchmark.
Task: {task.get('description', 'deliver onion soup')}.
Teammate: {other_name}.
Teammate's latest message: {teammate}
Shared task-phase hint: {observation.phase_hint}
Your current plan from last turn: {observation.current_plan or "[none]"}
Your previous action feedback: {observation.action_feedback}
Repeated no-op warning: {observation.no_op_warning or "[none]"}
{text_state_instruction}
Rules:
- Need 3 onions in a pot, then interact to cook if needed.
- When soup is ready, pick up a dish, interact with the pot to pick up soup, then deliver at the serve tile.
- If your facing tile is the station you need and `interact` is legal, choose `interact`.
- If your previous action was a no-op, do not repeat it unless the state changed to make it useful.
- Maintain a short plan across turns. Update it when feedback shows the plan is wrong or incomplete.
- If you see a repeated no-op warning, change position or orientation before trying the same interaction again.
- To use a station, stand on an adjacent floor tile and face it. Moving into a station tile itself is impossible.
- In `cramped_room`, useful adjacent floor tiles are: onion dispensers from (1,1) facing west or (3,1) facing east; pot from (2,1) facing north; dish from (1,2) facing south; serve from (3,2) facing south.
- You and your teammate act at the same time. Avoid blocking each other.
- Valid actions are exactly: {', '.join(LOW_LEVEL_ACTIONS)}.
- Respond with JSON only: {{"action":"up|down|left|right|stay|interact","message":"short optional teammate message","plan":"short current plan"}}.
Choose the single best next low-level action."""
