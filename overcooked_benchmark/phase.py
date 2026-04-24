from __future__ import annotations


def _held_names(state) -> list[str]:
    names = []
    for player in state.to_dict()["players"]:
        held = player["held_object"]
        names.append(held["name"] if held else "nothing")
    return names


def _pot_descriptions(state, mdp) -> list[dict]:
    descriptions = []
    for pos in mdp.get_pot_locations():
        if not state.has_object(pos):
            descriptions.append({"position": pos, "count": 0, "is_cooking": False, "is_ready": False, "stage": "empty"})
            continue
        soup = state.get_object(pos)
        count = len(soup.ingredients)
        if soup.is_ready:
            stage = "ready"
        elif soup.is_cooking:
            stage = "cooking"
        elif count >= 3:
            stage = "full_needs_cooking"
        else:
            stage = "filling"
        descriptions.append(
            {
                "position": pos,
                "count": count,
                "is_cooking": soup.is_cooking,
                "is_ready": soup.is_ready,
                "stage": stage,
            }
        )
    return descriptions


def task_phase_hint(state, mdp) -> str:
    """Return a concise task-phase hint shared by LLM and VLM agents."""
    pots = _pot_descriptions(state, mdp)
    held = _held_names(state)

    ready = [pot for pot in pots if pot["is_ready"]]
    if any(item in {"soup", "soup in plate"} for item in held):
        return "Task phase: soup is being carried. The player holding soup should go to a serve tile and interact."
    if ready:
        if "dish" in held:
            return "Task phase: soup is ready and someone holds a dish. That player should face the ready pot and interact to pick up soup."
        return "Task phase: soup is ready. Someone should get a dish, then face the ready pot and interact to pick up soup."

    cooking = [pot for pot in pots if pot["is_cooking"]]
    if cooking:
        return "Task phase: soup is cooking. Wait or prepare by getting a dish, then pick up soup when ready."

    full = [pot for pot in pots if pot["count"] >= 3]
    if full:
        return "Task phase: a pot has 3 onions and is not cooking. Someone adjacent to that pot should face it and interact to start cooking."

    max_count = max((pot["count"] for pot in pots), default=0)
    onion_carriers = sum(1 for item in held if item == "onion")
    needed = max(0, 3 - max_count)
    if onion_carriers:
        return (
            f"Task phase: pot has {max_count}/3 onions and {onion_carriers} player(s) hold onion. "
            "Onion carriers should go adjacent to the pot, face it, and interact to add onion."
        )
    return (
        f"Task phase: collect onions. Best pot has {max_count}/3 onions, so {needed} more onion(s) are needed. "
        "A player should face an onion dispenser and interact to pick up onion."
    )
