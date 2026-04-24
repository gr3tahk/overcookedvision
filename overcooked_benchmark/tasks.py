from __future__ import annotations

from typing import Any


def _delivery_reference(reference_id: str = "single_delivery") -> dict[str, Any]:
    return {
        "id": reference_id,
        "actions": [
            "pickup_onion",
            "place_onion_in_pot",
            "pickup_onion",
            "place_onion_in_pot",
            "pickup_onion",
            "place_onion_in_pot",
            "pickup_dish",
            "pickup_soup",
            "deliver_soup",
        ],
    }


TASKS: list[dict[str, Any]] = [
    {
        "task_id": "cramped_room_single_delivery",
        "layout": "cramped_room",
        "description": "Deliver one onion soup in the cramped_room layout.",
        "reference_trajectories": [
            _delivery_reference("single_agent_delivery"),
        ],
        "agent_references": {
            "0": [
                {
                    "id": "p0_all_steps",
                    "actions": [
                        "pickup_onion",
                        "place_onion_in_pot",
                        "pickup_onion",
                        "place_onion_in_pot",
                        "pickup_onion",
                        "place_onion_in_pot",
                        "pickup_dish",
                        "pickup_soup",
                        "deliver_soup",
                    ],
                },
                {
                    "id": "p0_onion_focus",
                    "actions": [
                        "pickup_onion",
                        "place_onion_in_pot",
                        "pickup_onion",
                        "place_onion_in_pot",
                    ],
                },
            ],
            "1": [
                {
                    "id": "p1_all_steps",
                    "actions": [
                        "pickup_onion",
                        "place_onion_in_pot",
                        "pickup_onion",
                        "place_onion_in_pot",
                        "pickup_onion",
                        "place_onion_in_pot",
                        "pickup_dish",
                        "pickup_soup",
                        "deliver_soup",
                    ],
                },
                {
                    "id": "p1_onion_plate_delivery",
                    "actions": [
                        "pickup_onion",
                        "place_onion_in_pot",
                        "pickup_dish",
                        "pickup_soup",
                        "deliver_soup",
                    ],
                },
            ],
        },
    },
    {
        "task_id": "cramped_room_divide_and_plate",
        "layout": "cramped_room",
        "description": (
            "Deliver one onion soup in cramped_room with a clean division of labor: "
            "one player focuses on loading onions while the teammate handles dish, soup pickup, and delivery."
        ),
        "reference_trajectories": [
            _delivery_reference("onion_loader_then_plater"),
        ],
        "agent_references": {
            "0": [
                {
                    "id": "p0_onion_loader",
                    "actions": [
                        "pickup_onion",
                        "place_onion_in_pot",
                        "pickup_onion",
                        "place_onion_in_pot",
                        "pickup_onion",
                        "place_onion_in_pot",
                    ],
                },
                {
                    "id": "p0_support_loader",
                    "actions": [
                        "pickup_onion",
                        "place_onion_in_pot",
                        "pickup_onion",
                        "place_onion_in_pot",
                    ],
                },
            ],
            "1": [
                {
                    "id": "p1_plater_server",
                    "actions": [
                        "pickup_dish",
                        "pickup_soup",
                        "deliver_soup",
                    ],
                },
                {
                    "id": "p1_one_onion_then_finish",
                    "actions": [
                        "pickup_onion",
                        "place_onion_in_pot",
                        "pickup_dish",
                        "pickup_soup",
                        "deliver_soup",
                    ],
                },
            ],
        },
    },
    {
        "task_id": "cramped_room_balanced_handoff",
        "layout": "cramped_room",
        "description": (
            "Deliver one onion soup in cramped_room using a balanced handoff: "
            "both players contribute onions, then one finishes with dish pickup and delivery."
        ),
        "reference_trajectories": [
            _delivery_reference("balanced_handoff_delivery"),
        ],
        "agent_references": {
            "0": [
                {
                    "id": "p0_two_onions_then_clear_lane",
                    "actions": [
                        "pickup_onion",
                        "place_onion_in_pot",
                        "pickup_onion",
                        "place_onion_in_pot",
                    ],
                },
                {
                    "id": "p0_one_onion_then_serve",
                    "actions": [
                        "pickup_onion",
                        "place_onion_in_pot",
                        "pickup_dish",
                        "pickup_soup",
                        "deliver_soup",
                    ],
                },
            ],
            "1": [
                {
                    "id": "p1_one_onion_then_finish",
                    "actions": [
                        "pickup_onion",
                        "place_onion_in_pot",
                        "pickup_dish",
                        "pickup_soup",
                        "deliver_soup",
                    ],
                },
                {
                    "id": "p1_two_onions_support",
                    "actions": [
                        "pickup_onion",
                        "place_onion_in_pot",
                        "pickup_onion",
                        "place_onion_in_pot",
                    ],
                },
            ],
        },
    },
]


def load_tasks() -> list[dict[str, Any]]:
    return TASKS


def list_task_ids() -> list[str]:
    return [task["task_id"] for task in TASKS]


def get_task_by_id(task_id: str) -> dict[str, Any]:
    for task in TASKS:
        if task["task_id"] == task_id:
            return task
    raise KeyError(f"Unknown task id: {task_id}")
