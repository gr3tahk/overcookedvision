from __future__ import annotations

from typing import Any


TASKS: list[dict[str, Any]] = [
    {
        "task_id": "cramped_room_single_delivery",
        "layout": "cramped_room",
        "description": "Deliver one onion soup in the cramped_room layout.",
        "reference_trajectories": [
            {
                "id": "single_agent_delivery",
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
    }
]


def load_tasks() -> list[dict[str, Any]]:
    return TASKS


def get_task_by_id(task_id: str) -> dict[str, Any]:
    for task in TASKS:
        if task["task_id"] == task_id:
            return task
    raise KeyError(f"Unknown task id: {task_id}")
