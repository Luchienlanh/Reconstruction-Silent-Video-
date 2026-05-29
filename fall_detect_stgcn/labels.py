"""Label definitions for the UP-Fall dataset."""

from __future__ import annotations

ACTIVITY_NAMES = {
    1: "fall_forward_hands",
    2: "fall_forward_knees",
    3: "fall_backward",
    4: "fall_sideward",
    5: "fall_sitting",
    6: "walking",
    7: "standing",
    8: "sitting",
    9: "picking_object",
    10: "jumping",
    11: "laying",
}

FALL_ACTIVITIES = {1, 2, 3, 4, 5}

MULTICLASS_LABEL_NAMES = [ACTIVITY_NAMES[i] for i in range(1, 12)]
BINARY_LABEL_NAMES = ["non_fall", "fall"]


def activity_to_label(activity: int, task: str) -> int:
    if task == "multiclass":
        if activity not in ACTIVITY_NAMES:
            raise ValueError(f"Unknown activity: {activity}")
        return activity - 1
    if task == "binary":
        return int(activity in FALL_ACTIVITIES)
    raise ValueError(f"Unknown task: {task}")


def label_names(task: str) -> list[str]:
    if task == "multiclass":
        return list(MULTICLASS_LABEL_NAMES)
    if task == "binary":
        return list(BINARY_LABEL_NAMES)
    raise ValueError(f"Unknown task: {task}")


def label_is_fall(label: int, task: str) -> bool:
    if task == "multiclass":
        return (label + 1) in FALL_ACTIVITIES
    if task == "binary":
        return label == 1
    raise ValueError(f"Unknown task: {task}")
