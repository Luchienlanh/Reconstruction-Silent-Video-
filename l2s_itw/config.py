from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")


def apply_overrides(config: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must look like section.key=value, got: {override}")
        key, raw_value = override.split("=", 1)
        set_by_dotted_key(updated, key, parse_scalar(raw_value))
    return updated


def set_by_dotted_key(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if not parts:
        raise ValueError("Empty config key")

    node = config
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value
