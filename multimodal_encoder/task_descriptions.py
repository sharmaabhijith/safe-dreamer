"""Task text registry backed by dmc_task_texts.json.

Loads the pre-generated pool of 100 diverse text descriptions per task and
provides random sampling so each training episode sees a different text.
"""

import json
import random
from pathlib import Path


_TEXTS_FILE = Path(__file__).parent / "dmc_task_texts.json"
_TASK_TEXTS: dict | None = None


def _load_task_texts() -> dict:
    """Load and cache the task texts JSON (lazy, once)."""
    global _TASK_TEXTS
    if _TASK_TEXTS is None:
        with open(_TEXTS_FILE) as f:
            data = json.load(f)
        _TASK_TEXTS = data["tasks"]
    return _TASK_TEXTS


def _parse_task_name(task_name: str) -> str:
    """Normalise an env name like 'dmc_walker_walk' -> 'walker_walk'.

    Also handles compound tasks like 'finger_turn_easy', 'cartpole_balance_sparse'.
    Returns the key used in dmc_task_texts.json (e.g. 'walker_walk').
    """
    if task_name.startswith("dmc_"):
        task_name = task_name[4:]
    return task_name


def get_task_texts(task_name: str) -> list[str]:
    """Return the full list of ~100 text descriptions for *task_name*.

    Args:
        task_name: e.g. 'walker_walk' or 'dmc_walker_walk'
    """
    key = _parse_task_name(task_name)
    texts = _load_task_texts()
    if key not in texts:
        raise KeyError(
            f"No texts found for task '{key}'. "
            f"Available: {sorted(texts.keys())}"
        )
    return [entry["text"] for entry in texts[key]["texts"]]


def sample_task_text(task_name: str) -> str:
    """Randomly sample one text description for *task_name*."""
    return random.choice(get_task_texts(task_name))
