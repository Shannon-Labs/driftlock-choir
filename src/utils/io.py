"""Shared IO helpers for DriftLock simulations."""

from __future__ import annotations

import csv
import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, MutableMapping, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy always available in repo
    np = None  # type: ignore


_JSON_INDENT = 2


def ensure_directory(path: str | os.PathLike[str]) -> str:
    """Create *path* if it does not already exist and return it."""
    resolved = Path(path)
    if not resolved.exists():
        resolved.mkdir(parents=True, exist_ok=True)
    return str(resolved)


def save_json(data: Any, path: str | os.PathLike[str]) -> str:
    """Persist *data* to *path* as formatted JSON and return the path."""
    dest = Path(path)
    if dest.parent:
        ensure_directory(dest.parent)
    with dest.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=_JSON_INDENT, sort_keys=True, default=_json_default)
    return str(dest)


def write_csv(
    path: str | os.PathLike[str],
    fieldnames: Sequence[str],
    rows: Iterable[MutableMapping[str, Any]],
) -> str:
    """Write *rows* (with *fieldnames*) to CSV at *path* and return the path."""
    dest = Path(path)
    if dest.parent:
        ensure_directory(dest.parent)
    with dest.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(dest)


def append_csv_row(
    path: str | os.PathLike[str],
    fieldnames: Sequence[str],
    row: MutableMapping[str, Any],
) -> str:
    """Append *row* to CSV at *path*, adding a header on first write."""
    dest = Path(path)
    if dest.parent:
        ensure_directory(dest.parent)
    file_exists = dest.exists()
    with dest.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return str(dest)


def dataclass_to_dict(instance: Any) -> Dict[str, Any]:
    """Convert dataclass *instance* to a dictionary."""
    if dataclasses.is_dataclass(instance):
        return dataclasses.asdict(instance)
    raise TypeError('dataclass_to_dict expects a dataclass instance')


def echo_config(config: Any, *, label: str = 'config') -> str:
    """Return a human-readable JSON string representing ``config``."""
    payload = config
    if dataclasses.is_dataclass(config):
        payload = dataclasses.asdict(config)
    text = json.dumps(payload, indent=_JSON_INDENT, sort_keys=True, default=_json_default)
    banner = f"=== {label} ==="
    return f"{banner}\n{text}\n{'=' * len(banner)}"


def dump_run_config(
    results_dir: str | os.PathLike[str],
    config: Any,
    *,
    filename: str = 'run_config.json',
) -> str:
    """Serialize *config* (dataclass or mapping) into ``results_dir/filename``."""
    if dataclasses.is_dataclass(config):
        payload = dataclasses.asdict(config)
    elif isinstance(config, MutableMapping):
        payload = dict(config)
    else:
        raise TypeError('dump_run_config expects a dataclass or mapping')
    dest = Path(results_dir) / filename
    return save_json(payload, dest)


def _json_default(value: Any) -> Any:
    """JSON serializer that handles numpy types automatically."""
    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    if isinstance(value, complex):
        return {'real': value.real, 'imag': value.imag}
    if isinstance(value, (Path,)):
        return str(value)
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')
