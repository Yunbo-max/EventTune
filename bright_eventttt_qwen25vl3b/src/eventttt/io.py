from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Iterator

from .schemas import Sample


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def read_samples(path: str | Path, resolve: bool = True) -> list[Sample]:
    manifest = Path(path).resolve()
    samples = [Sample.from_dict(row) for row in iter_jsonl(manifest)]
    if resolve:
        samples = [sample.resolve_paths(manifest.parent) for sample in samples]
    return samples


def write_samples(path: str | Path, samples: Iterable[Sample]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            row = sample.to_dict()
            for key in ("pre_image", "post_image", "mask_path"):
                value = row.get(key)
                if value is not None and Path(value).is_absolute():
                    row[key] = os.path.relpath(value, start=output.parent.resolve())
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: str | Path, value: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
