from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DAMAGE_LABELS = ("intact", "damaged", "destroyed")
LABEL_TO_ID = {name: idx for idx, name in enumerate(DAMAGE_LABELS)}


@dataclass(frozen=True)
class Sample:
    """A normalized building-level paired-image example."""

    sample_id: str
    event_id: str
    tile_id: str
    pre_image: str
    post_image: str
    label: str
    label_id: int
    dataset: str
    bbox_xyxy: tuple[float, float, float, float] | None = None
    mask_path: str | None = None
    region_id: str | None = None
    question: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in DAMAGE_LABELS:
            raise ValueError(f"Unknown damage label: {self.label!r}")
        if self.label_id != LABEL_TO_ID[self.label]:
            raise ValueError(
                f"label_id={self.label_id} disagrees with label={self.label!r}"
            )
        if not self.event_id or not self.tile_id:
            raise ValueError("event_id and tile_id must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        if self.bbox_xyxy is not None:
            row["bbox_xyxy"] = list(self.bbox_xyxy)
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Sample":
        values = dict(row)
        if values.get("bbox_xyxy") is not None:
            values["bbox_xyxy"] = tuple(float(x) for x in values["bbox_xyxy"])
        values.setdefault("metadata", {})
        return cls(**values)

    def resolve_paths(self, base_dir: str | Path) -> "Sample":
        base = Path(base_dir)

        def resolve(value: str | None) -> str | None:
            if value is None:
                return None
            path = Path(value)
            return str(path if path.is_absolute() else (base / path).resolve())

        values = self.to_dict()
        values["pre_image"] = resolve(self.pre_image)
        values["post_image"] = resolve(self.post_image)
        values["mask_path"] = resolve(self.mask_path)
        return Sample.from_dict(values)
