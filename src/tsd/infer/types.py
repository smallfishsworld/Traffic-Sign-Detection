"""Detection results."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Detections:
    """Detections for one image: ``xyxy`` boxes in original-image pixels, scores and class ids."""

    boxes: NDArray[np.float32]
    scores: NDArray[np.float32]
    class_ids: NDArray[np.int64]

    def __post_init__(self) -> None:
        n = len(self.scores)
        if self.boxes.shape != (n, 4) or self.class_ids.shape != (n,):
            raise ValueError("boxes, scores and class_ids must describe the same detections")

    def __len__(self) -> int:
        return len(self.scores)

    @classmethod
    def empty(cls) -> Detections:
        return cls(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))

    @classmethod
    def concatenate(cls, parts: Sequence[Detections]) -> Detections:
        if not parts:
            return cls.empty()
        return cls(
            np.concatenate([p.boxes for p in parts]).astype(np.float32),
            np.concatenate([p.scores for p in parts]).astype(np.float32),
            np.concatenate([p.class_ids for p in parts]).astype(np.int64),
        )

    def select(self, mask: NDArray[np.bool_] | NDArray[np.int64]) -> Detections:
        return Detections(self.boxes[mask], self.scores[mask], self.class_ids[mask])

    def to_dicts(self, names: Sequence[str]) -> list[dict[str, Any]]:
        """JSON-friendly list, highest score first."""
        order = np.argsort(-self.scores, kind="stable")
        return [
            {
                "class_id": int(self.class_ids[i]),
                "class_name": names[int(self.class_ids[i])],
                "score": round(float(self.scores[i]), 5),
                "box_xyxy": [round(float(v), 2) for v in self.boxes[i]],
            }
            for i in order
        ]
