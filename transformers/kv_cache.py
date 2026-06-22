from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class KVCache:
    """Minimal append-only cache for one transformer layer.

    Expected shape: (batch, heads, sequence, head_dim).
    """

    keys: np.ndarray | None = None
    values: np.ndarray | None = None

    def append(self, new_keys: np.ndarray, new_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if new_keys.shape != new_values.shape:
            raise ValueError("keys and values must have identical shapes")
        if self.keys is None:
            self.keys, self.values = new_keys.copy(), new_values.copy()
        else:
            if self.keys.shape[:2] != new_keys.shape[:2] or self.keys.shape[-1] != new_keys.shape[-1]:
                raise ValueError("batch, head count, and head dimension must stay fixed")
            self.keys = np.concatenate([self.keys, new_keys], axis=2)
            self.values = np.concatenate([self.values, new_values], axis=2)
        return self.keys, self.values

    @property
    def sequence_length(self) -> int:
        return 0 if self.keys is None else int(self.keys.shape[2])
