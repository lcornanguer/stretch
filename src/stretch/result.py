"""Structured results returned by STRETCH."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class STRETCHResult:
    """A learned summary graph and its source-time-indexed delay functions.

    ``adjacency.loc[source, target]`` is one when STRETCH learned the directed
    edge ``source -> target``. Each delay array is indexed by source time; the
    value is the learned delay to the target, and ``-1`` denotes an inactive or
    unassigned source time.
    """

    adjacency: pd.DataFrame
    delay_functions: Mapping[tuple[Hashable, Hashable], np.ndarray]
    variable_names: tuple[Hashable, ...]
    search: str
    max_lag: int

    @property
    def adjacency_array(self) -> np.ndarray:
        """Return a copy of the adjacency matrix as a NumPy array."""

        return self.adjacency.to_numpy(copy=True)

    def delays_frame(self) -> pd.DataFrame:
        """Return all delay functions in tidy, source-time-indexed form."""

        rows = []
        for (source, target), delays in self.delay_functions.items():
            rows.extend(
                {
                    "source": source,
                    "target": target,
                    "source_time": source_time,
                    "delay": int(delay),
                }
                for source_time, delay in enumerate(delays)
            )
        return pd.DataFrame(
            rows,
            columns=["source", "target", "source_time", "delay"],
        )
