"""Public estimator API for STRETCH."""

from __future__ import annotations

import contextlib
import io
import warnings
from collections.abc import Hashable, Sequence
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from ._core.exhaustive_search import Exhaustive_search
from ._core.globe.search import dag_tree_search
from ._core.topic.topic import Topic
from .result import STRETCHResult


_SEARCH_ALIASES = {
    "greedy": "greedy",
    "globe": "greedy",
    "exhaustive": "exhaustive",
    "topic": "topic",
}


class STRETCH:
    """Discover a causal graph with source-time-varying causal delays.

    Parameters use the settings from the final STRETCH paper evaluation by
    default. ``search="greedy"`` selects the GLOBE-wrapped search;
    ``search="exhaustive"`` enumerates candidate DAGs; and ``search="topic"``
    reproduces the additional STRETCH-TOPIC paper variant.
    """

    def __init__(
        self,
        *,
        search: str = "greedy",
        max_lag: int = 6,
        max_parents: int | None = None,
        allow_cycles: bool = False,
        use_lasso: bool = False,
        init_with_child_centric: bool = False,
        pruning_threshold: float = 0.0,
        fix_previous_parents: bool = True,
        use_bic: bool = False,
        regression: str = "poly",
        verbose: bool = False,
    ) -> None:
        self.search = search
        self.max_lag = max_lag
        self.max_parents = max_parents
        self.allow_cycles = allow_cycles
        self.use_lasso = use_lasso
        self.init_with_child_centric = init_with_child_centric
        self.pruning_threshold = pruning_threshold
        self.fix_previous_parents = fix_previous_parents
        self.use_bic = use_bic
        self.regression = regression
        self.verbose = verbose

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        *,
        variable_names: Sequence[Hashable] | None = None,
    ) -> STRETCH:
        """Fit STRETCH to rows of observations and columns of variables."""

        values, names = self._validate_input(X, variable_names)
        search = self._validate_parameters(values.shape[1])
        data = {index: values[:, index] for index in range(values.shape[1])}
        global_params = {
            "use_lasso": self.use_lasso,
            "verbose_iter": self.verbose,
            "init_with_child_centric": self.init_with_child_centric,
            "max_lag": self.max_lag,
            "pruning_threshold": self.pruning_threshold,
            "fix_previous_parents": self.fix_previous_parents,
            "use_bic": self.use_bic,
            "regression": self.regression,
        }

        output = contextlib.nullcontext()
        if not self.verbose:
            output = contextlib.redirect_stdout(io.StringIO())

        with output:
            adjacency, integer_delays = self._fit_core(search, data, global_params)

        labelled_delays = {
            (names[source], names[target]): np.asarray(delays, dtype=int).copy()
            for (source, target), delays in integer_delays.items()
        }
        adjacency_frame = pd.DataFrame(
            np.asarray(adjacency, dtype=int),
            index=pd.Index(names, name="source"),
            columns=pd.Index(names, name="target"),
        )
        self.n_features_in_ = values.shape[1]
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.result_ = STRETCHResult(
            adjacency=adjacency_frame,
            delay_functions=labelled_delays,
            variable_names=names,
            search=search,
            max_lag=self.max_lag,
        )
        self.adjacency_matrix_ = self.result_.adjacency
        self.delay_functions_ = self.result_.delay_functions
        return self

    def fit_result(
        self,
        X: pd.DataFrame | np.ndarray,
        *,
        variable_names: Sequence[Hashable] | None = None,
    ) -> STRETCHResult:
        """Fit STRETCH and return the structured result directly."""

        return self.fit(X, variable_names=variable_names).result_

    def _fit_core(self, search: str, data: dict[int, np.ndarray], global_params):
        if search == "greedy":
            fitted = dag_tree_search(
                N=len(data),
                data=data,
                global_params=global_params,
                verbosity=int(self.verbose),
            )
            return fitted.get_adj(), fitted.get_delay_functions()

        if search == "exhaustive":
            fitted = Exhaustive_search(
                data=data,
                true_dag=None,
                global_params=global_params,
                max_pa=self.max_parents,
                allow_cycles=self.allow_cycles,
                verbosity=self.verbose,
            )
            graph = fitted.main()
            adjacency = nx.to_numpy_array(
                graph,
                nodelist=range(len(data)),
                dtype=int,
            )
            return adjacency, fitted.get_delay_functions(graph)

        fitted = Topic(global_params=global_params, verbosity=self.verbose)
        graph, _ = fitted.fit(data)
        adjacency = nx.to_numpy_array(
            graph,
            nodelist=range(len(data)),
            dtype=int,
        )
        return adjacency, fitted.get_delay_functions()

    def _validate_parameters(self, n_variables: int) -> str:
        try:
            search = _SEARCH_ALIASES[self.search.lower()]
        except (AttributeError, KeyError) as error:
            choices = ", ".join(sorted(_SEARCH_ALIASES))
            raise ValueError(f"search must be one of: {choices}") from error

        if not isinstance(self.max_lag, int) or isinstance(self.max_lag, bool):
            raise TypeError("max_lag must be an integer")
        if self.max_lag < 0:
            raise ValueError("max_lag must be non-negative")
        if self.max_parents is not None:
            if not isinstance(self.max_parents, int) or isinstance(self.max_parents, bool):
                raise TypeError("max_parents must be an integer or None")
            if self.max_parents < 0:
                raise ValueError("max_parents must be non-negative")
        if not isinstance(self.allow_cycles, bool):
            raise TypeError("allow_cycles must be a boolean")
        if not 0 <= self.pruning_threshold <= 1:
            raise ValueError("pruning_threshold must lie between zero and one")
        if self.regression not in {"poly", "spline", "fourier"}:
            raise ValueError("regression must be 'poly', 'spline', or 'fourier'")
        if search == "exhaustive" and n_variables > 5:
            warnings.warn(
                "Exhaustive graph search grows factorially and may be impractical "
                "for more than five variables; consider search='greedy'.",
                RuntimeWarning,
                stacklevel=3,
            )
        return search

    def _validate_input(
        self,
        X: pd.DataFrame | np.ndarray,
        variable_names: Sequence[Hashable] | None,
    ) -> tuple[np.ndarray, tuple[Hashable, ...]]:
        if isinstance(X, pd.DataFrame):
            if variable_names is not None:
                raise ValueError(
                    "variable_names is only accepted for NumPy input; DataFrame "
                    "column labels are preserved automatically"
                )
            names = tuple(X.columns)
            raw_values: Any = X.to_numpy()
        else:
            raw_values = X
            array = np.asarray(X)
            if array.ndim != 2:
                raise ValueError("X must be a two-dimensional array or DataFrame")
            if variable_names is None:
                names = tuple(range(array.shape[1]))
            else:
                names = tuple(variable_names)

        try:
            values = np.asarray(raw_values, dtype=float)
        except (TypeError, ValueError) as error:
            raise TypeError("X must contain only numeric values") from error
        if values.ndim != 2:
            raise ValueError("X must be a two-dimensional array or DataFrame")
        n_samples, n_variables = values.shape
        if n_variables < 2:
            raise ValueError("X must contain at least two variables")
        if n_samples <= self.max_lag + 1:
            raise ValueError(
                f"X needs more than max_lag + 1 ({self.max_lag + 1}) observations"
            )
        if len(names) != n_variables:
            raise ValueError("variable_names must match the number of columns in X")
        if not all(isinstance(name, Hashable) for name in names):
            raise TypeError("all variable names must be hashable")
        if len(set(names)) != len(names):
            raise ValueError("variable names must be unique")
        if not np.isfinite(values).all():
            raise ValueError("X must not contain NaN or infinite values")

        constant_columns = [names[index] for index in np.flatnonzero(np.ptp(values, axis=0) == 0)]
        if constant_columns:
            warnings.warn(
                f"Constant variables may make causal fitting ill-conditioned: {constant_columns!r}",
                RuntimeWarning,
                stacklevel=3,
            )
        return values, names
