# STRETCH

STRETCH is a causal discovery method for time series where the delay from a cause to its effect changes over time.
This repository contains the reusable implementation accompanying the NeurIPS 2026 paper *Causal Discovery under Time-Varying Delays* (see citation below).

## Five-minute example

Install the package, then fit a labelled Pandas DataFrame:

```console
pip install stretch-tscd
```

```python
import numpy as np
import pandas as pd

from stretch import STRETCH

rng = np.random.default_rng(7)
n = 120
temperature = rng.normal(size=n)
demand = np.zeros(n)
for t in range(2, n):
    delay = 1 if t < n // 2 else 2
    demand[t] = 0.8 * temperature[t - delay] + 0.2 * demand[t - 1]
demand += 0.1 * rng.normal(size=n)

data = pd.DataFrame(
    {"outdoor temperature": temperature, "power demand": demand}
)
result = STRETCH(search="greedy", max_lag=3).fit_result(data)

print(result.adjacency)
for edge, delays in result.delay_functions.items():
    print(edge, delays)
```

Column names are preserved in the adjacency matrix and delay-function keys.
NumPy arrays are also accepted; pass `variable_names=[...]` to assign labels.

STRETCH supports:

- `search="greedy"`, the scalable GLOBE-wrapped search used for STRETCH-GLOBE;
- `search="exhaustive"`, the complete DAG enumeration used for the main STRETCH result;
- `search="topic"`, the STRETCH-TOPIC paper variant.

Exhaustive search is intended for small variable sets and may become impractical beyond five variables.
Set `allow_cycles=True` only when the application permits a cyclic summary graph; the default remains a DAG.

## CSV command

```console
stretch observations.csv --search greedy --max-lag 6 --output-dir results
```

The command writes a labelled adjacency matrix, tidy source-time-indexed delays, exact compressed delay arrays, and run metadata.

## Citation

If you use STRETCH, please cite:

> Lénaïg Cornanguer, David Kaltenpoth, and Jilles Vreeken. “Causal Discovery under Time-Varying Delays.” Advances in Neural Information Processing Systems, 2026.

Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## License

STRETCH is released under the BSD 3-Clause License.
