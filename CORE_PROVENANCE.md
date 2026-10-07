# Numerical core provenance

The inference core was taken from the STRETCH method submodule used by the final evaluation:

- upstream at the time of evaluation: `https://projects.cispa.saarland/eda/papers/deutschebahn.git`;
- audited revision: `1cde09c81d6282b192086f622ec38dbb6999431b`;
- revision date: 2026-03-31;
- commit subject: `small changes`.

The same inference files are byte-for-byte identical at the camera-ready paper revision `c46b17056296241a2124980d6e6f155ca8a31fec` from 2026-09-29.

Packaging changes are limited to package-relative imports and a public wrapper.
The TOPIC evaluation helper was reduced to the two functions imported by the search, removing an unused dependency on `cdt`.
No scoring, fitting, graph-search, stopping, or delay-recovery rule was intentionally changed.

The original SHA-256 values and the full comparison rationale are recorded in the adjacent release audit prepared before repository construction.
