import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from stretch import STRETCH


FIXTURE = Path(__file__).parent / "fixtures" / "legacy_expected.json"


def fixed_case() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    samples = 40
    cause = rng.normal(size=samples)
    effect = np.zeros(samples)
    for source_time in range(2, samples):
        delay = 1 if source_time < samples // 2 else 2
        effect[source_time] = 3 * cause[source_time - delay] + 0.01 * rng.normal()
    return pd.DataFrame({"cause variable": cause, "effect variable": effect})


class LegacyEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_all_paper_searches_match_legacy_outputs(self):
        frame = fixed_case()
        for search in ("greedy", "exhaustive", "topic"):
            with self.subTest(search=search):
                result = STRETCH(search=search, max_lag=2).fit_result(frame)
                expected = self.expected[search]
                np.testing.assert_array_equal(
                    result.adjacency_array,
                    np.asarray(expected["adjacency"]),
                )
                self.assertEqual(
                    set(result.delay_functions),
                    {("cause variable", "effect variable")},
                )
                np.testing.assert_array_equal(
                    result.delay_functions[("cause variable", "effect variable")],
                    np.asarray(expected["delay_functions"]["0->1"]),
                )

    def test_numpy_labels_are_preserved(self):
        frame = fixed_case()
        result = STRETCH(search="greedy", max_lag=2).fit_result(
            frame.to_numpy(),
            variable_names=("cause / α", 17),
        )
        self.assertEqual(result.variable_names, ("cause / α", 17))
        self.assertEqual(list(result.adjacency.index), ["cause / α", 17])
        self.assertIn(("cause / α", 17), result.delay_functions)


if __name__ == "__main__":
    unittest.main()
