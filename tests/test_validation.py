import unittest
import warnings

import numpy as np
import pandas as pd

from stretch import STRETCH


class ValidationTest(unittest.TestCase):
    def test_rejects_non_finite_input(self):
        values = np.arange(24, dtype=float).reshape(12, 2)
        values[3, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            STRETCH(max_lag=2).fit(values)

    def test_rejects_duplicate_dataframe_labels(self):
        frame = pd.DataFrame(np.ones((12, 2)), columns=["same", "same"])
        with self.assertRaisesRegex(ValueError, "unique"):
            STRETCH(max_lag=2).fit(frame)

    def test_warns_for_constant_variables(self):
        values = np.column_stack([np.arange(12), np.ones(12)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            STRETCH(max_lag=2).fit(values)
        self.assertTrue(any("Constant variables" in str(item.message) for item in caught))

    def test_rejects_unknown_search(self):
        with self.assertRaisesRegex(ValueError, "search must be"):
            STRETCH(search="unknown", max_lag=2).fit(np.ones((12, 2)))


if __name__ == "__main__":
    unittest.main()
