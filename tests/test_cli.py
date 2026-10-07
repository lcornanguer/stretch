import json
import tempfile
import unittest
from pathlib import Path

from stretch.cli import main

from tests.test_legacy_equivalence import fixed_case


class CommandLineTest(unittest.TestCase):
    def test_csv_command_writes_structured_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.csv"
            output_path = root / "output"
            fixed_case().to_csv(input_path, index=False)

            status = main(
                [
                    str(input_path),
                    "--search",
                    "greedy",
                    "--max-lag",
                    "2",
                    "--output-dir",
                    str(output_path),
                ]
            )

            self.assertEqual(status, 0)
            self.assertTrue((output_path / "adjacency.csv").is_file())
            self.assertTrue((output_path / "delays.csv").is_file())
            self.assertTrue((output_path / "delay-functions.npz").is_file())
            metadata = json.loads(
                (output_path / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["variables"], ["cause variable", "effect variable"])
            self.assertEqual(metadata["search"], "greedy")


if __name__ == "__main__":
    unittest.main()
