"""ODM solver-route comparison covers numerical equivalence and the executable
six-row command-line report.
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from scripts.fracture_analysis import compare_odm_solver_routes


class TestCompareOdmSolverRoutesScript(unittest.TestCase):
    def test_comparison_retains_equivalent_full_facade_results(self):
        rows = compare_odm_solver_routes._compare_solver_routes()

        self.assertEqual(len(rows), 6)
        expected_results = (
            ("williams_xy", "direct", 6),
            ("williams_xy", "iterative", 6),
            ("williams_xy", "legacy", 6),
            ("williams_z", "direct", 3),
            ("williams_z", "iterative", 3),
            ("williams_z", "legacy", 3),
        )
        required_result_fields = {
            "x",
            "fun",
            "cost",
            "jac",
            "solver",
            "rank",
            "singular_values",
            "success",
            "message",
            "status",
            "nfev",
            "njev",
        }
        for row, (fit, solver, coefficient_count) in zip(rows, expected_results, strict=True):
            with self.subTest(fit=fit, solver=solver):
                self.assertEqual((row.fit, row.solver), (fit, solver))
                self.assertTrue(row.result.success)
                self.assertEqual(row.result.x.size, coefficient_count)
                self.assertTrue(required_result_fields.issubset(row.result.keys()))
                self.assertTrue(np.isfinite(row.elapsed_ms))
                self.assertGreaterEqual(row.elapsed_ms, 0.0)

        for offset in (0, 3):
            direct = rows[offset]
            self.assertEqual(direct.result.rank, direct.result.x.size)
            for candidate in rows[offset + 1:offset + 3]:
                np.testing.assert_allclose(candidate.result.x, direct.result.x, rtol=1e-4, atol=1e-6)
                np.testing.assert_allclose(candidate.result.fun, direct.result.fun, rtol=1e-6, atol=1e-9)
                np.testing.assert_allclose(candidate.result.cost, direct.result.cost, rtol=0.0, atol=1e-12)

    def test_script_prints_stable_six_row_table(self):
        repository_root = Path(__file__).resolve().parents[3]

        completed = subprocess.run(
            [sys.executable, "-m", "scripts.fracture_analysis.compare_odm_solver_routes"],
            cwd=repository_root,
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        lines = completed.stdout.strip().splitlines()
        self.assertEqual(
            lines[0],
            "fit\tsolver\tsuccess\tcost\telapsed_ms\tmax_abs_coefficient_delta\tmax_abs_residual_delta\tcoefficients",
        )
        self.assertEqual(len(lines), 7)
        expected_rows = (
            ("williams_xy", "direct"),
            ("williams_xy", "iterative"),
            ("williams_xy", "legacy"),
            ("williams_z", "direct"),
            ("williams_z", "iterative"),
            ("williams_z", "legacy"),
        )
        for line, (fit, solver) in zip(lines[1:], expected_rows, strict=True):
            fields = line.split("\t")
            self.assertEqual(len(fields), 8)
            self.assertEqual(fields[:2], [fit, solver])
            self.assertEqual(fields[2], "True")
            for value in fields[3:7]:
                self.assertTrue(np.isfinite(float(value)))
            self.assertGreaterEqual(float(fields[4]), 0.0)
            self.assertIsInstance(json.loads(fields[7]), list)


if __name__ == "__main__":
    unittest.main()
