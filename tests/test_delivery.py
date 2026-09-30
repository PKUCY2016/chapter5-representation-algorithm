import json
import pathlib
import unittest

from reference import ALLOWED_STATUS, looped_crate, orthogonal_subspaces
import numpy as np


class DeliveryReceiptTests(unittest.TestCase):
    def test_run_result_schema_and_status(self):
        rng = np.random.default_rng(8)
        U = orthogonal_subspaces(8, 2, 2, rng)
        spec = {"R": 1, "seed": 8, "task_id": "delivery_schema", "layers": [{
            "U": U, "D": np.eye(8), "epsilon": .75, "kappa": .01,
            "eta": .05, "lambda_": .01, "nonnegative": True}],
            "include_state": True}
        result = looped_crate(rng.normal(size=(8, 4)), spec)
        required = {"run_id", "source_hashes", "config_hash", "seed", "task_id", "R",
                    "weights_mode", "dtype", "device", "budget", "started_at",
                    "finished_at", "status", "stop_reason", "coding_rate", "sparsity",
                    "SNR", "norms", "operator_residual", "wall_seconds", "peak_memory",
                    "FLOPs", "compile_seconds", "receipt"}
        self.assertTrue(required.issubset(result))
        self.assertIn(result["status"], ALLOWED_STATUS)
        self.assertEqual(len(result["receipt"]["final_state"]), 8)

    def test_all_round_evidence_is_json(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        records = sorted((root / "rounds").glob("*/evidence.json"))
        self.assertGreaterEqual(len(records), 4)
        for path in records:
            data = json.loads(path.read_text())
            self.assertIn("status", data)
