import unittest
import numpy as np

from reference import (
    causal_mask, coding_rate, crate_layer, group_snr, ista, mssa,
    orthogonal_subspaces, theorem_sample, looped_crate, causal_mssa_cached,
    exact_coding_rate_gradient, neumann_gradient,
    tssa, dense_attention, causal_mssa_incremental,
)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.Z = rng.normal(size=(8, 12))
        self.U = orthogonal_subspaces(8, 2, 2, rng)
        self.D = np.eye(8)

    def test_coding_rate_matches_direct_slogdet(self):
        eps = 0.75
        expected = 0.5 * np.linalg.slogdet(np.eye(8) + 8 / (12 * eps**2) * self.Z @ self.Z.T)[1]
        self.assertAlmostEqual(coding_rate(self.Z, eps), expected, places=12)

    def test_gradient_and_neumann_diagnostics(self):
        g, gd = exact_coding_rate_gradient(self.Z, self.U, 10.0)
        n, nd = neumann_gradient(self.Z, self.U, 10.0)
        self.assertEqual(g.shape, self.Z.shape)
        self.assertTrue(gd["finite"] and nd["finite"])
        self.assertTrue(all(x < 1 for x in nd["A_norms"]))
        self.assertTrue(all(x <= b + 1e-12 for x, b in zip(nd["truncation_error_norms"], nd["remainder_bounds"])))

    def test_mssa_shape_and_rows(self):
        update, diag = mssa(self.Z, self.U, 0.75, 0.1)
        self.assertEqual(update.shape, self.Z.shape)
        self.assertTrue(diag["finite"])
        self.assertLess(diag["row_sum_error"], 1e-14)

    def test_tssa_and_dense_control(self):
        update, diag = tssa(self.Z, self.U, .75, .1, .5)
        dense, ddiag = dense_attention(self.Z, self.U, .75)
        self.assertEqual(update.shape, self.Z.shape)
        self.assertEqual(dense.shape, self.Z.shape)
        self.assertTrue(diag["finite"] and ddiag["finite"])
        self.assertLess(diag["assignment_row_error"], 1e-14)

    def test_causal_mask(self):
        _, diag = mssa(self.Z, self.U, 0.75, 0.1, causal_mask(12))
        self.assertTrue(diag["causal"])

    def test_causal_prefix_is_invariant_to_future_tokens(self):
        z1 = self.Z[:, :8]
        z2 = self.Z.copy()
        z1_padded = np.concatenate([z1, np.zeros((8, 4))], axis=1)
        a, _ = mssa(z1_padded, self.U, .75, .1, causal_mask(12))
        b, _ = mssa(z2, self.U, .75, .1, causal_mask(12))
        np.testing.assert_allclose(a[:, :8], b[:, :8], rtol=1e-12, atol=1e-12)

    def test_causal_cached_path_matches_full_mask(self):
        full, _ = mssa(self.Z, self.U, .75, .1, causal_mask(12))
        cached, diag = causal_mssa_cached(self.Z, self.U, .75, .1)
        np.testing.assert_allclose(full, cached, rtol=1e-12, atol=1e-12)
        self.assertTrue(diag["finite"])
        incremental, idiag = causal_mssa_incremental(self.Z, self.U, .75, .1)
        np.testing.assert_allclose(full, incremental, rtol=1e-12, atol=1e-12)
        self.assertTrue(idiag["cache_used"])

    def test_ista_identity_soft_threshold(self):
        H = np.array([[2.0, -0.4], [0.1, -3.0]])
        out, _ = ista(H, np.eye(2), 0.5, 0.2, nonnegative=False)
        np.testing.assert_allclose(out, np.array([[1.9, -0.3], [0.0, -2.9]]))

    def test_layer(self):
        out, diag = crate_layer(self.Z, {"U": self.U, "D": self.D, "epsilon": .75,
            "kappa": .1, "eta": .1, "lambda_": .01, "nonnegative": True})
        self.assertEqual(out.shape, self.Z.shape)
        self.assertTrue(diag["finite"])

    def test_loop_receipt_fields(self):
        spec = {"R": 2, "seed": 3, "layers": [{"U": self.U, "D": self.D,
            "epsilon": .75, "kappa": .01, "eta": .1, "lambda_": .01,
            "nonnegative": True}]}
        result = looped_crate(self.Z, spec)
        for key in ("run_id", "source_hashes", "config_hash", "R", "status",
                    "stop_reason", "coding_rate", "sparsity", "norms", "receipt"):
            self.assertIn(key, result)

    def test_theorem_sample(self):
        X, U, groups = theorem_sample(16, 2, 4, 16, .1, .75, 0)
        self.assertEqual(X.shape, (16, 16))
        self.assertEqual(len(U), 2)
        self.assertTrue(np.isfinite(group_snr(X, U, groups)))

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            coding_rate(np.ones(3), 1.0)


if __name__ == "__main__":
    unittest.main()
