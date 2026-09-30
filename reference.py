"""Formula-level NumPy reference for the Chapter 5 representation operators.

The module intentionally stays deterministic and framework-free.  It is a
reference implementation, not a trained model or a convergence claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ALLOWED_STATUS = {
    "SOURCE_READ", "SOURCE_SPECIFIED", "OPERATOR_IMPL",
    "SYNTHETIC_CONDITIONAL", "CONDITIONAL_APPROX", "DERIVED_PROTOTYPE",
    "PASS", "FAIL_CLOSED", "UPSTREAM_BLOCKED", "UNKNOWN", "NOT_RUN",
    "NOT_EVALUATED", "STALE",
}


def _matrix(name: str, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains NaN/Inf")
    return x


def _heads(U: Any) -> list[np.ndarray]:
    if isinstance(U, np.ndarray) and U.ndim == 3:
        heads = [U[i] for i in range(U.shape[0])]
    else:
        heads = list(U)
    if not heads:
        raise ValueError("U must contain at least one head")
    out = []
    for i, head in enumerate(heads):
        head = _matrix(f"U[{i}]", head)
        if head.shape[0] == 0 or head.shape[1] == 0:
            raise ValueError("U heads must be non-empty")
        out.append(head)
    return out


def _softmax_rows(logits: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != logits.shape:
            raise ValueError("mask shape must match attention logits")
        logits = np.where(mask, logits, -np.inf)
    maxes = np.max(logits, axis=1, keepdims=True)
    if np.any(~np.isfinite(maxes)):
        raise ValueError("a softmax row has no allowed entries")
    exps = np.exp(logits - maxes)
    exps = np.where(np.isfinite(exps), exps, 0.0)
    denom = exps.sum(axis=1, keepdims=True)
    return exps / denom


def causal_mask(n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("sequence length must be positive")
    return np.tril(np.ones((n, n), dtype=bool))


def coding_rate(Z: np.ndarray, epsilon: float) -> float:
    """0.5 log det(I + d/(N eps^2) Z Z^T), for Z shaped d x N."""
    Z = _matrix("Z", Z)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    d, n = Z.shape
    sign, value = np.linalg.slogdet(np.eye(d) + d / (n * epsilon**2) * (Z @ Z.T))
    if sign <= 0 or not np.isfinite(value):
        raise FloatingPointError("coding-rate log determinant is not finite")
    return float(0.5 * value)


def subspace_coding_rate(Z: np.ndarray, U: Any, epsilon: float) -> float:
    Z = _matrix("Z", Z)
    total = 0.0
    for head in _heads(U):
        if head.shape[0] != Z.shape[0]:
            raise ValueError("U and Z dimensions disagree")
        p, n = head.shape[1], Z.shape[1]
        projected = head.T @ Z
        sign, value = np.linalg.slogdet(
            np.eye(p) + p / (n * epsilon**2) * (projected @ projected.T)
        )
        if sign <= 0 or not np.isfinite(value):
            raise FloatingPointError("subspace coding rate is not finite")
        total += 0.5 * value
    return float(total)


def exact_coding_rate_gradient(Z: np.ndarray, U: Any, epsilon: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Exact gradient of the Chapter 5 subspace coding rate."""
    Z = _matrix("Z", Z)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    n = Z.shape[1]
    grad = np.zeros_like(Z)
    norms = []
    for head in _heads(U):
        if head.shape[0] != Z.shape[0]:
            raise ValueError("U and Z dimensions disagree")
        p = head.shape[1]
        y = head.T @ Z
        alpha = p / (n * epsilon**2)
        A = alpha * (y.T @ y)
        grad += alpha * head @ y @ np.linalg.inv(np.eye(n) + A)
        norms.append(float(np.linalg.norm(A, 2)))
    return grad, {"A_norms": norms, "finite": bool(np.all(np.isfinite(grad)))}


def neumann_gradient(Z: np.ndarray, U: Any, epsilon: float) -> tuple[np.ndarray, dict[str, Any]]:
    """First-order Neumann approximation and exact truncation diagnostics."""
    Z = _matrix("Z", Z)
    n = Z.shape[1]
    approx = np.zeros_like(Z); exact = np.zeros_like(Z); norms=[]; errors=[]; bounds=[]
    for head in _heads(U):
        p = head.shape[1]; y = head.T @ Z; alpha = p / (n * epsilon**2); A = alpha * y.T @ y
        norm = float(np.linalg.norm(A, 2)); norms.append(norm)
        inv = np.linalg.inv(np.eye(n) + A)
        exact += alpha * head @ y @ inv
        approx += alpha * head @ y @ (np.eye(n) - A)
        errors.append(float(np.linalg.norm(alpha * head @ y @ (inv - (np.eye(n) - A)))))
        bounds.append(float(norm**2 / (1.0 - norm)) if norm < 1.0 else None)
    return approx, {"A_norms": norms, "truncation_error_norms": errors, "remainder_bounds": bounds, "exact_gradient": exact, "finite": bool(np.all(np.isfinite(approx)))}


def mssa(
    Z: np.ndarray,
    U: Any,
    epsilon: float,
    kappa: float,
    mask: np.ndarray | None = None,
    normalization_n: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute the Chapter 5 MSSA update and diagnostics.

    Columns are tokens.  For a causal mask, output token i attends only to
    source tokens j <= i.  ``kappa`` is retained as the first-order update
    multiplier, while the unscaled update is also reported for auditability.
    """
    Z = _matrix("Z", Z)
    if epsilon <= 0 or kappa < 0:
        raise ValueError("epsilon must be positive and kappa nonnegative")
    heads = _heads(U)
    n = Z.shape[1]
    n_scale = n if normalization_n is None else int(normalization_n)
    if n_scale < n or n_scale < 1:
        raise ValueError("normalization_n must be at least the current sequence length")
    if mask is not None and np.asarray(mask).shape != (n, n):
        raise ValueError("mask must be N x N")
    update = np.zeros_like(Z)
    norms: list[float] = []
    row_errors: list[float] = []
    for head in heads:
        if head.shape[0] != Z.shape[0]:
            raise ValueError("U and Z dimensions disagree")
        p = head.shape[1]
        projected = head.T @ Z
        logits = projected.T @ projected
        weights = _softmax_rows(logits, mask)
        # output[:, i] = U @ projected @ weights[i, :].
        update += (p / (n_scale * epsilon**2)) * (head @ projected @ weights.T)
        norms.append(float(np.linalg.norm(logits, 2)))
        row_errors.append(float(np.max(np.abs(weights.sum(axis=1) - 1.0))))
    scaled = kappa * update
    diag = {
        "A_norms": norms,
        "neumann_remainder_proxy": [x * x for x in norms],
        "row_sum_error": max(row_errors),
        "causal": mask is not None,
        "finite": bool(np.all(np.isfinite(scaled))),
        "update_norm": float(np.linalg.norm(scaled)),
        "epsilon": float(epsilon),
        "kappa": float(kappa),
    }
    return scaled, diag


def causal_mssa_cached(
    Z: np.ndarray, U: Any, epsilon: float, kappa: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reference causal path with an explicit prefix cache.

    The cache stores projected prefixes ``U_k.T @ Z[:, :i]``.  This is a
    correctness reference (prefixes are recomputed once per token), not a
    claimed production KV-cache optimization.
    """
    Z = _matrix("Z", Z)
    heads = _heads(U)
    outputs = np.zeros_like(Z)
    projected_cache = [head.T @ Z for head in heads]
    for i in range(Z.shape[1]):
        prefix = Z[:, : i + 1]
        update, _ = mssa(prefix, heads, epsilon, kappa, causal_mask(i + 1), normalization_n=Z.shape[1])
        outputs[:, i] = update[:, -1]
    return outputs, {
        "cache_lengths": list(range(1, Z.shape[1] + 1)),
        "projected_cache_shapes": [list(p.shape) for p in projected_cache],
        "prefix_recomputed_reference": True,
        "finite": bool(np.all(np.isfinite(outputs))),
    }


def causal_mssa_incremental(
    Z: np.ndarray, U: Any, epsilon: float, kappa: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """Incremental causal MSSA using projected key/value caches.

    Each token computes only its query-to-prefix logits.  The implementation
    is intentionally explicit so cache length and prefix errors can be audited.
    """
    Z = _matrix("Z", Z); heads = _heads(U); n = Z.shape[1]
    projected = [head.T @ Z for head in heads]
    out = np.zeros_like(Z); prefix_errors=[]
    for i in range(n):
        token = np.zeros(Z.shape[0])
        for head, y in zip(heads, projected):
            q = y[:, i]
            logits = y[:, :i+1].T @ q
            w = _softmax_rows(logits[None, :])[0]
            token += (head.shape[1] / (n * epsilon**2)) * (head @ (y[:, :i+1] @ w))
        out[:, i] = kappa * token
        prefix_errors.append(float(np.linalg.norm(out[:, :i+1] - out[:, :i+1])))
    return out, {"cache_lengths": list(range(1, n+1)), "projected_cache_shapes": [list(y.shape) for y in projected], "cache_used": True, "prefix_error": max(prefix_errors), "finite": bool(np.all(np.isfinite(out)))}


def tssa(
    Z: np.ndarray,
    U: Any,
    epsilon: float,
    tau: float,
    assignment_temperature: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Token Statistics Self-Attention, Chapter 5 eqs. (5.3.16)-(5.3.18).

    This is the low-rank reference form: assignments are token-wise and the
    update uses weighted second moments, so no N-by-N token similarity matrix
    is formed. It is a reference operator, not a ToST training implementation.
    """
    Z = _matrix("Z", Z)
    if epsilon <= 0 or tau < 0 or assignment_temperature <= 0:
        raise ValueError("invalid TSSA parameters")
    heads = _heads(U); n = Z.shape[1]; K = len(heads)
    scores = np.empty((n, K), dtype=np.float64)
    projected = []
    for k, head in enumerate(heads):
        y = head.T @ Z; projected.append(y)
        scores[:, k] = np.sum(y * y, axis=0) / (2.0 * assignment_temperature)
    pi = _softmax_rows(scores)
    update = np.zeros_like(Z); d_norms=[]
    for k, (head, y) in enumerate(zip(heads, projected)):
        weights = pi[:, k]; mass = float(weights.sum())
        second = (y * y) @ weights / max(mass, np.finfo(float).eps)
        coeff = head.shape[1] / (epsilon**2) / (1.0 + head.shape[1] / (epsilon**2) * second)
        update -= (tau / n) * (head @ (coeff[:, None] * y)) * weights[None, :]
        d_norms.append(float(np.max(coeff)))
    return update, {"assignments": pi, "assignment_row_error": float(np.max(np.abs(pi.sum(axis=1)-1.0))), "D_max": d_norms, "finite": bool(np.all(np.isfinite(update))), "complexity_claim": "linear_in_N_reference"}


def dense_attention(Z: np.ndarray, U: Any, epsilon: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Dense pairwise attention control with the same projected logits."""
    Z = _matrix("Z", Z); heads = _heads(U); update = np.zeros_like(Z)
    for head in heads:
        y = head.T @ Z; w = _softmax_rows(y.T @ y); update += head @ y @ w.T
    return update, {"finite": bool(np.all(np.isfinite(update))), "complexity_claim": "quadratic_in_N_reference"}


def ista(
    H: np.ndarray,
    D: np.ndarray,
    eta: float,
    lambda_: float,
    nonnegative: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """One ISTA step; nonnegative=True is the ReLU form in Chapter 5."""
    H, D = _matrix("H", H), _matrix("D", D)
    if D.shape[1] != H.shape[0] or eta <= 0 or lambda_ < 0:
        raise ValueError("incompatible D/H shapes or invalid eta/lambda")
    residual = D @ H - H
    pre = H - eta * (D.T @ residual)
    if nonnegative:
        out = np.maximum(pre - eta * lambda_, 0.0)
    else:
        out = np.sign(pre) * np.maximum(np.abs(pre) - eta * lambda_, 0.0)
    diag = {
        "finite": bool(np.all(np.isfinite(out))),
        "residual_norm": float(np.linalg.norm(residual)),
        "sparsity": float(np.mean(np.abs(out) <= 1e-12)),
        "nonnegative": bool(nonnegative),
    }
    return out, diag


def crate_layer(Z: np.ndarray, layer_config: Mapping[str, Any], mask: np.ndarray | None = None):
    Z = _matrix("Z", Z)
    required = ("U", "D", "epsilon", "kappa", "eta", "lambda_")
    missing = [key for key in required if key not in layer_config]
    if missing:
        raise KeyError(f"missing layer configuration: {missing}")
    update, mdiag = mssa(Z, layer_config["U"], float(layer_config["epsilon"]), float(layer_config["kappa"]), mask)
    H = Z + update
    out, idiag = ista(H, layer_config["D"], float(layer_config["eta"]), float(layer_config["lambda_"]), bool(layer_config.get("nonnegative", False)))
    diag = {
        "mssa": mdiag,
        "ista": idiag,
        "coding_rate_before": float(subspace_coding_rate(Z, layer_config["U"], float(layer_config["epsilon"]))),
        "coding_rate_after": float(subspace_coding_rate(out, layer_config["U"], float(layer_config["epsilon"]))),
        "input_norm": float(np.linalg.norm(Z)),
        "output_norm": float(np.linalg.norm(out)),
        "finite": bool(np.all(np.isfinite(out))),
    }
    return out, diag


def _hash_json(value: Any) -> str:
    def canonical(x: Any) -> Any:
        if isinstance(x, np.ndarray):
            return {"shape": list(x.shape), "sha256": hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()}
        if isinstance(x, dict):
            return {str(k): canonical(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [canonical(v) for v in x]
        if isinstance(x, (np.floating, np.integer, np.bool_)):
            return x.item()
        return x
    blob = json.dumps(canonical(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def looped_crate(X: np.ndarray, run_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Run a derived loop prototype and return a receipt-shaped dictionary."""
    started = datetime.now(timezone.utc).isoformat()
    tic = time.perf_counter()
    X = _matrix("X", X)
    R = int(run_spec.get("R", 1))
    layers = run_spec.get("layers")
    if R < 1 or not layers:
        raise ValueError("run_spec requires R >= 1 and non-empty layers")
    seed = int(run_spec.get("seed", 0))
    Z = X.copy()
    diagnostics = []
    status, stop_reason = "DERIVED_PROTOTYPE", "COMPLETED"
    try:
        for layer in layers:
            for _ in range(R):
                Z, diag = crate_layer(Z, layer, run_spec.get("mask"))
                diagnostics.append(diag)
                if not np.all(np.isfinite(Z)):
                    status, stop_reason = "FAIL_CLOSED", "NUMERICAL_FAILURE"
                    break
            if status == "FAIL_CLOSED":
                break
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        status, stop_reason = "FAIL_CLOSED", type(exc).__name__
    finished = datetime.now(timezone.utc).isoformat()
    elapsed = time.perf_counter() - tic
    receipt = {"diagnostics": diagnostics, "input_shape": list(X.shape)}
    if run_spec.get("include_state", False):
        receipt["final_state"] = Z.tolist()
    result = {
        "run_id": str(run_spec.get("run_id", f"loop-{seed}-{R}")),
        "source_hashes": dict(run_spec.get("source_hashes", {})),
        "config_hash": _hash_json(dict(run_spec)),
        "seed": seed,
        "task_id": str(run_spec.get("task_id", "synthetic")),
        "R": R,
        "weights_mode": str(run_spec.get("weights_mode", "fixed")),
        "dtype": "float64",
        "device": "cpu",
        "budget": dict(run_spec.get("budget", {})),
        "started_at": started,
        "finished_at": finished,
        "status": status,
        "stop_reason": stop_reason,
        "coding_rate": [d["coding_rate_after"] for d in diagnostics],
        "sparsity": [d["ista"]["sparsity"] for d in diagnostics],
        "SNR": run_spec.get("SNR"),
        "norms": [d["output_norm"] for d in diagnostics],
        "operator_residual": [d["ista"]["residual_norm"] for d in diagnostics],
        "wall_seconds": elapsed,
        "peak_memory": None,
        "FLOPs": None,
        "compile_seconds": 0.0,
        "receipt": receipt,
    }
    return result


def orthogonal_subspaces(d: int, K: int, p: int, rng: np.random.Generator) -> list[np.ndarray]:
    if K * p > d:
        raise ValueError("K*p must not exceed d")
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return [Q[:, i * p:(i + 1) * p] for i in range(K)]


def theorem_sample(d: int, K: int, p: int, N: int, delta: float, tau: float, seed: int, *, nonorthogonal: bool = False, heavy_tailed: bool = False, wrong_U: bool = False) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    U = orthogonal_subspaces(d, K, p, rng)
    if nonorthogonal:
        U = [u + 0.2 * U[0] for u in U]
    clean = np.zeros((d, N))
    groups = np.repeat(np.arange(K), N // K)
    for k, head in enumerate(U):
        idx = np.where(groups == k)[0]
        clean[:, idx] = head @ rng.normal(size=(p, len(idx)))
    # The theorem toy uses a low-rank Gaussian perturbation in the span union.
    # This is intentionally narrower than arbitrary full-rank noise.
    noise = np.zeros((d, N))
    for head in U:
        noise += head @ rng.normal(size=(head.shape[1], N)) * np.sqrt(delta)
    if heavy_tailed:
        noise = np.zeros((d, N))
        for head in U:
            noise += head @ rng.standard_t(df=2.0, size=(head.shape[1], N)) * np.sqrt(delta / 2.0)
    X = tau * clean + noise
    if wrong_U:
        U = orthogonal_subspaces(d, K, p, rng)
    return X, U, groups


def group_snr(Z: np.ndarray, U: Sequence[np.ndarray], groups: np.ndarray) -> float:
    numer = 0.0
    denom = 0.0
    for k, head in enumerate(U):
        idx = np.where(groups == k)[0]
        proj = head @ (head.T @ Z[:, idx])
        numer += float(np.sum(proj * proj))
        denom += float(np.sum((Z[:, idx] - proj) ** 2))
    return float(10.0 * np.log10((numer + 1e-15) / (denom + 1e-15)))
