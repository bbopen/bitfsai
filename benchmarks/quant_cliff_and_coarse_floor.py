#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as spla


def build_3d_aniso_contrast(L: int = 20) -> sp.csr_matrix:
    rng = np.random.default_rng(42)
    B = max(1, L // 4)
    patch = np.where(rng.integers(0, 2, size=(B, B, B)) == 1, 1e4, 1.0)

    def idx(x: int, y: int, z: int) -> int:
        return (z * L + y) * L + x

    def cval(x: int, y: int, z: int) -> float:
        return float(patch[min(z // 4, B - 1), min(y // 4, B - 1), min(x // 4, B - 1)])

    N = L * L * L
    rows, cols, vals = [], [], []
    for z in range(L):
        for y in range(L):
            for x in range(L):
                u = idx(x, y, z)
                cu = cval(x, y, z)
                diag = 1e-4
                for dx, dy, dz, eps in [
                    (-1, 0, 0, 1e-3), (1, 0, 0, 1e-3),
                    (0, -1, 0, 1.0),  (0, 1, 0, 1.0),
                    (0, 0, -1, 1e-2), (0, 0, 1, 1e-2),
                ]:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if 0 <= nx < L and 0 <= ny < L and 0 <= nz < L:
                        cv = cval(nx, ny, nz)
                        w = eps * (2.0 * cu * cv) / (cu + cv)
                        diag += w
                        rows.append(u)
                        cols.append(idx(nx, ny, nz))
                        vals.append(-w)
                rows.append(u)
                cols.append(u)
                vals.append(diag)
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N), dtype=np.float64)


def compute_ic0_lower(A: sp.csr_matrix) -> sp.csr_matrix:
    n = A.shape[0]
    L_rows = []
    for i in range(n):
        rs, re = A.indptr[i], A.indptr[i + 1]
        d = {}
        for k in range(rs, re):
            c = int(A.indices[k])
            if c <= i:
                d[c] = float(A.data[k])
        L_rows.append(d)

    for i in range(n):
        row_i = L_rows[i]
        cols_i = sorted(row_i.keys())
        for j in cols_i:
            if j == i:
                break
            row_j = L_rows[j]
            s = row_i[j]
            for k in cols_i:
                if k >= j:
                    break
                if k in row_j:
                    s -= row_i[k] * row_j[k]
            row_i[j] = s / row_j[j]
        s = row_i[i]
        for k in cols_i:
            if k < i:
                s -= row_i[k] * row_i[k]
        row_i[i] = math.sqrt(max(s, 1e-18))

    indptr = [0]
    indices, data = [], []
    for i in range(n):
        for c in sorted(L_rows[i].keys()):
            indices.append(c)
            data.append(L_rows[i][c])
        indptr.append(len(indices))
    return sp.csr_matrix((np.array(data), np.array(indices), np.array(indptr)), shape=(n, n))


def compute_fsai1_lower(A: sp.csr_matrix, patch_cap: int = 12) -> sp.csr_matrix:
    """Build the capped lower FSAI pattern used by ``bitfsai.cuh``."""
    n = A.shape[0]
    max_low = min(max(int(patch_cap), 1), 16) - 1
    indptr = [0]
    indices, data = [], []
    for i in range(n):
        rs, re = A.indptr[i], A.indptr[i + 1]
        selected: list[tuple[int, float]] = []
        has_diagonal = False
        for k in range(rs, re):
            col = int(A.indices[k])
            if col == i:
                has_diagonal = True
            elif col < i and max_low:
                weight = abs(float(A.data[k]))
                if len(selected) < max_low:
                    selected.append((col, weight))
                else:
                    least = min(range(max_low), key=lambda j: selected[j][1])
                    if weight > selected[least][1]:
                        selected[least] = (col, weight)
        if not has_diagonal:
            raise ValueError(f"FSAI row {i} is missing its diagonal")
        J = sorted(col for col, _ in selected)
        J.append(i)
        m = len(J)
        sub = A[J, :][:, J].toarray()
        diagonal = float(sub[-1, -1])
        if not (diagonal > 0.0 and math.isfinite(diagonal)):
            raise ValueError(f"FSAI row {i} has an invalid diagonal")
        rhs = np.zeros(m, dtype=np.float64)
        rhs[-1] = 1.0
        try:
            lower = np.linalg.cholesky(sub)
            g = la.solve_triangular(lower.T, rhs, lower=False)
        except np.linalg.LinAlgError:
            g = None
        if g is None or not np.all(np.isfinite(g)) or not (g[-1] > 0.0):
            g = np.zeros(m, dtype=np.float64)
            g[-1] = 1.0 / math.sqrt(diagonal)
        if not np.all(np.isfinite(g)) or not (g[-1] > 0.0):
            raise ValueError(f"FSAI local factor is not finite in row {i}")
        indices.extend(J)
        data.extend(g.tolist())
        indptr.append(len(indices))
    return sp.csr_matrix((np.array(data), np.array(indices), np.array(indptr)), shape=(n, n))


def quantize_lower_rowwise_inclusive(M: sp.csr_matrix, fmt: str) -> sp.csr_matrix:
    """Quantize every lower-factor entry with one row scale.

    This models diagonal-inclusive quantization. IC(0) uses this comparator.
    BitFSAI INT4-Split does not use this encoding.
    """
    if fmt == "FP64":
        return M.copy()
    Q = M.copy()
    n = Q.shape[0]
    for i in range(n):
        rs, re = Q.indptr[i], Q.indptr[i + 1]
        row = Q.data[rs:re].copy()
        if fmt == "FP16":
            row = row.astype(np.float16).astype(np.float64)
            if row[-1] == 0.0:
                row[-1] = M.data[re - 1]
        else:
            qmax = {"INT4": 7, "INT3": 3, "INT2": 1}[fmt]
            sc = float(np.max(np.abs(row)))
            if sc > 0.0:
                q = np.clip(np.round((row / sc) * qmax), -qmax, qmax)
                if q[-1] == 0.0:
                    q[-1] = 1.0 if row[-1] >= 0 else -1.0
                row = (q / qmax) * sc
        Q.data[rs:re] = row
    return Q


def quantize_lower_rowwise(M: sp.csr_matrix, fmt: str) -> sp.csr_matrix:
    """Compatibility name for the diagonal-inclusive comparator."""
    return quantize_lower_rowwise_inclusive(M, fmt)


def _round_away_from_zero(x: np.ndarray) -> np.ndarray:
    return np.copysign(np.floor(np.abs(x) + 0.5), x)


def _f32_metadata(value: float, row: int, name: str) -> float:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        stored = float(np.float32(value))
    if not (math.isfinite(value) and math.isfinite(stored) and stored != 0.0):
        raise ValueError(f"FSAI {name} does not fit FP32 in row {row}")
    return stored


def _row_energy(weights: np.ndarray, patch: np.ndarray) -> tuple[float, float]:
    energy = 0.0
    compensation = 0.0
    sum_abs = 0.0
    for a, wa in enumerate(weights):
        for b, wb in enumerate(weights):
            term = float(wa * patch[a, b] * wb)
            if not math.isfinite(term):
                return float("nan"), float("nan")
            adjusted = term - compensation
            next_energy = energy + adjusted
            compensation = (next_energy - energy) - adjusted
            energy = next_energy
            sum_abs += abs(term)
    return energy, sum_abs


def _unit_energy(stored: np.ndarray, patch: np.ndarray) -> bool:
    energy, sum_abs = _row_energy(stored, patch)
    uncertainty = 8.0 * np.finfo(np.float64).eps * sum_abs
    return abs(energy - 1.0) + uncertainty <= 1e-4


def _project_split_row(patch: np.ndarray, direction: np.ndarray, row: int):
    if not np.any(direction):
        return None
    try:
        lower = np.linalg.cholesky(patch)
    except np.linalg.LinAlgError:
        return None
    u = lower[:-1, :-1].T @ direction
    c11 = float(u @ u)
    c12 = float(u @ lower[-1, :-1])
    if not (c11 > 0.0 and math.isfinite(c11) and math.isfinite(c12)):
        return None
    projection = c12 / c11
    component = lower[-1, :-1] - projection * u
    pivot = float(component @ component + lower[-1, -1] ** 2)
    if not (pivot > 0.0 and math.isfinite(pivot)):
        return None
    diag = 1.0 / math.sqrt(pivot)
    try:
        scale_f = _f32_metadata(-projection * diag, row, "off-diagonal scale")
        diag_f = _f32_metadata(diag, row, "diagonal")
    except ValueError:
        return None
    stored = np.append(scale_f * direction, diag_f)
    return stored if _unit_energy(stored, patch) else None


def quantize_fsai_split_reprojected(A: sp.csr_matrix, G: sp.csr_matrix,
                                     fmt: str) -> sp.csr_matrix:
    """Model BitFSAI's split lower factor with FP32 row metadata.

    The diagonal stays in FP32. The off-diagonals use a separate row scale.
    After quantization, use the CUDA row projection and verify the row again
    after FP32 metadata storage.
    """
    if fmt == "FP64":
        return G.copy()
    if fmt not in {"FP16", "INT4", "INT3", "INT2"}:
        raise ValueError(f"Unknown format: {fmt}")

    Q = G.copy()
    for i in range(Q.shape[0]):
        rs, re = Q.indptr[i], Q.indptr[i + 1]
        cols = Q.indices[rs:re]
        row = Q.data[rs:re]
        if len(row) == 0 or cols[-1] != i:
            raise ValueError(f"FSAI row {i} must end with its diagonal")

        off = row[:-1]
        q_dec = np.zeros_like(off)
        off_max = float(np.max(np.abs(off))) if len(off) else 0.0
        if off_max > 0.0:
            normalized = off / off_max
            if fmt == "FP16":
                q_dec = normalized.astype(np.float16).astype(np.float64)
            else:
                qmax = {"INT4": 7, "INT3": 3, "INT2": 1}[fmt]
                codes = np.clip(_round_away_from_zero(normalized * qmax),
                                -qmax, qmax)
                q_dec = (codes.astype(np.float32) *
                         np.float32(1.0 / qmax)).astype(np.float64)

        a_sub = A[cols, :][:, cols].toarray()
        c22 = float(a_sub[-1, -1])
        if not (c22 > 0.0 and math.isfinite(c22)):
            raise ValueError(f"FSAI patch has invalid diagonal in row {i}")

        stored = _project_split_row(a_sub, q_dec, i)
        if stored is None:
            stored = np.zeros_like(row)
            stored[-1] = _f32_metadata(1.0 / math.sqrt(c22), i, "diagonal")
            if not _unit_energy(stored, a_sub):
                raise ValueError(f"FSAI FP32 fallback row energy is invalid in row {i}")
        Q.data[rs:re] = stored
    return Q


def quantize_fp8_array(x: np.ndarray, exp_bits: int, man_bits: int) -> np.ndarray:
    bias = (1 << (exp_bits - 1)) - 1
    max_exp = (1 << exp_bits) - 2 - bias
    min_norm_exp = 1 - bias
    sub_step = 2.0 ** (min_norm_exp - man_bits)
    max_val = (2.0 - 2.0 ** (-man_bits)) * (2.0 ** max_exp)

    out = np.zeros_like(x, dtype=np.float64)
    sgn = np.sign(x)
    ax = np.abs(x)
    ax = np.minimum(ax, max_val)

    norm_mask = ax >= (2.0 ** min_norm_exp)
    if np.any(norm_mask):
        v = ax[norm_mask]
        e = np.floor(np.log2(v))
        step = np.exp2(e - man_bits)
        out[norm_mask] = np.round(v / step) * step

    sub_mask = (~norm_mask) & (ax > 0.0)
    if np.any(sub_mask):
        out[sub_mask] = np.round(ax[sub_mask] / sub_step) * sub_step

    return sgn * out


def run_pcg(A: sp.csr_matrix, b: np.ndarray, apply_M, rtol: float = 1e-8, maxit: int = 400):
    n = len(b)
    bmax = float(np.max(np.abs(b))) if n > 0 else 0.0
    if bmax == 0.0:
        return 0, 0.0, "CONVERGED"
    b_scaled = b / bmax
    x = np.zeros(n, dtype=np.float64)
    r = b_scaled.copy()
    bnorm = float(np.linalg.norm(b_scaled))
    z = apply_M(r)
    if not np.all(np.isfinite(z)):
        return 1, float("nan"), "DIVERGED (NaN/Inf)"
    rz = float(np.dot(r, z))
    if rz <= 0.0:
        return 1, float(np.linalg.norm(r) / bnorm), "NON-SPD BREAKDOWN"
    p = z.copy()
    for it in range(1, maxit + 1):
        Ap = A @ p
        pAp = float(np.dot(p, Ap))
        if pAp <= 0.0 or not np.isfinite(pAp):
            return it, float(np.linalg.norm(b_scaled - A @ x) / bnorm), "BREAKDOWN"
        alpha = rz / pAp
        x += alpha * p
        r -= alpha * Ap
        rel = float(np.linalg.norm(r) / bnorm)
        if not np.isfinite(rel) or rel > 1e6:
            return it, rel, "DIVERGED"
        if rel <= rtol:
            true_rel = float(np.linalg.norm(b_scaled - A @ x) / bnorm)
            if true_rel <= rtol:
                return it, true_rel, "CONVERGED"
            return it, true_rel, "TRUE_RESIDUAL_FAIL"
        z = apply_M(r)
        if not np.all(np.isfinite(z)):
            return it, float("nan"), "DIVERGED (NaN/Inf)"
        rz_new = float(np.dot(r, z))
        if rz_new <= 0.0:
            return it, float(np.linalg.norm(b_scaled - A @ x) / bnorm), "NON-SPD BREAKDOWN"
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new
    return maxit, float(np.linalg.norm(b_scaled - A @ x) / bnorm), "MAX_ITERS"


def build_unsmoothed_agg_P(A: sp.csr_matrix, theta: float = 0.25) -> sp.csr_matrix:
    n = A.shape[0]
    agg = np.full(n, -1, dtype=np.int32)
    next_agg = 0
    for i in range(n):
        if agg[i] != -1:
            continue
        rs, re = A.indptr[i], A.indptr[i + 1]
        best_j, best_w, max_off = -1, 0.0, 0.0
        for k in range(rs, re):
            c = int(A.indices[k])
            if c != i and A.data[k] < 0:
                max_off = max(max_off, -A.data[k])
        thresh = theta * max_off
        for k in range(rs, re):
            c = int(A.indices[k])
            w = -A.data[k]
            if c != i and agg[c] == -1 and w >= thresh and w > best_w:
                best_w = w
                best_j = c
        agg[i] = next_agg
        if best_j != -1:
            agg[best_j] = next_agg
        next_agg += 1
    rows = np.arange(n, dtype=np.int32)
    return sp.csr_matrix((np.ones(n, dtype=np.float64), (rows, agg)), shape=(n, next_agg))


def verify_sddm_grounding_spd(Q: sp.csr_matrix) -> tuple[bool, str]:
    """Check whether a symmetric quantized matrix Q is a grounded SDDM M-matrix:
    all off-diagonals <= 0, positive diagonals, nonnegative row sums, and every
    connected component of the surviving nonzero off-diagonal graph contains at
    least one strictly grounded row (row_sum > 0)."""
    n = Q.shape[0]
    Q_csr = Q.tocsr()
    diag = Q_csr.diagonal().astype(np.float64)
    if not np.all(np.isfinite(diag) & (diag > 0.0)):
        return False, "repaired coarse diagonal is not positive finite"
    q_off = Q_csr.copy()
    q_off.setdiag(0.0)
    q_off.eliminate_zeros()
    if q_off.nnz > 0 and np.any(q_off.data > 0.0):
        return False, "quantized coarse matrix has positive off-diagonals (non-SDDM)"
    row_sum = np.array(
        [
            math.fsum(float(v) for v in Q_csr.data[Q_csr.indptr[i] : Q_csr.indptr[i + 1]])
            for i in range(n)
        ],
        dtype=np.float64,
    )
    if np.any(row_sum < 0.0) or not np.all(np.isfinite(row_sum)):
        return False, "quantized coarse matrix violates weak diagonal dominance"
    n_comp, labels = csgraph.connected_components(
        q_off, directed=False, return_labels=True
    )
    comp_grounded = np.zeros(n_comp, dtype=bool)
    for i in range(n):
        if row_sum[i] > 0.0:
            comp_grounded[labels[i]] = True
    if not np.all(comp_grounded):
        return False, "quantized coarse matrix has an ungrounded connected component"
    return True, "ok"


def quantize_coarse_matrix(
    Ac: sp.csr_matrix,
    fmt: str,
    preserve_rowsum: bool = False,
    persist_diag_fp32: bool = False,
    verify_sddm: bool | None = None,
) -> sp.csr_matrix:
    if fmt == "FP64":
        return Ac.copy()
    Q = Ac.copy()
    if fmt == "FP32":
        Q.data = Q.data.astype(np.float32).astype(np.float64)
        return Q

    if fmt == "FP8_E5M2":
        Q = (0.5 * (Q + Q.T)).tocsr()
        do_verify_range = preserve_rowsum and (persist_diag_fp32 if verify_sddm is None else verify_sddm)
        if do_verify_range:
            q_check = Q.copy()
            q_check.setdiag(0.0)
            q_check.eliminate_zeros()
            if q_check.nnz > 0:
                abs_off = np.abs(q_check.data)
                if np.any(abs_off > 57344.0):
                    raise ValueError(
                        "FP8 E5M2 off-diagonal exceeds normal max 57344.0 (saturation)"
                    )
                if np.any(abs_off < 2.0 ** -14):
                    raise ValueError(
                        "FP8 E5M2 off-diagonal below normal min 2^-14 (subnormal/underflow)"
                    )
        Q.data = quantize_fp8_array(Q.data, exp_bits=5, man_bits=2)
    else:
        n = Q.shape[0]
        for i in range(n):
            rs, re = Q.indptr[i], Q.indptr[i + 1]
            row = Q.data[rs:re]
            sc = float(np.max(np.abs(row)))
            if sc <= 0.0:
                continue
            u = row / sc
            if fmt == "FP16":
                Q.data[rs:re] = u.astype(np.float16).astype(np.float64) * sc
            elif fmt == "INT8":
                Q.data[rs:re] = (np.round(u * 127.0) / 127.0) * sc
            elif fmt == "FP8_E4M3":
                Q.data[rs:re] = quantize_fp8_array(u, exp_bits=4, man_bits=3) * sc
            else:
                raise ValueError(f"Unknown format: {fmt}")
        Q = (0.5 * (Q + Q.T)).tocsr()

    if preserve_rowsum:
        n_rows = Ac.shape[0]
        Ac_csr = Ac.tocsr()
        exact_row_sum = np.array(
            [
                math.fsum(float(v) for v in Ac_csr.data[Ac_csr.indptr[i] : Ac_csr.indptr[i + 1]])
                for i in range(n_rows)
            ],
            dtype=np.float64,
        )
        q_off = Q.copy().tocsr()
        q_off.setdiag(0.0)
        q_off.eliminate_zeros()
        dec_off_sum = np.array(
            [
                math.fsum(float(v) for v in q_off.data[q_off.indptr[i] : q_off.indptr[i + 1]])
                for i in range(n_rows)
            ],
            dtype=np.float64,
        )
        rep_diag = exact_row_sum - dec_off_sum
        if persist_diag_fp32:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                rep_f32 = rep_diag.astype(np.float32)
            weak_loss = (exact_row_sum >= 0.0) & (
                rep_f32.astype(np.float64) + dec_off_sum < 0.0
            )
            if np.any(weak_loss):
                rep_f32[weak_loss] = np.nextafter(
                    rep_f32[weak_loss], np.float32(np.inf)
                )
            grounded_loss = (exact_row_sum > 0.0) & (
                rep_f32.astype(np.float64) + dec_off_sum <= 0.0
            )
            if np.any(grounded_loss):
                rep_f32[grounded_loss] = np.nextafter(
                    rep_f32[grounded_loss], np.float32(np.inf)
                )
            if not np.all(np.isfinite(rep_f32) & (rep_f32 > 0.0)):
                raise ValueError("Repaired coarse diagonal is not positive finite in FP32")
            if np.any((exact_row_sum >= 0.0) & (rep_f32.astype(np.float64) + dec_off_sum < 0.0)):
                raise ValueError("FP32 coarse diagonal violated weak diagonal dominance")
            if np.any((exact_row_sum > 0.0) & (rep_f32.astype(np.float64) + dec_off_sum <= 0.0)):
                raise ValueError("FP32 coarse diagonal lost strict row-sum grounding")
            rep_diag = rep_f32.astype(np.float64)
        Q = (q_off + sp.diags(rep_diag, format="csr")).tocsr()
        do_verify = (persist_diag_fp32 if verify_sddm is None else verify_sddm)
        if do_verify:
            ok, reason = verify_sddm_grounding_spd(Q)
            if not ok:
                raise ValueError(f"Coarse SDDM grounding/SPD check failed: {reason}")
    return Q


def fmt_res(it: int, status: str) -> str:
    if status == "CONVERGED":
        return f"{it} it"
    if status == "TRUE_RESIDUAL_FAIL":
        return "TRUE-FAIL"
    if status == "MAX_ITERS":
        return "MAX_ITERS"
    return "BREAKDOWN"


def est_spectral_radius(apply_M, A: sp.csr_matrix, n: int, iters: int = 25) -> float:
    v = np.ones(n, dtype=np.float64)
    v[::2] = -1.0
    v /= np.linalg.norm(v)
    lam = 1.0
    for _ in range(iters):
        w = apply_M(A @ v)
        if not np.all(np.isfinite(w)):
            return float("inf")
        nrm = float(np.linalg.norm(w))
        if nrm == 0.0:
            return 0.0
        lam = float(np.dot(v, w))
        v = w / nrm
    return lam


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="CPU smoother quantization cliff and coarse Galerkin precision floor."
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Optional output CSV path for both Experiment A and Experiment B results.",
    )
    args = parser.parse_args(argv)

    L = 20
    A = build_3d_aniso_contrast(L)
    rng = np.random.default_rng(12345)
    b = rng.standard_normal(A.shape[0])

    P = build_unsmoothed_agg_P(A, theta=0.25)
    Ac_exact = (P.T @ A @ P).tocsr()
    lu_exact = spla.splu(Ac_exact.tocsc())

    csv_rows: list[dict[str, object]] = []

    print("=" * 102)
    print(f"EXPERIMENT A (Principle 1): Smoother Quantization Cliff — IC(0) vs. BitFSAI (3D {L}^3, N={A.shape[0]})")
    print("=" * 102)
    L_ic = compute_ic0_lower(A)
    G_fs = compute_fsai1_lower(A)

    print(f"  {'Format':6s} | {'IC lmax':>8s} | {'IC(w=1)':>10s} | {'IC(est. w)':>10s} || "
          f"{'FS lmax':>7s} | {'FS(w=.72)':>10s} | {'FS(est. w)':>10s}")
    print("  " + "-" * 78)
    for fmt in ["FP64", "FP16", "INT4", "INT3", "INT2"]:
        Lq = quantize_lower_rowwise_inclusive(L_ic, fmt).tocsc()
        LqT = Lq.T.tocsc()

        def ap_ic(v, Lq=Lq, LqT=LqT):
            y = spla.spsolve_triangular(Lq, v, lower=True)
            return spla.spsolve_triangular(LqT, y, lower=False)

        lmax_ic = est_spectral_radius(ap_ic, A, A.shape[0])

        Gq = quantize_fsai_split_reprojected(A, G_fs, fmt)
        GqT = Gq.T.tocsr()

        def ap_fs(v, Gq=Gq, GqT=GqT):
            return GqT @ (Gq @ v)

        lmax_fs = est_spectral_radius(ap_fs, A, A.shape[0])

        def mk_v(ap, om):
            def f(r):
                x = om * ap(r)
                x = x + P @ lu_exact.solve(P.T @ (r - A @ x))
                return x + om * ap(r - A @ x)
            return f

        it_ic1, rel_ic1, st_ic1 = run_pcg(A, b, mk_v(ap_ic, 1.0))
        om_ic = min(1.0, 1.4 / max(lmax_ic, 1e-12)) if np.isfinite(lmax_ic) else 0.0
        it_ics, rel_ics, st_ics = run_pcg(A, b, mk_v(ap_ic, om_ic)) if om_ic > 0 else (1, float("inf"), "EXPLODED")

        it_fs1, rel_fs1, st_fs1 = run_pcg(A, b, mk_v(ap_fs, 0.72))
        om_fs = min(0.72, 1.4 / max(lmax_fs, 1e-12))
        it_fss, rel_fss, st_fss = run_pcg(A, b, mk_v(ap_fs, om_fs))
        print(f"  {fmt:6s} | {lmax_ic:8.2f} | {fmt_res(it_ic1, st_ic1):>10s} | {fmt_res(it_ics, st_ics):>10s} || "
              f"{lmax_fs:7.2f} | {fmt_res(it_fs1, st_fs1):>10s} | {fmt_res(it_fss, st_fss):>10s}")
        csv_rows.append({
            "experiment": "A",
            "label": fmt,
            "fmt": fmt,
            "preserve_rowsum": "",
            "diag_precision": "",
            "ic_lmax": f"{lmax_ic:.6f}",
            "ic_w1_iters": it_ic1,
            "ic_w1_status": st_ic1,
            "ic_estw_iters": it_ics,
            "ic_estw_status": st_ics,
            "fs_lmax": f"{lmax_fs:.6f}",
            "fs_w072_iters": it_fs1,
            "fs_w072_status": st_fs1,
            "fs_estw_iters": it_fss,
            "fs_estw_status": st_fss,
            "coarse_iters": "",
            "coarse_true_rel": "",
            "coarse_status": "",
        })

    # Keep the diagonal-inclusive FSAI comparator explicit. It is not
    # BitFSAI INT4-Split because it scales and quantizes the diagonal with
    # the off-diagonals.
    Gq_inc = quantize_lower_rowwise_inclusive(G_fs, "INT4")
    GqT_inc = Gq_inc.T.tocsr()

    def ap_fs_inc(v):
        return GqT_inc @ (Gq_inc @ v)

    lmax_inc = est_spectral_radius(ap_fs_inc, A, A.shape[0])
    it_inc, _, st_inc = run_pcg(A, b, mk_v(ap_fs_inc, 0.72))
    print(f"  INT4inc | {'--':>8s} | {'--':>10s} | {'--':>10s} || "
          f"{lmax_inc:7.2f} | {fmt_res(it_inc, st_inc):>10s} | {'--':>10s}")
    csv_rows.append({
        "experiment": "A",
        "label": "INT4inc",
        "fmt": "INT4inc",
        "preserve_rowsum": "",
        "diag_precision": "",
        "ic_lmax": "",
        "ic_w1_iters": "",
        "ic_w1_status": "",
        "ic_estw_iters": "",
        "ic_estw_status": "",
        "fs_lmax": f"{lmax_inc:.6f}",
        "fs_w072_iters": it_inc,
        "fs_w072_status": st_inc,
        "fs_estw_iters": "",
        "fs_estw_status": "",
        "coarse_iters": "",
        "coarse_true_rel": "",
        "coarse_status": "",
    })

    print("\n" + "=" * 102)
    print(f"EXPERIMENT B (Principle 3): Coarse Galerkin A_c Dynamic-Range Floor (N_c={Ac_exact.shape[0]}, 10^4 Contrast)")
    print("=" * 102)
    Gq_int4 = quantize_fsai_split_reprojected(A, G_fs, "INT4")
    GqT_int4 = Gq_int4.T.tocsr()
    omega_fs = 0.72
    print(f"  {'A_c Format':28s} | {'PCG Iters':>9s} | {'True Rel Res':>12s} | {'Status':>18s}")
    print("  " + "-" * 76)
    for label, fmt, prs, f32_diag in [
        ("FP64 exact", "FP64", False, False),
        ("FP32", "FP32", False, False),
        ("FP16 (naive)", "FP16", False, False),
        ("FP16 + row-sum (FP64)", "FP16", True, False),
        ("FP16 + row-sum (FP32)", "FP16", True, True),
        ("INT8 (naive)", "INT8", False, False),
        ("INT8 + row-sum (FP64)", "INT8", True, False),
        ("INT8 + row-sum (FP32)", "INT8", True, True),
        ("FP8 E5M2 (naive)", "FP8_E5M2", False, False),
        ("FP8 E5M2 + row-sum (FP64)", "FP8_E5M2", True, False),
        ("FP8 E5M2 + row-sum (FP32)", "FP8_E5M2", True, True),
        ("FP8 E4M3 (naive)", "FP8_E4M3", False, False),
        ("FP8 E4M3 + row-sum (FP64)", "FP8_E4M3", True, False),
        ("FP8 E4M3 + row-sum (FP32)", "FP8_E4M3", True, True),
    ]:
        Ac_q = quantize_coarse_matrix(
            Ac_exact, fmt, preserve_rowsum=prs, persist_diag_fp32=f32_diag
        ).tocsc()
        try:
            lu = spla.splu(Ac_q)

            def apply_2lvl(r, lu=lu):
                x = omega_fs * (GqT_int4 @ (Gq_int4 @ r))
                res = r - A @ x
                rc = P.T @ res
                xc = lu.solve(rc)
                x = x + P @ xc
                res = r - A @ x
                return x + omega_fs * (GqT_int4 @ (Gq_int4 @ res))

            it, rel, st = run_pcg(A, b, apply_2lvl)
        except RuntimeError:
            it, rel, st = 0, float("nan"), "SINGULAR FACTOR"
        print(f"  {label:28s} | {it:9d} | {rel:12.2e} | {st:>18s}")
        csv_rows.append({
            "experiment": "B",
            "label": label,
            "fmt": fmt,
            "preserve_rowsum": int(prs),
            "diag_precision": "FP32" if f32_diag else "FP64",
            "ic_lmax": "",
            "ic_w1_iters": "",
            "ic_w1_status": "",
            "ic_estw_iters": "",
            "ic_estw_status": "",
            "fs_lmax": "",
            "fs_w072_iters": "",
            "fs_w072_status": "",
            "fs_estw_iters": "",
            "fs_estw_status": "",
            "coarse_iters": it,
            "coarse_true_rel": f"{rel:.6e}",
            "coarse_status": st,
        })

    if args.export_csv:
        out_path = Path(args.export_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(csv_rows[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nExported CPU precision sweep CSV to {out_path}")


if __name__ == "__main__":
    main()
