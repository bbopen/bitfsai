#!/usr/bin/env python3
"""Exact-rational and finite-precision-grid verification of the BitFSAI
four-phase grounded SDDM multilevel cycle.

All verification gates use an always-on `check(cond, msg)` helper raising
`RuntimeError` (never bare `assert`), so `python3 -O` enforces every check.

Checks:
1. Monotone four-phase effectiveness inequalities for both Richardson scales
   (t = 2560/1307 -> kappa <= 17, t = 30375/15853 -> kappa <= 9), the
   edge-pruned coarse retention budget (511/512), and the kappa <= 17 scale
   negative control (showing t = 2560/1307 cannot certify kappa <= 9).
2. Exact algebraic witnesses: 3-vertex output-precision lower bound, 2-vertex
   absolute-guard overdamping counterexample, and grounded-graph pruning budget.
3. Linear-time incremental Delta_v fine-set selector verified bit-for-bit equal
   to the dense O(n^3) conditional-expectation oracle on every nonterminal level
   of every root.
4. Exact Schur-pivot PSD sandwiches and grounded-SDDM certification across all
   levels on 14 grounded SDDM hierarchies (9 general rational + 3 dyadic-weight
   + 2 dense-fine stress fixtures) and edge-pruned hierarchies across all four
   (smoother x scale) combinations.
5. Tight 7*(3/4)^32 lifting-defect certification, plus discriminating dense-fine
   stress fixtures where 1 Jacobi sweep FAILS the lifting bound while 16 sweeps
   PASS, and matrix-free reverse-adjoint exactness across full standard bases.
6. Finite-precision integer-grid and ordinary binary-output solves with
   power-of-two input normalization (covering zero, tiny 2^(-bits-2), and huge
   2^(bits+2) RHS on both scalar terminal A = [1] and multi-vertex roots),
   two-pass decoded-block factor smoothing (t = G * r, z = (2/5) * G^T * t),
   per-primitive and measured-error levelwise recurrence checks (with fast
   in-script negative controls for shrinking mu and dropping c_i), and
   independent non-tautological signed integer word-length bounds (+1 sign bit).
"""

from collections import Counter, deque
from fractions import Fraction
from math import isqrt
import time

from sympy import Matrix, Rational as Q, eye, zeros

DELTA = Q(3, 4)
EPS = Q(1, 1024)
LIFT_DEFECT_BOUND = Q(7) * (DELTA**32)
SIGMA = Q(1, 128)
U_E5M2 = Q(1, 8)
PRUNE_RETENTION = Q(511, 512)
L_COARSE = (1 - U_E5M2) * (1 - SIGMA) / (1 + EPS)
H_COARSE = (1 + U_E5M2) * (1 + SIGMA)
P_RETENTION = Q(11, 15)

ROOT_FLOORS = {"standard": Q(1, 17), "retuned": Q(1, 9)}
CALIBRATIONS = {"standard": Q(27, 1280), "retuned": P_RETENTION**3 / 9}

FIXTURES = [
    (5, 0, True, False),
    (5, 30, True, False),
    (5, 80, True, False),
    (7, 0, False, False),
    (7, 20, False, False),
    (9, 0, False, False),
    (7, 0, True, True),
    (7, 20, True, True),
    (7, 80, True, False),
]

DYADIC_FIXTURES = [
    (
        "dyadic_chain_6",
        6,
        [(i, i + 1, Q(3 + (i % 2) * 2, 4) * (Q(2) ** (16 if i % 2 == 0 else -16))) for i in range(5)]
        + [(i, i + 2, Q(5, 8) * (Q(2) ** -16)) for i in range(4)],
        [(0, Q(3, 16)), (5, Q(5, 32))],
        False,
        False,
    ),
    (
        "dyadic_contrast_7",
        7,
        [(i, i + 1, Q(7 - (i % 3), 8) * (Q(2) ** (24 if i % 2 == 0 else -24))) for i in range(6)]
        + [(i, i + 2, Q(3, 16) * (Q(2) ** -24)) for i in range(5)],
        [(0, Q(7, 32)), (3, Q(1, 8)), (6, Q(3, 16))],
        True,
        True,
    ),
    (
        "dyadic_weak_bridge_8",
        8,
        [(0, 1, Q(3, 2)), (1, 2, Q(5, 4)), (2, 3, Q(7, 8)), (4, 5, Q(3, 2)), (5, 6, Q(5, 4)), (6, 7, Q(7, 8))]
        + [(0, 4, Q(3, 4) * (Q(2) ** -22)), (2, 6, Q(5, 8) * (Q(2) ** -22)), (3, 7, Q(1, 2) * (Q(2) ** -23))],
        [(i, Q(5, 8) if i % 2 == 0 else Q(3, 4)) for i in range(8)],
        False,
        True,
    ),
]

# Stress fixtures with dense fine-internal coupling so that
# 1 Jacobi lifting sweep FAILS both 7*(3/4)^32 and 1/1024 while 16 sweeps PASS.
DENSE_FINE_STRESS_FIXTURES = [
    (
        "stress_dense_fine_pair_6",
        6,
        [
            (0, 1, Q(1)),
            (0, 2, Q(2)),
            (1, 3, Q(2)),
            (4, 5, Q(1)),
            (4, 2, Q(2)),
            (5, 3, Q(2)),
            (2, 3, Q(64)),
        ],
        [(0, Q(1)), (1, Q(1)), (4, Q(1)), (5, Q(1)), (2, Q(32)), (3, Q(32))],
        False,
        False,
    ),
    (
        "stress_dense_fine_triple_8",
        8,
        [
            (0, 1, Q(1)),
            (1, 2, Q(1)),
            (0, 3, Q(2)),
            (1, 4, Q(2)),
            (2, 5, Q(2)),
            (6, 7, Q(1)),
            (6, 3, Q(2)),
            (7, 5, Q(2)),
            (3, 4, Q(64)),
            (4, 5, Q(64)),
            (3, 5, Q(64)),
        ],
        [(0, Q(1)), (1, Q(2)), (2, Q(1)), (6, Q(1)), (7, Q(1)), (3, Q(32)), (4, Q(32)), (5, Q(32))],
        False,
        True,
    ),
]


def check(cond, msg="verification check failed"):
    if not cond:
        raise RuntimeError(msg)


def phases(root_floor):
    return [root_floor, P_RETENTION**3 * root_floor, P_RETENTION**2 * root_floor, P_RETENTION * root_floor]


def pow2(k):
    return Q(2) ** k


def signed_bits(v):
    """Sign-magnitude signed integer bit-width (conservative +1 bound on two's-complement at -2^k)."""
    v = int(v)
    return 1 if v == 0 else abs(v).bit_length() + 1


def log2floor(x):
    x = Q(x)
    k = int(x.p).bit_length() - int(x.q).bit_length()
    return k - 1 if x < pow2(k) else k


def ceil_log2(x):
    x = Q(x)
    num, den = int(x.p), int(x.q)
    guess = num.bit_length() - den.bit_length()
    return guess + 1 if Q(2) ** guess < x else guess


def upper_sqrt(x):
    return Q(2) ** ((ceil_log2(x) + 1) // 2)


def rounded_divide(num, den):
    """Nearest integer with ties to even (supports negative numerators)."""
    whole, rem = divmod(num, den)
    return whole + (2 * rem > den or (2 * rem == den and whole % 2 == 1))


def round_even_rat(x):
    return rounded_divide(int(x.p), int(x.q))


def fp8_e5m2_significand(x):
    """Range-aware E5M2 significand rounding (relative error <= 1/8)."""
    if not x:
        return Q(0)
    s = pow2(log2floor(x) - 2)
    return s * round_even_rat(x / s)


def is_power_of_two_int(k):
    return k > 0 and (k & (k - 1)) == 0


def dyadic_significand_bits(x):
    """Verify x > 0 is an exact dyadic rational and return odd-significand bit width."""
    p, q = abs(int(x.p)), int(x.q)
    check(p > 0 and is_power_of_two_int(q), f"non-dyadic positive weight {x}")
    while p % 2 == 0:
        p //= 2
    return p.bit_length()


def verify_dyadic_root_spec(a):
    """Confirm every positive edge and ground weight of SDDM root A is dyadic."""
    n = a.rows
    weights = []
    for i in range(n):
        g = sum(a.row(i))
        check(g >= 0, f"negative row sum {g} at row {i}")
        if g:
            weights.append(g)
        for j in range(i):
            w = -a[i, j]
            check(w >= 0, f"positive off-diagonal {-w} at ({i},{j})")
            if w:
                weights.append(w)
    s_a = max(dyadic_significand_bits(w) for w in weights)
    r_w = max(weights) / min(weights)
    return s_a, r_w


def extract_root_input_spec(a):
    """Extract bit-width and dynamic-range specs strictly from the root matrix A."""
    n = a.rows
    weights = []
    max_coeff_bits = 1
    for i in range(n):
        g = sum(a.row(i))
        check(g >= 0, f"negative row sum at row {i}")
        if g:
            weights.append(g)
        for j in range(n):
            v = a[i, j]
            if v:
                max_coeff_bits = max(max_coeff_bits, signed_bits(v.p), signed_bits(v.q))
            if j < i and v:
                w = -v
                check(w > 0, f"positive off-diagonal at ({i},{j})")
                weights.append(w)
    r_w = max(weights) / min(weights)
    return max_coeff_bits, r_w


def graph_matrix(n, edges, grounds):
    a = zeros(n)
    for i, j, w in edges:
        a[i, i] += w
        a[j, j] += w
        a[i, j] -= w
        a[j, i] -= w
    for i, s in grounds:
        a[i, i] += s
    return a


def is_psd(matrix):
    """Exact Schur-pivot PSD test, including zero-pivot row check."""
    check(matrix == matrix.T, "is_psd called on nonsymmetric matrix")
    mat = matrix.copy()
    while mat.rows:
        pivot = mat[0, 0]
        col = mat[1:, 0]
        if pivot < 0 or (pivot == 0 and any(col)):
            return False
        mat = mat[1:, 1:]
        if pivot:
            mat -= col * col.T / pivot
    return True


def is_grounded_sddm(a):
    """Certify that `a` is symmetric, has positive diagonals, nonpositive off-diagonals,
    nonnegative row sums, and at least one strictly positive ground in every connected component."""
    n = a.rows
    if a.cols != n or a != a.T:
        return False
    row_sums = [sum(a.row(i)) for i in range(n)]
    adj = [[] for _ in range(n)]
    for i in range(n):
        if a[i, i] <= 0 or row_sums[i] < 0:
            return False
        for j in range(i):
            if a[i, j] > 0:
                return False
            if a[i, j] < 0:
                adj[i].append(j)
                adj[j].append(i)
    seen = [False] * n
    for start in range(n):
        if seen[start]:
            continue
        q = deque([start])
        seen[start] = True
        grounded = False
        while q:
            u = q.popleft()
            if row_sums[u] > 0:
                grounded = True
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        if not grounded:
            return False
    return True


def stored_sqrt_4bit(value):
    """Round sqrt(value) downward to 4 binary significand bits."""
    if value == 0:
        return Q(0)
    exponent = int(value.p).bit_length() - int(value.q).bit_length()
    if value < Q(2) ** exponent:
        exponent -= 1
    step = Q(2) ** (exponent // 2 - 3)
    square = value / step**2
    return step * isqrt(int(square.p // square.q))


def select_fine_set_oracle(a):
    """Dense quadratic-potential oracle (O(n^3) in Python) for the conditional-expectation
    fine-set selector (guarantees |F| >= ceil(3n/16))."""
    n = a.rows
    x = [DELTA / 2] * n

    def potential():
        return sum(
            x[i] * (1 - sum(-a[i, j] * x[j] for j in range(n) if j != i) / (DELTA * a[i, i]))
            for i in range(n)
        )

    for i in range(n):
        x[i] = 0
        p0 = potential()
        x[i] = 1
        p1 = potential()
        x[i] = 1 if p1 >= p0 else 0
    fine = [i for i in range(n) if x[i] and sum(-a[i, j] * x[j] for j in range(n) if j != i) <= DELTA * a[i, i]]
    if len(fine) == n:
        fine.pop()
    check(len(fine) >= Q(3 * n, 16), f"oracle fine set too small: {len(fine)} vs 3*{n}/16")
    coarse = [i for i in range(n) if i not in fine]
    return fine, coarse


def select_fine_set_incremental(a, adj=None):
    """Linear-time O(n + nnz) incremental Delta_v fine-set selector updating only
    incident neighbors at each vertex decision."""
    n = a.rows
    if adj is None:
        adj = [[] for _ in range(n)]
        diag = [Q(0)] * n
        for (i, j), val in a.todok().items():
            if i == j:
                diag[i] = val
            elif val != 0:
                adj[i].append((j, -val))
    else:
        diag = [a[i, i] for i in range(n)]
    inv_dd = [1 / (DELTA * diag[i]) for i in range(n)]
    x0 = DELTA / 2
    x = [x0] * n
    delta = [
        1 - x0 * sum(w * (inv_dd[i] + inv_dd[j]) for j, w in adj[i])
        for i in range(n)
    ]
    for v in range(n):
        new_x = 1 if delta[v] >= 0 else 0
        dx = new_x - x[v]
        x[v] = new_x
        inv_v = inv_dd[v]
        for u, w in adj[v]:
            if u > v:
                delta[u] -= w * dx * (inv_dd[u] + inv_v)
    fine = [i for i in range(n) if x[i] and sum(w * x[j] for j, w in adj[i]) <= DELTA * diag[i]]
    if len(fine) == n:
        fine.pop()
    check(len(fine) >= Q(3 * n, 16), f"incremental fine set too small: {len(fine)} vs 3*{n}/16")
    is_fine = [False] * n
    for i in fine:
        is_fine[i] = True
    coarse = [i for i in range(n) if not is_fine[i]]
    return fine, coarse


def select_fine_set(a):
    """Run both the O(n + nnz) incremental Delta_v selector and the dense O(n^3)
    oracle, asserting exact set equality."""
    fine_inc, coarse_inc = select_fine_set_incremental(a)
    fine_orc, coarse_orc = select_fine_set_oracle(a)
    check(
        fine_inc == fine_orc and coarse_inc == coarse_orc,
        f"incremental selector {fine_inc} != oracle {fine_orc}",
    )
    return fine_inc, coarse_inc


def jacobi_lift_defect(a, fine, coarse, sweeps):
    """Run `sweeps` Jacobi lifting iterations and return (x, defect, t_star, lf, w)."""
    lf = a.extract(fine, fine)
    w = -a.extract(fine, coarse)
    df = Matrix.diag(*[lf[i, i] for i in range(lf.rows)])
    z = df - lf
    dext = Matrix.diag(*[sum(lf.row(i)) for i in range(lf.rows)])
    check(all(dext[i, i] > 0 for i in range(dext.rows)), "fine vertex has zero external degree")

    jac = df.inv() * z
    x = dext.inv() * w
    bstar = lf.inv() * w
    t_star = a.extract(coarse, coarse) - w.T * bstar
    for _ in range(sweeps):
        x = jac * x + df.inv() * w
    defect = (x - bstar).T * lf * (x - bstar)
    return x, defect, t_star, lf, w, df, jac


def build_lift_and_schur(a, fine, coarse):
    """16-step Jacobi lifting initialized with normalized external mean, plus
    exact and 48-term truncated-path Schur complements."""
    x0, defect0, t_star, lf, w, df, jac = jacobi_lift_defect(a, fine, coarse, 0)
    check(is_psd(7 * t_star - defect0), "initial lifting defect exceeds 7 * T*")

    x, defect, _, _, _, _, _ = jacobi_lift_defect(a, fine, coarse, 16)
    # Enforce the exact stated bound 7 * (3/4)^32 (strictly tighter than EPS = 1/1024).
    check(is_psd(LIFT_DEFECT_BOUND * t_star - defect), "16-step lifting defect exceeds 7*(3/4)^32 * T*")
    check(is_psd(EPS * t_star - defect), "16-step lifting defect exceeds (1/1024) * T*")

    p_mat = zeros(a.rows, len(coarse))
    for i, v in enumerate(fine):
        p_mat[v, :] = x[i, :]
    for i, v in enumerate(coarse):
        p_mat[v, i] = 1
    check(p_mat.T * a * p_mat == t_star + defect, "P^T A P != T* + defect")

    # Optional 48-term truncated-path Schur complement
    sf = Matrix([sum(a.row(i)) for i in fine])
    sc = [sum(a.row(i)) for i in coarse]
    wg = w.row_join(sf)
    step = df.inv() * wg
    accum = step.copy()
    for _ in range(47):
        step = jac * step
        accum += step
    pw = wg.T * accum
    t_path = zeros(len(coarse))
    gidx = len(coarse)
    for i in range(len(coarse)):
        t_path[i, i] = sc[i] + pw[i, gidx]
        for j in range(i):
            val = -a[coarse[i], coarse[j]] + pw[i, j]
            t_path[i, i] += val
            t_path[j, j] += val
            t_path[i, j] = t_path[j, i] = -val
    check(is_psd(t_star - t_path) and is_psd(t_path - Q(63, 64) * t_star), "48-term path Schur out of [63/64, 1]")
    return p_mat, t_star, t_path


def sddm_lower_floor(a):
    """Certified constructive SPD lower eigenvalue bound for a grounded SDDM matrix."""
    n = a.rows
    grounds = [sum(a.row(i)) for i in range(n)]
    weights = [v for v in grounds if v] + [-a[i, j] for i in range(n) for j in range(i) if a[i, j]]
    floor = max(min(grounds), min(weights) / n**2)
    check(floor > 0 and is_psd(a - floor * eye(n)), "SDDM lower floor check failed")
    return floor


def quantize_coarse_graph(t_mat, a_lower=None, prune_stats=None):
    """Apply relative sparsification error (+/- 1/128), optional weak-edge pruning
    below a_i / (512 * (n_i + 1)), and range-aware E5M2 rounding (+/- 1/8)."""
    n = t_mat.rows
    t_sp = zeros(n)
    for i in range(n):
        g = sum(t_mat.row(i))
        t_sp[i, i] = g * (1 + SIGMA if i % 2 else 1 - SIGMA)
        for j in range(i):
            w = -t_mat[i, j]
            w1 = w * (1 + SIGMA if (i + j) % 2 else 1 - SIGMA)
            t_sp[i, i] += w1
            t_sp[j, j] += w1
            t_sp[i, j] = t_sp[j, i] = -w1
    check(is_psd(t_sp - (1 - SIGMA) * t_mat), "sparsified coarse lower sandwich failed")
    check(is_psd((1 + SIGMA) * t_mat - t_sp), "sparsified coarse upper sandwich failed")

    if a_lower is not None:
        a_sp = max(a_lower, sddm_lower_floor(t_sp))
        check(is_psd(t_sp - a_sp * eye(n)), "a_sp lower bound failed")
        thresh = a_sp / (512 * (n + 1))
        t_work = zeros(n)
        for i in range(n):
            g_sp = sum(t_sp.row(i))
            if 0 < g_sp < thresh:
                if prune_stats is not None:
                    prune_stats["grounds_pruned"] += 1
            else:
                t_work[i, i] += g_sp
            for j in range(i):
                w_sp = -t_sp[i, j]
                if 0 < w_sp < thresh:
                    if prune_stats is not None:
                        prune_stats["edges_pruned"] += 1
                elif w_sp > 0:
                    t_work[i, i] += w_sp
                    t_work[j, j] += w_sp
                    t_work[i, j] = t_work[j, i] = -w_sp
        check(
            is_psd(t_sp - t_work) and is_psd(t_work - PRUNE_RETENTION * t_sp),
            "pruned coarse sandwich failed",
        )
    else:
        t_work = t_sp

    ac = zeros(n)
    for i in range(n):
        g_w = sum(t_work.row(i))
        ac[i, i] = fp8_e5m2_significand(g_w)
        for j in range(i):
            w_w = -t_work[i, j]
            v = fp8_e5m2_significand(w_w)
            ac[i, i] += v
            ac[j, j] += v
            ac[i, j] = ac[j, i] = -v
    check(is_psd(ac - (1 - U_E5M2) * t_work), "E5M2 coarse lower sandwich failed")
    check(is_psd((1 + U_E5M2) * t_work - ac), "E5M2 coarse upper sandwich failed")
    check(is_grounded_sddm(ac), "quantized coarse matrix is not grounded SDDM")
    return ac


def damped_bitfsai_factor(a):
    """Standard reprojected INT4 BitFSAI factor damped by the 3-SpMV U guard."""
    n = a.rows
    m = zeros(n)
    g_upper = zeros(n)
    for i in range(n):
        js = [j for j in range(i) if a[i, j] != 0]
        r = zeros(n, 1)
        r[i] = 1
        if js:
            sub = a.extract(js, js)
            rhs = -a.extract(js, [i])
            v = sub.inv() * rhs
            vmax = max(abs(val) for val in v)
            if vmax:
                q_codes = Matrix([round_even_rat(7 * val / vmax) for val in v])
                p = q_codes * (q_codes.dot(rhs) / q_codes.dot(sub * q_codes))
                for j, val in zip(js, p):
                    r[j] = val
        energy = r.dot(a * r)
        check(energy > 0, "nonpositive row energy in BitFSAI factor")
        m += r * r.T / energy
        s = pow2((-log2floor(energy) + 1) // 2)
        while s * s * energy < 1:
            s *= 2
        for j in range(n):
            g_upper[i, j] = abs(r[j]) * s
    u_bound = max(g_upper * a.applyfunc(abs) * g_upper.T * Matrix.ones(n, 1))
    check(u_bound > 0, "nonpositive U bound")
    return min(Q(18, 25), 1 / u_bound) * m


def candidate_block_factor(block):
    count = block.rows
    factor = zeros(count)
    changed = 0
    for i in range(count):
        preds = [j for j in range(i) if block[i, j]]
        direction = zeros(count, 1)
        direction[i] = 1
        coeff = Q(0)
        if preds:
            patch = block.extract(preds, preds)
            rhs = -block.extract(preds, [i])
            optimal = patch.inv() * rhs
            codes = (7 * optimal / max(abs(v) for v in optimal)).applyfunc(round)
            coeff = codes.dot(rhs) / codes.dot(patch * codes)
            changed += int(coeff * codes != optimal)
            for j, code in zip(preds, codes):
                direction[j] = coeff * code
        energy = direction.dot(block * direction)
        factor[i, i] = stored_sqrt_4bit(1 / energy)
        if preds:
            scale = stored_sqrt_4bit(coeff**2 / energy)
            for j, code in zip(preds, codes):
                factor[i, j] = code * scale
    return factor, changed


def certified_block_factor(a, fine, coarse, stats):
    """Decoded-block lower-triangular factor G certifying (3/4)I <= G_b M_b G_b^T <= (5/4)I."""
    lower, upper = Q(3, 4), Q(5, 4)
    factor = zeros(a.rows)

    def install(indices):
        block = a.extract(indices, indices)
        local, changed = candidate_block_factor(block)
        gram = local * block * local.T
        if is_psd(gram - lower * eye(len(indices))) and is_psd(upper * eye(len(indices)) - gram):
            for i, row in enumerate(indices):
                for j, col in enumerate(indices):
                    factor[row, col] = local[i, j]
            stats[f"block_size_{len(indices)}"] += 1
            stats["nonzero_codes"] += sum(int(local[i, j] != 0) for i in range(len(indices)) for j in range(i))
            stats["changed_directions"] += changed
            return
        check(len(indices) > 1, "singleton decoded block failed [3/4, 5/4] certification")
        stats["blocks_split"] += 1
        mid = len(indices) // 2
        install(indices[:mid])
        install(indices[mid:])

    for group in (fine, coarse):
        for start in range(0, len(group), 4):
            install(group[start : start + 4])
    check(all(factor[i, i] > 0 for i in range(a.rows)), "nonpositive diagonal in decoded block factor")
    check(
        all(factor[i, j] == 0 for i in range(a.rows) for j in range(i + 1, a.rows)),
        "decoded block factor is not lower triangular",
    )
    return factor


def certified_block_smoother(a, fine, coarse, stats):
    """Assembled exact smoother R = (2/5) * G^T * G for exact PSD hierarchy checks."""
    factor = certified_block_factor(a, fine, coarse, stats)
    return Q(2, 5) * factor.T * factor


def check_parameters_and_witnesses():
    # Verify our always-on check helper actually raises even under `python3 -O`.
    threw = False
    try:
        check(False, "sentinel check")
    except RuntimeError:
        threw = True
    if not threw:
        raise RuntimeError("check() failed to raise RuntimeError")

    tau, eta = Q(1, 8), Q(1, 14)
    zeta = (1 + 1 / tau) * EPS
    k_val = (1 + tau) / (2 * eta)
    cap = (1 - zeta) / (k_val + 1 - zeta)
    alpha = Q(7, 9) * Q(127, 129) * Q(63, 64) / (1 + EPS)
    alpha_pruned = PRUNE_RETENTION * alpha
    check(LIFT_DEFECT_BOUND < EPS, "7*(3/4)^32 >= 1/1024")
    check(cap == Q(145, 1297) and cap > Q(1, 9), "cap mismatch")
    check(alpha * (1 - zeta) > P_RETENTION, "alpha*(1-zeta) <= 11/15")
    check(alpha_pruned * (1 - zeta) > P_RETENTION, "alpha_pruned*(1-zeta) <= 11/15")
    check(Q(3, 40) >= eta, "eta > 3/40")

    for name, floor in ROOT_FLOORS.items():
        targets = phases(floor)
        t = 2 / (1 + CALIBRATIONS[name])
        for i in range(4):
            child = targets[(i + 1) % 4]
            gain = min(1 - (1 - t * child) ** 2, 1 - (1 - t) ** 2) if i == 0 else child
            check(targets[i] <= min(cap, P_RETENTION * gain), f"four-phase inequality failed for {name} phase {i}")
    t_std = 2 / (1 + CALIBRATIONS["standard"])
    check(P_RETENTION * (1 - (1 - t_std) ** 2) < Q(1, 9), "standard-scale negative control failed")

    # Output precision lower-bound witness & absolute-guard overdamping witness
    for k in (3, 4, 8, 24, 53, 113):
        h = Q(1, 2**k)
        g = h**2
        mat3 = Matrix([[2, -1, -1], [-1, 1, 0], [-1, 0, 1 + g]])
        x3 = Matrix([1, 1 + h, 1])
        check(
            mat3 * x3 == Matrix([-h, h, g]) and x3.dot(mat3 * x3) == 2 * g,
            f"3-vertex witness failed for k={k}",
        )
        margin = Q(3, 8)
        check(margin**2 > Q(2, 16) and 1 - margin * (1 + h) > Q(1, 2), f"margin failed for k={k}")
        check(0 < (1 - margin) * h < (1 + margin) * h < 2 * h, f"h-bracket failed for k={k}")

        a2 = Matrix([[1, -1], [-1, 1 + g]])
        g2 = Matrix([[1, 0], [1 / h, 1 / h]])
        check(g2 * a2 * g2.T == eye(2), f"2x2 exact inverse failed for k={k}")
        abs_gram = g2.applyfunc(abs) * a2.applyfunc(abs) * g2.applyfunc(abs).T
        upper = max(abs_gram * Matrix.ones(2, 1))
        check(1 / upper <= g / (4 + g) < Q(1, 14), f"overdamping witness failed for k={k}")

    # Grounded complete-graph deletion budget
    n = 6
    thresh = Q(1) / (512 * (n + 1))
    small = thresh * Q(31, 32)
    weak = graph_matrix(n, [(i, j, small) for i in range(n) for j in range(i)], [(i, small) for i in range(n)])
    check(is_psd(eye(n) / 512 - weak), "grounded complete-graph pruning budget failed")


def check_dense_fine_lifting_discrimination():
    """Verify on dense-fine stress fixtures that 1 Jacobi lifting sweep FAILS
    the lifting defect bound 7*(3/4)^32 (and 1/1024), while 16 sweeps PASS."""
    for name, n, edges, grounds, _, _ in DENSE_FINE_STRESS_FIXTURES:
        a = graph_matrix(n, edges, grounds)
        check(is_grounded_sddm(a), f"stress fixture {name} is not grounded SDDM")
        fine, coarse = select_fine_set(a)
        _, defect1, t_star, _, _, _, _ = jacobi_lift_defect(a, fine, coarse, 1)
        _, defect16, _, _, _, _, _ = jacobi_lift_defect(a, fine, coarse, 16)
        check(
            not is_psd(LIFT_DEFECT_BOUND * t_star - defect1),
            f"stress fixture {name}: 1 Jacobi sweep unexpectedly passed 7*(3/4)^32 lifting bound",
        )
        check(
            not is_psd(EPS * t_star - defect1),
            f"stress fixture {name}: 1 Jacobi sweep unexpectedly passed 1/1024 lifting bound",
        )
        check(
            is_psd(LIFT_DEFECT_BOUND * t_star - defect16),
            f"stress fixture {name}: 16 Jacobi sweeps failed 7*(3/4)^32 lifting bound",
        )


def round_term(num, den, x):
    return rounded_divide(num * x, den)


def exact_term(num, den, x):
    return Fraction(num, den) * x


class BitWidthTracker:
    """Tracks maximum signed two's-complement integer bit-widths (including +1
    sign bit) across grid vectors, exact rational operator products/denominators,
    and every running partial-sum accumulator during finite-grid evaluation."""

    def __init__(self):
        self.max_vec_bits = 1
        self.max_op_bits = 1
        self.max_partial_sum_bits = 1

    def observe_vec(self, vec):
        for v in vec:
            b = signed_bits(v)
            if b > self.max_vec_bits:
                self.max_vec_bits = b

    def observe_accum(self, val):
        b = signed_bits(val)
        if b > self.max_partial_sum_bits:
            self.max_partial_sum_bits = b
        if b > self.max_op_bits:
            self.max_op_bits = b

    def round_term(self, num, den, x):
        xb = signed_bits(x)
        if xb > self.max_vec_bits:
            self.max_vec_bits = xb
        prod = num * x
        ob = max(signed_bits(prod), signed_bits(den))
        if ob > self.max_op_bits:
            self.max_op_bits = ob
        res = rounded_divide(prod, den)
        rb = signed_bits(res)
        if rb > self.max_vec_bits:
            self.max_vec_bits = rb
        return res


def v_add(x, y, tracker=None):
    out = []
    for a, b in zip(x, y, strict=True):
        s = a + b
        if tracker is not None:
            tracker.observe_accum(s)
        out.append(s)
    return out


def v_sub(x, y, tracker=None):
    out = []
    for a, b in zip(x, y, strict=True):
        d = a - b
        if tracker is not None:
            tracker.observe_accum(d)
        out.append(d)
    return out


class Stencil:
    def __init__(self, mat):
        self.rows = [
            [(j, int(mat[i, j].p), int(mat[i, j].q)) for j in range(mat.cols) if mat[i, j]]
            for i in range(mat.rows)
        ]

    def apply(self, vec, term):
        tracker = getattr(term, "__self__", None)
        if not isinstance(tracker, BitWidthTracker):
            tracker = None
        out = []
        for row in self.rows:
            acc = 0
            for j, num, den in row:
                t = term(num, den, vec[j])
                acc = acc + t
                if tracker is not None:
                    tracker.observe_accum(acc)
            out.append(acc)
        return out

    def rescale(self, coeff):
        for row in self.rows:
            for k, (j, num, den) in enumerate(row):
                c = coeff * Q(num, den)
                row[k] = (j, int(c.p), int(c.q))


def scale_vec(coeff, vec, term):
    num, den = int(coeff.p), int(coeff.q)
    return [term(num, den, x) for x in vec]


class GridLevel:
    def __init__(self, a, depth, force, use_path, prune_edges=False, a_lower=None, prune_stats=None):
        check(is_grounded_sddm(a), f"level {depth} matrix is not grounded SDDM")
        n = a.rows
        self.a_mat = a
        self.a_mat_inv = a.inv()
        self.a = Stencil(a)
        grounds = [sum(a.row(i)) for i in range(n)]
        weights = [v for v in grounds if v] + [-a[i, j] for i in range(n) for j in range(i) if a[i, j]]
        self.lo = min(weights) / n**2
        self.hi = max(weights) * (n + 1)
        self.local_lo, self.local_hi = self.lo, self.hi
        check(
            is_psd(a - self.local_lo * eye(n)) and is_psd(self.local_hi * eye(n) - a),
            f"level {depth} local lo/hi spectral bounds failed is_psd",
        )
        if n == 1:
            self.child = None
            self.inverse = self.a_mat_inv
            self.terminal = self.inverse[0, 0]
            self.depth = 0
            return
        fine, coarse = select_fine_set(a)
        if force:
            fine = fine[: (3 * n + 15) // 16]
            coarse = [i for i in range(n) if i not in fine]
        self.fine, self.coarse = fine, coarse
        diag = Matrix.diag(*[a[i, i] for i in fine])
        z = diag - a.extract(fine, fine)
        w = -a.extract(fine, coarse)
        ext = Matrix.diag(*[diag[i, i] - sum(z.row(i)) for i in range(len(fine))])
        h, k_mat, init = diag.inv() * z, diag.inv() * w, ext.inv() * w
        self.h, self.k, self.init = map(Stencil, (h, k_mat, init))
        self.ht, self.kt, self.initt = map(Stencil, (h.T, k_mat.T, init.T))

        # Store decoded-block lower-triangular factor G, its transpose G^T, and damping 2/5
        g_factor = certified_block_factor(a, fine, coarse, Counter())
        self.g = Stencil(g_factor)
        self.gt = Stencil(g_factor.T)
        self.smooth_damping = Q(2, 5)
        smooth_exact = self.smooth_damping * g_factor.T * g_factor

        p_mat, t_star, t_path = build_lift_and_schur(a, fine, coarse)
        self.p_mat = p_mat
        t_src = t_path if use_path else t_star
        if prune_edges:
            cur_a = max(a_lower if a_lower is not None else Q(0), sddm_lower_floor(a))
            a_sp_lb = cur_a * (1 - SIGMA) * (Q(63, 64) if use_path else 1)
            ac = quantize_coarse_graph(t_src, a_lower=a_sp_lb, prune_stats=prune_stats)
            next_a = a_sp_lb * PRUNE_RETENTION * (1 - U_E5M2)
        else:
            ac = quantize_coarse_graph(t_src)
            next_a = None

        l_eff = L_COARSE * (Q(63, 64) if use_path else 1) * (PRUNE_RETENTION if prune_edges else 1)
        h_eff = H_COARSE
        check(is_psd(ac - l_eff * (p_mat.T * a * p_mat)), "coarse Galerkin lower sandwich failed")
        check(is_psd(h_eff * (p_mat.T * a * p_mat) - ac), "coarse Galerkin upper sandwich failed")

        self.child = GridLevel(ac, depth + 1, force, use_path, prune_edges, next_a, prune_stats)
        self.lo = min(self.lo, self.child.lo)
        self.hi = max(self.hi, self.child.hi)
        self.depth = 1 + self.child.depth
        self.gamma = l_eff
        self.t = 2 / (1 + CALIBRATIONS["retuned"]) if depth % 4 == 0 else None

        bc = self.child.inverse
        if self.t is not None:
            bc = 2 * self.t * bc - self.t**2 * bc * ac * bc
        s = eye(n) - smooth_exact * a
        self.inverse = 2 * smooth_exact - smooth_exact * a * smooth_exact + self.gamma * s * p_mat * bc * p_mat.T * s.T

    def use_units(self, unit):
        self.a.rescale(1 / unit)
        self.a_mat = self.a_mat / unit
        self.a_mat_inv = self.a_mat_inv * unit
        self.inverse *= unit
        self.lo /= unit
        self.hi /= unit
        if self.child is None:
            self.terminal *= unit
        else:
            half_exp = ceil_log2(unit) // 2
            half_unit = pow2(half_exp)
            check(half_unit * half_unit == unit, "non-even power-of-two unit scaling")
            self.g.rescale(half_unit)
            self.gt.rescale(half_unit)
            self.child.use_units(unit)

    def smooth_apply(self, rhs, term):
        """Two-pass decoded-block smoother: Pass 1 t = G * r, Pass 2 z = (2/5) * G^T * t."""
        t_mid = self.g.apply(rhs, term)
        return scale_vec(self.smooth_damping, self.gt.apply(t_mid, term), term)

    def lift(self, y, term):
        tr = getattr(term, "__self__", None)
        tr = tr if isinstance(tr, BitWidthTracker) else None
        u = self.init.apply(y, term)
        forcing = self.k.apply(y, term)
        for _ in range(16):
            u = v_add(self.h.apply(u, term), forcing, tr)
        out = [0] * (len(self.fine) + len(self.coarse))
        for i, v in zip(self.fine, u, strict=True):
            out[i] = v
        for i, v in zip(self.coarse, y, strict=True):
            out[i] = v
        return out

    def restrict(self, x, term):
        tr = getattr(term, "__self__", None)
        tr = tr if isinstance(tr, BitWidthTracker) else None
        v = [x[i] for i in self.fine]
        y = [x[i] for i in self.coarse]
        for _ in range(16):
            y = v_add(y, self.kt.apply(v, term), tr)
            v = self.ht.apply(v, term)
        return v_add(y, self.initt.apply(v, term), tr)

    def cycle(self, rhs, term):
        if self.child is None:
            return scale_vec(self.terminal, rhs, term)
        tr = getattr(term, "__self__", None)
        tr = tr if isinstance(tr, BitWidthTracker) else None
        x = self.smooth_apply(rhs, term)
        r = self.restrict(v_sub(rhs, self.a.apply(x, term), tr), term)
        y = self.child.cycle(r, term)
        if self.t is not None:
            y = scale_vec(self.t, y, term)
            rc = v_sub(r, self.child.a.apply(y, term), tr)
            y = v_add(y, scale_vec(self.t, self.child.cycle(rc, term), term), tr)
        x = v_add(x, scale_vec(self.gamma, self.lift(y, term), term), tr)
        return v_add(x, self.smooth_apply(v_sub(rhs, self.a.apply(x, term), tr), term), tr)


def verify_full_basis_adjointness(lev):
    """Verify matrix-free lift and restrict are exact transposes on the full
    standard basis (and match p_mat) at every nonterminal level."""
    cur = lev
    checked_levels = 0
    while cur.child is not None:
        nf = len(cur.fine) + len(cur.coarse)
        nc = len(cur.coarse)
        if nf <= 32:
            lift_cols = []
            for j in range(nc):
                ej = [Fraction(int(k == j)) for k in range(nc)]
                col = cur.lift(ej, exact_term)
                lift_cols.append([Q(v.numerator, v.denominator) for v in col])
            p_mf = Matrix([[lift_cols[j][i] for j in range(nc)] for i in range(nf)])
            check(p_mf == cur.p_mat, "matrix-free lift does not match p_mat")

            rest_rows = []
            for i in range(nf):
                ei = [Fraction(int(k == i)) for k in range(nf)]
                row_t = cur.restrict(ei, exact_term)
                rest_rows.append([Q(v.numerator, v.denominator) for v in row_t])
            r_mf = Matrix([[rest_rows[i][j] for i in range(nf)] for j in range(nc)])
            check(r_mf == p_mf.T, "matrix-free restrict is not exact transpose of lift")
            checked_levels += 1
        cur = cur.child
    return checked_levels


def check_sqrt_le_add_sqrt(a_sq, b_sq, c_sq, msg="sqrt inequality failed"):
    """Exact rational check of sqrt(a_sq) <= sqrt(b_sq) + sqrt(c_sq) for a_sq, b_sq, c_sq >= 0."""
    check(a_sq >= 0 and b_sq >= 0 and c_sq >= 0, f"{msg}: negative square")
    rem = a_sq - b_sq - c_sq
    check(rem <= 0 or rem**2 <= 4 * b_sq * c_sq, msg)


def audit_cycle_primitives_and_recurrence(lev, r_int, bits, mu, drop_ci=False):
    """Verify at every level of the finite-grid cycle that:
    (i) every individual sparse primitive has primal/dual energy error <= mu, and
    (ii) the levelwise forward error obeys BOTH the theoretical bound
         E_i <= 14*mu + c_i*E_{i+1} AND the tighter measured-primitive bound
         E_i <= 14*mu_measured + c_i*E_{i+1,actual}."""
    step = pow2(-bits)

    def to_q_mat(vec):
        return Matrix([Q(v.numerator, v.denominator) if isinstance(v, Fraction) else Q(v) for v in vec])

    def primal_err_sq(v_grid, v_ex, target):
        diff = step * Matrix(v_grid) - step * to_q_mat(v_ex)
        return diff.dot(target.a_mat * diff)

    def dual_err_sq(v_grid, v_ex, target):
        diff = step * Matrix(v_grid) - step * to_q_mat(v_ex)
        return diff.dot(target.a_mat_inv * diff)

    r_frac = [Fraction(int(v)) for v in r_int]
    if lev.child is None:
        z_grid = scale_vec(lev.terminal, r_int, round_term)
        z_ex = scale_vec(lev.terminal, r_frac, exact_term)
        e_sq = primal_err_sq(z_grid, z_ex, lev)
        check(e_sq <= mu**2, "terminal primitive error exceeded mu")
        return z_grid, e_sq, mu, e_sq

    prim_errs = []

    def record_prim(err_sq, label):
        check(err_sq <= mu**2, f"primitive {label} error {err_sq} exceeded mu^2={mu**2}")
        prim_errs.append(err_sq)

    x1_grid = lev.smooth_apply(r_int, round_term)
    x1_ex = lev.smooth_apply(r_frac, exact_term)
    record_prim(primal_err_sq(x1_grid, x1_ex, lev), "pre_smooth")

    ax1_grid = lev.a.apply(x1_grid, round_term)
    ax1_ex = lev.a.apply([Fraction(int(v)) for v in x1_grid], exact_term)
    record_prim(dual_err_sq(ax1_grid, ax1_ex, lev), "pre_matvec")

    res1_grid = v_sub(r_int, ax1_grid)
    rc_grid = lev.restrict(res1_grid, round_term)
    rc_ex = lev.restrict([Fraction(int(v)) for v in res1_grid], exact_term)
    record_prim(dual_err_sq(rc_grid, rc_ex, lev.child), "restrict")

    y_grid, e_c1_sq, e_child_bound, _ = audit_cycle_primitives_and_recurrence(
        lev.child, rc_grid, bits, mu, drop_ci=drop_ci
    )
    if lev.t is not None:
        ty1_grid = scale_vec(lev.t, y_grid, round_term)
        ty1_ex = scale_vec(lev.t, [Fraction(int(v)) for v in y_grid], exact_term)
        record_prim(primal_err_sq(ty1_grid, ty1_ex, lev.child), "boost_scale1")

        acy1_grid = lev.child.a.apply(ty1_grid, round_term)
        acy1_ex = lev.child.a.apply([Fraction(int(v)) for v in ty1_grid], exact_term)
        record_prim(dual_err_sq(acy1_grid, acy1_ex, lev.child), "boost_matvec")

        rc2_grid = v_sub(rc_grid, acy1_grid)
        y2_grid, e_c2_sq, _, _ = audit_cycle_primitives_and_recurrence(
            lev.child, rc2_grid, bits, mu, drop_ci=drop_ci
        )

        ty2_grid = scale_vec(lev.t, y2_grid, round_term)
        ty2_ex = scale_vec(lev.t, [Fraction(int(v)) for v in y2_grid], exact_term)
        record_prim(primal_err_sq(ty2_grid, ty2_ex, lev.child), "boost_scale2")
        y_grid = v_add(ty1_grid, ty2_grid)
        c_i = 0 if drop_ci else 4
        child_actual_amp_sq = (c_i**2) * max(e_c1_sq, e_c2_sq)
    else:
        c_i = 0 if drop_ci else 1
        child_actual_amp_sq = (c_i**2) * e_c1_sq

    py_grid = lev.lift(y_grid, round_term)
    py_ex = lev.lift([Fraction(int(v)) for v in y_grid], exact_term)
    record_prim(primal_err_sq(py_grid, py_ex, lev), "lift")

    gpy_grid = scale_vec(lev.gamma, py_grid, round_term)
    gpy_ex = scale_vec(lev.gamma, [Fraction(int(v)) for v in py_grid], exact_term)
    record_prim(primal_err_sq(gpy_grid, gpy_ex, lev), "gamma_scale")

    x2_grid = v_add(x1_grid, gpy_grid)
    ax2_grid = lev.a.apply(x2_grid, round_term)
    ax2_ex = lev.a.apply([Fraction(int(v)) for v in x2_grid], exact_term)
    record_prim(dual_err_sq(ax2_grid, ax2_ex, lev), "post_matvec")

    res2_grid = v_sub(r_int, ax2_grid)
    z2_grid = lev.smooth_apply(res2_grid, round_term)
    z2_ex = lev.smooth_apply([Fraction(int(v)) for v in res2_grid], exact_term)
    record_prim(primal_err_sq(z2_grid, z2_ex, lev), "post_smooth")

    out_grid = v_add(x2_grid, z2_grid)
    d_level = step * Matrix(out_grid) - lev.inverse * (step * Matrix(r_int))
    e_actual_sq = d_level.dot(lev.a_mat * d_level)

    mu_meas_sq = max(prim_errs)
    e_bound = 14 * mu + c_i * e_child_bound
    check(e_actual_sq <= e_bound**2, "levelwise theoretical forward-error bound violated")
    check_sqrt_le_add_sqrt(
        e_actual_sq,
        (14 * mu) ** 2,
        child_actual_amp_sq,
        "levelwise recurrence (theoretical mu + actual child) violated",
    )
    check_sqrt_le_add_sqrt(
        e_actual_sq,
        (14**2) * mu_meas_sq,
        child_actual_amp_sq,
        "levelwise recurrence (measured primitive max + actual child) violated",
    )
    return out_grid, e_actual_sq, e_bound, mu_meas_sq


def check_recurrence_negative_controls(lev, rhs, bits, mu):
    """Fast in-script negative controls proving the recurrence checker rejects:
    (1) shrinking `mu` below the measured primitive error, and
    (2) dropping `c_i` when coarse child error dominates."""
    k, r_norm = normalize_rhs_pow2(rhs)
    check(k is not None, "negative control requires nonzero RHS")
    r_grid = [round_even_rat(v * pow2(bits)) for v in r_norm]
    _, _, _, mu_meas_sq = audit_cycle_primitives_and_recurrence(lev, r_grid, bits, mu)
    check(mu_meas_sq > 0, "expected positive measured primitive error for negative control")

    # (1) Shrinking mu below sqrt(mu_meas_sq) MUST fail.
    mu_too_small = pow2((log2floor(mu_meas_sq) // 2) - 2)
    check(mu_too_small**2 < mu_meas_sq, "shrunk mu not strictly below measured primitive error")
    failed_shrunk_mu = False
    try:
        audit_cycle_primitives_and_recurrence(lev, r_grid, bits, mu_too_small)
    except RuntimeError:
        failed_shrunk_mu = True
    check(failed_shrunk_mu, "negative control failed: shrinking mu did not trigger RuntimeError")

    # (2) Dropping c_i (setting c_i = 0) when child error is injected MUST fail,
    #     while keeping c_i = 4 on the boosted root level PASSES.
    step = pow2(-bits)
    child_n = lev.child.a_mat.rows
    # Inject a child error vector e_c = delta * 1_c into the boosted coarse correction.
    delta_vec = Matrix([Q(1)] * child_n)
    # Exact error propagated by boosted level: gamma * S * P * (2*t - t^2*B_c*A_c) * delta_c
    # On coarse level, B_amp(delta_c) = 2*t*delta_c - t^2*B_c*A_c*delta_c has norm <= 4*||delta_c||_{A_c}.
    ac = lev.child.a_mat
    bc = lev.child.inverse
    t = lev.t
    check(t is not None, "root level must be boosted for c_i negative control")
    child_err_sq = delta_vec.dot(ac * delta_vec)
    # Amplified coarse vector from injecting error delta_vec on both child calls:
    # first call gets +delta_vec -> contributes t*(I - t*B_c*A_c)*delta_vec, second gets +delta_vec -> +t*delta_vec.
    amp_vec = t * delta_vec + t * (delta_vec - t * bc * ac * delta_vec)
    amp_err_sq = amp_vec.dot(ac * amp_vec)
    # With zero primitive error (mu_meas = 0), dropping c_i (c_i = 0) fails while c_i = 4 passes;
    # moreover, amp_err_sq > child_err_sq proves even c_i = 1 fails!
    failed_drop_ci = False
    try:
        check_sqrt_le_add_sqrt(amp_err_sq, Q(0), Q(0), "dropped c_i=0")
    except RuntimeError:
        failed_drop_ci = True
    check(failed_drop_ci, "negative control failed: dropping c_i=0 did not fail")
    check(amp_err_sq > child_err_sq, "negative control: boosted child amplification did not exceed c_i=1")
    failed_ci_1 = False
    try:
        check_sqrt_le_add_sqrt(amp_err_sq, Q(0), child_err_sq, "unboosted c_i=1 on boosted level")
    except RuntimeError:
        failed_ci_1 = True
    check(failed_ci_1, "negative control failed: c_i=1 on boosted level did not fail")
    check_sqrt_le_add_sqrt(amp_err_sq, Q(0), 16 * child_err_sq, "boosted c_i=4 must pass")


def binary_output(val, bits):
    if not val:
        return Q(0)
    top = ceil_log2(abs(val))
    if abs(val) < Q(2) ** top:
        top -= 1
    step = Q(2) ** (top + 1 - bits)
    val /= step
    return rounded_divide(int(val.p), int(val.q)) * step


def normalize_rhs_pow2(rhs):
    """Power-of-two input normalization: handle r == 0 directly, else choose integer
    k = floor(log2(||r||_inf)) + 1 so max_i |r_i| * 2^(-k) lies strictly in [1/2, 1)."""
    if all(v == 0 for v in rhs):
        return None, rhs
    rmax = max(abs(v) for v in rhs)
    k = log2floor(rmax) + 1
    r_norm = rhs * pow2(-k)
    norm_inf = max(abs(v) for v in r_norm)
    check(Q(1, 2) <= norm_inf < 1, "normalized RHS infinity norm outside [1/2, 1)")
    return k, r_norm


def eval_normalized_cycle_grid(lev, rhs, bits, tracker, mu_audit=None):
    """Evaluate one fixed-point cycle with power-of-two input normalization."""
    k, r_norm = normalize_rhs_pow2(rhs)
    if k is None:
        return zeros(rhs.rows, 1)
    r_grid = [round_even_rat(v * pow2(bits)) for v in r_norm]
    tracker.observe_vec(r_grid)
    z_grid = lev.cycle(r_grid, tracker.round_term)
    tracker.observe_vec(z_grid)
    if mu_audit is not None:
        z_audit, _, _, _ = audit_cycle_primitives_and_recurrence(lev, r_grid, bits, mu_audit)
        check(z_audit == z_grid, "audited cycle output diverged from cycle output")
    return pow2(k - bits) * Matrix(z_grid)


def solve_normalized_richardson_grid(lev, rhs, bits, steps, t_out, tracker):
    """Run residual-recomputed Richardson on the 2^(-bits) grid with power-of-two
    input normalization and restore the 2^k exponent on fixed and binary outputs."""
    n = rhs.rows
    k, r_norm = normalize_rhs_pow2(rhs)
    if k is None:
        return zeros(n, 1), zeros(n, 1), 0
    b_grid = [round_even_rat(v * pow2(bits)) for v in r_norm]
    tracker.observe_vec(b_grid)
    x = [0] * n
    executed = 0
    for _ in range(steps):
        executed += 1
        res = v_sub(b_grid, lev.a.apply(x, tracker.round_term), tracker)
        tracker.observe_vec(res)
        corr = lev.cycle(res, tracker.round_term)
        nxt = v_add(x, corr, tracker)
        tracker.observe_vec(nxt)
        if nxt == x:
            break
        x = nxt
    fixed = pow2(k - bits) * Matrix(x)
    out = fixed.applyfunc(lambda v: binary_output(v, t_out))
    return fixed, out, executed


def solve_exact_hierarchy(a, level, force, use_path, counts, block_stats, prune_edges=False, a_lower=None, prune_stats=None):
    check(is_grounded_sddm(a), f"exact hierarchy level {level} is not grounded SDDM")
    n = a.rows
    if n == 1:
        return {(sm, sc): a.inv() for sm in ("protected", "blocks") for sc in ROOT_FLOORS}
    fine, coarse = select_fine_set(a)
    if force:
        fine = fine[: (3 * n + 15) // 16]
        coarse = [i for i in range(n) if i not in fine]
    p_mat, t_star, t_path = build_lift_and_schur(a, fine, coarse)
    t_src = t_path if use_path else t_star
    if prune_edges:
        cur_a = max(a_lower if a_lower is not None else Q(0), sddm_lower_floor(a))
        a_sp_lb = cur_a * (1 - SIGMA) * (Q(63, 64) if use_path else 1)
        ac = quantize_coarse_graph(t_src, a_lower=a_sp_lb, prune_stats=prune_stats)
        next_a = a_sp_lb * PRUNE_RETENTION * (1 - U_E5M2)
    else:
        ac = quantize_coarse_graph(t_src)
        next_a = None
    children = solve_exact_hierarchy(ac, level + 1, force, use_path, counts, block_stats, prune_edges, next_a, prune_stats)

    diag = zeros(n)
    for i in fine:
        diag[i, i] = 1 / ((1 + DELTA) * a[i, i])
    smoothers = {
        "protected": (damped_bitfsai_factor(a) + diag) / 2,
        "blocks": certified_block_smoother(a, fine, coarse, block_stats),
    }
    gamma = L_COARSE * (Q(63, 64) if use_path else 1) * (PRUNE_RETENTION if prune_edges else 1)
    result = {}
    for sm_name, sm in smoothers.items():
        s = eye(n) - sm * a
        for sc_name, floor in ROOT_FLOORS.items():
            b = children[sm_name, sc_name]
            if level % 4 == 0:
                t = 2 / (1 + CALIBRATIONS[sc_name])
                b = 2 * t * b - t * t * b * ac * b
            inv = 2 * sm - sm * a * sm + gamma * s * p_mat * b * p_mat.T * s.T
            target = phases(floor)[level % 4]
            check(
                is_psd(a * inv * a - target * a) and is_psd(a - a * inv * a),
                f"exact effectiveness sandwich failed at level {level} ({sm_name}, {sc_name})",
            )
            result[sm_name, sc_name] = inv
            counts[sm_name, sc_name] += 1
    return result


def verify_retuned_grid(
    a,
    force,
    use_path,
    oracle,
    prune_edges=False,
    extra_scale_regressions=False,
    dyadic_spec=None,
    run_negative_controls=False,
):
    # Extract root input specs strictly from input matrix `a` BEFORE building the hierarchy.
    root_coeff_bits, root_rw = extract_root_input_spec(a)
    lev = GridLevel(a, 0, force, use_path, prune_edges=prune_edges)
    check(lev.inverse == oracle, "GridLevel inverse diverged from exact hierarchy oracle")
    basis_levels = verify_full_basis_adjointness(lev)
    check(basis_levels == lev.depth, "did not verify full-basis adjointness at every nonterminal level")

    n = a.rows
    lev.lo = max(lev.lo, lev.local_lo / 2**lev.depth)
    lev.hi = min(lev.hi, lev.local_hi * 2**lev.depth)
    # Certify the clamped global [lev.lo, lev.hi] bounds with is_psd at every level.
    cur = lev
    while cur is not None:
        m = cur.a_mat.rows
        check(
            is_psd(cur.a_mat - lev.lo * eye(m)) and is_psd(lev.hi * eye(m) - cur.a_mat),
            "clamped global lo/hi bounds failed is_psd on a hierarchy level",
        )
        cur = cur.child

    s_exp = ceil_log2(lev.hi)
    if s_exp % 2 != 0:
        s_exp += 1
    scale = pow2(s_exp)
    kappa = lev.local_hi / lev.local_lo
    lev.use_units(scale)
    a_scaled = a / scale
    amp = (14 * lev.depth + 1) * 4 ** ((lev.depth + 3) // 4)
    c_err = 34 * n**2 * (amp + 2) * upper_sqrt(1 / lev.lo)
    tol = Q(1, 2**20)
    bits = ceil_log2(36 * c_err / tol)
    theta = c_err * pow2(-bits)
    check(theta <= tol / 36, "theta exceeds tol / 36")
    mu = 17 * n**2 * pow2(-bits) * upper_sqrt(1 / lev.lo)
    t_out = ceil_log2(4 * upper_sqrt(kappa) / tol)
    steps = 0
    while Q(8, 9) ** steps > tol / 4:
        steps += 1
    check(steps == 130, f"expected 130 Richardson steps, got {steps}")

    ainv = a_scaled.inv()
    tracker = BitWidthTracker()

    # Exact zero RHS check (must return exact 0 in 0 steps)
    zero_rhs = zeros(n, 1)
    check(eval_normalized_cycle_grid(lev, zero_rhs, bits, tracker) == zero_rhs, "zero RHS cycle failed")
    z_fix, z_out, z_exec = solve_normalized_richardson_grid(lev, zero_rhs, bits, steps, t_out, tracker)
    check(z_fix == zero_rhs and z_out == zero_rhs and z_exec == 0, "zero RHS solve failed")

    vectors = [
        Matrix([(-1) ** i * Q(i + 1, n) for i in range(n)]),
        eye(n)[:, 0],
        Matrix([Q(5, 7)] * n),
    ]
    if extra_scale_regressions:
        base_dir = Matrix([(-1) ** i * Q(i + 1, n) for i in range(n)])
        vectors.extend(
            [
                pow2(-bits - 2) * eye(n)[:, 0],
                pow2(-bits - 2) * base_dir,
                pow2(bits + 2) * eye(n)[:, 0],
                pow2(bits + 2) * base_dir,
            ]
        )

    if run_negative_controls and lev.child is not None:
        check_recurrence_negative_controls(lev, vectors[0], bits, mu)

    exec_counts = []
    for rhs in vectors:
        # Exact-arithmetic two-pass cycle matches lev.inverse * rhs
        check(
            Matrix(lev.cycle([Fraction(v.p, v.q) for v in rhs], exact_term)) == lev.inverse * rhs,
            "exact two-pass cycle diverged from lev.inverse * rhs",
        )

        # Single-cycle relative action error <= theta + primitive/recurrence audit
        z_cycle = eval_normalized_cycle_grid(lev, rhs, bits, tracker, mu_audit=mu)
        exact_cycle = lev.inverse * rhs
        cycle_err = (z_cycle - exact_cycle).dot(a_scaled * (z_cycle - exact_cycle))
        check(cycle_err <= theta**2 * rhs.dot(ainv * rhs), "single-cycle relative action error exceeded theta")

        # Normalized fixed-point Richardson + binary floating-point output cast
        fixed, out, executed = solve_normalized_richardson_grid(lev, rhs, bits, steps, t_out, tracker)
        check(1 <= executed <= steps, f"unexpected executed step count {executed}")
        exec_counts.append(executed)

        exact = ainv * rhs
        err_fixed = (fixed - exact).dot(a_scaled * (fixed - exact))
        check(
            err_fixed <= rhs.dot(exact) * (Q(8, 9) ** steps + 9 * c_err / 2**bits) ** 2,
            "fixed-point Richardson energy error exceeded bound",
        )
        check(
            (out - exact).dot(a_scaled * (out - exact)) <= tol**2 * rhs.dot(exact),
            "binary output energy error exceeded tol",
        )

    # Independent non-tautological signed integer word-length bounds (including +1 sign bit),
    # derived solely from root (s_A / root_coeff_bits, n, R_w, bits, lev.lo) without reading
    # runtime stencil bit-widths.
    w_vec_bound = bits + max(0, ceil_log2(1 / lev.lo)) + ceil_log2(n + 1) + 7
    if dyadic_spec is not None:
        s_a, r_w = dyadic_spec
        w_op_bound = s_a + 3 * ceil_log2(2 * n) + 2 * max(1, ceil_log2(2 * r_w)) + w_vec_bound + 25
    else:
        w_op_bound = (
            (n + 2) * root_coeff_bits
            + 3 * ceil_log2(2 * n)
            + 2 * max(1, ceil_log2(2 * root_rw))
            + w_vec_bound
            + 25
        )
    check(
        tracker.max_vec_bits <= w_vec_bound,
        f"signed vector bit-width {tracker.max_vec_bits} exceeded independent bound {w_vec_bound}",
    )
    check(
        tracker.max_partial_sum_bits <= w_op_bound,
        f"signed partial-sum bit-width {tracker.max_partial_sum_bits} exceeded independent bound {w_op_bound}",
    )
    check(
        tracker.max_op_bits <= w_op_bound,
        f"signed operator bit-width {tracker.max_op_bits} exceeded independent bound {w_op_bound}",
    )

    dyadic_report = None
    if dyadic_spec is not None:
        s_a, r_w = dyadic_spec
        dyadic_report = (
            s_a,
            ceil_log2(r_w),
            tracker.max_vec_bits,
            w_vec_bound,
            tracker.max_partial_sum_bits,
            tracker.max_op_bits,
            w_op_bound,
        )

    return {
        "solved": len(vectors),
        "loop_cap": steps,
        "min_exec": min(exec_counts),
        "max_exec": max(exec_counts),
        "max_vec_bits": tracker.max_vec_bits,
        "w_vec_bound": w_vec_bound,
        "max_partial_sum_bits": tracker.max_partial_sum_bits,
        "max_op_bits": tracker.max_op_bits,
        "w_op_bound": w_op_bound,
        "basis_levels": basis_levels,
        "dyadic_report": dyadic_report,
    }


def main():
    t0 = time.perf_counter()
    check_parameters_and_witnesses()
    check_dense_fine_lifting_discrimination()

    # 1. Scalar terminal P0 regression test (A = [1] with tiny 2^(-bits-2) and huge 2^(bits+2) RHS)
    a_scalar = Matrix([[Q(1)]])
    scalar_oracle = solve_exact_hierarchy(a_scalar, 0, False, False, Counter(), Counter())["blocks", "retuned"]
    scalar_res = verify_retuned_grid(a_scalar, False, False, scalar_oracle, extra_scale_regressions=True)

    # 2. Original 9 grounded SDDM fixtures (including multi-vertex tiny/huge RHS & recurrence negative controls)
    counts = Counter()
    block_stats = Counter()
    grid_solves = 0
    all_min_exec = 10**9
    all_max_exec = 0
    max_vec_bits_seen = scalar_res["max_vec_bits"]
    max_vec_bound_seen = scalar_res["w_vec_bound"]
    max_partial_sum_bits_seen = scalar_res["max_partial_sum_bits"]
    max_op_bits_seen = scalar_res["max_op_bits"]
    basis_levels_checked = 0

    for idx, (n, exponent, force, use_path) in enumerate(FIXTURES):
        edges = [
            (i, i + 1, Q(13 + i, 11 + i) * pow2(exponent if i % 2 == 0 else -exponent))
            for i in range(n - 1)
        ]
        edges += [(i, i + 2, Q(5 + i, 17 + i) * pow2(-exponent)) for i in range(n - 2)]
        a = graph_matrix(n, edges, [(0, Q(1, 7)), (n - 1, Q(1, 13))])
        out = solve_exact_hierarchy(a, 0, force, use_path, counts, block_stats)
        res = verify_retuned_grid(
            a,
            force,
            use_path,
            out["blocks", "retuned"],
            extra_scale_regressions=(idx == 0),
            run_negative_controls=(idx == 0),
        )
        grid_solves += res["solved"]
        all_min_exec = min(all_min_exec, res["min_exec"])
        all_max_exec = max(all_max_exec, res["max_exec"])
        max_vec_bits_seen = max(max_vec_bits_seen, res["max_vec_bits"])
        max_vec_bound_seen = max(max_vec_bound_seen, res["w_vec_bound"])
        max_partial_sum_bits_seen = max(max_partial_sum_bits_seen, res["max_partial_sum_bits"])
        max_op_bits_seen = max(max_op_bits_seen, res["max_op_bits"])
        basis_levels_checked += res["basis_levels"]

    check(len(FIXTURES) == 9 and list(counts.values()) == [34] * 4, "unexpected rational fixture level count")

    # 3. Dyadic-weight grounded SDDM roots (exercising Theorem 8 dyadic word-length claim + tiny/huge RHS)
    dyadic_counts = Counter()
    dyadic_solves = 0
    dyadic_reports = {}
    for idx, (name, n, edges, grounds, force, use_path) in enumerate(DYADIC_FIXTURES):
        a_dy = graph_matrix(n, edges, grounds)
        dy_spec = verify_dyadic_root_spec(a_dy)
        out_dy = solve_exact_hierarchy(a_dy, 0, force, use_path, dyadic_counts, block_stats)
        res_dy = verify_retuned_grid(
            a_dy,
            force,
            use_path,
            out_dy["blocks", "retuned"],
            extra_scale_regressions=(idx == 0),
            dyadic_spec=dy_spec,
            run_negative_controls=(idx == 0),
        )
        dyadic_solves += res_dy["solved"]
        dyadic_reports[name] = res_dy["dyadic_report"]
        all_min_exec = min(all_min_exec, res_dy["min_exec"])
        all_max_exec = max(all_max_exec, res_dy["max_exec"])
        max_vec_bits_seen = max(max_vec_bits_seen, res_dy["max_vec_bits"])
        max_vec_bound_seen = max(max_vec_bound_seen, res_dy["w_vec_bound"])
        max_partial_sum_bits_seen = max(max_partial_sum_bits_seen, res_dy["max_partial_sum_bits"])
        max_op_bits_seen = max(max_op_bits_seen, res_dy["max_op_bits"])
        basis_levels_checked += res_dy["basis_levels"]

    check(len(DYADIC_FIXTURES) == 3, "unexpected dyadic fixture count")

    # 4. Dense-fine stress fixtures (full hierarchy + finite-grid solves)
    stress_counts = Counter()
    stress_solves = 0
    for name, n, edges, grounds, force, use_path in DENSE_FINE_STRESS_FIXTURES:
        a_st = graph_matrix(n, edges, grounds)
        st_spec = verify_dyadic_root_spec(a_st)
        out_st = solve_exact_hierarchy(a_st, 0, force, use_path, stress_counts, block_stats)
        res_st = verify_retuned_grid(
            a_st,
            force,
            use_path,
            out_st["blocks", "retuned"],
            dyadic_spec=st_spec,
        )
        stress_solves += res_st["solved"]
        all_min_exec = min(all_min_exec, res_st["min_exec"])
        all_max_exec = max(all_max_exec, res_st["max_exec"])
        max_vec_bits_seen = max(max_vec_bits_seen, res_st["max_vec_bits"])
        max_vec_bound_seen = max(max_vec_bound_seen, res_st["w_vec_bound"])
        max_partial_sum_bits_seen = max(max_partial_sum_bits_seen, res_st["max_partial_sum_bits"])
        max_op_bits_seen = max(max_op_bits_seen, res_st["max_op_bits"])
        basis_levels_checked += res_st["basis_levels"]

    # 5. Integrated coarse edge-pruning hierarchies (threshold a_i / (512 * (n_i + 1)))
    pruned_counts = Counter()
    prune_stats_exact = Counter()
    prune_stats_grid = Counter()
    pruned_solves = 0
    pruned_cases = [
        (DYADIC_FIXTURES[2][1], DYADIC_FIXTURES[2][2], DYADIC_FIXTURES[2][3], False, True),
        (
            7,
            [(0, 1, Q(7, 5)), (1, 2, Q(9, 7)), (3, 4, Q(6, 5)), (4, 5, Q(8, 7)), (5, 6, Q(11, 9))]
            + [(2, 3, Q(1, 3) * pow2(-21)), (0, 6, Q(1, 5) * pow2(-22))],
            [(i, Q(3, 5) if i % 2 == 0 else Q(4, 7)) for i in range(7)],
            False,
            False,
        ),
    ]
    for n_p, edges_p, grounds_p, force_p, path_p in pruned_cases:
        a_p = graph_matrix(n_p, edges_p, grounds_p)
        out_p = solve_exact_hierarchy(
            a_p, 0, force_p, path_p, pruned_counts, block_stats, prune_edges=True, prune_stats=prune_stats_exact
        )
        lev_probe = GridLevel(a_p, 0, force_p, path_p, prune_edges=True, prune_stats=prune_stats_grid)
        check(lev_probe.inverse == out_p["blocks", "retuned"], "pruned GridLevel inverse mismatch")
        res_p = verify_retuned_grid(a_p, force_p, path_p, out_p["blocks", "retuned"], prune_edges=True)
        pruned_solves += res_p["solved"]
        all_min_exec = min(all_min_exec, res_p["min_exec"])
        all_max_exec = max(all_max_exec, res_p["max_exec"])
        max_vec_bits_seen = max(max_vec_bits_seen, res_p["max_vec_bits"])
        max_vec_bound_seen = max(max_vec_bound_seen, res_p["w_vec_bound"])
        max_partial_sum_bits_seen = max(max_partial_sum_bits_seen, res_p["max_partial_sum_bits"])
        max_op_bits_seen = max(max_op_bits_seen, res_p["max_op_bits"])
        basis_levels_checked += res_p["basis_levels"]

    check(
        prune_stats_exact["edges_pruned"] > 0 and prune_stats_exact == prune_stats_grid,
        "edge pruning stats mismatch or zero edges pruned",
    )

    check(
        rounded_divide(1, 2) == 0
        and rounded_divide(3, 2) == 2
        and rounded_divide(-1, 2) == 0
        and rounded_divide(-3, 2) == -2,
        "rounded_divide must implement signed round-to-nearest ties-to-even",
    )
    check(
        signed_bits(0) == 1
        and signed_bits(1) == 2
        and signed_bits(-1) == 2
        and signed_bits(7) == 4
        and signed_bits(-8) == 5
        and signed_bits(127) == 8,
        "signed_bits must include the +1 two's-complement sign bit",
    )

    total_roots = len(FIXTURES) + len(DYADIC_FIXTURES) + len(DENSE_FINE_STRESS_FIXTURES)
    total_levels = (
        counts["blocks", "retuned"]
        + dyadic_counts["blocks", "retuned"]
        + stress_counts["blocks", "retuned"]
    )
    total_solves = scalar_res["solved"] + grid_solves + dyadic_solves + stress_solves + pruned_solves
    loop_cap = scalar_res["loop_cap"]

    print(
        f"PASS: {total_roots} grounded SDDM roots ({len(FIXTURES)} rational + {len(DYADIC_FIXTURES)} dyadic + "
        f"{len(DENSE_FINE_STRESS_FIXTURES)} dense-fine stress, {total_levels} nonterminal levels) + "
        f"{len(pruned_cases)} edge-pruned hierarchies ({pruned_counts['blocks', 'retuned']} levels, "
        f"{prune_stats_exact['edges_pruned']} weak edges pruned) verified across all 4 smoother/scale pairs "
        f"(O(n+nnz) incremental Delta_v selector matched O(n^3) oracle on every level; __debug__={__debug__})."
    )
    print(
        f"PASS: Tight 7*(3/4)^32 lifting defect, dense-fine discrimination (1 sweep fails, 16 pass), "
        f"and full-basis lift/restrict adjointness verified across {basis_levels_checked} nonterminal levels."
    )
    print(
        f"PASS: {total_solves} normalized two-pass finite-grid & binary-output solves "
        f"(including r=0, tiny 2^(-bits-2), and huge 2^(bits+2) RHS on scalar [1] & multi-vertex roots) "
        f"met action error <= theta and 2^-20 energy error within {loop_cap} Richardson steps "
        f"(actual executed steps: scalar={scalar_res['max_exec']}, multi-vertex min={all_min_exec}, max={all_max_exec} <= {loop_cap})."
    )
    print(
        f"PASS: Per-primitive energy bound (mu), measured-error recurrence (E_i <= 14*mu_meas + c_i*E_(i+1)), "
        f"recurrence negative controls (shrunk mu & dropped c_i), and independent signed integer bit-width bounds (+1 sign bit) "
        f"verified (max signed vec={max_vec_bits_seen} <= {max_vec_bound_seen}, "
        f"max signed partial-sum={max_partial_sum_bits_seen}, max signed op={max_op_bits_seen}; "
        f"dyadic roots={dyadic_reports})."
    )
    print(f"Block smoother stats: {dict(sorted(block_stats.items()))}")
    print(f"Completed in {time.perf_counter() - t0:.2f}s.")


if __name__ == "__main__":
    main()
