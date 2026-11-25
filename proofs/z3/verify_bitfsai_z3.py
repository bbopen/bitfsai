#!/usr/bin/env python3
"""BitFSAI core algebraic obligations in Z3 exact real arithmetic.

Every implication obligation is guarded against vacuity via `assert_premises_sat`
before asserting the negated conclusion and checking `unsat`.
"""

import sys
import time
import z3


def assert_premises_sat(name: str, premises: list) -> None:
    if not premises:
        return
    s = z3.Solver()
    s.add(*premises)
    res = s.check()
    if res != z3.sat:
        print(f"  [{name:8s}] FAILED: premises are not satisfiable ({res})")
        sys.exit(1)


def prove_implication(name: str, theorem_ref: str, premises: list, neg_conclusion) -> None:
    assert_premises_sat(name, premises)
    s = z3.Solver()
    s.add(*premises)
    s.add(neg_conclusion)
    res = s.check()
    status = "PROVED (premises sat, negation unsat)" if res == z3.unsat else f"FAILED ({res})"
    print(f"  [{name:8s}] {status:38s} | {theorem_ref}")
    if res != z3.unsat:
        sys.exit(1)


def prove_impossibility(name: str, theorem_ref: str, premises: list, impossible_target: list) -> None:
    """Verify baseline domain premises are satisfiable, while adding the
    claimed impossible target constraints yields `unsat`."""
    assert_premises_sat(name, premises)
    s = z3.Solver()
    s.add(*premises)
    s.add(*impossible_target)
    res = s.check()
    status = "PROVED (premises sat, target unsat)" if res == z3.unsat else f"FAILED ({res})"
    print(f"  [{name:8s}] {status:38s} | {theorem_ref}")
    if res != z3.unsat:
        sys.exit(1)


def main() -> None:
    t0 = time.perf_counter()
    print("=" * 96)
    print("BitFSAI FORMAL VERIFICATION SUITE (Z3 Exact Real Arithmetic, Premise-SAT Guarded)")
    print("=" * 96)

    # Z3-B1: Anisotropic 2D Fourier stencil impossibility for 1-level local smoother (κ <= 50)
    eps, mu = z3.Q(1, 1000), z3.Q(1, 10000)
    modes_1 = [(1, z3.Q(0, 1)), (1, z3.Q(1, 2)), (1, z3.Q(1, 1))]
    c0, cx, cy, m_lo = z3.Reals("c0 cx cy m_lo")
    target_b1 = []
    for sx, sy in modes_1:
        a_sym = 4 * eps * sx + 4 * sy + mu
        g_sym = c0 + cx * sx + cy * sy
        lam = g_sym * a_sym
        target_b1.extend([lam >= m_lo, lam <= 50 * m_lo])
    prove_impossibility(
        "Z3-B1",
        "Background: anisotropic 1st-order Fourier stencil cannot bound κ <= 50",
        [m_lo > 0],
        target_b1,
    )

    # Z3-B2: Anisotropic 2D Fourier stencil impossibility for 2nd-order y-stencil (κ <= 10)
    modes_2 = [(1, z3.Q(0, 1)), (1, z3.Q(1, 10)), (1, z3.Q(1, 2)), (1, z3.Q(1, 1))]
    d0, d1, d2, m2 = z3.Reals("d0 d1 d2 m2")
    target_b2 = []
    for sx, sy in modes_2:
        a_sym = 4 * eps * sx + 4 * sy + mu
        g_sym = d0 + d1 * sy + d2 * sy * sy
        lam = g_sym * a_sym
        target_b2.extend([lam >= m2, lam <= 10 * m2])
    prove_impossibility(
        "Z3-B2",
        "Background: anisotropic 2nd-order y-stencil cannot bound κ <= 10",
        [m2 > 0],
        target_b2,
    )

    # Z3-B3: Theorem 1 (3x3 concrete instance) — lower-triangular nonzero diag => G^T G SPD
    g11, g21, g22, g31, g32, g33 = z3.Reals("g11 g21 g22 g31 g32 g33")
    x1, x2, x3 = z3.Reals("x1 x2 x3")
    y1 = g11 * x1
    y2 = g21 * x1 + g22 * x2
    y3 = g31 * x1 + g32 * x2 + g33 * x3
    prove_implication(
        "Z3-B3",
        "Theorem 1 (3x3 witness): G_ii != 0 => x^T G^T G x > 0 for x != 0",
        [g11 != 0, g22 != 0, g33 != 0, z3.Or(x1 != 0, x2 != 0, x3 != 0)],
        y1 * y1 + y2 * y2 + y3 * y3 <= 0,
    )

    # Z3-B4a: Theorem 2 — Euclidean norm distortion from relative perturbation eta < 1
    U, e1, e2, eta = z3.Reals("U e1 e2 eta")
    norm_sq = (U + e1) * (U + e1) + e2 * e2
    prove_implication(
        "Z3-B4a",
        "Theorem 2: ||e|| <= eta*U => (1-eta)^2 U^2 <= ||U+e||^2 <= (1+eta)^2 U^2",
        [U > 0, eta >= 0, eta < 1, e1 * e1 + e2 * e2 <= eta * eta * U * U],
        z3.Or(norm_sq < (1 - eta) ** 2 * U * U, norm_sq > (1 + eta) ** 2 * U * U),
    )

    # Z3-B4b: Theorem 2 numeric instantiation at illustrative eta = 5/28 (< 2.06)
    eta_val = z3.Q(5, 28)
    prove_implication(
        "Z3-B4b",
        "Theorem 2 (illustrative eta = 5/28): ((1+eta)/(1-eta))^2 < 2.06",
        [eta == eta_val],
        ((1 + eta) / (1 - eta)) ** 2 >= z3.Q(206, 100),
    )

    # Z3-B4c: Theorem 2 2x2 counterexample — diagonal-inclusive INT4 yields κ > 321
    tr, det = z3.Q(18, 1), z3.Q(1, 1)
    lam_max, lam_min = z3.Reals("lam_max lam_min")
    prove_implication(
        "Z3-B4c",
        "Theorem 2 (2x2 counterexample): tr=18, det=1 => lam_max / lam_min > 321",
        [lam_max + lam_min == tr, lam_max * lam_min == det, lam_max >= lam_min, lam_min > 0],
        lam_max / lam_min <= 321,
    )

    # Z3-B4d: Theorem 2 — generalized Rayleigh-quotient condition distortion ratio
    ux, uy, qx, qy, ax, ay = z3.Reals("ux uy qx qy ax ay")
    prove_implication(
        "Z3-B4d",
        "Theorem 2: Rayleigh quotient ratio distortion <= ((1+eta)/(1-eta))^2",
        [
            eta >= 0,
            eta < 1,
            ux >= 0,
            uy > 0,
            ax > 0,
            ay > 0,
            qx >= (1 - eta) ** 2 * ux,
            qx <= (1 + eta) ** 2 * ux,
            qy >= (1 - eta) ** 2 * uy,
            qy <= (1 + eta) ** 2 * uy,
        ],
        (qx / ax) / (qy / ay) > ((1 + eta) / (1 - eta)) ** 2 * ((ux / ax) / (uy / ay)),
    )

    # Z3-B5a/b: 1D Fourier high-frequency smoothing factor and stability check
    rq, sv = z3.Reals("rq sv")
    lam_hf = z3.Q(2, 3) * ((1 + rq) ** 2 - 4 * rq * sv) * (4 * sv)
    err_hf = 1 - z3.Q(9, 10) * lam_hf
    prove_implication(
        "Z3-B5a",
        "Smoothing (1D Fourier): rq in [1/3,2/3], sv in [1/2,1] => |1 - 0.9*lam| < 3/4",
        [rq >= z3.Q(1, 3), rq <= z3.Q(2, 3), sv >= z3.Q(1, 2), sv <= 1],
        z3.Or(err_hf >= z3.Q(3, 4), err_hf <= z3.Q(-3, 4)),
    )
    prove_implication(
        "Z3-B5b",
        "Smoothing (1D Fourier): rq in [1/3,2/3], sv in [0,1] => 0.9*lam < 1.9 < 2",
        [rq >= z3.Q(1, 3), rq <= z3.Q(2, 3), sv >= 0, sv <= 1],
        z3.Q(9, 10) * lam_hf >= z3.Q(19, 10),
    )

    # Z3-B6: Theorem 3 — zero-off-diagonal re-projection fallback optimality & unit energy
    c11, c12, c22, s_val, g_val, g_opt = z3.Reals("c11 c12 c22 s_val g_val g_opt")
    opt_energy = g_opt * g_opt * c22
    opt_ratio = opt_energy / (g_opt * g_opt)
    any_ratio = (s_val * s_val * c11 + 2 * s_val * g_val * c12 + g_val * g_val * c22) / (g_val * g_val)
    prove_implication(
        "Z3-B6",
        "Theorem 3 (zero-direction fallback): c11=c12=0 => s=0, g_opt=1/sqrt(c22) achieves unit energy & exact Kaporin ratio c22",
        [c11 == 0, c12 == 0, c22 > 0, g_opt > 0, g_opt * g_opt * c22 == 1, g_val > 0],
        z3.Or(opt_ratio > any_ratio, opt_energy != 1),
    )

    print(f"\nALL 10 Z3 OBLIGATIONS VERIFIED (PREMISE-SAT GUARDED) IN {time.perf_counter() - t0:.3f}s.")


if __name__ == "__main__":
    main()
