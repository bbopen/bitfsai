#!/usr/bin/env python3
"""Exact-real Z3 proof obligations for BitFSAI multilevel convergence,
quantized four-phase polynomial cycles, and negative controls."""

from fractions import Fraction as F
import sys
import time
import z3


def Q(val: F) -> z3.RatNumRef:
    return z3.Q(val.numerator, val.denominator)


def verify_obligation(name: str, premises, claim, expected=z3.unsat):
    s_prem = z3.Solver()
    s_prem.set("timeout", 10000)
    s_prem.add(*premises)
    prem_res = s_prem.check()
    if prem_res != z3.sat:
        print(f"  [FAIL] {name}: premises {prem_res} (expected sat)", flush=True)
        sys.exit(1)

    s_claim = z3.Solver()
    s_claim.set("timeout", 10000)
    s_claim.add(*premises, z3.Not(claim))
    res = s_claim.check()
    if res != expected:
        print(f"  [FAIL] {name}: negated claim {res} (expected {expected})", flush=True)
        sys.exit(1)
    return res


def main() -> None:
    t0 = time.perf_counter()
    unsat_count = 0
    sat_controls = 0

    def prove(name, premises, claim):
        nonlocal unsat_count
        verify_obligation(name, premises, claim, z3.unsat)
        unsat_count += 1

    def control(name, premises, claim):
        nonlocal sat_controls
        verify_obligation(name, premises, claim, z3.sat)
        sat_controls += 1

    # 1. 2D & 3D INT4 quantizer rounding cells (8 + 64 = 72 obligations)
    m, z, w, p0, p1 = z3.Reals("m z w p0 p1")
    for k in range(8):
        r = z3.Q(k, 7)
        prem = [
            m > 0, z >= 0, z <= m, m + z <= z3.Q(1, 2),
            7 * z - k * m <= m / 2, 7 * z - k * m >= -m / 2,
            (1 + r * r) * p0 == m + r * z, p1 == r * p0,
        ]
        err = (p0 - m) * (p0 - m) + (p1 - z) * (p1 - z)
        prove(
            f"int4_2d_cell_{k}",
            prem,
            z3.And(
                p0 >= 0, p1 >= 0, err <= z3.Q(1, 784),
                p0 * p0 + p1 * p1 <= m * m + z * z,
                p0 * (m - p0) + p1 * (z - p1) == 0,
                p0 <= m + z3.Q(1, 28), p1 <= z + z3.Q(1, 28),
            ),
        )

    for k in range(8):
        for l in range(8):
            rk, rl = z3.Q(k, 7), z3.Q(l, 7)
            prem = [
                m > 0, z >= 0, w >= 0, z <= m, w <= m, m + z + w <= z3.Q(1, 2),
                7 * z - k * m <= m / 2, 7 * z - k * m >= -m / 2,
                7 * w - l * m <= m / 2, 7 * w - l * m >= -m / 2,
            ]
            prove(
                f"int4_3d_cell_{k}_{l}",
                prem,
                (z - rk * m) ** 2 + (w - rl * m) ** 2 <= z3.Q(1, 392),
            )

    # 2. Cartesian 2D/3D variance, spectral overlap, smoothing, stored-factor robustness
    v2, err, t, u, a, b, e, lam = z3.Reals("v2 err t u a b e lam")
    prove("variance_2d", [v2 >= 0, v2 <= z3.Q(1, 4), err >= 0, err <= z3.Q(1, 784), t * (1 - v2) == err],
          z3.And(t >= 0, t <= z3.Q(1, 588)))
    prove("variance_3d", [v2 >= 0, v2 <= z3.Q(1, 4), err >= 0, err <= z3.Q(1, 392), t * (1 - v2) == err],
          z3.And(t >= 0, t <= z3.Q(1, 294)))
    prove("normalized_row_geometry", [t >= 0, u > 0, (1 + t) * u * u == 1],
          z3.And(u <= 1, 2 * (1 - u) <= t))
    prove("gram_2d", [a >= 0, a * a <= 2, b >= 0, b * b <= z3.Q(1, 49)],
          z3.And((a + b) ** 2 <= z3.Q(5, 2), z3.Q(18, 25) * (a + b) ** 2 < 2))
    prove("gram_3d", [a >= 0, a * a <= 2, b >= 0, b * b <= z3.Q(8, 147)],
          (a + b) ** 2 < z3.Q(11, 4))
    prove("smoothing_3d", [lam >= 0, lam <= z3.Q(11, 4)],
          z3.And(1 - (1 - z3.Q(18, 25) * lam) ** 2 >= z3.Q(18, 1250) * lam,
                 (1 - z3.Q(18, 25) * lam) ** 2 <= 1))
    prove("stored_factor_robust_upper",
          [a >= 0, a * a <= 2, b >= 0, b * b <= z3.Q(8, 147), e >= 0, e <= z3.Q(1, 1000)],
          (a + b + 3 * e) ** 2 < z3.Q(11, 4))
    prove("stored_factor_robust_lower", [e >= 0, e <= z3.Q(1, 1000)],
          z3.Q(1, 3) - 2 * e >= z3.Q(1, 4))

    # 3. Certified damping guard and relative E5M2 coarse sandwich
    omega, wq, d = z3.Reals("omega wq d")
    prove("certified_damping_guard", [u > 0, lam > 0, lam <= u, omega > 0, omega * u <= 1],
          z3.And(1 - omega * lam >= 0, 1 - omega * lam < 1, 2 - omega * lam > 0))
    prove("e5m2_coarse_sandwich",
          [w >= 0, d >= 0, wq >= z3.Q(7, 8) * w, wq <= z3.Q(9, 8) * w, z >= 0],
          z3.And(z3.Q(7, 8) * (d + w * z) <= d + wq * z, d + wq * z <= z3.Q(9, 8) * (d + w * z)))

    # 4. Monotone multilevel effectiveness recurrence and polynomial schedule
    x, f, p, s, zeta, K, alpha, gain = z3.Reals("x f p s zeta K alpha gain")
    prove("two_call_minimax", [e > 0, e <= 1, x >= e, x <= 1],
          (1 + e - 2 * x) ** 2 <= (1 - e) ** 2)
    prove("two_call_gain", [e > 0, e <= 1, f * (1 + e) ** 2 == 4 * e],
          z3.And(f > 0, f <= 1, f <= 4 * e, f >= 2 * e - e * e))
    prove("two_call_fixed_point_threshold", [e > 0, e < 1, p > 0, p <= z3.Q(1, 4)],
          4 * p * e < e * (1 + e) ** 2)
    prove("protected_smoothing_scalar", [x >= 0, x <= 1],
          2 * x * (1 - x) ** 2 <= 1 - (1 - x) ** 2)

    c_var, e_new = z3.Reals("c_var e_new")
    prove("monotone_effectiveness_step",
          [alpha >= 0, gain >= 0, zeta >= 0, zeta < 1, K >= 0,
           s >= 0, s <= 1, t <= zeta * s + K * (1 - s),
           c_var >= 0, c_var <= alpha * gain, c_var * (K + 1 - zeta) <= 1,
           e_new == (1 - zeta) * c_var],
          (1 - c_var) * s + c_var * t <= 1 - e_new)

    beta, deg = z3.Reals("beta deg")
    prove("work_convergence_frontier",
          [beta > 0, deg > 0, p <= beta * beta, beta * deg < 1],
          p * deg * deg < 1)

    # 5. Grounded SDDM four-phase budgets and exact kappa <= 17 / kappa <= 9 bounds
    eps = F(1, 1024)
    tau = F(1, 8)
    eta = F(1, 14)
    zeta_v = (1 + 1 / tau) * eps
    K_v = (1 + tau) / (2 * eta)
    cap_v = (1 - zeta_v) / (K_v + 1 - zeta_v)
    alpha_v = F(7, 9) * F(127, 129) * F(63, 64) / (1 + eps)
    p_v = F(11, 15)
    beta_v = F(13, 16)

    prove("sixteen_jacobi_steps", [z3.BoolVal(True)], Q(7 * F(3, 4) ** 32) < Q(eps))
    k_path = 48
    d_path = F(3, 4)
    tail = d_path ** (k_path - 1) * (
        F(k_path * (k_path + 2)) / (1 - d_path)
        + (2 * k_path + 2) * d_path / (1 - d_path) ** 2
        + d_path * (1 + d_path) / (1 - d_path) ** 3
    )
    prove("path_tail_48", [z3.BoolVal(True)], Q(tail) < z3.Q(1, 64))
    prove("level_cap_above_one_ninth", [z3.BoolVal(True)],
          z3.And(Q(cap_v) == z3.Q(145, 1297), Q(cap_v) > z3.Q(1, 9)))
    prove("retention_above_11_15", [z3.BoolVal(True)], Q(alpha_v * (1 - zeta_v)) > Q(p_v))
    prove("pruned_retention_above_11_15", [z3.BoolVal(True)],
          Q(F(511, 512) * alpha_v * (1 - zeta_v)) > Q(p_v))
    prove("four_phase_linear_work", [z3.BoolVal(True)], Q(2 * beta_v ** 4) < 1)
    prove("frontier_converse_four_phase_witness", [z3.BoolVal(True)],
          z3.And(Q(beta_v ** 2) < Q(p_v), Q(beta_v ** 4 * 2) < 1, Q(p_v ** 4 * 4) > 1))
    prove("block_smoother_eta", [z3.BoolVal(True)], z3.Q(3, 40) >= Q(eta))

    a_op, r_sm, t_gal, ac_op, bc_op, gam = z3.Reals("a_op r_sm t_gal ac_op bc_op gam")
    v_b = 2 * r_sm - a_op * r_sm * r_sm + gam * (1 - a_op * r_sm) ** 2 * bc_op
    prove("vcycle_galerkin_upper_scalar",
          [a_op > 0, r_sm >= 0, r_sm * a_op <= 1, t_gal >= a_op,
           gam > 0, ac_op >= gam * t_gal, bc_op >= 0, bc_op * ac_op <= 1],
          z3.And(gam / ac_op <= 1 / t_gal, r_sm <= v_b, v_b * a_op <= 1))

    # Kappa <= 17 at t = 2560/1307
    t17 = F(2560, 1307)
    x17 = F(1, 17)
    prove("kappa_17_endpoints", [z3.BoolVal(True)],
          z3.And(
              Q(x17) <= Q(cap_v),
              Q(x17) <= Q(p_v * (1 - (1 - t17 * (p_v ** 3 * x17)) ** 2)),
              Q(x17) <= Q(p_v * (1 - (1 - t17) ** 2)),
          ))

    # Kappa <= 9 at retuned scale t = 30375/15853
    t9 = F(30375, 15853)
    x9 = F(1, 9)
    prove("kappa_9_endpoints", [z3.BoolVal(True)],
          z3.And(
              Q(x9) <= Q(cap_v),
              Q(x9) <= Q(p_v * (1 - (1 - t9 * (p_v ** 3 * x9)) ** 2)),
              Q(x9) <= Q(p_v * (1 - (1 - t9) ** 2)),
          ))

    # 6. Negative controls (expected SAT on negated claim)
    r = z3.Real("r")
    control("ctrl_unit_energy_not_fixed_damping", [r == z3.Q(99, 100)],
            z3.Q(18, 25) * (1 + 2 * r) < 2)
    control("ctrl_coarse_rank_not_approximation", [p == z3.Q(9, 10), s == z3.Q(99, 100)],
            s <= p)
    control("ctrl_unscaled_coarse_overshoot", [s == z3.Q(9, 10), b == z3.Q(5, 2)],
            s * s * (b - 1) < 1)
    control("ctrl_t17_cannot_certify_kappa_9", [z3.BoolVal(True)],
            Q(p_v * (1 - (1 - t17) ** 2)) >= z3.Q(1, 9))
    control("ctrl_full_w_cycle_exceeds_linear_work", [z3.BoolVal(True)],
            Q(2 * beta_v) < 1)
    control("ctrl_one_call_repeated_fp8", [e > 0, e < 1],
            z3.Q(7, 9) * e >= e)
    control("ctrl_zero_start_near_null", [s > 0, s < z3.Q(1, 100)],
            1 <= 7 * s)

    elapsed = time.perf_counter() - t0
    print(
        f"PASS: {unsat_count} UNSAT proof obligations + {sat_controls} SAT negative controls "
        f"verified with Z3 {z3.get_version_string()} in {elapsed:.2f}s."
    )


if __name__ == "__main__":
    main()
