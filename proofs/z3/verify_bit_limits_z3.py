#!/usr/bin/env python3
"""BitFSAI Extreme-Low-Bit Limits & Frozen-Word Drift Z3 Verification Suite.

Verifies 21 exact-real (QF_NRA) UNSAT obligations and 2 expected SAT controls
covering:
  - 3-vertex grounded SDDM sign-only vs. 1.58-bit ternary separation (kappa <= 3)
  - 3x3 / 4x4 arrow-patch ternary vs. asymmetric/symmetric 2-bit (INT2) optima
  - Exact-prefix spectral threshold kappa <= K iff D / sigma <= (K - 1)^2 / (4 K)
  - Projective packing boundary (sub-bit 2-axis obstruction & strict K^2 separation)
  - Harmonic-mean minimax coarse weight bin relative error on [1, 2]
  - Frozen-word Newton drift (general h/l Loewner degradation & exact-row Kantorovich
    quadratic bound 1 + (h - l)^2 / (4 h l))
  - Grounded 2-slot resistor-network threshold cut lemma (C_k <= 3 * sigma)
"""
from itertools import product
from time import perf_counter
import z3 as z

Q = z.Q
x, y, t, h, k, a, b, c, u, v = z.Reals("x y t h k a b c u v")
checks = []


def probe(name, premises, counterexample, expected=z.unsat):
    solver = z.SolverFor("QF_NRA")
    solver.set(timeout=10000)
    solver.add(*premises)
    start = perf_counter()
    if solver.check() != z.sat:
        raise RuntimeError(f"{name}: invalid or undecided premises")
    solver.add(counterexample)
    result = solver.check()
    elapsed = 1000 * (perf_counter() - start)
    print(f"  {name:<46} {str(result):5} {elapsed:8.3f} ms")
    if result != expected:
        raise RuntimeError(f"{name}: {result}, expected {expected}")
    checks.append(result)


def residuals(alphabet, target):
    """Exact squared Euclidean distances to all alphabet-generated lines."""
    for code in product(alphabet, repeat=len(target)):
        size = sum(i * i for i in code)
        if size:
            dot = sum(i * j for i, j in zip(code, target))
            yield sum(j * j for j in target) - dot * dot * Q(1, size)
        else:
            yield sum(j * j for j in target)


def sharp(name, alphabet, target, premises, bound):
    errors = list(residuals(alphabet, target))
    probe(
        name,
        premises,
        z.Or(*(e < bound for e in errors), z.And(*(e != bound for e in errors))),
    )


def main():
    t0 = perf_counter()
    print("=" * 96)
    print(f"BitFSAI BIT-LIMITS & DRIFT VERIFICATION SUITE (Z3 {z.get_version_string()}, Premise-SAT Guarded)")
    print("=" * 96)

    domain = [t > 0, t < 1]
    sd = t * (1 - t)
    ds = (1 - t) ** 2 / 2
    dt = z.If(t * t <= ds, t * t, ds)

    sharp("grounded star, sign minimum", [-1, 1], [1, t], domain, ds)
    sharp("grounded star, ternary minimum", [-1, 0, 1], [1, t], domain, dt)

    # kappa <= 3 iff rho <= 1/2 iff 3 * D <= Schur pivot.
    probe("sign kappa<=3 iff t>=3/5", domain, (3 * ds <= sd) != (5 * t >= 3))
    probe(
        "ternary kappa<=3 iff t<=1/4 or t>=3/5",
        domain,
        (3 * dt <= sd) != z.Or(4 * t <= 1, 5 * t >= 3),
    )
    probe("sign kappa<=3 at t=1/5 (refutation)", [t == Q(1, 5)], 3 * ds > sd, z.sat)

    sharp("3x3 arrow ternary slope 2:1", [-1, 0, 1], [2, 1], [], Q(1, 2))
    sharp("3x3 arrow asymmetric INT2 slope 2:1", [-2, -1, 0, 1], [2, 1], [], 0)
    sharp("3x3 arrow symmetric four-level slope 3:1", [-3, -1, 1, 3], [3, 1], [], 0)
    sharp("4x4 arrow ternary slope 3:2:1", [-1, 0, 1], [3, 2, 1], [], Q(3, 2))

    # Normalized last row: off-diagonal block has squared length u^2 + v^2.
    lam = k
    probe(
        "3x3 arrow characteristic polynomial",
        [],
        ((1 - lam) * ((1 - lam) ** 2 - v * v) - u * u * (1 - lam))
        != (1 - lam) * ((1 - lam) ** 2 - (u * u + v * v)),
    )
    probe(
        "general spectral threshold after projection",
        [k >= 1, h > 0, a >= 0, b >= 0, b < 1, b * b * (h + a) == a],
        ((1 + b) <= k * (1 - b)) != (4 * k * a <= h * (k - 1) ** 2),
    )
    probe(
        "one line cannot beat 1/2 on both axes",
        [x * x + y * y == 1],
        z.And(1 - x * x < Q(1, 2), 1 - y * y < Q(1, 2)),
    )
    probe(
        "equal line attains packing boundary",
        [x == y, x * x + y * y == 1],
        z.Or(1 - x * x != Q(1, 2), 1 - y * y != Q(1, 2)),
    )

    # Single positive representative for a ratio-2 coarse weight bin.
    probe(
        "coarse [1,2] relerr below 1/3 impossible",
        [u >= 0, u < Q(1, 3)],
        z.And(1 - u <= c, c <= 1 + u, 2 * (1 - u) <= c, c <= 2 * (1 + u)),
    )
    probe(
        "coarse [1,2] harmonic center attains 1/3",
        [1 <= x, x <= 2],
        z.Or(Q(4, 3) < Q(2, 3) * x, Q(4, 3) > Q(4, 3) * x),
    )

    # General SPD fiber: minimum of h - 2*b*t + a*t^2.
    probe(
        "row Pythagorean identity",
        [a > 0],
        a * (h - 2 * b * t + a * t * t) != (a * h - b * b) + (a * t - b) ** 2,
    )

    # Fixed decoder diagonal ratios compatible with both encoded matrices.
    probe(
        "projective packing at ratio >K^2",
        [k >= 1, h > k * k, c > 0],
        z.And(k * c >= 1, c <= k, k * c >= h, c <= k * h),
    )
    probe(
        "packing equality can share decoder",
        [k == 3, h == 9, c == 3],
        z.And(k * c >= 1, c <= k, k * c >= h, c <= k * h),
        z.sat,
    )

    # Frozen-line optimal/unrestricted local minima old (a, b) and new (x, y).
    probe(
        "Loewner drift gives h/l local degradation",
        [a > 0, b > 0, h >= t, t > 0, x > 0, y > 0, x <= h * a, y >= t * b],
        t * x * b > h * a * y,
    )

    # Spectral averaging gives a = E(lambda), b = E(1 / lambda).
    probe("Kantorovich pointwise majorant", [t > 0, h >= t, x >= t, x <= h], x * x + t * h > (h + t) * x)
    probe(
        "exact old row: quadratic drift bound",
        [t > 0, h >= t, a > 0, b > 0, a + t * h * b <= t + h],
        4 * t * h * a * b > (t + h) ** 2,
    )
    probe(
        "equal endpoint weights attain drift bound",
        [t > 0, h >= t, 2 * a == t + h, 2 * t * h * b == t + h],
        4 * t * h * a * b != (t + h) ** 2,
    )

    # Three successive voltage gaps for a two-predecessor grounded patch.
    gaps = [t, x, 1 - t - x]
    caps = [a, b, c]
    probe(
        "grounded 2-slot threshold cut lemma",
        [
            k > 0,
            *(gap >= 0 for gap in gaps),
            *(cap >= 0 for cap in caps),
            *(gap * cap <= k for gap, cap in zip(gaps, caps)),
        ],
        z.And(*(cap > 3 * k for cap in caps)),
    )

    unsat_cnt = sum(r == z.unsat for r in checks)
    sat_cnt = sum(r == z.sat for r in checks)
    assert unsat_cnt == 21 and sat_cnt == 2
    print(f"\nALL {unsat_cnt} UNSAT BIT-LIMIT OBLIGATIONS + {sat_cnt} SAT CONTROLS VERIFIED IN {perf_counter() - t0:.3f}s.")


if __name__ == "__main__":
    main()
