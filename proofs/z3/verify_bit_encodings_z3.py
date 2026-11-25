#!/usr/bin/env python3
"""Bit-level Z3 BitVec + exact-real verification of BitFSAI's 4-bit and FP8 E5M2 encodings.

Verifies the exact bit-packing, row-aligned nibble addressing, unpacking, sign
symmetry, normal-range predicates, and non-tautological piecewise encoder
rounding bounds implemented in reference/bitfsai.cuh (offset-binary INT4 and
FP4 E2M1) and reference/standalone_amg.cu (two's-complement INT4 and IEEE-754
RN-even FP8 E5M2) across all 4-bit nibble pairs and all 256 FP8 E5M2 bytes.
"""

import time
import z3


def prove_unsat(name: str, desc: str, constraints) -> None:
    s = z3.Solver()
    s.add(constraints)
    res = s.check()
    if res != z3.unsat:
        raise RuntimeError(
            f"[{name}] FAILED ({res}): {desc}\nModel: {s.model() if res == z3.sat else 'none'}"
        )
    print(f"  [{name:<12}] PROVED (unsat) | {desc}")


def prove_sat(name: str, desc: str, constraints) -> None:
    s = z3.Solver()
    s.add(constraints)
    res = s.check()
    if res != z3.sat:
        raise RuntimeError(f"[{name}] NEGATIVE CONTROL FAILED ({res}): {desc}")
    print(f"  [{name:<12}] PROVED (sat)   | {desc}")


def z3_abs(x):
    return z3.If(x >= 0, x, -x)


def llround_clamped_7(r):
    """Exact piecewise model of clamp(llround(7 * r), -7, 7) (ties away from zero)."""
    x = 7 * r
    # Positive half-integers round away from zero: [k - 1/2, k + 1/2) -> k (for k = 1..6), >= 6.5 -> 7
    pos = z3.If(
        x < z3.Q(1, 2), 0,
        z3.If(x < z3.Q(3, 2), 1,
        z3.If(x < z3.Q(5, 2), 2,
        z3.If(x < z3.Q(7, 2), 3,
        z3.If(x < z3.Q(9, 2), 4,
        z3.If(x < z3.Q(11, 2), 5,
        z3.If(x < z3.Q(13, 2), 6, 7))))))
    )
    # Negative half-integers round away from zero: (-k - 1/2, -k + 1/2] -> -k, <= -6.5 -> -7
    neg = z3.If(
        x > z3.Q(-1, 2), 0,
        z3.If(x > z3.Q(-3, 2), -1,
        z3.If(x > z3.Q(-5, 2), -2,
        z3.If(x > z3.Q(-7, 2), -3,
        z3.If(x > z3.Q(-9, 2), -4,
        z3.If(x > z3.Q(-11, 2), -5,
        z3.If(x > z3.Q(-13, 2), -6, -7))))))
    )
    return z3.If(x >= 0, pos, neg)


def int_to_bv4_offset(q):
    """bitfsai.cuh:66-71 offset-binary nibble: static_cast<uint8_t>(q + 8) in BitVec(4)."""
    val = z3.BitVecVal(15, 4)
    for k in range(-7, 7):
        val = z3.If(q == k, z3.BitVecVal(k + 8, 4), val)
    return val


def int_to_bv4_twos(q):
    """standalone_amg.cu:526 two's-complement nibble: (q & 15) in BitVec(4)."""
    val = z3.BitVecVal(7, 4)
    for k in range(-7, 7):
        val = z3.If(q == k, z3.BitVecVal(k & 0xF, 4), val)
    return val


def main() -> None:
    t0 = time.perf_counter()
    print("=" * 96)
    print("BitFSAI BIT-LEVEL ENCODING VERIFICATION SUITE (Z3 BitVec + Exact Real Arithmetic)")
    print("=" * 96)

    # 1. Nibble packing/unpacking round-trip across all 256 (lo, hi) nibble pairs (bitfsai.cuh)
    lo = z3.BitVec("lo", 8)
    hi = z3.BitVec("hi", 8)
    packed = (lo & z3.BitVecVal(0x0F, 8)) | ((hi & z3.BitVecVal(0x0F, 8)) << 4)
    unpacked_lo = packed & z3.BitVecVal(0x0F, 8)
    unpacked_hi = z3.LShR(packed, 4) & z3.BitVecVal(0x0F, 8)
    prove_unsat(
        "BV-PACK",
        "bitfsai.cuh pack_nibbles_kernel / unpack round-trip exact over all 256 nibble pairs",
        [
            z3.ULE(lo, 15),
            z3.ULE(hi, 15),
            z3.Or(unpacked_lo != lo, unpacked_hi != hi),
        ],
    )

    # 2. Row-aligned nibble addressing & byte-disjointness (standalone_amg.cu:266,527,628,941)
    #    Row i starts at byte (rp[i] + i) / 2 and writes offset t / 2 for 0 <= t < rp[i+1] - rp[i].
    #    Buffer has (nnz + n + 1) / 2 bytes.
    n_rows, nnz, row_i, rp_i, rp_next, t_slot = z3.Ints("n_rows nnz row_i rp_i rp_next t_slot")
    pos_curr = (rp_i + row_i) / 2 + t_slot / 2
    next_row_start = (rp_next + (row_i + 1)) / 2
    packed_buf_len = (nnz + n_rows + 1) / 2
    # Also verify standalone_amg.cu:527 lower `|=` and :628 upper `(t & 1) ? ((b & 15) | (q << 4)) : q`
    q_even = z3.BitVec("q_even", 8)
    q_odd = z3.BitVec("q_odd", 8)
    b0 = z3.BitVecVal(0, 8)
    b_after_lo = b0 | q_even
    b_after_hi = b_after_lo | (q_odd << 4)
    b_overwrite_even = q_even
    b_overwrite_odd = (b_overwrite_even & z3.BitVecVal(15, 8)) | (q_odd << 4)
    prove_unsat(
        "BV-PACK-ROW",
        "standalone_amg.cu row-aligned addressing: rows never share a byte, in-bounds, no nibble clobber",
        [
            n_rows >= 1,
            row_i >= 0,
            row_i < n_rows,
            rp_i >= 0,
            rp_next >= rp_i + 1,
            rp_next <= nnz,
            t_slot >= 0,
            t_slot < rp_next - rp_i,
            z3.ULE(q_even, 15),
            z3.ULE(q_odd, 15),
            z3.Or(
                pos_curr >= next_row_start,
                pos_curr < 0,
                pos_curr >= packed_buf_len,
                (b_after_hi & z3.BitVecVal(15, 8)) != q_even,
                (z3.LShR(b_after_hi, 4) & z3.BitVecVal(15, 8)) != q_odd,
                (b_overwrite_odd & z3.BitVecVal(15, 8)) != q_even,
                (z3.LShR(b_overwrite_odd, 4) & z3.BitVecVal(15, 8)) != q_odd,
            ),
        ],
    )

    # 3. bitfsai.cuh offset-binary INT4_Split (codes 1..15 -> [-7..7]/7, code 8 == 0, code 0 unused)
    c = z3.BitVec("c", 4)

    def decode_int4_offset(nib):
        return (z3.ToReal(z3.BV2Int(nib, is_signed=False)) - 8) / z3.Q(7, 1)

    dec_off = decode_int4_offset(c)
    c_neg_off = z3.BitVecVal(16, 5) - z3.ZeroExt(1, c)
    dec_off_neg = (z3.ToReal(z3.BV2Int(c_neg_off, is_signed=False)) - 8) / z3.Q(7, 1)
    prove_unsat(
        "BV-INT4-1",
        "bitfsai.cuh offset-binary INT4 [1..15]: code 8 -> 0, exact sign symmetry, range [-1, 1]",
        [
            z3.UGE(c, 1),
            z3.ULE(c, 15),
            z3.Or(
                z3.And(c == 8, dec_off != 0),
                dec_off_neg != -dec_off,
                dec_off < -1,
                dec_off > 1,
            ),
        ],
    )

    # Non-tautological piecewise encoder -> BitVec(4) -> decoder round-trip for bitfsai.cuh
    r = z3.Real("r")
    q_enc = llround_clamped_7(r)
    enc_off_bv = int_to_bv4_offset(q_enc)
    dec_off_rt = decode_int4_offset(enc_off_bv)
    prove_unsat(
        "BV-INT4-2",
        "bitfsai.cuh encode_int4(r) on [-1, 1]: never emits unused code 0 & |r - decode(enc(r))| <= 1/14",
        [
            r >= -1,
            r <= 1,
            z3.Or(
                enc_off_bv == z3.BitVecVal(0, 4),
                z3_abs(r - dec_off_rt) > z3.Q(1, 14),
            ),
        ],
    )

    # 4. standalone_amg.cu 4-bit two's-complement INT4 (codes != 8, decode ((c ^ 8) - 8) / 7)
    def decode_int4_twos(nib):
        shifted = nib ^ z3.BitVecVal(8, 4)
        return (z3.ToReal(z3.BV2Int(shifted, is_signed=False)) - 8) / z3.Q(7, 1)

    dec_tc = decode_int4_twos(c)
    c_neg_tc = -c  # mod 16 two's-complement negation ((-c) & 15)
    dec_tc_neg = decode_int4_twos(c_neg_tc)
    prove_unsat(
        "BV-INT4-TC-1",
        "standalone_amg.cu two's-complement INT4 (c != 8): 0 -> 0, sign symmetry, range [-1, 1]; c=8 -> -8/7",
        [
            z3.Or(
                z3.And(
                    c != z3.BitVecVal(8, 4),
                    z3.Or(
                        z3.And(c == 0, dec_tc != 0),
                        dec_tc_neg != -dec_tc,
                        dec_tc < -1,
                        dec_tc > 1,
                    ),
                ),
                z3.And(c == z3.BitVecVal(8, 4), dec_tc != z3.Q(-8, 7)),
            )
        ],
    )

    enc_tc_bv = int_to_bv4_twos(q_enc)
    dec_tc_rt = decode_int4_twos(enc_tc_bv)
    prove_unsat(
        "BV-INT4-TC-2",
        "standalone_amg.cu two's-complement encode(r) on [-1, 1]: never emits code 8 & |r - decode| <= 1/14",
        [
            r >= -1,
            r <= 1,
            z3.Or(
                enc_tc_bv == z3.BitVecVal(8, 4),
                z3_abs(r - dec_tc_rt) > z3.Q(1, 14),
            ),
        ],
    )

    # 5. FP4 E2M1 split nibble encoding (bit 3 = sign, bits 2..0 -> {0, 0.5, 1, 1.5, 2, 3, 4, 6} / 6)
    e2m1_mags = [
        z3.Q(0, 12), z3.Q(1, 12), z3.Q(2, 12), z3.Q(3, 12),
        z3.Q(4, 12), z3.Q(6, 12), z3.Q(8, 12), z3.Q(12, 12),
    ]

    def decode_e2m1_bv(nib):
        sgn = z3.Extract(3, 3, nib)
        mag = z3.Extract(2, 0, nib)
        val = e2m1_mags[7]
        for idx in range(6, -1, -1):
            val = z3.If(mag == z3.BitVecVal(idx, 3), e2m1_mags[idx], val)
        return z3.If(sgn == z3.BitVecVal(1, 1), -val, val)

    n4 = z3.BitVec("n4", 4)
    dec_e2 = decode_e2m1_bv(n4)
    dec_e2_flip = decode_e2m1_bv(n4 ^ z3.BitVecVal(0x8, 4))
    prove_unsat(
        "BV-FP4-1",
        "FP4 E2M1 all 16 nibbles: +0/-0 -> 0, exact sign-bit symmetry, range [-1, 1]",
        [
            z3.Or(
                z3.And(z3.Or(n4 == 0, n4 == 8), dec_e2 != 0),
                dec_e2_flip != -dec_e2,
                dec_e2 < -1,
                dec_e2 > 1,
            )
        ],
    )

    # Exact piecewise encoder encode_fp4_e2m1 from bitfsai.cuh:51-64
    ar = z3_abs(r) * 6
    enc_mag = z3.If(
        ar < z3.Q(1, 4), z3.BitVecVal(0, 3),
        z3.If(ar < z3.Q(3, 4), z3.BitVecVal(1, 3),
        z3.If(ar < z3.Q(5, 4), z3.BitVecVal(2, 3),
        z3.If(ar < z3.Q(7, 4), z3.BitVecVal(3, 3),
        z3.If(ar < z3.Q(5, 2), z3.BitVecVal(4, 3),
        z3.If(ar < z3.Q(7, 2), z3.BitVecVal(5, 3),
        z3.If(ar < z3.Q(5, 1), z3.BitVecVal(6, 3), z3.BitVecVal(7, 3)))))))
    )
    enc_sgn = z3.If(r < 0, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1))
    enc_nib = z3.If(enc_mag == z3.BitVecVal(0, 3), z3.BitVecVal(0, 4), z3.Concat(enc_sgn, enc_mag))
    dec_roundtrip = decode_e2m1_bv(enc_nib)
    prove_unsat(
        "BV-FP4-2",
        "encode_fp4_e2m1 never emits -0 (0x8) and has absolute error <= 1/6 on [-1, 1]",
        [
            r >= -1,
            r <= 1,
            z3.Or(enc_nib == z3.BitVecVal(0x8, 4), z3_abs(r - dec_roundtrip) > z3.Q(1, 6)),
        ],
    )
    prove_unsat(
        "BV-FP4-3",
        "encode_fp4_e2m1 achieves exact relative error <= 1/5 on normal band [1/6, 1] (<= 1/3 on [1/12, 1])",
        [
            z3.Or(
                z3.And(z3_abs(r) >= z3.Q(1, 6), z3_abs(r) <= 1, z3_abs(r - dec_roundtrip) > z3.Q(1, 5) * z3_abs(r)),
                z3.And(z3_abs(r) >= z3.Q(1, 12), z3_abs(r) <= 1, z3_abs(r - dec_roundtrip) > z3.Q(1, 3) * z3_abs(r)),
            )
        ],
    )

    # 6. FP8 E5M2 exhaustive bit-level verification across all 256 byte encodings (BitVec(8))
    #    Format: 1 sign bit, 5 exponent bits (bias 15), 2 mantissa bits.
    #    Normal encodings: exp in [1..30] => value = (-1)^s * (1 + m/4) * 2^(exp - 15)
    #    Normal range: [2^-14, 7/4 * 2^15 = 57344]
    def fp8_pow2(exp_bv):
        scale = z3.Q(1, 1)
        for e_code in range(1, 31):
            e_unbiased = e_code - 15
            pow2 = z3.Q(2 ** e_unbiased, 1) if e_unbiased >= 0 else z3.Q(1, 2 ** (-e_unbiased))
            scale = z3.If(exp_bv == z3.BitVecVal(e_code, 5), pow2, scale)
        return scale

    def fp8_e5m2_normal_mag(exp_bv, man_bv):
        m_val = z3.Q(1, 1) + z3.ToReal(z3.BV2Int(man_bv, is_signed=False)) / z3.Q(4, 1)
        return m_val * fp8_pow2(exp_bv)

    b8 = z3.BitVec("b8", 8)
    s8 = z3.Extract(7, 7, b8)
    e8 = z3.Extract(6, 2, b8)
    m8 = z3.Extract(1, 0, b8)
    mag8 = fp8_e5m2_normal_mag(e8, m8)
    val8 = z3.If(s8 == z3.BitVecVal(1, 1), -mag8, mag8)
    b8_flip = b8 ^ z3.BitVecVal(0x80, 8)
    s8_f = z3.Extract(7, 7, b8_flip)
    e8_f = z3.Extract(6, 2, b8_flip)
    m8_f = z3.Extract(1, 0, b8_flip)
    val8_flip = z3.If(
        s8_f == z3.BitVecVal(1, 1),
        -fp8_e5m2_normal_mag(e8_f, m8_f),
        fp8_e5m2_normal_mag(e8_f, m8_f),
    )

    min_normal = z3.Q(1, 2 ** 14)
    max_normal = z3.Q(57344, 1)
    prove_unsat(
        "BV-FP8-1",
        "FP8 E5M2 normal bytes (exp in [1..30]): exact range [2^-14, 57344] & exact sign-bit symmetry",
        [
            z3.UGE(e8, 1),
            z3.ULE(e8, 30),
            z3.Or(mag8 < min_normal, mag8 > max_normal, val8_flip != -val8),
        ],
    )

    # Subnormal encodings (exp == 0, man in 1..3): strictly below 2^-14
    sub_mag = z3.ToReal(z3.BV2Int(m8, is_signed=False)) * z3.Q(1, 2 ** 16)
    prove_unsat(
        "BV-FP8-2",
        "FP8 E5M2 subnormal bytes (exp == 0, man != 0): strictly inside (0, 2^-14)",
        [
            e8 == 0,
            m8 != 0,
            z3.Or(sub_mag <= 0, sub_mag >= min_normal),
        ],
    )

    # IEEE-754 RN-even encoder connected directly to BitVec(5)/BitVec(2) decoding across all
    # 30 normal exponent bands [2^-14, 57344], including top-band (exp=30) non-overflow into Inf/NaN.
    e_in = z3.BitVec("e_in", 5)
    u = z3.Real("u")
    x_in = u * fp8_pow2(e_in)
    # IEEE-754 round-to-nearest-even in significand band u in [1, 2):
    #   even mantissas 0 (u=1), 2 (u=3/2), and next-band 0 (u=2) win ties (<= / >=);
    #   odd mantissas 1 (u=5/4) and 3 (u=7/4) lose ties (< / >).
    e_out = z3.If(u >= z3.Q(15, 8), e_in + z3.BitVecVal(1, 5), e_in)
    m_out = z3.If(
        u <= z3.Q(9, 8), z3.BitVecVal(0, 2),
        z3.If(u < z3.Q(11, 8), z3.BitVecVal(1, 2),
        z3.If(u <= z3.Q(13, 8), z3.BitVecVal(2, 2),
        z3.If(u < z3.Q(15, 8), z3.BitVecVal(3, 2), z3.BitVecVal(0, 2))))
    )
    x_dec = fp8_e5m2_normal_mag(e_out, m_out)
    prove_unsat(
        "BV-FP8-3",
        "FP8 E5M2 IEEE RN-even BitVec round-trip on [2^-14, 57344]: exp_out in [1..30] (no Inf) & relerr <= 1/8",
        [
            z3.UGE(e_in, 1),
            z3.ULE(e_in, 30),
            u >= 1,
            u < 2,
            z3.Implies(e_in == z3.BitVecVal(30, 5), u <= z3.Q(7, 4)),
            z3.Or(
                z3.ULT(e_out, 1),
                z3.UGT(e_out, 30),
                z3_abs(x_in - x_dec) > z3.Q(1, 8) * x_in,
            ),
        ],
    )

    # Negative control: subnormal region (0, 2^-14) violates the 1/8 relative error bound
    xs = z3.Real("xs")
    sub_step = z3.Q(1, 2 ** 16)
    nearest_sub = z3.If(
        xs <= z3.Q(1, 2) * sub_step, z3.Q(0, 1),
        z3.If(xs < z3.Q(3, 2) * sub_step, 1 * sub_step,
        z3.If(xs <= z3.Q(5, 2) * sub_step, 2 * sub_step,
        z3.If(xs < z3.Q(7, 2) * sub_step, 3 * sub_step, 4 * sub_step)))
    )
    prove_sat(
        "BV-FP8-NC",
        "Negative control: FP8 E5M2 subnormal x in (0, 2^-14) exceeds 1/8 relative error (proves guard necessity)",
        [
            xs > 0,
            xs < min_normal,
            z3_abs(xs - nearest_sub) > z3.Q(1, 8) * xs,
        ],
    )

    print(f"\nALL 12 BIT-LEVEL ENCODING OBLIGATIONS + 1 NEGATIVE CONTROL VERIFIED IN {time.perf_counter() - t0:.3f}s.")


if __name__ == "__main__":
    main()
