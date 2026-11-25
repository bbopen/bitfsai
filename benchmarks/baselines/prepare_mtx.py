#!/usr/bin/env python3
"""Prepare and verify SuiteSparse Matrix Market inputs for BitFSAI benchmarks.

Usage:
  python3 benchmarks/baselines/prepare_mtx.py --mtx-dir /path/to/mtx [--negate-rail]

With --negate-rail, reads Oberwolfach/rail_79841.mtx (negative definite) from
--mtx-dir and writes the negated SPD rail_79841_neg.mtx. Always verifies the
SHA-256 checksums of present benchmark matrices.
"""
import argparse
import hashlib
from pathlib import Path

EXPECTED_SHA256 = {
    "bcsstk14.mtx": "68e263d540c7be28aa9345387118e47201a6b935d3443fb98d0a8cac7731efaa",
    "bcsstk15.mtx": "8c74572f6e4bd9c2362a3d4cce360a0cebf9fd01c55c462e189efec727ddd9fb",
    "bcsstk16.mtx": "380aef6ba9ce95ef7934a0c3dfb080337f9d9234ec621b9f39af6d49ea86d3c0",
    "rail_79841.mtx": "b829d365878f7bee26c2fff3b987e87786d88a17d4fb308c9fe47468abc2db3f",
    "rail_79841_neg.mtx": "5b2a9a259fefd00845d82ab787285da35122b68c8f278394de2248543dbce156",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def negate_mtx(src: Path, dst: Path) -> None:
    saw_dims = False
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if line.startswith("%") or not line.strip():
                fout.write(line)
                continue
            if not saw_dims:
                saw_dims = True
                fout.write(line)
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Malformed coordinate line in {src}: {line!r}")
            val = parts[2]
            neg_val = val[1:] if val.startswith("-") else ("-" + val)
            fout.write(f"{parts[0]} {parts[1]} {neg_val}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mtx-dir", type=Path, required=True, help="Directory holding .mtx files")
    ap.add_argument("--negate-rail", action="store_true", help="Create rail_79841_neg.mtx from rail_79841.mtx")
    args = ap.parse_args()

    if args.negate_rail:
        src = args.mtx_dir / "rail_79841.mtx"
        dst = args.mtx_dir / "rail_79841_neg.mtx"
        if not src.exists():
            raise FileNotFoundError(f"Missing source matrix: {src}")
        negate_mtx(src, dst)
        print(f"Wrote {dst}")

    for name, expected in EXPECTED_SHA256.items():
        p = args.mtx_dir / name
        if not p.exists():
            print(f"[MISSING] {p}")
            continue
        actual = sha256_file(p)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {name}: got {actual}, expected {expected}")
        print(f"[OK] {name}  {actual}")


if __name__ == "__main__":
    main()
