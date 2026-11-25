#!/usr/bin/env python3
"""Deterministically rebuild benchmarks/results/summary.json from tracked CSVs."""
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def med(vals):
    return round(statistics.median(vals), 3)


def r3(val):
    return round(val, 3)


def summarize_standalone(csv_path: Path, key_col: str) -> dict:
    groups = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["phase"] != "timed":
                continue
            k = f"{row['case_label']}|{row[key_col]}"
            groups.setdefault(k, []).append(row)
    out = {}
    for k, rows in groups.items():
        passed = [r for r in rows if int(r["pass"]) == 1]
        entry = {"pass": f"{len(passed)}/{len(rows)}"}
        if passed:
            reset = [float(r["reset_ms"]) for r in passed]
            solve = [float(r["solve_ms"]) for r in passed]
            paired = [float(r["reset_ms"]) + float(r["solve_ms"]) for r in passed]
            iters = [int(r["iters"]) for r in passed]
            rels = [float(r["true_rel"]) for r in passed]
            r0 = passed[0]
            entry.update({
                "iters": int(statistics.median(iters)),
                "reset_ms": med(reset),
                "solve_ms": med(solve),
                "solve_ms_min": r3(min(solve)),
                "solve_ms_max": r3(max(solve)),
                "paired_ms": med(paired),
                "paired_ms_min": r3(min(paired)),
                "paired_ms_max": r3(max(paired)),
                "max_rel": max(rels),
                "factored_levels": r0["factored_levels"],
                "fp8coarse": int(r0["fp8coarse"]),
                "poly_stride": int(r0["poly_stride"]),
                "poly_offset": int(r0["poly_offset"]),
                "poly_scale": float(r0["poly_scale"]),
                "weight_bytes": int(r0["weight_bytes"]),
                "coarse_bytes": int(r0["coarse_bytes"]),
                "resident_bytes": int(r0["resident_bytes"]),
                "fallbacks": max(int(r["diagonal_fallbacks"]) for r in passed),
            })
        out[k] = entry
    return out


def summarize_amgx(csv_path: Path) -> dict:
    timed_groups, warm_groups = {}, {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            k = f"{row['case']}|{row['method']}"
            if row["phase"] == "timed":
                timed_groups.setdefault(k, []).append(row)
            elif row["phase"] == "warm":
                warm_groups.setdefault(k, []).append(row)
    out = {}
    for k, rows in timed_groups.items():
        passed = [r for r in rows if int(r["pass"]) == 1]
        wrows = warm_groups.get(k, [])
        warm_rec = sum(int(r["recovered"]) for r in wrows)
        warm_trigs = sorted({r["trigger"] for r in wrows if r["trigger"] != "none"})
        entry = {"pass": f"{len(passed)}/{len(rows)}"}
        if passed:
            reset = [float(r["reset_ms"]) for r in passed]
            solve = [float(r["solve_ms"]) for r in passed]
            paired = [float(r["reset_ms"]) + float(r["solve_ms"]) for r in passed]
            iters = [int(r["iters"]) for r in passed]
            rels = [float(r["true_rel"]) for r in passed]
            entry.update({
                "iters": int(statistics.median(iters)),
                "reset_ms": med(reset),
                "solve_ms": med(solve),
                "solve_ms_min": r3(min(solve)),
                "solve_ms_max": r3(max(solve)),
                "paired_ms": med(paired),
                "paired_ms_min": r3(min(paired)),
                "paired_ms_max": r3(max(paired)),
                "max_rel": max(rels),
                "recovered": sum(int(r["recovered"]) for r in passed),
            })
        else:
            all_iters = [int(r["iters"]) for r in rows]
            all_rels = [float(r["true_rel"]) for r in rows if math.isfinite(float(r["true_rel"]))]
            entry.update({
                "all_trials_iters": int(statistics.median(all_iters)),
                "all_trials_min_rel": min(all_rels) if all_rels else None,
                "all_trials_max_rel": max(all_rels) if all_rels else None,
                "triggers": sorted({r["trigger"] for r in rows}),
                "recovered": sum(int(r["recovered"]) for r in rows),
            })
        entry["warm_recovered"] = warm_rec
        if warm_trigs:
            entry["warm_triggers"] = warm_trigs
        out[k] = entry
    return out


def summarize_prior_art(csv_path: Path) -> dict:
    groups = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["phase"] != "timed":
                continue
            k = f"{row['case']}|{row['engine']}|{row['method']}"
            groups.setdefault(k, []).append(row)
    out = {}
    for k, rows in groups.items():
        passed = [r for r in rows if int(r["pass"]) == 1]
        entry = {"pass": f"{len(passed)}/{len(rows)}"}
        if passed:
            setup = [float(r["setup_ms"]) for r in passed]
            solve = [float(r["solve_ms"]) for r in passed]
            paired = [float(r["setup_ms"]) + float(r["solve_ms"]) for r in passed]
            iters = [int(r["iters"]) for r in passed]
            rels = [float(r["true_rel"]) for r in passed]
            entry.update({
                "iters": int(statistics.median(iters)),
                "reset_ms": med(setup),
                "solve_ms": med(solve),
                "solve_ms_min": r3(min(solve)),
                "solve_ms_max": r3(max(solve)),
                "paired_ms": med(paired),
                "paired_ms_min": r3(min(paired)),
                "paired_ms_max": r3(max(paired)),
                "max_rel": max(rels),
            })
            if len(passed) < len(rows):
                all_setup = [float(r["setup_ms"]) for r in rows]
                all_solve = [float(r["solve_ms"]) for r in rows]
                all_paired = [float(r["setup_ms"]) + float(r["solve_ms"]) for r in rows]
                all_iters = [int(r["iters"]) for r in rows]
                all_rels = [float(r["true_rel"]) for r in rows if math.isfinite(float(r["true_rel"]))]
                entry.update({
                    "all_trials_iters": int(statistics.median(all_iters)),
                    "all_trials_reset_ms": med(all_setup),
                    "all_trials_solve_ms": med(all_solve),
                    "all_trials_paired_ms": med(all_paired),
                    "all_trials_max_rel": max(all_rels) if all_rels else None,
                })
        else:
            entry["notes"] = sorted({r["note"] for r in rows})
        out[k] = entry
    return out


def summarize_pattern_storage(csv_path: Path, standalone_8cases: dict) -> dict:
    """Summarize exact same-pattern (K<=16), full-pattern, and mirrored storage ratios."""
    out = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            case = row["case_label"]
            levels = int(row["levels"])
            rows = int(row["rows"])
            nnz = int(row["nnz"])
            full_lower = int(row["full_lower_nnz"])
            sel_lower = int(row["selected_lower_nnz"])
            trunc = int(row["truncated_rows"])
            wi4 = int(row["int4_weight_bytes"])
            w64 = int(row["fp64_mirrored_bytes"])
            w16 = int(row["fp16_mirrored_bytes"])

            # Cross-check against standalone_8cases.csv weight_bytes and nonterminal level count.
            for m, expected in (("int4", wi4), ("double", w64), ("half", w16)):
                key = f"{case}|{m}"
                if key not in standalone_8cases:
                    raise RuntimeError(f"pattern_storage.csv case missing from standalone_8cases: {key}")
                got = standalone_8cases[key]["weight_bytes"]
                if got != expected:
                    raise RuntimeError(f"weight_bytes mismatch for {key}: csv={got} vs pattern={expected}")
            got_lvls = int(standalone_8cases[f"{case}|int4"]["factored_levels"].split("/")[1])
            if got_lvls != levels:
                raise RuntimeError(f"level count mismatch for {case}: csv={got_lvls} vs pattern={levels}")
            if full_lower != (nnz + rows) // 2:
                raise RuntimeError(f"full_lower_nnz identity failed for {case}")

            same_val = 8 * sel_lower
            same_vi = 12 * sel_lower + 4 * (rows + levels)
            full_val = 8 * full_lower
            full_vi = 12 * full_lower + 4 * (rows + levels)
            out[case] = {
                "levels": levels,
                "rows": rows,
                "nnz": nnz,
                "full_lower_nnz": full_lower,
                "selected_lower_nnz": sel_lower,
                "truncated_rows": trunc,
                "truncated_pct": round(100.0 * trunc / rows, 2),
                "int4_weight_bytes": wi4,
                "csv_int4_weight_bytes": standalone_8cases[f"{case}|int4"]["weight_bytes"],
                "same_pattern_conv64_val_ratio": r3(same_val / wi4),
                "same_pattern_conv64_vi_ratio": r3(same_vi / wi4),
                "full_pattern_conv64_val_ratio": r3(full_val / wi4),
                "full_pattern_conv64_vi_ratio": r3(full_vi / wi4),
                "mirrored_fp64_ratio": r3(w64 / wi4),
                "mirrored_fp16_ratio": r3(w16 / wi4),
            }
    return out


def main() -> None:
    standalone_8 = summarize_standalone(ROOT / "standalone_8cases.csv", "method")
    summary = {
        "standalone_8cases": standalone_8,
        "standalone_multilevel": summarize_standalone(ROOT / "standalone_multilevel.csv", "config"),
        "amgx_8cases": summarize_amgx(ROOT / "amgx_8cases.csv"),
        "prior_art_8cases": summarize_prior_art(ROOT / "prior_art_8cases.csv"),
        "pattern_storage": summarize_pattern_storage(ROOT / "pattern_storage.csv", standalone_8),
    }
    dst = ROOT / "summary.json"
    dst.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {dst} ({sum(len(v) for v in summary.values())} entries)")


if __name__ == "__main__":
    main()
