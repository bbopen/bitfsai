# BitFSAI raw trial data and provenance manifest

This directory holds the trial-by-trial raw CSV outputs, the deterministic aggregation script `build_summary.py`, and aggregated medians in `summary.json` for all four GPU benchmark suites in the paper, along with `cpu_precision_sweep.csv`. Every passing timed solve (`pass = 1`) clears an independent host IEEE `FP64` check of `||b - Ax||_2 / ||b||_2 <= 1e-8` using seeded right-hand sides (`std::mt19937(12345)`, `N(0, 1)`).

## Hardware and software environment

All GPU runs executed sequentially on an NVIDIA RTX Spark N1X (6144-core Blackwell GPU, Driver `615.78.08`, ARM64 Cortex-X925/A725 host running Linux, `nvcc` 13.4 `V13.4.92` with `-O3 -std=c++17 -arch=native`), one GPU process at a time with zero concurrent GPU workloads. Each trial runs an untimed `warm` pass at `step = 0` followed by a `timed` coefficient-refresh pass at `step = 2`.

## SuiteSparse matrix preparation and SHA-256 checksums

Download `HB/bcsstk14`, `HB/bcsstk15`, `HB/bcsstk16`, and `Oberwolfach/rail_79841` from SuiteSparse into `MTX_DIR`. Then negate `rail_79841.mtx` into SPD `rail_79841_neg.mtx` and verify all checksums.

```bash
python3 benchmarks/baselines/prepare_mtx.py --mtx-dir "$MTX_DIR" --negate-rail
```

| File | SHA-256 |
| :--- | :--- |
| `bcsstk14.mtx` | `68e263d540c7be28aa9345387118e47201a6b935d3443fb98d0a8cac7731efaa` |
| `bcsstk15.mtx` | `8c74572f6e4bd9c2362a3d4cce360a0cebf9fd01c55c462e189efec727ddd9fb` |
| `bcsstk16.mtx` | `380aef6ba9ce95ef7934a0c3dfb080337f9d9234ec621b9f39af6d49ea86d3c0` |
| `rail_79841.mtx` | `b829d365878f7bee26c2fff3b987e87786d88a17d4fb308c9fe47468abc2db3f` |
| `rail_79841_neg.mtx` | `5b2a9a259fefd00845d82ab787285da35122b68c8f278394de2248543dbce156` |

## Dataset contents

| File | Description |
| :--- | :--- |
| `cpu_precision_sweep.csv` | CPU two-grid smoother quantization cliff (`IC(0)` vs. `BitFSAI`, plus `INT4inc` control) and coarse Galerkin $A_c$ precision sweep emitted by `benchmarks/quant_cliff_and_coarse_floor.py --export-csv`. |
| `standalone_8cases.csv` | Raw `warm` + `timed` trials from `reference/standalone_amg` across all 8 SPD workloads (`grid128` $n=5$, `grid192` $n=3$, `tet48` $n=5$, `elastic28` $n=5$, `rail_79841_neg.mtx` $n=5$, `bcsstk14.mtx` $n=5$, `bcsstk15.mtx` $n=5$, `bcsstk16.mtx` $n=5$) and all 4 smoother modes (`jacobi`, `half`, `int4`, `double`), tagged with `case_label`. |
| `standalone_multilevel.csv` | Raw `warm` + `timed` trials from `reference/standalone_amg` (`int4` smoother-only allocation) on `grid128` ($n=5$) and `grid192` ($n=3$) across 5 multilevel cycle configurations (`vcycle_fp32`, `vcycle_fp8`, `poly4_fp32`, `poly4_shift2_fp32`, `poly4_shift2_fp8`), tagged with `case_label` and `config`. |
| `amgx_8cases.csv` | Raw `warm` + `timed` trials from `benchmarks/baselines/bench_amgx.cu` against [`bbopen/AMGX`](https://github.com/bbopen/AMGX) (branch `bitfsai-smoother`, commit `8f289e59047a000c9a7362aa1564784644c8fdc9`) across all 8 workloads in matched `dDDI` (`jacobi`, `fp16_rep`, `int4_rep`) and matched `dFFI` (`mix_jacobi`, `mix_fp16_rep`, `mix_int4_rep`), plus the Theorem 3 row-reprojection ablation (`int4_norep`, `mix_int4_norep`) on `bcsstk14.mtx` and `bcsstk15.mtx`. |
| `prior_art_8cases.csv` | Raw `warm` + `timed` trials from `benchmarks/baselines/bench_ginkgo.cpp` (Ginkgo `develop` 2.0.0-dev, `ginkgo-project/ginkgo` `796a388db0e8fb8c2a2d9918c658858bcab9c798`: `mg_jacobi_f32`, `mg_isai_f32`, and our unscaled `mg_isai_3p`) and `benchmarks/baselines/bench_hypre.cu` (LLNL `hypre` 3.2.1+, `hypre-space/hypre` `1da70dfc55ba6b88d43cf14c881beb810b92f079`, `FP64` CUDA `BoomerAMG`: `l1_jacobi`, `fsai`) across all 8 workloads, with an appended `note` column (`ok`, or `trial=-1, note=timeout, pass=0` for the two 240 s `mg_isai_3p` timeouts on `grid128` and `grid192`). |
| `pattern_storage.csv` | Per-workload symbolic hierarchy pattern counts (`levels`, `rows`, `nnz`, `full_lower_nnz`, `selected_lower_nnz` under `K <= 16`, `truncated_rows`, and `weight_bytes`) emitted directly by `reference/standalone_amg --pattern-stats` across all 8 workloads, matching `standalone_8cases.csv` byte-for-byte. |
| `build_summary.py` & `summary.json` | Deterministic aggregator (`python3 benchmarks/results/build_summary.py`) emitting `timed`-phase medians, `[min-max]` ranges, `max_rel`, pass ratios, failed-trial diagnostics, `amgx_8cases` `warm_recovered`/`warm_triggers`, and exact same-pattern (`K <= 16`) / full-pattern / mirrored storage ratios cross-checked against `standalone_8cases.csv` and `pattern_storage.csv`. |

## Column definitions (`standalone_8cases.csv` and `standalone_multilevel.csv`)

| Column | Meaning |
| :--- | :--- |
| `fp8coarse` | `1` when symmetrized row-sum-repaired `FP8 E5M2` off-diagonal + `FP32` diagonal storage (enforcing grounded SDDM and normal `E5M2` range `[2^-14, 57344]`) is enabled on nonterminal coarse levels ($j > 0$), `0` for `FP32` coarse residual operators. |
| `poly_stride`, `poly_offset`, `poly_scale` | Recursive second-degree polynomial boost parameters $K_c = 2 t B_c - t^2 B_c A_c B_c$ (`polySTRIDE[@OFFSET][:SCALE]`, where `poly_stride = 0` denotes standard V-cycle). |
| `factored_levels` | `k/L`, where $L$ is the number of nonterminal smoother levels and $k$ is the count of nonterminal levels running the Theorem 4 scale-factored zero-index `FP32` smooth kernel. |
| `weight_bytes` | Active smoother representation bytes across nonterminal levels for the active `method`. |
| `coarse_bytes` | Active nonterminal coarse operator value bytes ($j > 0$, excluding the finest matrix and terminal dense Cholesky leaf). |
| `resident_bytes` | Stored-value buffer bytes allocated across smoother representations and nonterminal coarse apply-value buffers, distinct from total process GPU VRAM. |

## Reproducing benchmarks and summary

```bash
# 1. Standalone 8-workload suite
make -C reference standalone_amg
for c in grid128 tet48 elastic28; do ./reference/standalone_amg "$c" 5; done
./reference/standalone_amg grid192 3
for c in rail_79841_neg.mtx bcsstk14.mtx bcsstk15.mtx bcsstk16.mtx; do
  MTX_DIR="$MTX_DIR" ./reference/standalone_amg "$c" 5
done

# 2. Standalone multilevel FP8 E5M2 coarse + polynomial boost study (grid128 n=5, grid192 n=3)
for pair in "grid128 5" "grid192 3"; do
  set -- $pair
  ./reference/standalone_amg "$1" "$2" int4
  ./reference/standalone_amg "$1" "$2" int4 fp8coarse
  ./reference/standalone_amg "$1" "$2" int4 poly4:1.25
  ./reference/standalone_amg "$1" "$2" int4 poly4@2:1.25
  ./reference/standalone_amg "$1" "$2" int4 fp8coarse poly4@2:1.25
done

# 3. NVIDIA AmgX, LLNL hypre, and Ginkgo develop (see benchmarks/baselines/README.md for build commands)
for m in jacobi fp16_rep int4_rep mix_jacobi mix_fp16_rep mix_int4_rep; do
  ./bench_amgx grid128 "$m" 5
  ./bench_amgx grid192 "$m" 3
  for c in tet48 elastic28 rail_79841_neg.mtx bcsstk14.mtx bcsstk15.mtx bcsstk16.mtx; do
    MTX_DIR="$MTX_DIR" ./bench_amgx "$c" "$m" 5
  done
done
for c in bcsstk14.mtx bcsstk15.mtx; do
  for m in int4_norep mix_int4_norep; do
    MTX_DIR="$MTX_DIR" ./bench_amgx "$c" "$m" 5 || true
  done
done

# 4. Regenerate pattern_storage.csv and summary.json from tracked CSVs
MTX_DIR="$MTX_DIR" ./reference/standalone_amg --pattern-stats \
  grid128 grid192 tet48 elastic28 rail_79841_neg.mtx bcsstk14.mtx bcsstk15.mtx bcsstk16.mtx \
  > benchmarks/results/pattern_storage.csv
python3 benchmarks/results/build_summary.py
```
