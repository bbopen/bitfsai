# External GPU AMG baseline drivers

These standalone C++/CUDA drivers reproduce the NVIDIA AmgX, Ginkgo `develop` (2.0.0-dev), and LLNL `hypre` benchmarks reported in the paper against the same deterministic SPD workloads (`reference/workloads.hpp`) and host-side `FP64` true relative residual gate (`||b - Ax||_2 / ||b||_2 <= 1e-8`).

| Driver | Target library and configurations |
|---|---|
| `bench_amgx.cu` | NVIDIA AmgX ([`bbopen/AMGX`](https://github.com/bbopen/AMGX), branch `bitfsai-smoother`, commit `8f289e59047a000c9a7362aa1564784644c8fdc9`) in matched `dDDI` (`jacobi`, `fp16_rep`, `int4_rep`, `int4_norep`) and matched `dFFI` (`mix_jacobi`, `mix_fp16_rep`, `mix_int4_rep`, `mix_int4_norep`) with `8e-9` inner tolerance, fixed $\omega = 0.72$, and host `<= 1e-8` true residual check. Exits `0` when all trials pass and `2` when any trial fails (as expected for `*_norep`). |
| `bench_ginkgo.cpp` | Ginkgo `develop` 2.0.0-dev ([`ginkgo-project/ginkgo`](https://github.com/ginkgo-project/ginkgo), commit `796a388db0e8fb8c2a2d9918c658858bcab9c798`, ahead of `v1.11.0`) `Cg<double>` preconditioned by `Pgm` multigrid in three configs: `mg_jacobi_f32` (`FP32` Jacobi), `mg_isai_f32` (`FP32` `SpdIsai(1)`), and `mg_isai_3p` (our unscaled three-precision config with `FP32` finest `SpdIsai` and unscaled `FP16` coarse levels). |
| `bench_hypre.cu` | LLNL `hypre` 3.2.1+ ([`hypre-space/hypre`](https://github.com/hypre-space/hypre), `master` commit `1da70dfc55ba6b88d43cf14c881beb810b92f079`, 4 commits past `v3.2.1`, CUDA `FP64` backend) `HYPRE_ParCSRPCG` with `BoomerAMG` (`PMIS` + `ext+i`) using `l1_jacobi` (`relax_type = 18`) or native GPU adaptive `fsai` (`smooth_type = 4`). |
| `prepare_mtx.py` | Negates `Oberwolfach/rail_79841.mtx` into SPD `rail_79841_neg.mtx` and checks SHA-256 hashes for all four SuiteSparse `.mtx` inputs. |

## Building

Point the compiler at your local AmgX, Ginkgo, or `hypre` install prefix.

```bash
# NVIDIA AmgX (built from bbopen/AMGX @ 8f289e59047a000c9a7362aa1564784644c8fdc9)
nvcc -O3 -std=c++17 -arch=native bench_amgx.cu -I$AMGX_PREFIX/include -L$AMGX_PREFIX/lib \
  -Xlinker -rpath,$AMGX_PREFIX/lib -lamgxsh -o bench_amgx

# Ginkgo develop / 2.0.0-dev (CUDA executor)
g++ -O3 -std=c++17 bench_ginkgo.cpp -I$GINKGO_PREFIX/include -L$GINKGO_PREFIX/lib \
  -Wl,-rpath,$GINKGO_PREFIX/lib -lginkgo -lginkgo_cuda -lginkgo_omp -lginkgo_reference \
  -lginkgo_device -lginkgo_hip -lginkgo_dpcpp -lcudart -o bench_ginkgo

# LLNL hypre (CUDA, no-MPI single-GPU build)
nvcc -O3 -std=c++17 -arch=native bench_hypre.cu -I$HYPRE_PREFIX/include \
  $HYPRE_PREFIX/lib/libHYPRE.a -lcusparse -lcublas -lcusolver -lcurand -o bench_hypre
```

Run any workload (`grid128`, `grid192`, `tet48`, `elastic28`, or a SuiteSparse `.mtx` file via `MTX_DIR`).

```bash
./bench_amgx   grid128 mix_int4_rep 5
./bench_ginkgo grid128 mg_isai_f32 5
./bench_hypre  grid128 fsai 5
```
