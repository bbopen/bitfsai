#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_utilities.h"
#include "../../reference/workloads.hpp"

#define CHECK_HYPRE(expr)                                                      \
  do {                                                                         \
    HYPRE_Int rc_ = (expr);                                                    \
    if (rc_ != 0)                                                              \
      throw std::runtime_error(std::string(#expr) + " failed rc=" +            \
                               std::to_string(static_cast<int>(rc_)));         \
  } while (0)

static HostCSR load_case(const std::string &name, int step) {
  if (name == "grid128" || name == "grid192" || name == "grid16")
    return build_3d_structured(std::stoi(name.substr(4)), step);
  if (name == "elastic28")
    return build_3d_elasticity(28, step);
  if (name == "tet48")
    return build_p1_3d_tet_fem(48, step);
  if (std::filesystem::exists(name))
    return load_mtx_csr(name);
  const char *dir = std::getenv("MTX_DIR");
  std::string base = dir ? std::string(dir) : ".";
  return load_mtx_csr(base + "/" + name);
}

static double cpu_true_rel(const HostCSR &h, const std::vector<double> &b,
                           const std::vector<double> &x) {
  double r2 = 0.0, b2 = 0.0;
  for (int i = 0; i < h.n; ++i) {
    double ax = 0.0;
    for (int k = h.rp[i]; k < h.rp[i + 1]; ++k) ax += h.val[k] * x[h.ci[k]];
    double d = b[i] - ax;
    r2 += d * d;
    b2 += b[i] * b[i];
  }
  return std::sqrt(r2 / b2);
}

static HYPRE_IJMatrix build_hypre_matrix(const HostCSR &h) {
  HYPRE_IJMatrix A;
  CHECK_HYPRE(HYPRE_IJMatrixCreate(0, 0, h.n - 1, 0, h.n - 1, &A));
  CHECK_HYPRE(HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR));
  CHECK_HYPRE(HYPRE_IJMatrixInitialize_v2(A, HYPRE_MEMORY_HOST));

  std::vector<HYPRE_Int> ncols(h.n);
  std::vector<HYPRE_BigInt> rows(h.n);
  for (int i = 0; i < h.n; ++i) {
    ncols[i] = h.rp[i + 1] - h.rp[i];
    rows[i] = i;
  }
  std::vector<HYPRE_BigInt> cols(h.nnz);
  for (int k = 0; k < h.nnz; ++k) cols[k] = h.ci[k];

  CHECK_HYPRE(HYPRE_IJMatrixSetValues(A, h.n, ncols.data(), rows.data(), cols.data(), h.val.data()));
  CHECK_HYPRE(HYPRE_IJMatrixAssemble(A));
  CHECK_HYPRE(HYPRE_IJMatrixMigrate(A, HYPRE_MEMORY_DEVICE));
  return A;
}

static HYPRE_IJVector build_hypre_vec(int n, const double *vals) {
  HYPRE_IJVector v;
  CHECK_HYPRE(HYPRE_IJVectorCreate(0, 0, n - 1, &v));
  CHECK_HYPRE(HYPRE_IJVectorSetObjectType(v, HYPRE_PARCSR));
  CHECK_HYPRE(HYPRE_IJVectorInitialize_v2(v, HYPRE_MEMORY_HOST));
  std::vector<HYPRE_BigInt> idx(n);
  for (int i = 0; i < n; ++i) idx[i] = i;
  CHECK_HYPRE(HYPRE_IJVectorSetValues(v, n, idx.data(), vals));
  CHECK_HYPRE(HYPRE_IJVectorAssemble(v));
  CHECK_HYPRE(HYPRE_IJVectorMigrate(v, HYPRE_MEMORY_DEVICE));
  return v;
}

int main(int argc, char **argv) try {
  if (argc < 4) {
    std::fprintf(stderr, "usage: bench_hypre CASE METHOD REPEATS\n");
    return 1;
  }
  std::string name = argv[1];
  std::string method = argv[2];
  if (method != "l1_jacobi" && method != "fsai") {
    std::fprintf(stderr, "unknown method '%s' (expected l1_jacobi or fsai)\n", method.c_str());
    return 1;
  }
  int repeats = std::stoi(argv[3]);
  if (repeats < 1) {
    std::fprintf(stderr, "repeats must be positive\n");
    return 1;
  }

  HYPRE_Initialize();
  HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
  HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
  HYPRE_SetSpGemmUseVendor(0);

  HostCSR h0 = load_case(name, 0);
  HostCSR h2 = load_case(name, 2);
  assert_same_pattern(h0, h2, name, 2);

  std::mt19937 rng(12345);
  std::normal_distribution<double> nd(0.0, 1.0);
  std::vector<double> b_vec(h0.n), zero_vec(h0.n, 0.0);
  for (auto &v : b_vec) v = nd(rng);

  HYPRE_IJMatrix ijA0 = build_hypre_matrix(h0);
  HYPRE_IJMatrix ijA2 = build_hypre_matrix(h2);
  HYPRE_IJVector ijB = build_hypre_vec(h0.n, b_vec.data());

  HYPRE_ParCSRMatrix parA0, parA2;
  HYPRE_ParVector parB;
  HYPRE_IJMatrixGetObject(ijA0, (void **)&parA0);
  HYPRE_IJMatrixGetObject(ijA2, (void **)&parA2);
  HYPRE_IJVectorGetObject(ijB, (void **)&parB);

  std::vector<HYPRE_BigInt> idx(h0.n);
  for (int i = 0; i < h0.n; ++i) idx[i] = i;

  bool all_pass = true;
  std::printf("engine,case,method,trial,phase,setup_ms,solve_ms,iters,true_rel,pass\n");
  for (int t = 0; t < repeats; ++t) {
    for (int phase = 0; phase < 2; ++phase) {
      HYPRE_ParCSRMatrix parA = (phase == 0) ? parA0 : parA2;
      const HostCSR &h = (phase == 0) ? h0 : h2;

      HYPRE_IJVector ijX = build_hypre_vec(h0.n, zero_vec.data());
      HYPRE_ParVector parX;
      HYPRE_IJVectorGetObject(ijX, (void **)&parX);

      HYPRE_Solver pcg, amg;
      HYPRE_ParCSRPCGCreate(0, &pcg);
      HYPRE_PCGSetMaxIter(pcg, 2000);
      HYPRE_PCGSetTol(pcg, 7e-9);
      HYPRE_PCGSetTwoNorm(pcg, 1);
      HYPRE_PCGSetPrintLevel(pcg, 0);

      HYPRE_BoomerAMGCreate(&amg);
      HYPRE_BoomerAMGSetPrintLevel(amg, 0);
      HYPRE_BoomerAMGSetMaxIter(amg, 1);
      HYPRE_BoomerAMGSetTol(amg, 0.0);
      HYPRE_BoomerAMGSetCoarsenType(amg, 8);
      HYPRE_BoomerAMGSetInterpType(amg, 6);
      HYPRE_BoomerAMGSetPMaxElmts(amg, 4);
      HYPRE_BoomerAMGSetStrongThreshold(amg, 0.5);
      HYPRE_BoomerAMGSetNumSweeps(amg, 1);
      if (method == "fsai") {
        HYPRE_BoomerAMGSetSmoothType(amg, 4);
        HYPRE_BoomerAMGSetSmoothNumLevels(amg, 10);
        HYPRE_BoomerAMGSetSmoothNumSweeps(amg, 1);
        HYPRE_BoomerAMGSetFSAIMaxSteps(amg, 5);
        HYPRE_BoomerAMGSetFSAIMaxStepSize(amg, 3);
        HYPRE_BoomerAMGSetFSAIKapTolerance(amg, 1e-3);
        HYPRE_BoomerAMGSetRelaxType(amg, 18);
      } else {
        HYPRE_BoomerAMGSetRelaxType(amg, 18);
      }
      HYPRE_PCGSetPrecond(pcg, (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
                          (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup, amg);

      auto check_cuda_rc = [&](const char *step) {
        cudaError_t cerr = cudaDeviceSynchronize();
        if (cerr != cudaSuccess) {
          std::fprintf(stderr, "CUDA error in %s (trial=%d, phase=%s): %s\n",
                       step, t, phase ? "timed" : "warm", cudaGetErrorString(cerr));
          return false;
        }
        return true;
      };
      if (!check_cuda_rc("pre-setup")) return 1;
      auto t0 = std::chrono::steady_clock::now();
      HYPRE_Int setup_rc = HYPRE_ParCSRPCGSetup(pcg, parA, parB, parX);
      if (setup_rc != 0) {
        std::fprintf(stderr, "HYPRE_ParCSRPCGSetup failed (rc=%d, trial=%d, phase=%s)\n",
                     (int)setup_rc, t, phase ? "timed" : "warm");
        return 1;
      }
      if (!check_cuda_rc("post-setup")) return 1;
      auto t1 = std::chrono::steady_clock::now();
      double setup_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

      auto t2 = std::chrono::steady_clock::now();
      HYPRE_Int solve_rc = HYPRE_ParCSRPCGSolve(pcg, parA, parB, parX);
      if (solve_rc != 0 && (solve_rc & ~HYPRE_ERROR_CONV) != 0) {
        std::fprintf(stderr, "HYPRE_ParCSRPCGSolve failed (rc=%d, trial=%d, phase=%s)\n",
                     (int)solve_rc, t, phase ? "timed" : "warm");
        return 1;
      }
      if (solve_rc & HYPRE_ERROR_CONV) HYPRE_ClearAllErrors();
      if (!check_cuda_rc("post-solve")) return 1;
      auto t3 = std::chrono::steady_clock::now();
      double solve_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

      HYPRE_Int iters = 0;
      HYPRE_PCGGetNumIterations(pcg, &iters);

      HYPRE_IJVectorMigrate(ijX, HYPRE_MEMORY_HOST);
      std::vector<double> x_vec(h.n);
      HYPRE_IJVectorGetValues(ijX, h.n, idx.data(), x_vec.data());
      double rel = cpu_true_rel(h, b_vec, x_vec);
      int pass = (std::isfinite(rel) && rel <= 1e-8) ? 1 : 0;
      if (!pass)
        all_pass = false;

      std::printf("hypre,%s,%s,%d,%s,%.5f,%.5f,%d,%.12e,%d\n",
                  name.c_str(), method.c_str(), t, phase ? "timed" : "warm",
                  setup_ms, solve_ms, (int)iters, rel, pass);
      std::fflush(stdout);

      HYPRE_BoomerAMGDestroy(amg);
      HYPRE_ParCSRPCGDestroy(pcg);
      HYPRE_IJVectorDestroy(ijX);
    }
  }

  HYPRE_IJMatrixDestroy(ijA0);
  HYPRE_IJMatrixDestroy(ijA2);
  HYPRE_IJVectorDestroy(ijB);
  HYPRE_Finalize();
  return all_pass ? 0 : 2;
} catch (const std::exception &e) {
  std::fprintf(stderr, "%s\n", e.what());
  return 1;
}
