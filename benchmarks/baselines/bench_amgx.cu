// SPDX-FileCopyrightText: 2026 Brett Bonner
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Benchmark driver for NVIDIA AmgX (bbopen/AMGX, branch bitfsai-smoother,
// commit 8f289e59047a000c9a7362aa1564784644c8fdc9):
//   - Matched dDDI (FP64 inner V-cycle): jacobi, fp16_rep, int4_rep, int4_norep
//   - Matched dFFI (FP32 inner V-cycle + FP64 outer PCG with r^T z guard and
//     dDDI fallback): mix_jacobi, mix_fp16_rep, mix_int4_rep, mix_int4_norep
//   - Inner stopping tolerance 8e-9, gated by independent host CPU FP64 true
//     relative residual ||b - Ax||_2 / ||b||_2 <= 1e-8 on every trial.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <amgx_c.h>
#include "../../reference/workloads.hpp"

#define CHECK_CUDA(expr)                                                               \
  do {                                                                                 \
    cudaError_t err = (expr);                                                          \
    if (err != cudaSuccess)                                                            \
      throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(err));   \
  } while (0)

#define CHECK_AMGX(expr)                                                               \
  do {                                                                                 \
    AMGX_RC rc = (expr);                                                               \
    if (rc != AMGX_RC_OK)                                                              \
      throw std::runtime_error(std::string(#expr) + " rc=" + std::to_string(rc));      \
  } while (0)

__global__ void spmv_f64_k(int n, const int *rp, const int *ci, const double *val,
                           const double *x, double *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double s = 0.0;
  for (int k = rp[i]; k < rp[i + 1]; ++k) s += val[k] * __ldg(&x[ci[k]]);
  y[i] = s;
}

__global__ void axpy_f64_k(int n, double a, const double *x, double *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += a * x[i];
}

__global__ void xpay_f64_k(int n, const double *z, double beta, double *p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = z[i] + beta * p[i];
}

__global__ void d2f_k(int n, const double *src, float *dst) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = (float)src[i];
}

__global__ void f2d_k(int n, const float *src, double *dst) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = (double)src[i];
}

__global__ void dot_f64_k(int n, const double *x, const double *y, double *out) {
  __shared__ double sh[256];
  double s = 0.0;
  for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256)
    s += x[i] * (y ? y[i] : 1.0);
  int t = threadIdx.x;
  sh[t] = s;
  __syncthreads();
  for (int st = 128; st > 0; st >>= 1) {
    if (t < st) sh[t] += sh[t + st];
    __syncthreads();
  }
  if (t == 0) out[blockIdx.x] = sh[0];
}

struct DevPCG {
  int n = 0;
  double *d_r = nullptr, *d_z = nullptr, *d_p = nullptr, *d_q = nullptr;
  double *d_part = nullptr, *d_scal = nullptr;
  float *d_rf = nullptr, *d_zf = nullptr;

  void alloc(int n_) {
    n = n_;
    CHECK_CUDA(cudaMalloc(&d_r, (size_t)n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_z, (size_t)n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_p, (size_t)n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_q, (size_t)n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_part, 512 * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_scal, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_rf, (size_t)n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_zf, (size_t)n * sizeof(float)));
  }
  ~DevPCG() {
    cudaFree(d_r); cudaFree(d_z); cudaFree(d_p); cudaFree(d_q);
    cudaFree(d_part); cudaFree(d_scal); cudaFree(d_rf); cudaFree(d_zf);
  }
  double dot(const double *x, const double *y) {
    int bl = std::min(512, 1 + (n - 1) / 256);
    dot_f64_k<<<bl, 256>>>(n, x, y, d_part);
    CHECK_CUDA(cudaGetLastError());
    dot_f64_k<<<1, 256>>>(bl, d_part, nullptr, d_scal);
    CHECK_CUDA(cudaGetLastError());
    double v = 0.0;
    CHECK_CUDA(cudaMemcpy(&v, d_scal, sizeof(double), cudaMemcpyDeviceToHost));
    return v;
  }
};

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

static std::string smoother_json(const std::string &smoother, const std::string &mode, int rep) {
  if (smoother == "BITFSAI") {
    return "\"smoother\":{\"solver\":\"BITFSAI\",\"bitfsai_mode\":\"" + mode +
           "\",\"bitfsai_omega\":0.72,\"bitfsai_reproject\":" + std::to_string(rep) + "}";
  }
  return "\"smoother\":\"JACOBI_L1\"";
}

static std::string make_pcg_json(const std::string &smoother, const std::string &mode, int rep) {
  return "{\"config_version\":2,\"determinism_flag\":0,\"solver\":{"
         "\"solver\":\"PCG\",\"max_iters\":2000,\"tolerance\":8e-9,"
         "\"convergence\":\"RELATIVE_INI_CORE\",\"monitor_residual\":1,"
         "\"store_res_history\":0,\"norm\":\"L2\",\"preconditioner\":{"
         "\"solver\":\"AMG\",\"algorithm\":\"CLASSICAL\",\"selector\":\"PMIS\","
         "\"interpolator\":\"D2\",\"cycle\":\"V\",\"presweeps\":1,\"postsweeps\":1,"
         "\"coarsest_sweeps\":2,\"max_iters\":1," +
         smoother_json(smoother, mode, rep) + "}}}";
}

static std::string make_vcycle_json(const std::string &smoother, const std::string &mode, int rep) {
  return "{\"config_version\":2,\"determinism_flag\":0,\"solver\":{"
         "\"solver\":\"AMG\",\"algorithm\":\"CLASSICAL\",\"selector\":\"PMIS\","
         "\"interpolator\":\"D2\",\"cycle\":\"V\",\"presweeps\":1,\"postsweeps\":1,"
         "\"coarsest_sweeps\":2,\"max_iters\":1,\"monitor_residual\":0,"
         "\"store_res_history\":0," +
         smoother_json(smoother, mode, rep) + "}}";
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: bench_amgx CASE METHOD REPEATS\n");
    return 1;
  }
  std::string name = argv[1];
  std::string method = argv[2];
  int repeats = std::stoi(argv[3]);
  if (repeats < 1) {
    std::fprintf(stderr, "error: REPEATS must be >= 1\n");
    return 1;
  }
  const bool valid_method =
      (method == "jacobi" || method == "fp16_rep" || method == "int4_rep" ||
       method == "int4_norep" || method == "mix_jacobi" ||
       method == "mix_fp16_rep" || method == "mix_int4_rep" ||
       method == "mix_int4_norep");
  if (!valid_method) {
    std::fprintf(stderr, "error: unknown METHOD '%s'\n", method.c_str());
    return 1;
  }

  AMGX_initialize();
  try {
    HostCSR h0 = load_case(name, 0);
    HostCSR h2 = load_case(name, 2);
    assert_same_pattern(h0, h2, name, 2);
    int n = h0.n, nnz = h0.nnz;

    std::mt19937 rng(12345);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> b_vec(n), x_vec(n);
    for (auto &v : b_vec) v = nd(rng);

    std::vector<float> val0f(nnz), val2f(nnz);
    for (int k = 0; k < nnz; ++k) { val0f[k] = (float)h0.val[k]; val2f[k] = (float)h2.val[k]; }

    bool is_mix = (method.rfind("mix_", 0) == 0);
    int rep = (method.find("_rep") != std::string::npos) ? 1 : 0;
    std::string mode = (method.find("fp16") != std::string::npos) ? "fp16" : "int4_split";
    std::string smoother = (method.find("jacobi") != std::string::npos) ? "JACOBI_L1" : "BITFSAI";

    bool all_pass = true;
    std::printf("case,method,trial,phase,reset_ms,solve_ms,iters,recovered,trigger,true_rel,pass\n");
    cudaEvent_t e0, e1, e2, e3;
    CHECK_CUDA(cudaEventCreate(&e0)); CHECK_CUDA(cudaEventCreate(&e1));
    CHECK_CUDA(cudaEventCreate(&e2)); CHECK_CUDA(cudaEventCreate(&e3));

    if (!is_mix) {
      std::string cfg_str = make_pcg_json(smoother, mode, rep);
      AMGX_config_handle cfg; AMGX_resources_handle rsrc;
      AMGX_matrix_handle A; AMGX_vector_handle vb, vx; AMGX_solver_handle s;
      CHECK_AMGX(AMGX_config_create(&cfg, cfg_str.c_str()));
      CHECK_AMGX(AMGX_resources_create_simple(&rsrc, cfg));
      CHECK_AMGX(AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI));
      CHECK_AMGX(AMGX_vector_create(&vb, rsrc, AMGX_mode_dDDI));
      CHECK_AMGX(AMGX_vector_create(&vx, rsrc, AMGX_mode_dDDI));
      CHECK_AMGX(AMGX_solver_create(&s, rsrc, AMGX_mode_dDDI, cfg));

      CHECK_AMGX(AMGX_matrix_upload_all(A, n, nnz, 1, 1, h0.rp.data(), h0.ci.data(), h0.val.data(), nullptr));
      CHECK_AMGX(AMGX_vector_upload(vb, n, 1, b_vec.data()));
      CHECK_AMGX(AMGX_solver_setup(s, A));

      for (int t = 0; t < repeats; ++t) {
        for (int phase = 0; phase < 2; ++phase) {
          const HostCSR &h = phase ? h2 : h0;
          CHECK_CUDA(cudaEventRecord(e0));
          CHECK_AMGX(AMGX_matrix_replace_coefficients(A, n, nnz, h.val.data(), nullptr));
          CHECK_AMGX(AMGX_solver_resetup(s, A));
          CHECK_CUDA(cudaEventRecord(e1));
          CHECK_CUDA(cudaEventSynchronize(e1));
          float rst_ms = 0.0f;
          CHECK_CUDA(cudaEventElapsedTime(&rst_ms, e0, e1));

          CHECK_AMGX(AMGX_vector_set_zero(vx, n, 1));
          CHECK_CUDA(cudaEventRecord(e2));
          CHECK_AMGX(AMGX_solver_solve(s, vb, vx));
          CHECK_CUDA(cudaEventRecord(e3));
          CHECK_CUDA(cudaEventSynchronize(e3));
          float sol_ms = 0.0f;
          CHECK_CUDA(cudaEventElapsedTime(&sol_ms, e2, e3));

          int iters = 0;
          CHECK_AMGX(AMGX_solver_get_iterations_number(s, &iters));
          CHECK_AMGX(AMGX_vector_download(vx, x_vec.data()));
          double rel = cpu_true_rel(h, b_vec, x_vec);
          int pass = (std::isfinite(rel) && rel <= 1e-8) ? 1 : 0;
          all_pass &= (pass == 1);
          std::printf("%s,%s,%d,%s,%.5f,%.5f,%d,0,none,%.12e,%d\n",
                      name.c_str(), method.c_str(), t, phase ? "timed" : "warm",
                      rst_ms, sol_ms, iters, rel, pass);
          std::fflush(stdout);
        }
      }
      AMGX_solver_destroy(s); AMGX_vector_destroy(vb); AMGX_vector_destroy(vx);
      AMGX_matrix_destroy(A); AMGX_resources_destroy(rsrc); AMGX_config_destroy(cfg);
    } else {
      std::string cfg_str = make_vcycle_json(smoother, mode, rep);
      AMGX_config_handle cf, cd; AMGX_resources_handle rf, rd;
      AMGX_matrix_handle Af, Ad; AMGX_vector_handle vfb, vfx, vdb, vdx;
      AMGX_solver_handle sf, sd;
      CHECK_AMGX(AMGX_config_create(&cf, cfg_str.c_str()));
      CHECK_AMGX(AMGX_config_create(&cd, cfg_str.c_str()));
      CHECK_AMGX(AMGX_resources_create_simple(&rf, cf));
      CHECK_AMGX(AMGX_resources_create_simple(&rd, cd));
      CHECK_AMGX(AMGX_matrix_create(&Af, rf, AMGX_mode_dFFI));
      CHECK_AMGX(AMGX_matrix_create(&Ad, rd, AMGX_mode_dDDI));
      CHECK_AMGX(AMGX_vector_create(&vfb, rf, AMGX_mode_dFFI));
      CHECK_AMGX(AMGX_vector_create(&vfx, rf, AMGX_mode_dFFI));
      CHECK_AMGX(AMGX_vector_create(&vdb, rd, AMGX_mode_dDDI));
      CHECK_AMGX(AMGX_vector_create(&vdx, rd, AMGX_mode_dDDI));
      CHECK_AMGX(AMGX_solver_create(&sf, rf, AMGX_mode_dFFI, cf));
      CHECK_AMGX(AMGX_solver_create(&sd, rd, AMGX_mode_dDDI, cd));

      int *d_rp, *d_ci; double *d_val0, *d_val2, *d_b, *d_x; float *d_val0f, *d_val2f;
      CHECK_CUDA(cudaMalloc(&d_rp, (size_t)(n + 1) * sizeof(int)));
      CHECK_CUDA(cudaMalloc(&d_ci, (size_t)nnz * sizeof(int)));
      CHECK_CUDA(cudaMalloc(&d_val0, (size_t)nnz * sizeof(double)));
      CHECK_CUDA(cudaMalloc(&d_val2, (size_t)nnz * sizeof(double)));
      CHECK_CUDA(cudaMalloc(&d_val0f, (size_t)nnz * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&d_val2f, (size_t)nnz * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&d_b, (size_t)n * sizeof(double)));
      CHECK_CUDA(cudaMalloc(&d_x, (size_t)n * sizeof(double)));
      CHECK_CUDA(cudaMemcpy(d_rp, h0.rp.data(), (size_t)(n + 1) * sizeof(int), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(d_ci, h0.ci.data(), (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(d_val0, h0.val.data(), (size_t)nnz * sizeof(double), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(d_val2, h2.val.data(), (size_t)nnz * sizeof(double), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(d_b, b_vec.data(), (size_t)n * sizeof(double), cudaMemcpyHostToDevice));

      CHECK_AMGX(AMGX_matrix_upload_all(Af, n, nnz, 1, 1, h0.rp.data(), h0.ci.data(), val0f.data(), nullptr));
      CHECK_AMGX(AMGX_matrix_upload_all(Ad, n, nnz, 1, 1, h0.rp.data(), h0.ci.data(), h0.val.data(), nullptr));
      CHECK_AMGX(AMGX_solver_setup(sf, Af));
      bool sd_ready = false;

      DevPCG buf; buf.alloc(n);
      int bl = 1 + (n - 1) / 256, bl_nz = 1 + (nnz - 1) / 256;
      double bnorm = std::sqrt(buf.dot(d_b, d_b));

      for (int t = 0; t < repeats; ++t) {
        for (int phase = 0; phase < 2; ++phase) {
          const HostCSR &h = phase ? h2 : h0;
          const double *d_val = phase ? d_val2 : d_val0;
          float *d_valf = phase ? d_val2f : d_val0f;

          CHECK_CUDA(cudaEventRecord(e0));
          d2f_k<<<bl_nz, 256>>>(nnz, d_val, d_valf);
          CHECK_CUDA(cudaGetLastError());
          CHECK_AMGX(AMGX_matrix_replace_coefficients(Af, n, nnz, d_valf, nullptr));
          CHECK_AMGX(AMGX_solver_resetup(sf, Af));
          CHECK_CUDA(cudaEventRecord(e1));
          CHECK_CUDA(cudaEventSynchronize(e1));
          float rst_ms = 0.0f;
          CHECK_CUDA(cudaEventElapsedTime(&rst_ms, e0, e1));

          CHECK_CUDA(cudaEventRecord(e2));
          CHECK_CUDA(cudaMemset(d_x, 0, (size_t)n * sizeof(double)));
          CHECK_CUDA(cudaMemcpy(buf.d_r, d_b, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice));
          int iters = 0, recovered = 0;
          std::string trigger = "none";

          // Stage 0 runs the dFFI inner V-cycle; if a guard triggers (rz, curvature,
          // drift, stagnation, or max_iters), stage 1 sets recovered=1 and retries
          // with the dDDI inner V-cycle using the same smoother configuration.
          for (int stage = 0; stage < 2; ++stage) {
            if (stage == 1) {
              recovered = 1;
              if (!sd_ready) {
                CHECK_AMGX(AMGX_matrix_replace_coefficients(Ad, n, nnz, d_val, nullptr));
                CHECK_AMGX(AMGX_solver_setup(sd, Ad));
                sd_ready = true;
              } else {
                CHECK_AMGX(AMGX_matrix_replace_coefficients(Ad, n, nnz, d_val, nullptr));
                CHECK_AMGX(AMGX_solver_resetup(sd, Ad));
              }
              spmv_f64_k<<<bl, 256>>>(n, d_rp, d_ci, d_val, d_x, buf.d_q);
              CHECK_CUDA(cudaGetLastError());
              CHECK_CUDA(cudaMemcpy(buf.d_r, d_b, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice));
              axpy_f64_k<<<bl, 256>>>(n, -1.0, buf.d_q, buf.d_r);
              CHECK_CUDA(cudaGetLastError());
              if (!std::isfinite(buf.dot(buf.d_r, buf.d_r))) {
                CHECK_CUDA(cudaMemset(d_x, 0, (size_t)n * sizeof(double)));
                CHECK_CUDA(cudaMemcpy(buf.d_r, d_b, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice));
              }
            }
            double prev_rho = 0.0, best_rel = INFINITY;
            int stale = 0;
            std::string reason = "max_iters";
            for (int it = 0; it < 2000; ++it) {
              if (stage == 0) {
                d2f_k<<<bl, 256>>>(n, buf.d_r, buf.d_rf);
                CHECK_CUDA(cudaGetLastError());
                CHECK_AMGX(AMGX_vector_upload(vfb, n, 1, buf.d_rf));
                CHECK_AMGX(AMGX_vector_set_zero(vfx, n, 1));
                CHECK_AMGX(AMGX_solver_solve(sf, vfb, vfx));
                CHECK_AMGX(AMGX_vector_download(vfx, buf.d_zf));
                f2d_k<<<bl, 256>>>(n, buf.d_zf, buf.d_z);
                CHECK_CUDA(cudaGetLastError());
              } else {
                CHECK_AMGX(AMGX_vector_upload(vdb, n, 1, buf.d_r));
                CHECK_AMGX(AMGX_vector_set_zero(vdx, n, 1));
                CHECK_AMGX(AMGX_solver_solve(sd, vdb, vdx));
                CHECK_AMGX(AMGX_vector_download(vdx, buf.d_z));
              }
              double rho = buf.dot(buf.d_r, buf.d_z);
              if (!(rho > 0.0 && std::isfinite(rho))) { reason = "rz"; break; }
              double beta = (it == 0) ? 0.0 : (rho / prev_rho);
              if (it == 0) CHECK_CUDA(cudaMemcpy(buf.d_p, buf.d_z, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice));
              else { xpay_f64_k<<<bl, 256>>>(n, buf.d_z, beta, buf.d_p); CHECK_CUDA(cudaGetLastError()); }
              spmv_f64_k<<<bl, 256>>>(n, d_rp, d_ci, d_val, buf.d_p, buf.d_q);
              CHECK_CUDA(cudaGetLastError());
              double den = buf.dot(buf.d_p, buf.d_q);
              if (!(den > 0.0 && std::isfinite(den))) { reason = "curvature"; break; }
              double alpha = rho / den;
              axpy_f64_k<<<bl, 256>>>(n, alpha, buf.d_p, d_x);
              CHECK_CUDA(cudaGetLastError());
              axpy_f64_k<<<bl, 256>>>(n, -alpha, buf.d_q, buf.d_r);
              CHECK_CUDA(cudaGetLastError());
              iters++;
              double rel = std::sqrt(buf.dot(buf.d_r, buf.d_r)) / bnorm;
              prev_rho = rho;
              if (rel <= 8e-9) {
                spmv_f64_k<<<bl, 256>>>(n, d_rp, d_ci, d_val, d_x, buf.d_q);
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaMemcpy(buf.d_r, d_b, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice));
                axpy_f64_k<<<bl, 256>>>(n, -1.0, buf.d_q, buf.d_r);
                CHECK_CUDA(cudaGetLastError());
                double tr = std::sqrt(buf.dot(buf.d_r, buf.d_r)) / bnorm;
                if (std::isfinite(tr) && tr <= 1e-8) { reason = "ok"; break; }
                reason = "drift"; break;
              }
              if (rel < best_rel * 0.99) { best_rel = rel; stale = 0; }
              else if (++stale >= 100) { reason = "stagnation"; break; }
            }
            if (reason == "ok") break;
            if (stage == 0) trigger = reason;
          }

          CHECK_CUDA(cudaEventRecord(e3));
          CHECK_CUDA(cudaEventSynchronize(e3));
          float sol_ms = 0.0f;
          CHECK_CUDA(cudaEventElapsedTime(&sol_ms, e2, e3));

          CHECK_CUDA(cudaMemcpy(x_vec.data(), d_x, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
          double rel = cpu_true_rel(h, b_vec, x_vec);
          int pass = (std::isfinite(rel) && rel <= 1e-8) ? 1 : 0;
          all_pass &= (pass == 1);
          std::printf("%s,%s,%d,%s,%.5f,%.5f,%d,%d,%s,%.12e,%d\n",
                      name.c_str(), method.c_str(), t, phase ? "timed" : "warm",
                      rst_ms, sol_ms, iters, recovered, trigger.c_str(), rel, pass);
          std::fflush(stdout);
        }
      }
      cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_val0); cudaFree(d_val2);
      cudaFree(d_val0f); cudaFree(d_val2f); cudaFree(d_b); cudaFree(d_x);
      AMGX_solver_destroy(sf); AMGX_solver_destroy(sd);
      AMGX_vector_destroy(vfb); AMGX_vector_destroy(vfx);
      AMGX_vector_destroy(vdb); AMGX_vector_destroy(vdx);
      AMGX_matrix_destroy(Af); AMGX_matrix_destroy(Ad);
      AMGX_resources_destroy(rf); AMGX_resources_destroy(rd);
      AMGX_config_destroy(cf); AMGX_config_destroy(cd);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
    AMGX_finalize();
    return all_pass ? 0 : 2;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    AMGX_finalize();
    return 1;
  }
}
