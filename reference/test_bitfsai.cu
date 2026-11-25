#include "bitfsai.cuh"
#include "certified_damping.hpp"
#include "workloads.hpp"
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <unistd.h>
#include <vector>

static void require(bool cond, const char *msg) {
  if (!cond) throw std::runtime_error(std::string("CHECK FAILED: ") + msg);
}

static std::vector<double> host_spmv(const HostCSR &h, const std::vector<double> &x) {
  std::vector<double> y(h.n, 0.0);
  for (int i = 0; i < h.n; ++i) {
    double s = 0.0;
    for (int k = h.rp[i]; k < h.rp[i + 1]; ++k) s += h.val[k] * x[h.ci[k]];
    y[i] = s;
  }
  return y;
}

static double dot(const std::vector<double> &a, const std::vector<double> &b) {
  double s = 0.0;
  for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

static void test_2x2_counterexample() {
  HostCSR h;
  h.n = 2;
  h.rp = {0, 2, 4};
  h.ci = {0, 1, 0, 1};
  h.val = {1.0, 0.04, 0.04, 0.0017};
  h.nnz = 4;

  int *d_rp = nullptr, *d_ci = nullptr;
  double *d_val = nullptr;
  cudaMalloc(&d_rp, h.rp.size() * sizeof(int));
  cudaMalloc(&d_ci, h.ci.size() * sizeof(int));
  cudaMalloc(&d_val, h.val.size() * sizeof(double));
  cudaMemcpy(d_rp, h.rp.data(), h.rp.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ci, h.ci.data(), h.ci.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_val, h.val.data(), h.val.size() * sizeof(double), cudaMemcpyHostToDevice);

  bitfsai::Smoother sm_spl;
  sm_spl.setup(h.n, h.nnz, d_rp, d_ci, d_val, bitfsai::Mode::INT4_Split, true, 12);
  float gd[2], sc[2];
  cudaMemcpy(gd, sm_spl.d_diag, 2 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(sc, sm_spl.d_off_scale, 2 * sizeof(float), cudaMemcpyDeviceToHost);
  require(std::fabs(gd[0] - 1.0f) < 1e-5f && std::fabs(gd[1] - 100.0f) < 1e-3f,
          "Split-scale INT4 must recover exact diagonal [1, 100] on A_ex");

  bitfsai::Smoother sm_inc_raw;
  sm_inc_raw.setup(h.n, h.nnz, d_rp, d_ci, d_val, bitfsai::Mode::INT4_Inclusive, false, 12);
  cudaMemcpy(gd, sm_inc_raw.d_diag, 2 * sizeof(float), cudaMemcpyDeviceToHost);
  double diag_energy_raw = static_cast<double>(gd[1]) * static_cast<double>(gd[1]) * 0.0017;
  require(std::fabs(diag_energy_raw - 17.0) < 1e-2,
          "Unreprojected inclusive INT4 must reproduce (G_inc A G_inc^T)_{11} = 17");

  bitfsai::Smoother sm_inc_rep;
  sm_inc_rep.setup(h.n, h.nnz, d_rp, d_ci, d_val, bitfsai::Mode::INT4_Inclusive, true, 12);
  cudaMemcpy(gd, sm_inc_rep.d_diag, 2 * sizeof(float), cudaMemcpyDeviceToHost);
  double diag_energy_rep = static_cast<double>(gd[1]) * static_cast<double>(gd[1]) * 0.0017;
  require(std::fabs(diag_energy_rep - 1.0) < 1e-4,
          "Theorem 3 re-projection must enforce exact unit diagonal energy (G_q A G_q^T)_{11} = 1");

  cudaFree(d_rp);
  cudaFree(d_ci);
  cudaFree(d_val);
  std::printf("[PASS] 2x2 counterexample (Theorems 2 & 3): raw diag energy=%.2f -> reprojected=%.6f\n",
              diag_energy_raw, diag_energy_rep);
}

static void test_theorem4_transpose_equivalence() {
  HostCSR h = build_3d_structured(6, 0);
  int *d_rp = nullptr, *d_ci = nullptr;
  double *d_val = nullptr, *d_r = nullptr, *d_x = nullptr;
  cudaMalloc(&d_rp, h.rp.size() * sizeof(int));
  cudaMalloc(&d_ci, h.ci.size() * sizeof(int));
  cudaMalloc(&d_val, h.val.size() * sizeof(double));
  cudaMalloc(&d_r, h.n * sizeof(double));
  cudaMalloc(&d_x, h.n * sizeof(double));
  cudaMemcpy(d_rp, h.rp.data(), h.rp.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ci, h.ci.data(), h.ci.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_val, h.val.data(), h.val.size() * sizeof(double), cudaMemcpyHostToDevice);

  std::mt19937 rng(42);
  std::normal_distribution<double> nd(0.0, 1.0);
  std::vector<double> r(h.n), x(h.n);
  for (auto &v : r) v = nd(rng);
  cudaMemcpy(d_r, r.data(), h.n * sizeof(double), cudaMemcpyHostToDevice);

  std::vector<double> r2(h.n);
  for (auto &v : r2) v = nd(rng);
  double *d_r2 = nullptr, *d_x2 = nullptr;
  cudaMalloc(&d_r2, h.n * sizeof(double));
  cudaMalloc(&d_x2, h.n * sizeof(double));
  cudaMemcpy(d_r2, r2.data(), h.n * sizeof(double), cudaMemcpyHostToDevice);

  float *d_rf = nullptr, *d_xf = nullptr, *d_r2f = nullptr, *d_x2f = nullptr;
  cudaMalloc(&d_rf, h.n * sizeof(float));
  cudaMalloc(&d_xf, h.n * sizeof(float));
  cudaMalloc(&d_r2f, h.n * sizeof(float));
  cudaMalloc(&d_x2f, h.n * sizeof(float));
  std::vector<float> rf(h.n), r2f(h.n), xf(h.n), x2f(h.n);
  for (int i = 0; i < h.n; ++i) {
    rf[i] = static_cast<float>(r[i]);
    r2f[i] = static_cast<float>(r2[i]);
  }
  cudaMemcpy(d_rf, rf.data(), h.n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_r2f, r2f.data(), h.n * sizeof(float), cudaMemcpyHostToDevice);

  for (bitfsai::Mode m : {bitfsai::Mode::INT4_Split, bitfsai::Mode::INT4_Inclusive,
                          bitfsai::Mode::FP4_E2M1_Split,
                          bitfsai::Mode::FP16_Split, bitfsai::Mode::FP64_Exact}) {
    bitfsai::Smoother sm;
    sm.setup(h.n, h.nnz, d_rp, d_ci, d_val, m, true, 12);
    sm.apply(d_rp, d_ci, d_r, d_x, 1.0, false);
    sm.apply(d_rp, d_ci, d_r2, d_x2, 1.0, false);
    sm.apply(d_rp, d_ci, d_rf, d_xf, 1.0, false);
    sm.apply(d_rp, d_ci, d_r2f, d_x2f, 1.0, false);
    cudaDeviceSynchronize();
    cudaMemcpy(x.data(), d_x, h.n * sizeof(double), cudaMemcpyDeviceToHost);
    std::vector<double> x2(h.n);
    cudaMemcpy(x2.data(), d_x2, h.n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(xf.data(), d_xf, h.n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(x2f.data(), d_x2f, h.n * sizeof(float), cudaMemcpyDeviceToHost);

    double rMr = dot(r, x);
    double sym1 = dot(r2, x), sym2 = dot(r, x2);
    double rel_sym = std::fabs(sym1 - sym2) / (std::fabs(sym1) + std::fabs(sym2) + 1e-30);
    require(rMr > 0.0, "Gram smoother M = G_q^T G_q must be strictly positive definite");
    require(rel_sym < 1e-6, "Theorem 4 scale-factored zero-index gather must be strictly symmetric (double)");

    double rMr_f = 0.0, sym1_f = 0.0, sym2_f = 0.0;
    for (int i = 0; i < h.n; ++i) {
      rMr_f += double(rf[i]) * double(xf[i]);
      sym1_f += double(r2f[i]) * double(xf[i]);
      sym2_f += double(rf[i]) * double(x2f[i]);
    }
    double rel_sym_f = std::fabs(sym1_f - sym2_f) / (std::fabs(sym1_f) + std::fabs(sym2_f) + 1e-30);
    require(rMr_f > 0.0, "float-vector Gram smoother must be strictly positive definite");
    require(rel_sym_f < 1e-5, "Theorem 4 scale-factored zero-index gather must be strictly symmetric (float)");
  }

  cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_val);
  cudaFree(d_r); cudaFree(d_x); cudaFree(d_r2); cudaFree(d_x2);
  cudaFree(d_rf); cudaFree(d_xf); cudaFree(d_r2f); cudaFree(d_x2f);
  std::printf("[PASS] Theorem 4 scale-factored zero-index symmetry & SPD verified across 5 encodings (double & float)\n");
}

static void test_pcg_convergence() {
  HostCSR h = build_3d_structured(16, 0);
  int *d_rp = nullptr, *d_ci = nullptr;
  double *d_val = nullptr, *d_r = nullptr, *d_z = nullptr;
  cudaMalloc(&d_rp, h.rp.size() * sizeof(int));
  cudaMalloc(&d_ci, h.ci.size() * sizeof(int));
  cudaMalloc(&d_val, h.val.size() * sizeof(double));
  cudaMalloc(&d_r, h.n * sizeof(double));
  cudaMalloc(&d_z, h.n * sizeof(double));
  cudaMemcpy(d_rp, h.rp.data(), h.rp.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ci, h.ci.data(), h.ci.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_val, h.val.data(), h.val.size() * sizeof(double), cudaMemcpyHostToDevice);

  std::mt19937 rng(12345);
  std::normal_distribution<double> nd(0.0, 1.0);
  std::vector<double> b(h.n);
  for (auto &v : b) v = nd(rng);
  double bnorm = std::sqrt(dot(b, b));

  const char *names[] = {"FP4_E2M1_spl", "FP16_spl", "FP64_exact", "INT4_inc", "INT4_spl"};
  for (bitfsai::Mode m : {bitfsai::Mode::INT4_Split, bitfsai::Mode::FP4_E2M1_Split,
                          bitfsai::Mode::FP16_Split, bitfsai::Mode::FP64_Exact}) {
    bitfsai::Smoother sm;
    sm.setup(h.n, h.nnz, d_rp, d_ci, d_val, m, true, 12);

    std::vector<double> x(h.n, 0.0), r = b, z(h.n), p(h.n);
    auto apply_M = [&](const std::vector<double> &in, std::vector<double> &out) {
      cudaMemcpy(d_r, in.data(), h.n * sizeof(double), cudaMemcpyHostToDevice);
      sm.apply(d_rp, d_ci, d_r, d_z, 1.0f, false);
      cudaMemcpy(out.data(), d_z, h.n * sizeof(double), cudaMemcpyDeviceToHost);
    };

    apply_M(r, z);
    double rz = dot(r, z);
    require(rz > 0.0, "Initial r^T z must be positive");
    p = z;
    int it = 0;
    for (it = 1; it <= 2500; ++it) {
      auto Ap = host_spmv(h, p);
      double pAp = dot(p, Ap);
      double alpha = rz / pAp;
      for (int i = 0; i < h.n; ++i) {
        x[i] += alpha * p[i];
        r[i] -= alpha * Ap[i];
      }
      if (std::sqrt(dot(r, r)) / bnorm <= 5e-9) break;
      apply_M(r, z);
      double rz_new = dot(r, z);
      require(rz_new > 0.0, "PCG r^T z must stay positive");
      double beta = rz_new / rz;
      for (int i = 0; i < h.n; ++i) p[i] = z[i] + beta * p[i];
      rz = rz_new;
    }
    auto Ax = host_spmv(h, x);
    double r2 = 0.0;
    for (int i = 0; i < h.n; ++i) {
      double d = b[i] - Ax[i];
      r2 += d * d;
    }
    double true_rel = std::sqrt(r2) / bnorm;
    require(true_rel <= 1e-8, "Host FP64 true relative residual must be <= 1e-8");
    std::printf("[PASS] PCG + %-13s (16^3 aniso+10^4): %3d iters | true_rel=%.2e | factor=%zu B\n",
                names[static_cast<int>(m)], it, true_rel, sm.factor_bytes());
  }

  cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_val);
  cudaFree(d_r); cudaFree(d_z);
}

template <typename T> struct Managed {
  T *data = nullptr;

  explicit Managed(size_t count) {
    bitfsai::detail::check_cuda(cudaMallocManaged(&data, count * sizeof(T)), "test allocation");
  }
  ~Managed() { cudaFree(data); }
  Managed(const Managed &) = delete;
  Managed &operator=(const Managed &) = delete;
};

static void test_scaled_patches() {
  using bitfsai::Mode;
  Managed<int> rp(3), ci(4);
  Managed<double> a(4), r(2), x(2);
  rp.data[0] = 0; rp.data[1] = 2; rp.data[2] = 4;
  ci.data[0] = ci.data[2] = 0;
  ci.data[1] = ci.data[3] = 1;
  bitfsai::Smoother sm;
  auto matrix = [&](double a00, double a10, double a11) {
    a.data[0] = a00; a.data[1] = a.data[2] = a10; a.data[3] = a11;
  };
  auto apply = [&](Mode mode, bool project = true) {
    sm.setup(2, 4, rp.data, ci.data, a.data, mode, project);
    sm.apply(rp.data, ci.data, r.data, x.data);
    bitfsai::detail::check_cuda(cudaDeviceSynchronize(), "scaled patch apply");
  };
  for (Mode mode : {Mode::FP64_Exact, Mode::INT4_Split}) {
    matrix(1e-28, 0, 1e-28);
    r.data[0] = r.data[1] = 1;
    apply(mode);
    require(std::abs(x.data[0] / 1e28 - 1) < 1e-6, "positive tiny Cholesky pivot");
  }
  for (Mode mode : {Mode::INT4_Split, Mode::FP4_E2M1_Split, Mode::FP16_Split}) {
    matrix(0x1p42, 0x1p21, 1 + 0x1p-10);
    r.data[0] = 0; r.data[1] = 1;
    for (bool project : {false, true}) {
      apply(mode, project);
      require(std::abs(x.data[0] + 0x1p-11) < 1e-9 && std::abs(x.data[1] - 1024) < .001,
              "small off-diagonal scale must preserve the exact inverse column");
    }
    for (double diagonal : {1e-20, 1e20}) {
      matrix(diagonal, 0, diagonal);
      double rhs = diagonal < 1 ? 1e-48 : 1e40;
      r.data[0] = rhs; r.data[1] = -rhs;
      apply(mode);
      double expected = rhs / diagonal;
      require(std::abs(x.data[0] / expected - 1) < 1e-6 &&
              std::abs(x.data[1] / expected + 1) < 1e-6, "FP64 residual range");
    }
    matrix(1e100, 0, 1e100);
    bool rejected = false;
    try {
      sm.setup(2, 4, rp.data, ci.data, a.data, mode);
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    require(rejected && sm.n == 0, "reject and clear an unrepresentable FP32 factor");
  }

  matrix(1.0, 1e-30, 1.0);
  sm.setup(2, 4, rp.data, ci.data, a.data);
  uint8_t small_codes[2];
  cudaMemcpy(small_codes, sm.d_q_packed, sizeof(small_codes), cudaMemcpyDeviceToHost);
  require((small_codes[1] & 15) != 8, "retain tiny nonzero INT4 coupling");
  Managed<float> small_rhs(2), small_out(2);
  small_rhs.data[0] = 0; small_rhs.data[1] = 1e-20f;
  sm.apply(rp.data, ci.data, small_rhs.data, small_out.data);
  bitfsai::detail::check_cuda(cudaDeviceSynchronize(), "small-scale FP32 apply");
  require(std::abs(small_out.data[1] / small_rhs.data[1] - 1) < 1e-6f,
          "a retained tiny coupling must preserve the FP32 diagonal update");

  const double k = 7000;
  matrix(k*k + 1, k*(k+1) + 1, (k+1)*(k+1) + 1);
  sm.setup(2, 4, rp.data, ci.data, a.data);
  require(sm.used_diagonal_fallback, "report a diagonal fallback for the ill-conditioned patch");
  float gd[2], sc[2];
  uint8_t packed[2];
  cudaMemcpy(gd, sm.d_diag, sizeof(gd), cudaMemcpyDeviceToHost);
  cudaMemcpy(sc, sm.d_off_scale, sizeof(sc), cudaMemcpyDeviceToHost);
  cudaMemcpy(packed, sm.d_q_packed, sizeof(packed), cudaMemcpyDeviceToHost);
  double off = double(sc[1]) * (float(int(packed[1] & 15) - 8) * (1.0f / 7.0f));
  double u0 = k * off + (k + 1) * gd[1], u1 = off + gd[1];
  require(std::abs(u0 * u0 + u1 * u1 - 1) < 1e-4, "stored row must pass the energy check");
  std::puts("[PASS] scaled pivots, small couplings, stored energy, and FP64 residuals");
}

template <typename T> static void check_concurrent_apply(bitfsai::Smoother &sm, int *rp, int *ci) {
  const int n = sm.n;
  Managed<T> a(n), b(n), x(n), y(n);
  for (int i = 0; i < n; ++i) {
    a.data[i] = T(1 + i % 7);
    b.data[i] = T(-1 - i % 11);
    x.data[i] = y.data[i] = T(0);
  }
  cudaStream_t first, second;
  bitfsai::detail::check_cuda(cudaStreamCreateWithFlags(&first, cudaStreamNonBlocking), "first stream");
  bitfsai::detail::check_cuda(cudaStreamCreateWithFlags(&second, cudaStreamNonBlocking), "second stream");
  for (int j = 0; j < 16; ++j) {
    sm.apply(rp, ci, a.data, x.data, 0.5f, true, first);
    sm.apply(rp, ci, b.data, y.data, 0.25f, true, second);
  }
  bitfsai::detail::check_cuda(cudaStreamSynchronize(first), "first apply");
  bitfsai::detail::check_cuda(cudaStreamSynchronize(second), "second apply");
  for (int i = 0; i < n; ++i)
    require(x.data[i] == 8 * a.data[i] && y.data[i] == 4 * b.data[i], "cross-stream Gram output");
  void *workspace = nullptr;
  bitfsai::detail::check_cuda(cudaMalloc(&workspace, sm.workspace_bytes<T>()), "reusable workspace");
  sm.apply(rp, ci, a.data, x.data, 1.0f, false, first, workspace);
  bitfsai::detail::check_cuda(cudaStreamSynchronize(first), "reused workspace apply");
  for (int i = 0; i < n; ++i) require(x.data[i] == a.data[i], "caller-owned workspace");
  cudaFree(workspace);
  cudaStreamDestroy(first);
  cudaStreamDestroy(second);
}

static void test_concurrent_apply() {
  constexpr int n = 32768;
  Managed<int> rp(n + 1), ci(n);
  Managed<double> a(n);
  for (int i = 0; i < n; ++i) {
    rp.data[i] = ci.data[i] = i;
    a.data[i] = 1;
  }
  rp.data[n] = n;
  bitfsai::Smoother sm;
  for (auto mode : {bitfsai::Mode::INT4_Split, bitfsai::Mode::FP64_Exact}) {
    sm.setup(n, n, rp.data, ci.data, a.data, mode);
    check_concurrent_apply<float>(sm, rp.data, ci.data);
    check_concurrent_apply<double>(sm, rp.data, ci.data);
  }
  std::puts("[PASS] concurrent damped Gram applies on two streams");
}

static void test_float_factored_exponent_extremes() {
  using bitfsai::Mode;
  // 1) Normal-scale float-vector test across factored modes vs FP64 vector reference
  {
    HostCSR h = build_3d_structured(6, 0);
    Managed<int> rp(h.rp.size()), ci(h.ci.size());
    Managed<double> a(h.val.size()), rd(h.n), xd(h.n);
    Managed<float> rf(h.n), xf(h.n);
    for (size_t i = 0; i < h.rp.size(); ++i) rp.data[i] = h.rp[i];
    for (size_t i = 0; i < h.ci.size(); ++i) ci.data[i] = h.ci[i];
    for (size_t i = 0; i < h.val.size(); ++i) a.data[i] = h.val[i];
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int i = 0; i < h.n; ++i) {
      rf.data[i] = nd(rng);
      rd.data[i] = double(rf.data[i]);
    }
    for (Mode m : {Mode::INT4_Split, Mode::FP4_E2M1_Split, Mode::FP16_Split}) {
      bitfsai::Smoother sm;
      sm.setup(h.n, h.nnz, rp.data, ci.data, a.data, m, true, 12);
      require(!sm.wide_apply, "6^3 structured mesh must exercise the factored FP32 kernel");
      sm.apply(rp.data, ci.data, rf.data, xf.data, 0.72f, false);
      sm.apply(rp.data, ci.data, rd.data, xd.data, 0.72f, false);
      bitfsai::detail::check_cuda(cudaDeviceSynchronize(), "normal float factored apply");
      double diff2 = 0.0, ref2 = 0.0;
      for (int i = 0; i < h.n; ++i) {
        double d = double(xf.data[i]) - xd.data[i];
        diff2 += d * d;
        ref2 += xd.data[i] * xd.data[i];
      }
      require(std::sqrt(diff2 / ref2) < 1e-5,
              "normal-scale float factored apply must match double-accumulator reference");
    }
  }

  // 2) Upper exponent extreme: r = 2^127 with scale = 0.25, diag = 0.25, two off-diags = 1.0
  //    G = [[0.25, 0, 0], [0, 0.25, 0], [0.25, 0.25, 0.25]]  =>  A = (G^T G)^{-1} = [[16,0,-16],[0,16,-16],[-16,-16,48]]
  {
    Managed<int> rp(4), ci(7);
    Managed<double> a(7);
    Managed<float> r(3), x(3);
    int h_rp[4] = {0, 2, 4, 7};
    int h_ci[7] = {0, 2, 1, 2, 0, 1, 2};
    double h_a[7] = {16.0, -16.0, 16.0, -16.0, -16.0, -16.0, 48.0};
    for (int i = 0; i < 4; ++i) rp.data[i] = h_rp[i];
    for (int i = 0; i < 7; ++i) { ci.data[i] = h_ci[i]; a.data[i] = h_a[i]; }
    const float r_hi = std::ldexp(1.0f, 127);
    r.data[0] = r.data[1] = r.data[2] = r_hi;
    // Exact G*r = [2^125, 2^125, 3*2^125]; G^T(G*r) = [2^125, 2^125, 3*2^123]
    const float exp0 = std::ldexp(1.0f, 125);
    const float exp2 = 3.0f * std::ldexp(1.0f, 123);

    for (Mode m : {Mode::INT4_Split, Mode::FP4_E2M1_Split, Mode::FP16_Split}) {
      bitfsai::Smoother sm;
      sm.setup(3, 7, rp.data, ci.data, a.data, m, true, 12);
      require(!sm.wide_apply, "scale=0.25 patch must stay on factored FP32 path");
      float sc[3];
      cudaMemcpy(sc, sm.d_off_scale, sizeof(sc), cudaMemcpyDeviceToHost);
      require(std::fabs(sc[2] - 0.25f) < 1e-6f, "row 2 must have scale=0.25");
      sm.apply(rp.data, ci.data, r.data, x.data, 1.0f, false);
      bitfsai::detail::check_cuda(cudaDeviceSynchronize(), "r=2^127 apply");
      require(std::isfinite(x.data[0]) && std::isfinite(x.data[1]) && std::isfinite(x.data[2]),
              "r=2^127 with scale=0.25 must not overflow in Pass 1");
      require(std::fabs(x.data[0] - exp0) / exp0 < 1e-6f &&
              std::fabs(x.data[1] - exp0) / exp0 < 1e-6f &&
              std::fabs(x.data[2] - exp2) / exp2 < 1e-6f,
              "r=2^127 with scale=0.25 must match exact Gram product");
    }
  }

  // 3) Lower exponent extreme: r = 2^-126 with scale = 4.0, diag = 4.0, off-diags in {1.0, 1/7}
  //    G = [[4, 0, 0], [0, 4, 0], [4, 4/7, 4]]  =>  L = G^{-1} = [[1/4,0,0],[0,1/4,0],[-1/4,-1/28,1/4]]
  {
    Managed<int> rp(4), ci(7);
    Managed<double> a(7), rd(3), xd(3);
    Managed<float> rf(3), xf(3);
    double L[3][3] = {{0.25, 0.0, 0.0}, {0.0, 0.25, 0.0}, {-0.25, -1.0 / 28.0, 0.25}};
    int h_rp[4] = {0, 2, 4, 7};
    int h_ci[7] = {0, 2, 1, 2, 0, 1, 2};
    auto entryA = [&](int r0, int c0) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k) s += L[r0][k] * L[c0][k];
      return s;
    };
    double h_a[7] = {entryA(0, 0), entryA(0, 2), entryA(1, 1), entryA(1, 2),
                     entryA(2, 0), entryA(2, 1), entryA(2, 2)};
    for (int i = 0; i < 4; ++i) rp.data[i] = h_rp[i];
    for (int i = 0; i < 7; ++i) { ci.data[i] = h_ci[i]; a.data[i] = h_a[i]; }
    const float r_lo = std::ldexp(1.0f, -126);
    rf.data[0] = rf.data[1] = rf.data[2] = r_lo;
    rd.data[0] = rd.data[1] = rd.data[2] = double(r_lo);

    for (Mode m : {Mode::INT4_Split, Mode::FP16_Split}) {
      bitfsai::Smoother sm;
      sm.setup(3, 7, rp.data, ci.data, a.data, m, true, 12);
      require(!sm.wide_apply, "scale=4.0 patch must stay on factored FP32 path");
      float sc[3];
      cudaMemcpy(sc, sm.d_off_scale, sizeof(sc), cudaMemcpyDeviceToHost);
      require(std::fabs(sc[2] - 4.0f) < 1e-3f, "row 2 must have scale=4.0");
      sm.apply(rp.data, ci.data, rf.data, xf.data, 1.0f, false);
      sm.apply(rp.data, ci.data, rd.data, xd.data, 1.0f, false);
      bitfsai::detail::check_cuda(cudaDeviceSynchronize(), "r=2^-126 apply");
      for (int i = 0; i < 3; ++i) {
        require(std::isfinite(xf.data[i]) && xf.data[i] > 0.0f,
                "r=2^-126 with scale=4.0 must stay finite and positive");
        require(std::fabs(double(xf.data[i]) - xd.data[i]) / xd.data[i] < 1e-6,
                "r=2^-126 with scale=4.0 must avoid subnormal precision loss");
      }
    }

    // Cross-regime check: huge r = [0, 0, 2^125] with scale = 4.0 (y_2 = 4 * 4 * 2^125 = 2^129 overflows y_2!)
    // With damp = 0.125, true update is 0.125 * G^T * [0, 0, 2^127] = [2^126, (1/7)*2^126, 2^126], all < FLT_MAX!
    rf.data[0] = 0.0f; rf.data[1] = 0.0f; rf.data[2] = std::ldexp(1.0f, 125);
    rd.data[0] = 0.0;  rd.data[1] = 0.0;  rd.data[2] = double(rf.data[2]);
    bitfsai::Smoother sm;
    sm.setup(3, 7, rp.data, ci.data, a.data, Mode::INT4_Split, true, 12);
    const float init_x = std::ldexp(1.0f, 125);
    for (int i = 0; i < 3; ++i) {
      xf.data[i] = init_x;
      xd.data[i] = double(init_x);
    }
    sm.apply(rp.data, ci.data, rf.data, xf.data, 0.125f, true);
    sm.apply(rp.data, ci.data, rd.data, xd.data, 0.125f, true);
    bitfsai::detail::check_cuda(cudaDeviceSynchronize(), "cross-regime fallback apply");
    for (int i = 0; i < 3; ++i) {
      require(std::isfinite(xf.data[i]) &&
                  std::fabs(double(xf.data[i]) - xd.data[i]) / xd.data[i] < 1e-6,
              "overflowing y_i = scale * t_i must fall back cleanly and preserve accumulate");
    }
  }
  std::puts("[PASS] float-vector factored kernel verified at normal scales and both FP32 exponent extremes");
}

static void test_csr_symmetry_and_mtx_guards() {
  using bitfsai::Mode;
  // 1) Upper-only asymmetric edge: row 0 has {0, 1}, row 1 has {1}
  {
    Managed<int> rp(3), ci(3);
    Managed<double> a(3);
    rp.data[0] = 0; rp.data[1] = 2; rp.data[2] = 3;
    ci.data[0] = 0; ci.data[1] = 1; ci.data[2] = 1;
    a.data[0] = 4.0; a.data[1] = -1.0; a.data[2] = 4.0;
    for (Mode m : {Mode::INT4_Split, Mode::FP16_Split, Mode::FP64_Exact}) {
      bitfsai::Smoother sm;
      bool caught = false;
      try {
        sm.setup(2, 3, rp.data, ci.data, a.data, m, true, 12);
      } catch (const std::runtime_error &e) {
        caught = std::string(e.what()).find("sparsity pattern is not symmetric") != std::string::npos;
      }
      require(caught, "upper-only CSR edge must trigger kSetupNonsymmetricPattern");
    }
  }

  // 2) Lower edge dropped by patch_cap = 2 (max_low = 1):
  //    row 0 has {0} (missing mirror (0,2)!), row 1 has {1, 2}, row 2 has {0 (weak, dropped), 1 (strong, kept), 2}
  {
    Managed<int> rp(4), ci(6);
    Managed<double> a(6);
    int h_rp[4] = {0, 1, 3, 6};
    int h_ci[6] = {0, 1, 2, 0, 1, 2};
    double h_a[6] = {10.0, 10.0, -2.0, -0.1, -2.0, 10.0};
    for (int i = 0; i < 4; ++i) rp.data[i] = h_rp[i];
    for (int i = 0; i < 6; ++i) { ci.data[i] = h_ci[i]; a.data[i] = h_a[i]; }
    for (Mode m : {Mode::INT4_Split, Mode::FP16_Split, Mode::FP64_Exact}) {
      bitfsai::Smoother sm;
      bool caught = false;
      try {
        sm.setup(3, 6, rp.data, ci.data, a.data, m, true, 2);
      } catch (const std::runtime_error &e) {
        caught = std::string(e.what()).find("sparsity pattern is not symmetric") != std::string::npos;
      }
      require(caught, "dropped lower CSR edge must still trigger kSetupNonsymmetricPattern");
    }
  }

  // 3) Matrix Market loader strict validation checks (PID-unique temp file)
  const std::string tmp_path = "/tmp/bitfsai_mtx_guard_test_" + std::to_string(getpid()) + ".mtx";
  auto expect_mtx_fail = [&](const std::string &contents, const char *why) {
    {
      std::ofstream out(tmp_path);
      out << contents;
    }
    bool failed = false;
    try {
      (void)load_mtx_csr(tmp_path);
    } catch (const std::runtime_error &) {
      failed = true;
    }
    std::filesystem::remove(tmp_path);
    require(failed, why);
  };
  expect_mtx_fail("%%MatrixMarket matrix coordinate real skew-symmetric\n2 2 2\n1 1 2.0\n2 2 2.0\n",
                  "reject skew-symmetric Matrix Market banner");
  expect_mtx_fail("%%MatrixMarket matrix coordinate pattern symmetric\n2 2 2\n1 1\n2 2\n",
                  "reject pattern Matrix Market banner");
  expect_mtx_fail("%%MatrixMarket matrix array real symmetric\n2 2\n1.0\n2.0\n",
                  "reject array Matrix Market banner");
  expect_mtx_fail("%%MatrixMarket matrix coordinate complex hermitian\n2 2 2\n1 1 1 0\n2 2 1 0\n",
                  "reject complex/hermitian Matrix Market banner");
  expect_mtx_fail("%%MatrixMarket matrix coordinate real symmetric\n2 3 2\n1 1 1.0\n2 2 1.0\n",
                  "reject rectangular Matrix Market dimensions");
  expect_mtx_fail("%%MatrixMarket matrix coordinate real symmetric\n2 2 3\n1 1 1.0\n2 2 1.0\n",
                  "reject truncated Matrix Market entry count");
  expect_mtx_fail("%%MatrixMarket matrix coordinate real symmetric\n2 2 2\n1 1 1.0\n3 2 1.0\n",
                  "reject out-of-bounds Matrix Market coordinates");
  expect_mtx_fail("%%MatrixMarket matrix coordinate real symmetric\n2 2 2\n1 1 nan\n2 2 1.0\n",
                  "reject NaN Matrix Market values");
  expect_mtx_fail("%%MatrixMarket matrix coordinate real symmetric\n2 2 3\n1 1 2.0\n1 2 -0.5\n2 2 2.0\n",
                  "reject upper-triangle entry in symmetric Matrix Market file");
  expect_mtx_fail("%%MatrixMarket matrix coordinate real general\n2 2 3\n1 1 2.0\n1 2 -1.0\n2 2 2.0\n",
                  "reject structurally asymmetric general Matrix Market input");
  expect_mtx_fail("%%MatrixMarket matrix coordinate real general\n2 2 4\n1 1 2.0\n1 2 -1.0\n2 1 1.0\n2 2 2.0\n",
                  "reject numerically asymmetric general Matrix Market input [[2,-1],[+1,2]]");
  expect_mtx_fail("%%MatrixMarket matrix coordinate real symmetric\n2 2 1\n1 1 2.0\n",
                  "reject Matrix Market input missing positive diagonal");
  std::puts("[PASS] full CSR symmetry check (upper-only & dropped-lower) and Matrix Market guards verified");
}

static void test_malformed_csr_validation() {
  using bitfsai::Mode;
  auto expect_setup_fail = [](int n, int nnz, const std::vector<int> &h_rp,
                              const std::vector<int> &h_ci, const std::vector<double> &h_val,
                              const char *why) {
    Managed<int> rp(std::max<size_t>(1, h_rp.size()));
    Managed<int> ci(std::max<size_t>(1, h_ci.size()));
    Managed<double> val(std::max<size_t>(1, h_val.size()));
    for (size_t i = 0; i < h_rp.size(); ++i) rp.data[i] = h_rp[i];
    for (size_t i = 0; i < h_ci.size(); ++i) ci.data[i] = h_ci[i];
    for (size_t i = 0; i < h_val.size(); ++i) val.data[i] = h_val[i];
    bitfsai::Smoother sm;
    bool threw = false;
    try {
      sm.setup(n, nnz, rp.data, ci.data, val.data, Mode::INT4_Split, true, 12);
    } catch (const std::runtime_error &) {
      threw = true;
    }
    require(threw && sm.n == 0, why);
    require(cudaGetLastError() == cudaSuccess, "CUDA context must stay clean after malformed CSR rejection");
  };

  expect_setup_fail(0, 0, {0}, {}, {}, "reject n <= 0");
  expect_setup_fail(2, -1, {0, 1, 2}, {0, 1}, {1.0, 1.0}, "reject nnz < 0");
  expect_setup_fail(2, 2, {1, 1, 2}, {0, 1}, {1.0, 1.0}, "reject rp[0] != 0");
  expect_setup_fail(2, 2, {0, 2, 1}, {0, 1}, {1.0, 1.0}, "reject non-monotonic rp");
  expect_setup_fail(2, 3, {0, 1, 2}, {0, 1, 1}, {1.0, 1.0, 1.0}, "reject rp[n] != nnz");
  expect_setup_fail(2, 2, {0, 1, 2}, {0, 2}, {1.0, 1.0}, "reject out-of-range column ci >= n");
  expect_setup_fail(2, 2, {0, 1, 2}, {-1, 1}, {1.0, 1.0}, "reject negative column ci < 0");
  expect_setup_fail(2, 3, {0, 2, 3}, {0, 0, 1}, {2.0, 2.0, 2.0}, "reject duplicate column in sorted row");
  expect_setup_fail(2, 4, {0, 3, 4}, {1, 0, 1, 1}, {-0.1, 2.0, -0.1, 2.0}, "reject duplicate column in unsorted row");

  // Confirm subsequent valid setup + apply succeeds cleanly on the same CUDA context
  Managed<int> rp(3), ci(4);
  Managed<double> a(4), r(2), x(2);
  rp.data[0] = 0; rp.data[1] = 2; rp.data[2] = 4;
  ci.data[0] = 0; ci.data[1] = 1; ci.data[2] = 0; ci.data[3] = 1;
  a.data[0] = 2.0; a.data[1] = -0.5; a.data[2] = -0.5; a.data[3] = 2.0;
  r.data[0] = 1.0; r.data[1] = 2.0;
  bitfsai::Smoother sm;
  sm.setup(2, 4, rp.data, ci.data, a.data, Mode::INT4_Split, true, 12);
  sm.apply(rp.data, ci.data, r.data, x.data, 1.0, false);
  bitfsai::detail::check_cuda(cudaDeviceSynchronize(), "post-rejection valid apply");
  require(std::isfinite(x.data[0]) && std::isfinite(x.data[1]) && x.data[0] > 0.0 && x.data[1] > 0.0,
          "valid setup + apply must succeed after malformed CSR rejections");
  std::puts("[PASS] malformed CSR validation (offsets, bounds, duplicates) and CUDA context reuse verified");
}

static void test_pass2_overflow_and_f64_damping() {
  using bitfsai::Mode;
  // 1) Pass 2 overflow before damping:
  //    G = [[1, 0], [0.25, 2]] => L = G^{-1} = [[1, 0], [-0.125, 0.5]],
  //    A = L L^T = [[1, -0.125], [-0.125, 0.265625]].
  //    r = [0, 2^126], damp = 0.5 => G r = [0, 2^127], G^T G r = [2^125, 2^128] (overflows float!),
  //    0.5 * G^T G r = [2^124, 2^127] (finite in float!).
  {
    Managed<int> rp(3), ci(4);
    Managed<double> a(4);
    Managed<float> r(2), x(2);
    rp.data[0] = 0; rp.data[1] = 2; rp.data[2] = 4;
    ci.data[0] = 0; ci.data[1] = 1; ci.data[2] = 0; ci.data[3] = 1;
    a.data[0] = 1.0; a.data[1] = -0.125; a.data[2] = -0.125; a.data[3] = 0.265625;
    r.data[0] = 0.0f;
    r.data[1] = std::ldexp(1.0f, 126);
    const float exp0 = std::ldexp(1.0f, 124);
    const float exp1 = std::ldexp(1.0f, 127);

    for (Mode m : {Mode::INT4_Split, Mode::FP4_E2M1_Split, Mode::FP16_Split}) {
      bitfsai::Smoother sm;
      sm.setup(2, 4, rp.data, ci.data, a.data, m, true, 12);
      require(!sm.wide_apply, "Pass 2 overflow witness must run on factored FP32 path");

      // accumulate = false
      x.data[0] = 0.0f; x.data[1] = 0.0f;
      sm.apply(rp.data, ci.data, r.data, x.data, 0.5, false);
      bitfsai::detail::check_cuda(cudaDeviceSynchronize(), "Pass 2 overflow accumulate=false");
      require(std::isfinite(x.data[0]) && std::isfinite(x.data[1]),
              "Pass 2 overflow before damping must produce finite output (accumulate=false)");
      require(std::fabs(x.data[0] - exp0) / exp0 < 1e-5f &&
                  std::fabs(x.data[1] - exp1) / exp1 < 1e-5f,
              "Pass 2 overflow witness must match exact [2^124, 2^127] (accumulate=false)");

      // accumulate = true (starting from [2^124, -2^126] -> [2^125, 2^126])
      x.data[0] = std::ldexp(1.0f, 124);
      x.data[1] = -std::ldexp(1.0f, 126);
      sm.apply(rp.data, ci.data, r.data, x.data, 0.5, true);
      bitfsai::detail::check_cuda(cudaDeviceSynchronize(), "Pass 2 overflow accumulate=true");
      const float exp_acc0 = std::ldexp(1.0f, 125);
      const float exp_acc1 = std::ldexp(1.0f, 126);
      require(std::isfinite(x.data[0]) && std::isfinite(x.data[1]),
              "Pass 2 overflow before damping must produce finite output (accumulate=true)");
      require(std::fabs(x.data[0] - exp_acc0) / exp_acc0 < 1e-5f &&
                  std::fabs(x.data[1] - exp_acc1) / exp_acc1 < 1e-5f,
              "Pass 2 overflow witness must preserve accumulate=true exact sum");
    }
  }

  // 1b) Binary64 Gram apply pre-damping overflow witness:
  //     A = [1/4], G = [2], r = 2^1022, damp = 0.5 => G r = 2^1023, damp * G^T G r = 2^1023.
  {
    Managed<int> rp(2), ci(1);
    Managed<double> a(1), r(1), x(1);
    rp.data[0] = 0; rp.data[1] = 1; ci.data[0] = 0; a.data[0] = 0.25;
    r.data[0] = std::ldexp(1.0, 1022);
    const double expected = std::ldexp(1.0, 1023);
    for (Mode m : {Mode::INT4_Split, Mode::FP16_Split, Mode::FP64_Exact}) {
      bitfsai::Smoother sm;
      sm.setup(1, 1, rp.data, ci.data, a.data, m, true, 12);
      x.data[0] = 0.0;
      sm.apply(rp.data, ci.data, r.data, x.data, 0.5, false);
      bitfsai::detail::check_cuda(cudaDeviceSynchronize(), "f64 damp=0.5 overflow witness acc=false");
      require(std::isfinite(x.data[0]) && x.data[0] == expected,
              "binary64 apply must keep damp * G^T G r = 2^1023 finite (accumulate=false)");
      x.data[0] = 0.0;
      sm.apply(rp.data, ci.data, r.data, x.data, 0.5, true);
      bitfsai::detail::check_cuda(cudaDeviceSynchronize(), "f64 damp=0.5 overflow witness acc=true");
      require(std::isfinite(x.data[0]) && x.data[0] == expected,
              "binary64 apply must keep damp * G^T G r = 2^1023 finite (accumulate=true)");
    }
  }

  // 1c) Off-diagonal FP32 subnormal underflow witness in Pass 1:
  //     A = [2^-48, -2^-52; -2^-52, 2^-4 + 2^-56], r = [2^-149, 0], damp = 1.0.
  //     Row 1 has r[1] == 0 but nonzero off-diagonal coupling whose FP32 Pass-1 product
  //     underflows below FLT_MIN; the any_nz guard must trigger the double fallback path.
  {
    Managed<int> rp(3), ci(4);
    Managed<double> a(4);
    Managed<float> rf(2), xf(2);
    Managed<double> rd(2), xd(2);
    rp.data[0] = 0; rp.data[1] = 2; rp.data[2] = 4;
    ci.data[0] = 0; ci.data[1] = 1; ci.data[2] = 0; ci.data[3] = 1;
    a.data[0] = std::ldexp(1.0, -48);
    a.data[1] = -std::ldexp(1.0, -52);
    a.data[2] = -std::ldexp(1.0, -52);
    a.data[3] = std::ldexp(1.0, -4) + std::ldexp(1.0, -56);
    const float tiny = std::ldexp(1.0f, -149);
    rf.data[0] = tiny; rf.data[1] = 0.0f;
    rd.data[0] = static_cast<double>(tiny); rd.data[1] = 0.0;
    for (Mode m : {Mode::INT4_Split, Mode::FP16_Split}) {
      bitfsai::Smoother sm;
      sm.setup(2, 4, rp.data, ci.data, a.data, m, true, 12);
      xd.data[0] = 0.0; xd.data[1] = 0.0;
      sm.apply(rp.data, ci.data, rd.data, xd.data, 1.0, false);
      bitfsai::detail::check_cuda(cudaDeviceSynchronize(), "off-diag subnormal reference double apply");
      xf.data[0] = 0.0f; xf.data[1] = 0.0f;
      sm.apply(rp.data, ci.data, rf.data, xf.data, 1.0, false);
      bitfsai::detail::check_cuda(cudaDeviceSynchronize(), "off-diag subnormal float apply");
      require(std::isfinite(xf.data[0]) && std::isfinite(xf.data[1]) && xf.data[1] > 0.0f,
              "Pass 1 off-diagonal FP32 subnormal underflow must trigger double fallback");
      require(std::fabs(static_cast<double>(xf.data[1]) - xd.data[1]) <=
                  1e-5 * std::fabs(xd.data[1]),
              "Pass 1 off-diagonal subnormal fallback must match double Gram apply");
    }
  }

  // 2) Certified binary64 damping preservation: A = [3], G = [1] => 3 * damp_applied <= 1.0 strictly
  {
    bitfsai::certificate::Csr certA{{0, 1}, {0}, {3.0}};
    bitfsai::certificate::SplitFactor certG{{0, 0}, {}, {}, {0.0}, {1.0}};
    auto cert = bitfsai::certificate::certify_damping(certA, certG);
    require(3.0L * static_cast<long double>(cert.omega_lower) <= 1.0L,
            "certified binary64 omega_lower must satisfy 3 * omega <= 1");

    Managed<int> rp(2), ci(1);
    Managed<double> a(1), r(1), x(1);
    rp.data[0] = 0; rp.data[1] = 1; ci.data[0] = 0; a.data[0] = 1.0; // G = [1]
    r.data[0] = 1.0; x.data[0] = 0.0;
    bitfsai::Smoother sm;
    sm.setup(1, 1, rp.data, ci.data, a.data, Mode::INT4_Split, true, 12);
    sm.apply(rp.data, ci.data, r.data, x.data, cert.omega_lower, false);
    bitfsai::detail::check_cuda(cudaDeviceSynchronize(), "certified binary64 damp apply");
    require(x.data[0] == cert.omega_lower &&
                3.0L * static_cast<long double>(x.data[0]) <= 1.0L,
            "bitfsai::Smoother::apply must preserve certified binary64 damping without upward FP32 rounding");
  }
  // 3) Invalid Mode enum validation in setup()
  {
    Managed<int> rp(2), ci(1);
    Managed<double> a(1);
    rp.data[0] = 0; rp.data[1] = 1; ci.data[0] = 0; a.data[0] = 2.0;
    bitfsai::Smoother sm;
    bool rejected = false;
    try {
      sm.setup(1, 1, rp.data, ci.data, a.data, static_cast<bitfsai::Mode>(99));
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    require(rejected && sm.n == 0, "setup must reject invalid Mode enum value");
  }

  // 4) Real Smoother (INT4_Split) -> SplitFactor adapter -> certified damping -> GPU apply
  //    Verify directly on both a nontrivial 4x4 SPD matrix and a 6^3 3D anisotropic SPD matrix
  //    that omega_lower * lambda_max(M * A) <= 1.0.
  auto verify_real_cert = [](const HostCSR &h, const char *label) {
    Managed<int> rp(h.rp.size()), ci(h.ci.size());
    Managed<double> a(h.val.size()), r(h.n), x(h.n);
    for (size_t i = 0; i < h.rp.size(); ++i) rp.data[i] = h.rp[i];
    for (size_t i = 0; i < h.ci.size(); ++i) ci.data[i] = h.ci[i];
    for (size_t i = 0; i < h.val.size(); ++i) a.data[i] = h.val[i];

    bitfsai::Smoother sm;
    sm.setup(h.n, h.nnz, rp.data, ci.data, a.data, bitfsai::Mode::INT4_Split, true, 12);

    std::vector<float> h_diag(h.n), h_scale(h.n);
    std::vector<uint8_t> h_packed((static_cast<size_t>(h.nnz) + 1) / 2);
    bitfsai::detail::check_cuda(
        cudaMemcpy(h_diag.data(), sm.d_diag, h.n * sizeof(float), cudaMemcpyDeviceToHost), "copy diag");
    bitfsai::detail::check_cuda(
        cudaMemcpy(h_scale.data(), sm.d_off_scale, h.n * sizeof(float), cudaMemcpyDeviceToHost), "copy scale");
    if (!h_packed.empty()) {
      bitfsai::detail::check_cuda(
          cudaMemcpy(h_packed.data(), sm.d_q_packed, h_packed.size(), cudaMemcpyDeviceToHost), "copy packed");
    }

    bitfsai::certificate::Csr certA;
    certA.row.assign(h.rp.begin(), h.rp.end());
    certA.col.assign(h.ci.begin(), h.ci.end());
    certA.value = h.val;
    auto certG = bitfsai::certificate::make_int4_split_factor_from_packed_csr(
        h.n, h.rp.data(), h.ci.data(), h_diag.data(), h_scale.data(), h_packed.data());
    auto cert = bitfsai::certificate::certify_damping(certA, certG);
    require(cert.omega_lower > 0.0 && cert.U_upper > 0.0,
            "real INT4_Split damping certificate must be positive");

    // Power iteration in A-inner-product on T = omega_lower * M * A applied via GPU Smoother::apply
    std::mt19937 rng(2026);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> v(h.n);
    for (auto &vi : v) vi = nd(rng);
    double vnorm = std::sqrt(dot(v, v));
    for (auto &vi : v) vi /= vnorm;
    double lmax = 0.0;
    for (int it = 0; it < 60; ++it) {
      auto Av = host_spmv(h, v);
      for (int i = 0; i < h.n; ++i) r.data[i] = Av[i];
      sm.apply(rp.data, ci.data, r.data, x.data, cert.omega_lower, false);
      bitfsai::detail::check_cuda(cudaDeviceSynchronize(), "certified real apply");
      std::vector<double> w(x.data, x.data + h.n);
      double vAv = dot(v, Av);
      double vAw = dot(Av, w);
      lmax = vAw / vAv;
      require(lmax > 0.0 && lmax <= 1.0 + 1e-12,
              "certified real INT4_Split apply must satisfy omega_lower * lambda_max(M A) <= 1");
      double wn = std::sqrt(dot(w, w));
      for (int i = 0; i < h.n; ++i) v[i] = w[i] / wn;
    }
    std::printf("[PASS] Real Smoother -> SplitFactor certified damping (%s): U=%.3e, omega=%.3e, max(omega*lam)=%.6f <= 1\n",
                label, cert.U_upper, cert.omega_lower, lmax);
  };

  {
    HostCSR h4;
    h4.n = 4;
    h4.rp = {0, 3, 7, 11, 14};
    h4.ci = {0, 1, 2,  0, 1, 2, 3,  0, 1, 2, 3,  1, 2, 3};
    h4.val = {
        10.0, -3.0, -1.0,
        -3.0, 12.0, -4.0, -2.0,
        -1.0, -4.0, 14.0, -5.0,
        -2.0, -5.0, 11.0,
    };
    h4.nnz = static_cast<int>(h4.ci.size());
    verify_real_cert(h4, "4x4 SPD");
  }
  verify_real_cert(build_3d_structured(6, 0), "6^3 aniso+10^4");

  std::puts("[PASS] Pass 2 pre-damping overflow recovery (acc=false/true) and binary64 certified damping verified");
}

int main() {
  test_malformed_csr_validation();
  test_pass2_overflow_and_f64_damping();
  test_scaled_patches();
  test_float_factored_exponent_extremes();
  test_csr_symmetry_and_mtx_guards();
  test_concurrent_apply();
  test_2x2_counterexample();
  test_theorem4_transpose_equivalence();
  test_pcg_convergence();
  std::printf("ALL BITFSAI REFERENCE CHECKS PASSED.\n");
  return 0;
}
