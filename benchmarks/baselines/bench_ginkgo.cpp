#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <ginkgo/ginkgo.hpp>
#include "../../reference/workloads.hpp"

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

static std::shared_ptr<gko::matrix::Csr<double, int>> to_gko_csr(
    std::shared_ptr<gko::Executor> exec, const HostCSR &h) {
  auto host_exec = exec->get_master();
  auto A_host = gko::matrix::Csr<double, int>::create(
      host_exec, gko::dim<2>(h.n, h.n),
      gko::array<double>::const_view(host_exec, h.nnz, h.val.data()).copy_to_array(),
      gko::array<int>::const_view(host_exec, h.nnz, h.ci.data()).copy_to_array(),
      gko::array<int>::const_view(host_exec, h.n + 1, h.rp.data()).copy_to_array());
  return gko::share(gko::clone(exec, A_host));
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

static std::shared_ptr<gko::LinOpFactory> make_mg_factory(
    std::shared_ptr<gko::Executor> exec, const std::string &method) {
  using ir_f = gko::solver::Ir<float>;
  using bj_f = gko::preconditioner::Jacobi<float, int>;
  using isai_f = gko::preconditioner::SpdIsai<float, int>;
  using pgm_f = gko::multigrid::Pgm<float, int>;

  auto pgm_gen_f = gko::share(pgm_f::build().with_deterministic(true).on(exec));
  auto bj_gen_f = gko::share(bj_f::build().with_max_block_size(1u).on(exec));
  auto isai_gen_f = gko::share(
      isai_f::build().with_sparsity_power(1).with_excess_limit(0u).on(exec));

  auto smooth_bj_f = gko::share(
      ir_f::build()
          .with_solver(bj_gen_f)
          .with_relaxation_factor(0.9f)
          .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
          .on(exec));
  auto smooth_isai_f = gko::share(
      ir_f::build()
          .with_solver(isai_gen_f)
          .with_relaxation_factor(0.8f)
          .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
          .on(exec));
  auto coarse_f = gko::share(
      ir_f::build()
          .with_solver(bj_gen_f)
          .with_relaxation_factor(0.9f)
          .with_criteria(gko::stop::Iteration::build().with_max_iters(8u))
          .on(exec));

  if (method == "mg_jacobi_f32") {
    return gko::solver::Multigrid::build()
        .with_max_levels(12u)
        .with_min_coarse_rows(32u)
        .with_pre_smoother(smooth_bj_f)
        .with_post_uses_pre(true)
        .with_mg_level(pgm_gen_f)
        .with_coarsest_solver(coarse_f)
        .with_default_initial_guess(gko::solver::initial_guess_mode::zero)
        .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
        .on(exec);
  }
  if (method == "mg_isai_f32") {
    return gko::solver::Multigrid::build()
        .with_max_levels(12u)
        .with_min_coarse_rows(32u)
        .with_pre_smoother(smooth_isai_f)
        .with_post_uses_pre(true)
        .with_mg_level(pgm_gen_f)
        .with_coarsest_solver(coarse_f)
        .with_default_initial_guess(gko::solver::initial_guess_mode::zero)
        .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
        .on(exec);
  }

  using ir_h = gko::solver::Ir<gko::half>;
  using bj_h = gko::preconditioner::Jacobi<gko::half, int>;
  using pgm_h = gko::multigrid::Pgm<gko::half, int>;
  auto pgm_gen_h = gko::share(pgm_h::build().with_deterministic(true).on(exec));
  auto bj_gen_h = gko::share(bj_h::build().with_max_block_size(1u).on(exec));
  auto smooth_bj_h = gko::share(
      ir_h::build()
          .with_solver(bj_gen_h)
          .with_relaxation_factor(static_cast<gko::half>(0.9f))
          .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
          .on(exec));
  auto coarse_h = gko::share(
      ir_h::build()
          .with_solver(bj_gen_h)
          .with_relaxation_factor(static_cast<gko::half>(0.9f))
          .with_criteria(gko::stop::Iteration::build().with_max_iters(8u))
          .on(exec));
  return gko::solver::Multigrid::build()
      .with_max_levels(12u)
      .with_min_coarse_rows(32u)
      .with_pre_smoother(smooth_isai_f, smooth_bj_h)
      .with_post_uses_pre(true)
      .with_mg_level(pgm_gen_f, pgm_gen_h)
      .with_level_selector([](const gko::size_type level, const gko::LinOp *) -> gko::size_type {
        return level >= 1 ? 1 : 0;
      })
      .with_coarsest_solver(coarse_h)
      .with_default_initial_guess(gko::solver::initial_guess_mode::zero)
      .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
      .on(exec);
}

int main(int argc, char **argv) try {
  if (argc < 4) {
    std::fprintf(stderr, "usage: bench_ginkgo CASE METHOD REPEATS\n");
    return 1;
  }
  std::string name = argv[1];
  std::string method = argv[2];
  if (method != "mg_jacobi_f32" && method != "mg_isai_f32" && method != "mg_isai_3p") {
    std::fprintf(stderr, "unknown method '%s' (expected mg_jacobi_f32, mg_isai_f32, or mg_isai_3p)\n",
                 method.c_str());
    return 1;
  }
  int repeats = std::stoi(argv[3]);
  if (repeats < 1) {
    std::fprintf(stderr, "repeats must be positive\n");
    return 1;
  }

  auto exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
  auto host_exec = exec->get_master();

  HostCSR h0 = load_case(name, 0);
  HostCSR h2 = load_case(name, 2);
  assert_same_pattern(h0, h2, name, 2);

  std::mt19937 rng(12345);
  std::normal_distribution<double> nd(0.0, 1.0);
  std::vector<double> b_vec(h0.n);
  for (auto &v : b_vec) v = nd(rng);

  auto A0 = to_gko_csr(exec, h0);
  auto A2 = to_gko_csr(exec, h2);

  using vec = gko::matrix::Dense<double>;
  auto b_host = vec::create(host_exec, gko::dim<2>(h0.n, 1));
  for (int i = 0; i < h0.n; ++i) b_host->at(i, 0) = b_vec[i];
  auto b_dev = gko::clone(exec, b_host);
  auto x_dev = vec::create(exec, gko::dim<2>(h0.n, 1));
  auto x_host = vec::create(host_exec, gko::dim<2>(h0.n, 1));

  bool all_pass = true;
  std::printf("engine,case,method,trial,phase,setup_ms,solve_ms,iters,true_rel,pass\n");
  for (int t = 0; t < repeats; ++t) {
    for (int phase = 0; phase < 2; ++phase) {
      const auto &A = (phase == 0) ? A0 : A2;
      const auto &h = (phase == 0) ? h0 : h2;
      auto iter_stop = gko::share(
          gko::stop::Iteration::build().with_max_iters(2000u).on(exec));
      auto tol_stop = gko::share(
          gko::stop::ResidualNorm<double>::build()
              .with_baseline(gko::stop::mode::rhs_norm)
              .with_reduction_factor(7e-9)
              .on(exec));
      std::shared_ptr<const gko::log::Convergence<double>> logger =
          gko::log::Convergence<double>::create();
      iter_stop->add_logger(logger);
      tol_stop->add_logger(logger);

      auto mg_gen = make_mg_factory(exec, method);
      auto cg_gen = gko::solver::Cg<double>::build()
                        .with_criteria(iter_stop, tol_stop)
                        .with_preconditioner(mg_gen)
                        .on(exec);

      exec->synchronize();
      auto t0 = std::chrono::steady_clock::now();
      auto solver = cg_gen->generate(A);
      exec->synchronize();
      auto t1 = std::chrono::steady_clock::now();
      double setup_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

      x_dev->fill(0.0);
      exec->synchronize();
      auto t2 = std::chrono::steady_clock::now();
      solver->apply(b_dev, x_dev);
      exec->synchronize();
      auto t3 = std::chrono::steady_clock::now();
      double solve_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

      x_host->copy_from(x_dev);
      std::vector<double> x_vec(h.n);
      for (int i = 0; i < h.n; ++i) x_vec[i] = x_host->at(i, 0);
      double rel = cpu_true_rel(h, b_vec, x_vec);
      int pass = (std::isfinite(rel) && rel <= 1e-8) ? 1 : 0;
      if (!pass)
        all_pass = false;
      std::printf("ginkgo,%s,%s,%d,%s,%.5f,%.5f,%zu,%.12e,%d\n",
                  name.c_str(), method.c_str(), t, phase ? "timed" : "warm",
                  setup_ms, solve_ms, logger->get_num_iterations(), rel, pass);
      std::fflush(stdout);
    }
  }
  return all_pass ? 0 : 2;
} catch (const std::exception &e) {
  std::fprintf(stderr, "%s\n", e.what());
  return 1;
}
