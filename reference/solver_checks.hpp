template <class T> vector<T> download(const Buffer<T> &buffer, size_t count) {
  vector<T> host(count);
  CHECK(cudaMemcpy(host.data(), buffer.p, count * sizeof(T),
                   cudaMemcpyDeviceToHost));
  return host;
}

void test_numeric_refresh() {
  constexpr int n = 96;
  HostCSR h;
  h.n = n;
  h.rp.push_back(0);
  vector<double> changed;
  for (int i = 0; i < n; ++i) {
    for (int j = std::max(0, i - 18); j <= std::min(n - 1, i + 18); ++j) {
      h.ci.push_back(j);
      double a = i == j ? 72.0 : -1.0 / (1 + (i + j) % 4);
      h.val.push_back(a);
      changed.push_back(a * (1.1 + .2 * sin(i + j)));
    }
    h.rp.push_back(h.ci.size());
  }
  h.nnz = h.ci.size();
  Buffer<double> values;
  values.init(changed);
  vector<double> rhs(n);
  for (int i = 0; i < n; ++i)
    rhs[i] = sin(i + .3);
  for (bool sa : {false, true}) {
    Engine engine(h, rhs, !sa, false, sa);
    Level &fine = *engine.ls[0], &coarse = *engine.ls[1];
    for (Method method : {Half, Int4, Double}) {
      reset(engine.ls, method, values.p);
      auto &w = work<double>(fine);
      CHECK(cudaMemcpyAsync(w.rhs.p, rhs.data(), n * sizeof(double),
                            cudaMemcpyHostToDevice, stream));
      smooth<double>(fine, method, false);
      CHECK(cudaStreamSynchronize(stream));
      auto x = download(w.x, n);
      auto d = download(fine.diag, n), scale = download(fine.scale, n);
      auto half = download(fine.half, h.nnz);
      auto exact = download(fine.exact, h.nnz);
      auto packed = download(fine.packed, (h.nnz + n + 1) / 2);
      vector<double> g(n * n, 0), tmp(n, 0);
      for (int i = 0; i < n; ++i) {
        for (int k = h.rp[i]; k < h.rp[i + 1] && h.ci[k] <= i; ++k) {
          int c = h.ci[k], j = k - h.rp[i];
          double v;
          if (method == Double)
            v = exact[k];
          else if (c == i)
            v = d[i];
          else if (method == Half)
            v = scale[i] * __half2float(half[k]);
          else {
            int q = (packed[(h.rp[i] + i) / 2 + j / 2] >> (4 * (j & 1))) & 15;
            v = double(scale[i]) * (((q ^ 8) - 8) / 7.0);
          }
          g[i * n + c] = v;
          tmp[i] += v * rhs[c];
        }
      }
      for (int i = 0; i < n; ++i) {
        double expected = 0;
        for (int j = i; j < n; ++j)
          expected += g[j * n + i] * tmp[j];
        if (!(std::abs(x[i] - .35 * expected) < 1e-9))
          throw std::runtime_error(
              "mirrored factor disagrees with dense transpose");
      }
      if (method != Double) {
        vector<float> rhs_f(n);
        for (int i = 0; i < n; ++i)
          rhs_f[i] = static_cast<float>(rhs[i]);
        auto &wf = work<float>(fine);
        CHECK(cudaMemcpyAsync(wf.rhs.p, rhs_f.data(), n * sizeof(float),
                              cudaMemcpyHostToDevice, stream));
        smooth<float>(fine, method, false);
        CHECK(cudaStreamSynchronize(stream));
        auto xf = download(wf.x, n);
        double diff2 = 0, ref2 = 0;
        for (int i = 0; i < n; ++i) {
          double dlt = double(xf[i]) - x[i];
          diff2 += dlt * dlt;
          ref2 += x[i] * x[i];
        }
        if (!(sqrt(diff2 / std::max(1e-30, ref2)) < 1e-5))
          throw std::runtime_error(
              "factored FP32 smooth disagrees with FP64 reference");
      }
    }
    auto col = download(fine.interp, 2 * n);
    auto pw = download(fine.pw, 2 * n);
    auto rp = download(coarse.rp, coarse.n + 1),
         ci = download(coarse.ci, coarse.nz);
    auto ac = download(coarse.a, coarse.nz);
    vector<double> reference(coarse.n * coarse.n, 0);
    for (int i = 0; i < n; i++)
      for (int k = h.rp[i]; k < h.rp[i + 1]; k++)
        for (int left = 2 * i; left < 2 * i + 2; left++)
          for (int right = 2 * h.ci[k]; right < 2 * h.ci[k] + 2; right++)
            if (col[left] >= 0 && col[right] >= 0)
              reference[col[left] * coarse.n + col[right]] +=
                  pw[left] * changed[k] * pw[right];
    for (int i = 0; i < coarse.n; i++)
      for (int k = rp[i]; k < rp[i + 1]; k++)
        reference[i * coarse.n + ci[k]] -= ac[k];
    for (double e : reference)
      if (!(std::abs(e) < 1e-10))
        throw std::runtime_error(
            "cached RAP disagrees with independent host product");
  }
  puts("Changed-coefficient RAP and mirrored-factor checks passed for both "
       "hierarchies and three encodings.");
}

HostCSR isolated_smoothed_aggregation_matrix(bool invalid = false) {
  constexpr int n = 66;
  HostCSR h;
  h.n = n;
  h.rp.push_back(0);
  for (int i = 0; i < n; ++i) {
    if (i < 2) {
      h.ci.push_back(i);
      h.val.push_back(invalid ? -1.0 : 1.0);
    } else {
      if (i > 2) {
        h.ci.push_back(i - 1);
        h.val.push_back(-1.0);
      }
      h.ci.push_back(i);
      h.val.push_back(6.0);
      if (i + 1 < n) {
        h.ci.push_back(i + 1);
        h.val.push_back(-1.0);
      }
    }
    h.rp.push_back(h.ci.size());
  }
  h.nnz = h.ci.size();
  return h;
}

double host_relative_residual(const HostCSR &h, const vector<double> &b,
                              const vector<double> &x) {
  double r2 = 0, b2 = 0;
  for (int i = 0; i < h.n; ++i) {
    double ax = 0;
    for (int k = h.rp[i]; k < h.rp[i + 1]; ++k)
      ax += h.val[k] * x[h.ci[k]];
    double r = b[i] - ax;
    r2 += r * r;
    b2 += b[i] * b[i];
  }
  return sqrt(r2 / b2);
}

void require_smoothed_aggregation_anchors(const Level &fine) {
  auto anchor = download(fine.anchor, fine.nc);
  auto pw = download(fine.pw, 2 * fine.n);
  for (int c = 0; c < fine.nc; ++c) {
    int i = anchor[c];
    if (!(i >= 0 && i < fine.n && pw[2 * i] == 1.0 && pw[2 * i + 1] == 0.0))
      throw std::runtime_error(
          "smoothed aggregation lost its full-rank injection anchor");
  }
}

void test_smoothed_aggregation_anchor_refresh() {
  HostCSR h = isolated_smoothed_aggregation_matrix();
  vector<double> changed = h.val;
  for (int i = 0; i < h.n; ++i)
    for (int k = h.rp[i]; k < h.rp[i + 1]; ++k)
      changed[k] *= h.ci[k] < 2 ? 2.0 : 1.25;
  HostCSR refreshed = h;
  refreshed.val = changed;
  vector<double> rhs(h.n);
  for (int i = 0; i < h.n; ++i)
    rhs[i] = 1.0 + 0.1 * i;
  Buffer<double> initial, updated;
  initial.init(h.val);
  updated.init(changed);
  Engine engine(h, rhs, false, true, true);
  for (int pass = 0; pass < 2; ++pass) {
    reset(engine.ls, Int4, pass ? updated.p : initial.p);
    engine.capture(Int4);
    CHECK(cudaStreamSynchronize(stream));
    require_smoothed_aggregation_anchors(*engine.ls[0]);
    engine.solve(Int4);
    CHECK(cudaStreamSynchronize(stream));
    auto x = download(engine.solution, h.n);
    const HostCSR &current = pass ? refreshed : h;
    if (!(host_relative_residual(current, rhs, x) <= 1e-8))
      throw std::runtime_error(
          "smoothed aggregation graph solve failed after numeric refresh");
  }
  puts("Smoothed aggregation retained full-rank anchors through graph refresh.");
}

void test_invalid_coarse_cholesky_rejected() {
  HostCSR h = isolated_smoothed_aggregation_matrix(true);
  vector<double> rhs(h.n, 1.0);
  Buffer<double> values;
  values.init(h.val);
  Engine engine(h, rhs, false, false, true);
  bool rejected = false;
  try {
    reset(engine.ls, Double, values.p);
  } catch (const std::runtime_error &e) {
    rejected = std::string(e.what()).find("dense coarse operator is not SPD") !=
               std::string::npos;
  }
  if (!rejected)
    throw std::runtime_error("invalid coarse Cholesky pivot was not rejected");
  puts("Invalid dense coarse Cholesky pivot was rejected.");
}

HostCSR raw_split_int4_control_matrix() {
  constexpr int n = 66;
  HostCSR h;
  h.n = n;
  h.rp.push_back(0);
  for (int i = 0; i < n; ++i) {
    if (i == 0) {
      h.ci.insert(h.ci.end(), {0, 2});
      h.val.insert(h.val.end(), {1.0, -1.0});
    } else if (i == 1) {
      h.ci.insert(h.ci.end(), {1, 2});
      h.val.insert(h.val.end(), {10000.0, -400.0});
    } else if (i == 2) {
      h.ci.insert(h.ci.end(), {0, 1, 2});
      h.val.insert(h.val.end(), {-1.0, -400.0, 18.0});
    } else if (i == 3) {
      h.ci.insert(h.ci.end(), {3, 5});
      h.val.insert(h.val.end(), {1.0, 1.0});
    } else if (i == 4) {
      h.ci.insert(h.ci.end(), {4, 5});
      h.val.insert(h.val.end(), {10000.0, 400.0});
    } else if (i == 5) {
      h.ci.insert(h.ci.end(), {3, 4, 5});
      h.val.insert(h.val.end(), {1.0, 400.0, 18.0});
    } else {
      h.ci.push_back(i);
      h.val.push_back(1.0);
    }
    h.rp.push_back(h.ci.size());
  }
  h.nnz = h.ci.size();
  return h;
}

double host_entry(const HostCSR &h, int row, int col) {
  for (int k = h.rp[row]; k < h.rp[row + 1]; ++k)
    if (h.ci[k] == col)
      return h.val[k];
  return 0;
}

double stored_weight(const HostCSR &h, int row, int k, Method method,
                     const vector<float> &diag, const vector<float> &scale,
                     const vector<__half> &half,
                     const vector<unsigned char> &packed) {
  if (h.ci[k] == row)
    return diag[row];
  int local = k - h.rp[row];
  if (method == Half)
    return double(scale[row]) * __half2float(half[k]);
  int q = (packed[(h.rp[row] + row) / 2 + local / 2] >>
           (4 * (local & 1))) &
          15;
  return method == Int4Raw ? double(scale[row]) * ((q ^ 8) - 8)
                            : double(scale[row]) * (((q ^ 8) - 8) / 7.0);
}

double stored_row_energy(const HostCSR &h, int row, Method method,
                         const vector<float> &diag,
                         const vector<float> &scale,
                         const vector<__half> &half,
                         const vector<unsigned char> &packed) {
  double energy = 0;
  for (int left = h.rp[row]; left < h.rp[row + 1] && h.ci[left] <= row;
       ++left)
    for (int right = h.rp[row]; right < h.rp[row + 1] && h.ci[right] <= row;
         ++right)
      energy += stored_weight(h, row, left, method, diag, scale, half, packed) *
                host_entry(h, h.ci[left], h.ci[right]) *
                stored_weight(h, row, right, method, diag, scale, half, packed);
  return energy;
}

void require_stored_smoother_matches_dense(const HostCSR &h, Level &fine,
                                           Method method,
                                           const vector<double> &rhs) {
  auto diag = download(fine.diag, h.n), scale = download(fine.scale, h.n);
  auto half = download(fine.half, h.nnz);
  auto packed = download(fine.packed, (h.nnz + h.n + 1) / 2);
  vector<double> g(h.n * h.n, 0), tmp(h.n, 0), expected(h.n, 0);
  for (int i = 0; i < h.n; ++i)
    for (int k = h.rp[i]; k < h.rp[i + 1] && h.ci[k] <= i; ++k)
      g[i * h.n + h.ci[k]] =
          stored_weight(h, i, k, method, diag, scale, half, packed);
  for (int i = 0; i < h.n; ++i)
    for (int j = 0; j <= i; ++j)
      tmp[i] += g[i * h.n + j] * rhs[j];
  for (int i = 0; i < h.n; ++i)
    for (int j = i; j < h.n; ++j)
      expected[i] += g[j * h.n + i] * tmp[j];
  auto &w = work<double>(fine);
  CHECK(cudaMemcpyAsync(w.rhs.p, rhs.data(), h.n * sizeof(double),
                        cudaMemcpyHostToDevice, stream));
  smooth<double>(fine, method, false);
  CHECK(cudaStreamSynchronize(stream));
  auto actual = download(w.x, h.n);
  for (int i = 0; i < h.n; ++i)
    if (!(std::abs(actual[i] - 0.65 * expected[i]) < 1e-9))
      throw std::runtime_error(
          "persisted mirrored factor disagrees with its dense Gram application");
}

void test_int4_raw_projection_control() {
  HostCSR h = raw_split_int4_control_matrix();
  vector<double> rhs(h.n);
  for (int i = 0; i < h.n; ++i)
    rhs[i] = 0.25 + i;
  Buffer<double> values;
  values.init(h.val);
  Engine engine(h, rhs, true, false);
  Level &fine = *engine.ls[0];
  for (Method method : {Int4Raw, Int4, Half}) {
    reset(engine.ls, method, values.p);
    auto diag = download(fine.diag, h.n), scale = download(fine.scale, h.n);
    auto half = download(fine.half, h.nnz);
    auto packed = download(fine.packed, (h.nnz + h.n + 1) / 2);
    if (method == Int4Raw) {
      int plus = (packed[(h.rp[2] + 2) / 2] >>
                  (4 * ((h.rp[2] - h.rp[2]) & 1))) &
                 15;
      int minus = (packed[(h.rp[5] + 5) / 2] >>
                   (4 * ((h.rp[5] - h.rp[5]) & 1))) &
                  15;
      if (!(plus == 7 && ((minus ^ 8) - 8) == -7 &&
            std::abs(stored_row_energy(h, 2, method, diag, scale, half, packed) -
                     17.0) < 1e-6 &&
            std::abs(stored_row_energy(h, 5, method, diag, scale, half, packed) -
                     17.0) < 1e-6))
        throw std::runtime_error(
            "raw split INT4 control did not retain the 17-energy rows");
    } else if (!(std::abs(stored_row_energy(h, 2, method, diag, scale, half,
                                             packed) -
                        1.0) < 1e-4 &&
                 std::abs(stored_row_energy(h, 5, method, diag, scale, half,
                                             packed) -
                        1.0) < 1e-4)) {
      throw std::runtime_error("row projection did not store unit-energy rows");
    }
    require_stored_smoother_matches_dense(h, fine, method, rhs);
  }
  puts("Raw split INT4 rows reached energy=17; projected INT4 and Half stored unit energy.");
}

void test_fp8_coarse_and_poly_boost() {
  HostCSR h = build_3d_structured(16, 0);
  vector<double> rhs(h.n);
  for (int i = 0; i < h.n; ++i)
    rhs[i] = sin(0.13 * (i + 1));
  Buffer<double> values;
  values.init(h.val);

  // Verify row-sum-repaired FP8 E5M2 coarse operators + CUDA-Graph poly4 solve
  Engine engine(h, rhs, true, true, false, true, 4, 1.0f);
  reset(engine.ls, Int4, values.p, true);
  engine.capture(Int4);
  CHECK(cudaStreamSynchronize(stream));

  // Verify real resident allocation discipline: finest level allocates af only,
  // nonterminal FP8 coarse levels allocate ac_fp8 + ac_diag and NOT af, and
  // only_int4 mode omits half and exact buffers.
  if (engine.ls[0]->ac_fp8.p != nullptr || engine.ls[0]->ac_diag.p != nullptr ||
      engine.ls[0]->af.p == nullptr)
    throw std::runtime_error("finest level did not isolate FP32 af allocation");
  Level &c1 = *engine.ls[1];
  if (c1.af.p != nullptr || c1.ac_fp8.p == nullptr || c1.ac_diag.p == nullptr)
    throw std::runtime_error("FP8 coarse level still allocated unused FP32 af");
  {
    Engine slim(h, rhs, true, false, false, true, 0, 1.0f, 0, true);
    if (slim.ls[0]->half.p != nullptr || slim.ls[0]->exact.p != nullptr ||
        slim.ls[0]->packed.p == nullptr)
      throw std::runtime_error("only_int4 mode still allocated half/exact buffers");
  }

  // Verify row-sum conservation A_tilde * 1 == A * 1 on coarse level 1
  auto rp = download(c1.rp, c1.n + 1);
  auto ci = download(c1.ci, c1.nz);
  auto a = download(c1.a, c1.nz);
  auto ac_fp8 = download(c1.ac_fp8, c1.nz);
  auto ac_diag = download(c1.ac_diag, c1.n);
  double max_rel_row_err = 0.0;
  for (int i = 0; i < c1.n; ++i) {
    double exact_sum = 0.0, rep_sum = 0.0, row_l1 = 0.0;
    for (int k = rp[i]; k < rp[i + 1]; ++k) {
      exact_sum += a[k];
      row_l1 += std::abs(a[k]);
      if (ci[k] == i) {
        rep_sum += ac_diag[i];
      } else {
        __half_raw hr = __nv_cvt_fp8_to_halfraw(
            static_cast<__nv_fp8_storage_t>(ac_fp8[k]), __NV_E5M2);
        rep_sum += static_cast<double>(__half2float(__half(hr)));
      }
    }
    max_rel_row_err = std::max(max_rel_row_err,
                               std::abs(exact_sum - rep_sum) / std::max(1.0, row_l1));
  }
  if (!(max_rel_row_err < 1e-6))
    throw std::runtime_error("FP8 E5M2 coarse row-sum repair violated conservation");

  Result res = engine.solve(Int4);
  auto x = download(engine.solution, h.n);
  double rel = host_relative_residual(h, rhs, x);
  if (!(rel <= 1e-8) || res.trigger != "none")
    throw std::runtime_error("FP8 E5M2 coarse + poly4 CUDA-Graph solve failed");
  printf("FP8 E5M2 coarse row-sum repair (max_row_err=%.2e) + poly4 CUDA-Graph solve passed (%d iters, rel=%.2e).\n",
         max_rel_row_err, res.iterations, rel);
}

void test_fp8_symmetrization_and_grounding_preservation() {
  // Build a 132-DOF SPD matrix of 66 strongly-coupled 2-DOF pairs (so injection
  // coarsening produces nc = 66 > 64 nonterminal coarse level 1 with P_{2u,u}=P_{2u+1,u}=1):
  //  (a) Coarse pair (0, 1) has exact RAP block [[1 + 2^-26, -1], [-1, 1 + 2^-26]],
  //      where naive FP32 rounding turns 1 + 2^-26 into 1.0f (singular!), whereas
  //      nextafterf upward rounding preserves strict positive grounding and SPD.
  //  (b) Coarse pair (2, 3) has symmetric fine cross-edges {-2^-53, -2^-53, -1.0, -0.125}
  //      whose independent RAP accumulation orders produce -1.125 - 2^-52 in (2, 3)
  //      and -1.125 in (3, 2), straddling the FP8 E5M2 rounding midpoint (-1.25 vs -1.0);
  //      symmetrization guarantees ac_fp8[k_23] == ac_fp8[k_32].
  constexpr int nc_target = 66;
  constexpr int n = 2 * nc_target;
  std::vector<std::vector<std::pair<int, double>>> adj(n);
  auto add_sym = [&](int r, int c, double v) {
    adj[r].push_back({c, v});
    if (r != c)
      adj[c].push_back({r, v});
  };
  const double eps26 = std::ldexp(1.0, -26);
  const double eps27 = std::ldexp(1.0, -27);
  const double eps53 = std::ldexp(1.0, -53);
  for (int u = 0; u < nc_target; ++u) {
    int i0 = 2 * u, i1 = 2 * u + 1;
    if (u == 0 || u == 1) {
      add_sym(i0, i0, 1.5 + eps27);
      add_sym(i1, i1, 1.5 + eps27);
      add_sym(i0, i1, -1.0);
    } else {
      add_sym(i0, i0, 20.0);
      add_sym(i1, i1, 20.0);
      add_sym(i0, i1, -9.0);
    }
  }
  // Cross-aggregate grounding-loss block on coarse (0, 1): Kronecker-SPD with
  // coarse diagonal (1.5+2^-27)-1-1+(1.5+2^-27) = 1 + 2^-26 and off-diag -0.5-0.5 = -1.0
  add_sym(0, 2, -0.5);
  add_sym(1, 3, -0.5);
  // Cross-aggregate FP8 E5M2 midpoint split on coarse (2, 3)
  add_sym(4, 6, -eps53);
  add_sym(4, 7, -eps53);
  add_sym(5, 6, -1.0);
  add_sym(5, 7, -0.125);

  HostCSR h;
  h.n = n;
  h.rp.push_back(0);
  for (int i = 0; i < n; ++i) {
    auto &row = adj[i];
    std::sort(row.begin(), row.end(),
              [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
    for (const auto &kv : row) {
      h.ci.push_back(kv.first);
      h.val.push_back(kv.second);
    }
    h.rp.push_back(static_cast<int>(h.ci.size()));
  }
  h.nnz = static_cast<int>(h.ci.size());

  vector<double> rhs(n);
  for (int i = 0; i < n; ++i)
    rhs[i] = 1.0 + 0.05 * ((i % 7) - 3);
  Buffer<double> values;
  values.init(h.val);

  // direct = false ("injection") gives exact 2-to-1 pair aggregates of size 66
  Engine engine(h, rhs, false, true, false, true);
  reset(engine.ls, Int4, values.p, true);
  engine.capture(Int4);
  CHECK(cudaStreamSynchronize(stream));

  Level &c1 = *engine.ls[1];
  if (c1.dense.p != nullptr || c1.n != nc_target || !c1.use_fp8_level)
    throw std::runtime_error("coarse level 1 did not stay on nonterminal FP8 path");
  auto rp = download(c1.rp, c1.n + 1);
  auto ci = download(c1.ci, c1.nz);
  auto a = download(c1.a, c1.nz);
  auto ac_fp8 = download(c1.ac_fp8, c1.nz);
  auto ac_diag = download(c1.ac_diag, c1.n);

  auto find_slot = [&](int row, int col) {
    for (int k = rp[row]; k < rp[row + 1]; ++k)
      if (ci[k] == col)
        return k;
    return -1;
  };
  auto decode_e5m2 = [](uint8_t byte) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw(
        static_cast<__nv_fp8_storage_t>(byte), __NV_E5M2);
    return static_cast<double>(__half2float(__half(hr)));
  };

  // Check (a): diag = 1 + 2^-26, off = -1 preserves strict grounding and SPD
  int k01 = find_slot(0, 1), k10 = find_slot(1, 0);
  int k00 = find_slot(0, 0), k11 = find_slot(1, 1);
  if (k01 < 0 || k10 < 0 || k00 < 0 || k11 < 0)
    throw std::runtime_error("missing coarse (0,1) block");
  if (!(std::abs(a[k00] - (1.0 + eps26)) == 0.0 && a[k01] == -1.0))
    throw std::runtime_error("unexpected RAP values in grounding-loss test");
  double d0 = ac_diag[0], d1 = ac_diag[1];
  double o01 = decode_e5m2(ac_fp8[k01]), o10 = decode_e5m2(ac_fp8[k10]);
  double det01 = d0 * d1 - o01 * o10;
  if (!(d0 > 1.0 && d1 > 1.0 && o01 == -1.0 && o10 == -1.0 && det01 > 0.0))
    throw std::runtime_error("FP8+FP32 coarse block lost strict positive grounding");

  // Check (b): RAP asymmetric midpoint values encode to the identical E5M2 byte
  int k23 = find_slot(2, 3), k32 = find_slot(3, 2);
  if (k23 < 0 || k32 < 0)
    throw std::runtime_error("missing coarse (2,3) block");
  if (!(a[k23] < -1.125 && a[k32] == -1.125))
    throw std::runtime_error("RAP did not exercise the midpoint rounding split");
  if (ac_fp8[k23] != ac_fp8[k32])
    throw std::runtime_error("FP8 E5M2 off-diagonal mirror bytes differed at midpoint");

  // Verify the full 66x66 decoded coarse matrix is strictly symmetric and Cholesky-SPD
  vector<double> dec_mat(c1.n * c1.n, 0.0);
  for (int i = 0; i < c1.n; ++i)
    for (int k = rp[i]; k < rp[i + 1]; ++k)
      dec_mat[i * c1.n + ci[k]] =
          (ci[k] == i) ? double(ac_diag[i]) : decode_e5m2(ac_fp8[k]);
  for (int i = 0; i < c1.n; ++i)
    for (int j = 0; j < i; ++j)
      if (dec_mat[i * c1.n + j] != dec_mat[j * c1.n + i])
        throw std::runtime_error("decoded FP8 coarse matrix is not symmetric");
  vector<double> chol(c1.n * c1.n, 0.0);
  for (int i = 0; i < c1.n; ++i)
    for (int j = 0; j <= i; ++j) {
      double v = dec_mat[i * c1.n + j];
      for (int k = 0; k < j; ++k)
        v -= chol[i * c1.n + k] * chol[j * c1.n + k];
      if (i == j) {
        if (!(v > 0.0 && std::isfinite(v)))
          throw std::runtime_error("decoded FP8 coarse matrix is not SPD");
        chol[i * c1.n + j] = std::sqrt(v);
      } else {
        chol[i * c1.n + j] = v / chol[j * c1.n + j];
      }
    }

  Result rr = engine.solve(Int4);
  auto x = download(engine.solution, n);
  if (!(host_relative_residual(h, rhs, x) <= 1e-8))
    throw std::runtime_error("solve failed residual gate on grounding-loss + midpoint FP8 test");

  // Also verify midpoint-only well-conditioned matrix converges in stage 0 without recovery trigger
  for (int u : {0, 1}) {
    int i0 = 2 * u, i1 = 2 * u + 1;
    for (int r : {i0, i1})
      for (int k = h.rp[r]; k < h.rp[r + 1]; ++k)
        if (h.ci[k] == r)
          h.val[k] = 20.0;
  }
  values.init(h.val);
  reset(engine.ls, Int4, values.p, true);
  engine.capture(Int4);
  Result rr_mid = engine.solve(Int4);
  auto x_mid = download(engine.solution, n);
  if (!(host_relative_residual(h, rhs, x_mid) <= 1e-8) || rr_mid.trigger != "none")
    throw std::runtime_error("midpoint FP8 coarse solve triggered unexpected recovery");
  puts("FP8 E5M2 midpoint mirror symmetry and 1+2^-26 grounding preservation passed.");
}

vector<double> mat_mul(int m, int k, int n, const vector<double> &A,
                       const vector<double> &B) {
  vector<double> C(m * n, 0.0);
  for (int i = 0; i < m; ++i)
    for (int p = 0; p < k; ++p) {
      double aip = A[i * k + p];
      if (aip == 0.0)
        continue;
      for (int j = 0; j < n; ++j)
        C[i * n + j] += aip * B[p * n + j];
    }
  return C;
}

vector<double> build_dense_level_matrix(const Level &l, bool use_fp8) {
  int n = l.n;
  auto rp = download(l.rp, n + 1);
  auto ci = download(l.ci, l.nz);
  vector<double> M(n * n, 0.0);
  if (use_fp8 && l.use_fp8_level && !l.dense.p) {
    auto ac_fp8 = download(l.ac_fp8, l.nz);
    auto ac_diag = download(l.ac_diag, n);
    for (int i = 0; i < n; ++i)
      for (int k = rp[i]; k < rp[i + 1]; ++k) {
        if (ci[k] == i) {
          M[i * n + i] = ac_diag[i];
        } else {
          __half_raw hr = __nv_cvt_fp8_to_halfraw(
              static_cast<__nv_fp8_storage_t>(ac_fp8[k]), __NV_E5M2);
          M[i * n + ci[k]] = static_cast<double>(__half2float(__half(hr)));
        }
      }
  } else {
    auto a = download(l.a, l.nz);
    for (int i = 0; i < n; ++i)
      for (int k = rp[i]; k < rp[i + 1]; ++k)
        M[i * n + ci[k]] = l.dense.p ? a[k] : double(float(a[k]));
  }
  return M;
}

vector<double> build_dense_cycle_oracle(const Levels &ls, size_t j, Method m,
                                        bool coarse_fp8, int poly_stride,
                                        int poly_offset, float poly_scale) {
  const Level &l = *ls[j];
  int n = l.n;
  if (l.dense.p) {
    auto L = download(l.dense, n * n);
    vector<double> inv(n * n, 0.0);
    for (int col = 0; col < n; ++col) {
      vector<double> y(n, 0.0);
      for (int i = 0; i < n; ++i) {
        double v = (i == col) ? 1.0 : 0.0;
        for (int k = 0; k < i; ++k) v -= L[i * n + k] * y[k];
        y[i] = v / L[i * n + i];
      }
      for (int i = n - 1; i >= 0; --i) {
        double v = y[i];
        for (int k = i + 1; k < n; ++k) v -= L[k * n + i] * y[k];
        inv[i * n + col] = y[i] = v / L[i * n + i];
      }
    }
    return inv;
  }

  vector<double> A = build_dense_level_matrix(l, coarse_fp8 && j > 0);
  auto rp = download(l.rp, n + 1);
  auto ci = download(l.ci, l.nz);
  auto d = download(l.diag, n), s = download(l.scale, n);
  auto packed = download(l.packed, (size_t(l.nz) + n + 1) / 2);
  vector<double> G(n * n, 0.0), GT(n * n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int k = rp[i]; k < rp[i + 1] && ci[k] <= i; ++k) {
      double v = 0.0;
      if (ci[k] == i) {
        v = d[i];
      } else {
        int local = k - rp[i];
        int q = (packed[(rp[i] + i) / 2 + local / 2] >> (4 * (local & 1))) & 15;
        float sc = (m == Int4) ? (s[i] * (1.0f / 7.0f)) : s[i];
        v = double(sc * float((q ^ 8) - 8));
      }
      G[i * n + ci[k]] = v;
      GT[ci[k] * n + i] = v;
    }
  }
  double damp = double(float(l.max_lower <= MaxLower ? 0.65f : 0.35f));
  vector<double> S = mat_mul(n, n, n, GT, G);
  for (double &v : S)
    v *= damp;

  int nc = l.nc;
  auto col = download(l.interp, 2 * n);
  auto pw = download(l.pw, 2 * n);
  vector<double> P(n * nc, 0.0), PT(nc * n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int slot = 0; slot < 2; ++slot) {
      int c = col[2 * i + slot];
      if (c >= 0) {
        double w = double(float(pw[2 * i + slot]));
        P[i * nc + c] += w;
        PT[c * n + i] += w;
      }
    }
  }

  vector<double> B_child = build_dense_cycle_oracle(
      ls, j + 1, m, coarse_fp8, poly_stride, poly_offset, poly_scale);
  vector<double> K_child = B_child;
  if (poly_stride > 0 && (static_cast<int>(j) % poly_stride == poly_offset) &&
      !ls[j + 1]->dense.p) {
    // Exact polynomial boost operator: K = 2*t*B - t^2 * B * A_c * B
    double t = double(poly_scale);
    vector<double> Ac = build_dense_level_matrix(*ls[j + 1], coarse_fp8);
    vector<double> AcB = mat_mul(nc, nc, nc, Ac, B_child);
    vector<double> BAcB = mat_mul(nc, nc, nc, B_child, AcB);
    for (int idx = 0; idx < nc * nc; ++idx)
      K_child[idx] = 2.0 * t * B_child[idx] - (t * t) * BAcB[idx];
  }

  // E_pre = I - A * S;  X2 = S + P * K_child * P^T * E_pre;  B = X2 + S * (I - A * X2)
  vector<double> AS = mat_mul(n, n, n, A, S);
  vector<double> E_pre(n * n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int c = 0; c < n; ++c)
      E_pre[i * n + c] = (i == c ? 1.0 : 0.0) - AS[i * n + c];
  }
  vector<double> PTE = mat_mul(nc, n, n, PT, E_pre);
  vector<double> KPTE = mat_mul(nc, nc, n, K_child, PTE);
  vector<double> PKPTE = mat_mul(n, nc, n, P, KPTE);
  vector<double> X2(n * n, 0.0);
  for (int idx = 0; idx < n * n; ++idx)
    X2[idx] = S[idx] + PKPTE[idx];
  vector<double> AX2 = mat_mul(n, n, n, A, X2);
  vector<double> E2(n * n, 0.0);
  for (int i = 0; i < n; ++i)
    for (int c = 0; c < n; ++c)
      E2[i * n + c] = (i == c ? 1.0 : 0.0) - AX2[i * n + c];
  vector<double> SE2 = mat_mul(n, n, n, S, E2);
  vector<double> B(n * n, 0.0);
  for (int idx = 0; idx < n * n; ++idx)
    B[idx] = X2[idx] + SE2[idx];
  return B;
}

void test_dense_polynomial_boost_oracle() {
  // Verify invalid poly_scale values (<= 0, NaN, Inf) and bad offsets are rejected.
  HostCSR tiny{2, 4, {0, 2, 4}, {0, 1, 0, 1}, {2.0, -1.0, -1.0, 2.0}};
  vector<double> tiny_b{1.0, 1.0};
  for (float bad_scale : {0.0f, -1.0f, std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity()}) {
    bool rejected = false;
    try {
      Engine e(tiny, tiny_b, true, false, false, false, 2, bad_scale, 0);
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("Engine accepted non-positive or nonfinite poly_scale");
  }

  // Build a well-conditioned 8^3 anisotropic 3D SPD mesh (N = 512, >= 4 levels:
  // 512 -> 256 -> 128 -> 64) so float roundoff does not mask operator bugs and
  // both poly2@0:1.25 (boosts level 1) and poly4@1:1.25 (boosts level 2) are active.
  constexpr int L = 8, N = L * L * L;
  auto idx3 = [&](int x, int y, int z) { return (z * L + y) * L + x; };
  HostCSR h;
  h.n = N;
  h.rp.push_back(0);
  for (int z = 0; z < L; ++z)
    for (int y = 0; y < L; ++y)
      for (int x = 0; x < L; ++x) {
        int u = idx3(x, y, z);
        double diag = 0.25;
        std::vector<std::pair<int, double>> row;
        auto add_nb = [&](int nx, int ny, int nz, double w) {
          if (nx < 0 || nx >= L || ny < 0 || ny >= L || nz < 0 || nz >= L)
            return;
          diag += w;
          row.push_back({idx3(nx, ny, nz), -w});
        };
        double c_mod = 1.0 + 0.5 * ((x + y + z) & 1);
        add_nb(x, y, z - 1, 1e-2 * c_mod);
        add_nb(x, y - 1, z, 1.0 * c_mod);
        add_nb(x - 1, y, z, 1e-3 * c_mod);
        add_nb(x + 1, y, z, 1e-3 * c_mod);
        add_nb(x, y + 1, z, 1.0 * c_mod);
        add_nb(x, y, z + 1, 1e-2 * c_mod);
        row.push_back({u, diag});
        std::sort(row.begin(), row.end(),
                  [](const auto &a, const auto &b) { return a.first < b.first; });
        for (const auto &kv : row) {
          h.ci.push_back(kv.first);
          h.val.push_back(kv.second);
        }
        h.rp.push_back(static_cast<int>(h.ci.size()));
      }
  h.nnz = static_cast<int>(h.ci.size());
  vector<double> rhs(h.n);
  for (int i = 0; i < h.n; ++i)
    rhs[i] = std::sin(0.31 * (i + 1)) + 0.5 * std::cos(0.17 * i);
  Buffer<double> values;
  values.init(h.val);

  struct PolyCase {
    int stride;
    int offset;
    float scale;
    bool fp8;
  };
  for (PolyCase cfg : {PolyCase{2, 0, 1.25f, false}, PolyCase{4, 1, 1.25f, false},
                       PolyCase{4, 1, 1.25f, true}}) {
    Engine engine(h, rhs, true, true, false, cfg.fp8, cfg.stride, cfg.scale,
                  cfg.offset, true);
    if (engine.ls.size() < 4 || engine.ls[2]->dense.p != nullptr)
      throw std::runtime_error("oracle mesh must have >= 4 levels with nonterminal level 2");
    reset(engine.ls, Int4, values.p, cfg.fp8);
    engine.capture(Int4);
    CHECK(cudaStreamSynchronize(stream));

    vector<double> B0 = build_dense_cycle_oracle(
        engine.ls, 0, Int4, cfg.fp8, cfg.stride, cfg.offset, cfg.scale);
    vector<double> expected(h.n, 0.0);
    for (int i = 0; i < h.n; ++i)
      for (int c = 0; c < h.n; ++c)
        expected[i] += B0[i * h.n + c] * double(float(rhs[c]));

    // Check both eager apply<float> and captured CUDA-Graph launch against 2*t*B - t^2*B*A_c*B
    CHECK(cudaMemcpyAsync(engine.res.p, rhs.data(), h.n * sizeof(double),
                          cudaMemcpyHostToDevice, stream));
    engine.apply<float>(Int4);
    CHECK(cudaStreamSynchronize(stream));
    auto z_eager = download(engine.z, h.n);

    CHECK(cudaMemsetAsync(engine.z.p, 0, h.n * sizeof(double), stream));
    CHECK(cudaGraphLaunch(engine.graphs[Int4], stream));
    CHECK(cudaStreamSynchronize(stream));
    auto z_graph = download(engine.z, h.n);

    double err_eager2 = 0.0, err_graph2 = 0.0, ref2 = 0.0;
    for (int i = 0; i < h.n; ++i) {
      double de = z_eager[i] - expected[i];
      double dg = z_graph[i] - expected[i];
      err_eager2 += de * de;
      err_graph2 += dg * dg;
      ref2 += expected[i] * expected[i];
    }
    if (!(std::sqrt(err_eager2 / ref2) < 1e-5 &&
          std::sqrt(err_graph2 / ref2) < 1e-5))
      throw std::runtime_error("GPU boosted cycle disagreed with dense 2*t*B - t^2*B*A_c*B oracle");
  }
  puts("Dense polynomial boost oracle (2*t*B - t^2*B*A_c*B) matched GPU eager & graph cycles.");
}

void test_factored_refresh_and_guards() {
  // 1) Toggle use_factored true -> false -> true across numeric refreshes and verify
  //    CUDA-Graph re-capture matches eager execution at every step.
  constexpr int n = 96;
  HostCSR h;
  h.n = n;
  h.rp.push_back(0);
  for (int i = 0; i < n; ++i) {
    if (i > 0) {
      h.ci.push_back(i - 1);
      h.val.push_back(-0.25);
    }
    h.ci.push_back(i);
    h.val.push_back(4.0);
    if (i + 1 < n) {
      h.ci.push_back(i + 1);
      h.val.push_back(-0.25);
    }
    h.rp.push_back(static_cast<int>(h.ci.size()));
  }
  h.nnz = static_cast<int>(h.ci.size());

  HostCSR tiny_coupling = h;
  for (int i = 0; i < n; ++i)
    for (int k = tiny_coupling.rp[i]; k < tiny_coupling.rp[i + 1]; ++k)
      if ((i == 0 && tiny_coupling.ci[k] == 1) ||
          (i == 1 && tiny_coupling.ci[k] == 0))
        tiny_coupling.val[k] = -1e-20;

  vector<double> rhs(n);
  for (int i = 0; i < n; ++i)
    rhs[i] = std::cos(0.23 * i) + 1.0;
  Buffer<double> v_norm, v_tiny;
  v_norm.init(h.val);
  v_tiny.init(tiny_coupling.val);

  Engine graph_eng(h, rhs, true, true);
  Engine eager_eng(h, rhs, true, false);
  const bool expected_factored[3] = {true, false, true};
  for (int pass = 0; pass < 3; ++pass) {
    const double *vals = (pass == 1) ? v_tiny.p : v_norm.p;
    const HostCSR &cur = (pass == 1) ? tiny_coupling : h;
    reset(graph_eng.ls, Int4, vals);
    graph_eng.capture(Int4);
    reset(eager_eng.ls, Int4, vals);
    if (graph_eng.ls[0]->use_factored != expected_factored[pass])
      throw std::runtime_error("use_factored did not toggle as expected across refresh");

    CHECK(cudaMemcpyAsync(graph_eng.res.p, rhs.data(), n * sizeof(double),
                          cudaMemcpyHostToDevice, stream));
    CHECK(cudaGraphLaunch(graph_eng.graphs[Int4], stream));
    CHECK(cudaMemcpyAsync(eager_eng.res.p, rhs.data(), n * sizeof(double),
                          cudaMemcpyHostToDevice, stream));
    eager_eng.apply<float>(Int4);
    CHECK(cudaStreamSynchronize(stream));

    auto zg = download(graph_eng.z, n);
    auto ze = download(eager_eng.z, n);
    double diff2 = 0.0, ref2 = 0.0;
    for (int i = 0; i < n; ++i) {
      double d = zg[i] - ze[i];
      diff2 += d * d;
      ref2 += ze[i] * ze[i];
    }
    if (!(std::sqrt(diff2 / ref2) < 1e-6))
      throw std::runtime_error("CUDA-Graph replay diverged from eager apply after use_factored toggle");

    Result rr = graph_eng.solve(Int4);
    auto x = download(graph_eng.solution, n);
    if (!(host_relative_residual(cur, rhs, x) <= 1e-8) || rr.trigger != "none")
      throw std::runtime_error("CUDA-Graph solve failed after use_factored toggle");
  }

  // 2) Verify smooth<float> exponent safety on standalone_amg (r = 2^127, scale = 0.25 and r = 2^-126, scale = 4.0)
  {
    constexpr int m = 66;
    auto make_block_csr = [&](double a00, double a02, double a11, double a12, double a22) {
      HostCSR c;
      c.n = m;
      c.rp.push_back(0);
      auto push = [&](int col, double val) { c.ci.push_back(col); c.val.push_back(val); };
      push(0, a00); push(2, a02); c.rp.push_back(c.ci.size());
      push(1, a11); push(2, a12); c.rp.push_back(c.ci.size());
      push(0, a02); push(1, a12); push(2, a22); c.rp.push_back(c.ci.size());
      for (int i = 3; i < m; ++i) {
        push(i, a00);
        c.rp.push_back(c.ci.size());
      }
      c.nnz = static_cast<int>(c.ci.size());
      return c;
    };
    vector<double> dummy_b(m, 1.0);

    // Upper extreme: G_2 = [0.25, 0.25, 0.25], r = 2^127
    HostCSR hi_csr = make_block_csr(16.0, -16.0, 16.0, -16.0, 48.0);
    Buffer<double> hi_vals;
    hi_vals.init(hi_csr.val);
    Engine hi_eng(hi_csr, dummy_b, true, false);
    reset(hi_eng.ls, Int4, hi_vals.p);
    if (!hi_eng.ls[0]->use_factored)
      throw std::runtime_error("upper extreme test did not use factored smooth<float>");
    vector<float> r_hi(m, 0.0f);
    r_hi[0] = r_hi[1] = r_hi[2] = std::ldexp(1.0f, 127);
    auto &wf_hi = work<float>(*hi_eng.ls[0]);
    CHECK(cudaMemcpyAsync(wf_hi.rhs.p, r_hi.data(), m * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    smooth<float>(*hi_eng.ls[0], Int4, false);
    CHECK(cudaStreamSynchronize(stream));
    auto x_hi = download(wf_hi.x, m);
    float exp_hi2 = 0.65f * 3.0f * std::ldexp(1.0f, 123);
    if (!std::isfinite(x_hi[2]) || std::fabs(x_hi[2] - exp_hi2) / exp_hi2 > 1e-5f)
      throw std::runtime_error("standalone smooth<float> overflowed at r=2^127, scale=0.25");

    // Lower extreme: G_2 = [4, 4, 4], r = 2^-126
    HostCSR lo_csr = make_block_csr(0.0625, -0.0625, 0.0625, -0.0625, 0.1875);
    Buffer<double> lo_vals;
    lo_vals.init(lo_csr.val);
    Engine lo_eng(lo_csr, dummy_b, true, false);
    reset(lo_eng.ls, Int4, lo_vals.p);
    if (!lo_eng.ls[0]->use_factored)
      throw std::runtime_error("lower extreme test did not use factored smooth<float>");
    vector<float> r_lo(m, 0.0f);
    r_lo[0] = r_lo[1] = r_lo[2] = std::ldexp(1.0f, -126);
    auto &wf_lo = work<float>(*lo_eng.ls[0]);
    CHECK(cudaMemcpyAsync(wf_lo.rhs.p, r_lo.data(), m * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    smooth<float>(*lo_eng.ls[0], Int4, false);
    CHECK(cudaStreamSynchronize(stream));
    auto x_lo = download(wf_lo.x, m);
    float exp_lo2 = 0.65f * 3.0f * std::ldexp(1.0f, -122);
    if (!std::isfinite(x_lo[2]) || std::fabs(x_lo[2] - exp_lo2) / exp_lo2 > 1e-5f)
      throw std::runtime_error("standalone smooth<float> underflowed at r=2^-126, scale=4.0");
  }

  // 3) Verify bounded diagonal scan in numeric() rejects a CSR row missing its diagonal cleanly
  {
    constexpr int m = 66;
    HostCSR bad;
    bad.n = m;
    bad.rp.push_back(0);
    // Row 0 has only column 1 (no diagonal!), row 1 has {0, 1}, rows 2..65 have {i}
    bad.ci.push_back(1); bad.val.push_back(-0.1); bad.rp.push_back(bad.ci.size());
    bad.ci.push_back(0); bad.val.push_back(-0.1);
    bad.ci.push_back(1); bad.val.push_back(2.0); bad.rp.push_back(bad.ci.size());
    for (int i = 2; i < m; ++i) {
      bad.ci.push_back(i);
      bad.val.push_back(2.0);
      bad.rp.push_back(bad.ci.size());
    }
    bad.nnz = static_cast<int>(bad.ci.size());
    vector<double> b_bad(m, 1.0);
    Buffer<double> v_bad;
    v_bad.init(bad.val);
    Level lvl;
    lvl.n = bad.n;
    lvl.nz = bad.nnz;
    lvl.rp.init(bad.rp);
    lvl.ci.init(bad.ci);
    lvl.a.init(bad.val);
    lvl.af.alloc(bad.nnz);
    lvl.l1.alloc(m);
    lvl.inv.alloc(m);
    lvl.diag.alloc(m);
    lvl.scale.alloc(m);
    lvl.packed.alloc((size_t(bad.nnz) + m + 1) / 2);
    lvl.failures.alloc(1);
    lvl.invalid.alloc(1);
    CHECK(cudaMemsetAsync(lvl.packed.p, 0, lvl.packed.bytes(), stream));
    CHECK(cudaMemsetAsync(lvl.failures.p, 0, sizeof(unsigned), stream));
    CHECK(cudaMemsetAsync(lvl.invalid.p, 0, sizeof(unsigned), stream));
    numeric<3>(lvl, Int4);
    unsigned inv_flag = 0;
    CHECK(cudaMemcpyAsync(&inv_flag, lvl.invalid.p, sizeof(unsigned),
                          cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));
    if ((inv_flag & 1u) == 0)
      throw std::runtime_error("numeric() failed to flag missing diagonal row");
  }

  // 4) Pass 2 overflow witness (G=[[1,0],[0.25,2]], r=[0,2^126]) in standalone smooth<float>
  //    (both add=false and add=true) AND CUDA-Graph replay with ordinary + extreme residuals.
  {
    constexpr int m = 66;
    HostCSR c;
    c.n = m;
    c.rp.push_back(0);
    auto push = [&](int col, double val) { c.ci.push_back(col); c.val.push_back(val); };
    // G = [[1,0],[0.25,2]] => A = L L^T = [[1,-0.125],[-0.125,0.265625]]
    push(0, 1.0); push(1, -0.125); c.rp.push_back(c.ci.size());
    push(0, -0.125); push(1, 0.265625); c.rp.push_back(c.ci.size());
    for (int i = 2; i < m; ++i) {
      push(i, 1.0);
      c.rp.push_back(c.ci.size());
    }
    c.nnz = static_cast<int>(c.ci.size());
    vector<double> dummy_b(m, 1.0);
    Buffer<double> c_vals;
    c_vals.init(c.val);
    Engine eng(c, dummy_b, true, true);
    reset(eng.ls, Int4, c_vals.p);
    if (!eng.ls[0]->use_factored)
      throw std::runtime_error("Pass 2 overflow witness did not use factored smooth<float>");

    vector<float> r_ovf(m, 0.0f);
    r_ovf[1] = std::ldexp(1.0f, 126);
    auto &wf = work<float>(*eng.ls[0]);
    CHECK(cudaMemcpyAsync(wf.rhs.p, r_ovf.data(), m * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    smooth<float>(*eng.ls[0], Int4, false);
    CHECK(cudaStreamSynchronize(stream));
    auto x_ovf = download(wf.x, m);
    float exp0 = float(0.65 * std::ldexp(1.0, 125));
    float exp1 = float(0.65 * std::ldexp(1.0, 128));
    if (!std::isfinite(x_ovf[0]) || !std::isfinite(x_ovf[1]) ||
        std::fabs(x_ovf[0] - exp0) / exp0 > 1e-5f ||
        std::fabs(x_ovf[1] - exp1) / exp1 > 1e-5f)
      throw std::runtime_error("standalone smooth<float> Pass 2 overflowed before damping (add=false)");

    // add = true starting from [-0.65*2^125, -0.65*2^127]
    vector<float> x_init(m, 0.0f);
    x_init[0] = -exp0;
    x_init[1] = -float(0.65 * std::ldexp(1.0, 127));
    CHECK(cudaMemcpyAsync(wf.x.p, x_init.data(), m * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(wf.res.p, r_ovf.data(), m * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    smooth<float>(*eng.ls[0], Int4, true);
    CHECK(cudaStreamSynchronize(stream));
    auto x_acc = download(wf.x, m);
    float exp_acc1 = float(0.65 * std::ldexp(1.0, 127));
    if (!std::isfinite(x_acc[0]) || !std::isfinite(x_acc[1]) ||
        std::fabs(x_acc[0]) > 1e-5f * exp0 ||
        std::fabs(x_acc[1] - exp_acc1) / exp_acc1 > 1e-5f)
      throw std::runtime_error("standalone smooth<float> Pass 2 overflowed before damping (add=true)");

    // Same captured CUDA Graph replayed with an ordinary residual AND then an extreme residual,
    // comparing both against apply<double> oracle!
    // Use scale = 4.0 block so r = 2^125 triggers Pass 1 oor_flag -> double fallback inside CUDA Graph.
    HostCSR g_csr;
    g_csr.n = m;
    g_csr.rp.push_back(0);
    auto gpush = [&](int col, double val) { g_csr.ci.push_back(col); g_csr.val.push_back(val); };
    // G_2 = [4, 4, 4] => A_block = [[0.0625, 0, -0.0625], [0, 0.0625, -0.0625], [-0.0625, -0.0625, 0.1875]]
    gpush(0, 0.0625); gpush(2, -0.0625); g_csr.rp.push_back(g_csr.ci.size());
    gpush(1, 0.0625); gpush(2, -0.0625); g_csr.rp.push_back(g_csr.ci.size());
    gpush(0, -0.0625); gpush(1, -0.0625); gpush(2, 0.1875); g_csr.rp.push_back(g_csr.ci.size());
    for (int i = 3; i < m; ++i) {
      gpush(i, 0.0625);
      g_csr.rp.push_back(g_csr.ci.size());
    }
    g_csr.nnz = static_cast<int>(g_csr.ci.size());
    Buffer<double> g_vals;
    g_vals.init(g_csr.val);
    Engine g_eng(g_csr, dummy_b, true, true);
    reset(g_eng.ls, Int4, g_vals.p);
    g_eng.capture(Int4);
    if (!g_eng.ls[0]->use_factored)
      throw std::runtime_error("CUDA-Graph fallback test did not use factored smooth<float>");

    for (double scale_r : {1.0, std::ldexp(1.0, -126)}) {
      vector<double> r_vec(m, 0.0);
      r_vec[0] = scale_r; r_vec[1] = -0.5 * scale_r; r_vec[2] = scale_r;
      CHECK(cudaMemcpyAsync(g_eng.res.p, r_vec.data(), m * sizeof(double),
                            cudaMemcpyHostToDevice, stream));
      CHECK(cudaGraphLaunch(g_eng.graphs[Int4], stream));
      CHECK(cudaStreamSynchronize(stream));
      auto z_graph = download(g_eng.z, m);

      CHECK(cudaMemcpyAsync(g_eng.res.p, r_vec.data(), m * sizeof(double),
                            cudaMemcpyHostToDevice, stream));
      g_eng.apply<double>(Int4);
      CHECK(cudaStreamSynchronize(stream));
      auto z_ref = download(g_eng.z, m);

      double diff2 = 0.0, ref2 = 0.0;
      for (int i = 0; i < m; ++i) {
        if (!std::isfinite(z_graph[i]))
          throw std::runtime_error("CUDA-Graph replay produced nonfinite output");
        double dlt = z_graph[i] - z_ref[i];
        diff2 += dlt * dlt;
        ref2 += z_ref[i] * z_ref[i];
      }
      if (!(std::sqrt(diff2 / std::max(1e-300, ref2)) < 1e-5))
        throw std::runtime_error("CUDA-Graph replay diverged from double oracle on ordinary/extreme residual");
    }
  }
  puts("Factored refresh toggle, smooth<float> exponent extremes, Pass 2 overflow, CUDA-Graph fallback, and bounded diagonal scan passed.");
}

void test_fp8_sddm_counterexamples_and_neumaier() {
  // Build 132-DOF SPD matrices whose pair-injection coarse level 1 has nc = 66 > 64:
  // (a) Counterexample A: coarse 3-node path {0,1,2} with edges -2^-18 and -1, unit grounding
  //     only at node 0. E5M2 rounds -2^-18 to 0, leaving component {1,2} ungrounded/singular.
  //     reset(..., true) must detect this and cleanly fall back to FP32 af on level 1.
  // (b) Counterexample B: coarse 2-node positive off-diagonal block [[1.14, 1.13], [1.13, 1.14]].
  //     reset(..., true) must reject positive off-diagonals and fall back to FP32 af on level 1.
  // (c) Neumaier nearly-ungrounded multi-order row: large -1.0 plus tiny -2^-26 terms in
  //     disparate order must preserve strict grounding in FP8+FP32.
  constexpr int nc_target = 66;
  constexpr int n = 2 * nc_target;
  auto build_pair_base = [&]() {
    std::vector<std::vector<std::pair<int, double>>> adj(n);
    auto add_sym = [&](int r, int c, double v) {
      adj[r].push_back({c, v});
      if (r != c) adj[c].push_back({r, v});
    };
    for (int u = 0; u < nc_target; ++u) {
      int i0 = 2 * u, i1 = 2 * u + 1;
      add_sym(i0, i0, 20.0);
      add_sym(i1, i1, 20.0);
      add_sym(i0, i1, -9.0);
    }
    return adj;
  };
  auto finalize_csr = [&](std::vector<std::vector<std::pair<int, double>>> &adj) {
    HostCSR h;
    h.n = n;
    h.rp.push_back(0);
    for (int i = 0; i < n; ++i) {
      auto &row = adj[i];
      std::sort(row.begin(), row.end(),
                [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
      for (size_t k = 0; k < row.size();) {
        int col = row[k].first;
        double s = 0.0;
        while (k < row.size() && row[k].first == col) s += row[k++].second;
        h.ci.push_back(col);
        h.val.push_back(s);
      }
      h.rp.push_back(static_cast<int>(h.ci.size()));
    }
    h.nnz = static_cast<int>(h.ci.size());
    return h;
  };
  vector<double> rhs(n);
  for (int i = 0; i < n; ++i) rhs[i] = 1.0 + 0.03 * ((i % 5) - 2);

  // (a) Counterexample A: weak edge 2^-18 disconnects ungrounded {1, 2}
  {
    auto adj = build_pair_base();
    const double w = std::ldexp(1.0, -18);
    // Replace pairs 0, 1, 2 so coarse 3x3 block is:
    //   [[1 + w, -w, 0], [-w, 1 + w, -1], [0, -1, 1]]
    // With P_{2u,u}=P_{2u+1,u}=1, a pair with fine diag d and internal -1 gives coarse diag 2d - 2.
    auto set_pair = [&](int u, double coarse_d) {
      int i0 = 2 * u, i1 = 2 * u + 1;
      adj[i0].clear(); adj[i1].clear();
      double fd = 0.5 * coarse_d + 1.0;
      adj[i0].push_back({i0, fd}); adj[i0].push_back({i1, -1.0});
      adj[i1].push_back({i0, -1.0}); adj[i1].push_back({i1, fd});
    };
    set_pair(0, 1.0 + w);
    set_pair(1, 1.0 + w);
    set_pair(2, 1.0);
    adj[0].push_back({2, -w}); adj[2].push_back({0, -w});
    adj[2].push_back({4, -0.5}); adj[4].push_back({2, -0.5});
    adj[3].push_back({5, -0.5}); adj[5].push_back({3, -0.5});
    HostCSR h = finalize_csr(adj);
    Buffer<double> vals;
    vals.init(h.val);
    Engine eng(h, rhs, false, true, false, true);
    reset(eng.ls, Int4, vals.p, true);
    if (eng.ls[1]->use_fp8_level)
      throw std::runtime_error("FP8 coarse guard failed to fall back on 2^-18 disconnected component");
    eng.capture(Int4);
    Result r = eng.solve(Int4);
    auto x = download(eng.solution, n);
    if (!(host_relative_residual(h, rhs, x) <= 1e-8))
      throw std::runtime_error("FP32 fallback solve failed on Counterexample A");
  }

  // (b) Counterexample B: positive off-diagonal block [[1.14, 1.13], [1.13, 1.14]]
  {
    auto adj = build_pair_base();
    auto set_pair = [&](int u, double coarse_d) {
      int i0 = 2 * u, i1 = 2 * u + 1;
      adj[i0].clear(); adj[i1].clear();
      double fd = 0.5 * coarse_d + 1.0;
      adj[i0].push_back({i0, fd}); adj[i0].push_back({i1, -1.0});
      adj[i1].push_back({i0, -1.0}); adj[i1].push_back({i1, fd});
    };
    set_pair(0, 1.14);
    set_pair(1, 1.14);
    // Fine positive off-diagonals summing to +1.13 on coarse (0, 1)
    adj[0].push_back({2, 0.565}); adj[2].push_back({0, 0.565});
    adj[1].push_back({3, 0.565}); adj[3].push_back({1, 0.565});
    HostCSR h = finalize_csr(adj);
    Buffer<double> vals;
    vals.init(h.val);
    Engine eng(h, rhs, false, true, false, true);
    reset(eng.ls, Int4, vals.p, true);
    if (eng.ls[1]->use_fp8_level)
      throw std::runtime_error("FP8 coarse guard failed to fall back on positive off-diagonal block");
    eng.capture(Int4);
    Result r = eng.solve(Int4);
    auto x = download(eng.solution, n);
    if (!(host_relative_residual(h, rhs, x) <= 1e-8))
      throw std::runtime_error("FP32 fallback solve failed on Counterexample B");
  }

  // (c) Neumaier multi-order nearly-ungrounded row:
  //     coarse node 0 connects to node 1 (-1.0), node 2 (-2^-14), node 3 (-2^-14),
  //     with exact grounding 2^-52 so sign error in Kahan compensation would lose grounding!
  {
    auto adj = build_pair_base();
    const double eps14 = std::ldexp(1.0, -14);
    const double eps50 = std::ldexp(1.0, -50);
    auto set_pair = [&](int u, double coarse_d) {
      int i0 = 2 * u, i1 = 2 * u + 1;
      adj[i0].clear(); adj[i1].clear();
      double fd = 0.5 * coarse_d + 2.0;
      adj[i0].push_back({i0, fd}); adj[i0].push_back({i1, -2.0});
      adj[i1].push_back({i0, -2.0}); adj[i1].push_back({i1, fd});
    };
    set_pair(0, 1.0 + 2.0 * eps14 + eps50);
    set_pair(1, 2.0);
    set_pair(2, 2.0);
    set_pair(3, 2.0);
    // Connect coarse 0 to 1 (-1.0), 2 (-eps14), 3 (-eps14)
    adj[0].push_back({2, -0.5}); adj[2].push_back({0, -0.5});
    adj[1].push_back({3, -0.5}); adj[3].push_back({1, -0.5});
    adj[0].push_back({4, -eps14}); adj[4].push_back({0, -eps14});
    adj[0].push_back({6, -eps14}); adj[6].push_back({0, -eps14});
    HostCSR h = finalize_csr(adj);
    Buffer<double> vals;
    vals.init(h.val);
    Engine eng(h, rhs, false, true, false, true);
    reset(eng.ls, Int4, vals.p, true);
    if (!eng.ls[1]->use_fp8_level)
      throw std::runtime_error("Neumaier multi-order grounded SDDM row unexpectedly fell back");
    auto ac_diag = download(eng.ls[1]->ac_diag, eng.ls[1]->n);
    if (!(double(ac_diag[0]) - (1.0 + 2.0 * eps14) > 0.0))
      throw std::runtime_error("Neumaier multi-order row lost strict positive grounding");
  }

  // (d) 10^6-scaled matrix saturation check (|a_sym| > 57344 must trigger FP32 fallback)
  {
    auto adj = build_pair_base();
    for (int i = 0; i < n; ++i)
      for (auto &kv : adj[i]) kv.second *= 1e6;
    adj[0].push_back({2, -1e6}); adj[2].push_back({0, -1e6});
    HostCSR h = finalize_csr(adj);
    Buffer<double> vals;
    vals.init(h.val);
    Engine eng(h, rhs, false, true, false, true);
    eng.reset(Int4, vals.p);
    if (eng.ls[1]->use_fp8_level)
      throw std::runtime_error("FP8 coarse guard failed to fall back on 10^6-scaled saturation (>57344)");
    eng.capture(Int4);
    Result r = eng.solve(Int4);
    auto x = download(eng.solution, n);
    if (!(host_relative_residual(h, rhs, x) <= 1e-8) || r.trigger != "none")
      throw std::runtime_error("FP32 fallback solve failed on 10^6-scaled saturation test");
  }

  // (e) Subnormal E5M2 check (0 < |a_sym| < 2^-14 must trigger FP32 fallback)
  {
    auto adj = build_pair_base();
    const double sub = std::ldexp(1.0, -15);
    adj[0].push_back({2, -sub}); adj[2].push_back({0, -sub});
    HostCSR h = finalize_csr(adj);
    Buffer<double> vals;
    vals.init(h.val);
    Engine eng(h, rhs, false, true, false, true);
    eng.reset(Int4, vals.p);
    if (eng.ls[1]->use_fp8_level)
      throw std::runtime_error("FP8 coarse guard failed to fall back on 2^-15 subnormal off-diagonal");
    eng.capture(Int4);
    Result r = eng.solve(Int4);
    auto x = download(eng.solution, n);
    if (!(host_relative_residual(h, rhs, x) <= 1e-8) || r.trigger != "none")
      throw std::runtime_error("FP32 fallback solve failed on 2^-15 subnormal test");
  }
  // (f) 3-node star with sole 2^-53 ground + disconnected identity block (>64 rows):
  //     Center diag = 2, edges = -1, -(1 - 2^-53), leaf diags = 1, 1 - 2^-53.
  //     Exact center ground is 2^-53; summing diagonal + off-diagonals together preserves
  //     strict positive grounding in the repaired FP32 diagonal and unconditional component check.
  {
    auto adj = build_pair_base();
    const double eps53 = std::ldexp(1.0, -53);
    const double w = 1.0 - eps53;
    auto set_pair = [&](int u, double coarse_d) {
      int i0 = 2 * u, i1 = 2 * u + 1;
      adj[i0].clear(); adj[i1].clear();
      double fd = 0.5 * coarse_d + 2.0;
      adj[i0].push_back({i0, fd}); adj[i0].push_back({i1, -2.0});
      adj[i1].push_back({i0, -2.0}); adj[i1].push_back({i1, fd});
    };
    set_pair(0, 2.0);
    set_pair(1, 1.0);
    set_pair(2, w);
    adj[0].push_back({2, -1.0}); adj[2].push_back({0, -1.0});
    adj[0].push_back({4, -w});   adj[4].push_back({0, -w});
    HostCSR h = finalize_csr(adj);
    Buffer<double> vals;
    vals.init(h.val);
    Engine eng(h, rhs, false, true, false, true);
    eng.reset(Int4, vals.p);
    if (!eng.ls[1]->use_fp8_level)
      throw std::runtime_error("3-node 2^-53 grounded star unexpectedly fell back from FP8");
    auto ac_diag = download(eng.ls[1]->ac_diag, eng.ls[1]->n);
    auto ac_fp8 = download(eng.ls[1]->ac_fp8, eng.ls[1]->nz);
    auto rp1 = download(eng.ls[1]->rp, eng.ls[1]->n + 1);
    auto ci1 = download(eng.ls[1]->ci, eng.ls[1]->nz);
    // Evaluate decoded quadratic form v^T A_tilde v on the star constant vector v = (1, 1, 1, 0, ...):
    // must be strictly positive (non-singular), proving the sole 2^-53 ground was preserved.
    double quad = 0.0;
    for (int i = 0; i < 3; ++i) {
      for (int k = rp1[i]; k < rp1[i + 1]; ++k) {
        int c = ci1[k];
        if (c >= 3) continue;
        if (c == i) quad += static_cast<double>(ac_diag[i]);
        else {
          __half_raw hr = __nv_cvt_fp8_to_halfraw(
              static_cast<__nv_fp8_storage_t>(ac_fp8[k]), __NV_E5M2);
          quad += static_cast<double>(__half2float(__half(hr)));
        }
      }
    }
    if (!(double(ac_diag[0]) - 2.0 > 0.0) || !(quad > 0.0))
      throw std::runtime_error("FP8 coarse guard lost sole 2^-53 ground on 3-node star component");
  }
  puts("FP8 SDDM counterexamples (2^-18 disconnect, positive off-diag, 10^6 saturation, 2^-15 subnormal, 2^-53 star) and Neumaier test passed.");
}

void test_fp8_graph_reallocation_safety() {
  // Toggle level 1 between FP8 -> FP32 -> FP8 across methods (Int4 & Half) on
  // the same Engine and verify CUDA-Graph cache invalidation prevents dangling
  // device-pointer replays after buffer free/reallocation.
  constexpr int nc_target = 66;
  constexpr int n = 2 * nc_target;
  auto make_case = [&](double scale_factor) {
    std::vector<std::vector<std::pair<int, double>>> adj(n);
    auto add_sym = [&](int r, int c, double v) {
      adj[r].push_back({c, v});
      if (r != c) adj[c].push_back({r, v});
    };
    for (int u = 0; u < nc_target; ++u) {
      int i0 = 2 * u, i1 = 2 * u + 1;
      add_sym(i0, i0, 20.0 * scale_factor);
      add_sym(i1, i1, 20.0 * scale_factor);
      add_sym(i0, i1, -9.0 * scale_factor);
    }
    for (int u = 0; u + 1 < nc_target; ++u)
      add_sym(2 * u, 2 * (u + 1), -0.5 * scale_factor);
    HostCSR h;
    h.n = n;
    h.rp.push_back(0);
    for (int i = 0; i < n; ++i) {
      auto &row = adj[i];
      std::sort(row.begin(), row.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });
      for (const auto &kv : row) {
        h.ci.push_back(kv.first);
        h.val.push_back(kv.second);
      }
      h.rp.push_back(static_cast<int>(h.ci.size()));
    }
    h.nnz = static_cast<int>(h.ci.size());
    return h;
  };

  HostCSR h_norm = make_case(1.0);
  HostCSR h_sat = make_case(1e6);
  vector<double> rhs(n);
  for (int i = 0; i < n; ++i) rhs[i] = std::sin(0.19 * (i + 1)) + 0.5;

  Buffer<double> v_norm, v_sat;
  v_norm.init(h_norm.val);
  v_sat.init(h_sat.val);

  Engine eng(h_norm, rhs, false, true, false, true);
  // 1) Start in FP8 and capture both Int4 and Half graphs.
  eng.reset(Int4, v_norm.p);
  if (!eng.ls[1]->use_fp8_level)
    throw std::runtime_error("graph reallocation test expected initial FP8 level");
  eng.capture(Int4);
  eng.reset(Half, v_norm.p);
  eng.capture(Half);
  const uint64_t gen0 = eng.ls[1]->alloc_generation;

  // 2) Switch to 10^6-scaled matrix: level 1 frees ac_fp8/ac_diag and allocates af.
  eng.reset(Int4, v_sat.p);
  if (eng.ls[1]->use_fp8_level || eng.ls[1]->alloc_generation == gen0)
    throw std::runtime_error("graph reallocation test expected FP32 fallback and generation bump");
  if (eng.graphs[Half] != nullptr)
    throw std::runtime_error("stale Half CUDA graph was not invalidated after FP8->FP32 reallocation");
  Result r_sat = eng.solve(Int4);
  auto x_sat = download(eng.solution, n);
  if (!(host_relative_residual(h_sat, rhs, x_sat) <= 1e-8) || r_sat.trigger != "none")
    throw std::runtime_error("FP32 fallback graph solve failed");
  const uint64_t gen1 = eng.ls[1]->alloc_generation;

  // 3) Switch back to normal SDDM matrix: level 1 frees af and reallocates ac_fp8/ac_diag.
  eng.reset(Half, v_norm.p);
  if (!eng.ls[1]->use_fp8_level || eng.ls[1]->alloc_generation == gen1)
    throw std::runtime_error("graph reallocation test expected FP8 restoration and generation bump");
  if (eng.graphs[Int4] != nullptr)
    throw std::runtime_error("stale Int4 CUDA graph was not invalidated after FP32->FP8 reallocation");
  Result r_half = eng.solve(Half);
  auto x_half = download(eng.solution, n);
  if (!(host_relative_residual(h_norm, rhs, x_half) <= 1e-8) || r_half.trigger != "none")
    throw std::runtime_error("Half graph solve failed after FP8->FP32->FP8 toggle");

  eng.reset(Int4, v_norm.p);
  Result r_int4 = eng.solve(Int4);
  auto x_int4 = download(eng.solution, n);
  if (!(host_relative_residual(h_norm, rhs, x_int4) <= 1e-8) || r_int4.trigger != "none")
    throw std::runtime_error("Int4 graph solve failed after FP8->FP32->FP8 toggle");
  puts("CUDA-Graph FP8->FP32->FP8 reallocation invalidation and multi-method replay passed.");
}

void test_terminal_tiny_cholesky_and_poly_boundaries() {
  // 1) Terminal dense Cholesky solve on A = [1e-310]:
  //    (a) Direct cycle<double> with b = [1e-310] must give exact x = [1.0] (since
  //        L = [1e-155] avoids forming A^{-1} = [1e310] which overflows double).
  //    (b) Full Engine::solve with b = [1e-150] (nonzero b2 = 1e-300) must converge to x = [1e160].
  {
    HostCSR tiny;
    tiny.n = 1; tiny.nnz = 1;
    tiny.rp = {0, 1}; tiny.ci = {0}; tiny.val = {1e-310};
    vector<double> rhs_tiny{1e-310};
    Buffer<double> vals;
    vals.init(tiny.val);
    Engine eng(tiny, rhs_tiny, true, false);
    reset(eng.ls, Int4, vals.p);
    CHECK(cudaMemcpyAsync(eng.res.p, rhs_tiny.data(), sizeof(double),
                          cudaMemcpyHostToDevice, stream));
    eng.apply<double>(Int4);
    CHECK(cudaStreamSynchronize(stream));
    auto z_tiny = download(eng.z, 1);
    if (!(std::isfinite(z_tiny[0]) && std::fabs(z_tiny[0] - 1.0) < 1e-12))
      throw std::runtime_error("terminal dense Cholesky apply<double> failed on A=[1e-310], b=[1e-310]");

    vector<double> rhs_sol{1e-150};
    Engine eng_sol(tiny, rhs_sol, true, true);
    reset(eng_sol.ls, Int4, vals.p);
    eng_sol.capture(Int4);
    (void)eng_sol.solve(Int4);
    auto x_sol = download(eng_sol.solution, 1);
    if (!(std::isfinite(x_sol[0]) && std::fabs(x_sol[0] / 1e160 - 1.0) < 1e-12 &&
          host_relative_residual(tiny, rhs_sol, x_sol) <= 1e-8))
      throw std::runtime_error("terminal dense Cholesky solve failed on A=[1e-310], b=[1e-150]");
  }

  // 2) Polynomial boost boundary tests:
  //    On n = 130 diagonal A = I with pair injection (130 -> 65 -> 33),
  //    level 1 (n=65, A1 = 2I) is exact on range(P1) (including the constant vector 1):
  //    - poly_scale = 2.0 yields ZERO coarse correction on constant mode 1.
  //    - poly_scale = 3.0 on a high-degree (damping=0.35) level 0 yields NEGATIVE definite
  //      correction (r^T z < 0), triggering the PCG rz guard + FP64 Jacobi recovery!
  {
    constexpr int n = 130;
    HostCSR diag_csr;
    diag_csr.n = n;
    diag_csr.rp.push_back(0);
    for (int i = 0; i < n; ++i) {
      diag_csr.ci.push_back(i);
      diag_csr.val.push_back(1.0);
      diag_csr.rp.push_back(i + 1);
    }
    diag_csr.nnz = n;
    vector<double> ones(n, 1.0);
    Buffer<double> vals;
    vals.init(diag_csr.val);

    // Default Engine must reject poly_scale >= 2.0f unless unsafe_poly is explicitly enabled.
    for (float unsafe_scale : {2.0f, 3.0f}) {
      bool rejected = false;
      try {
        Engine bad_eng(diag_csr, ones, false, false, false, false, 1, unsafe_scale, 0);
      } catch (const std::runtime_error &) {
        rejected = true;
      }
      if (!rejected)
        throw std::runtime_error("Engine accepted poly_scale >= 2.0f without unsafe_poly opt-in");
    }

    Engine eng2(diag_csr, ones, false, false, false, false, 1, 2.0f, 0, false, true);
    reset(eng2.ls, Int4, vals.p);
    CHECK(cudaMemcpyAsync(eng2.res.p, ones.data(), n * sizeof(double),
                          cudaMemcpyHostToDevice, stream));
    eng2.apply<float>(Int4);
    CHECK(cudaStreamSynchronize(stream));
    auto z2 = download(eng2.z, n);
    // Pure smoother (2*0.65 - 0.65^2) = 0.8775 with ZERO coarse correction from poly_scale=2.0
    for (int i = 0; i < n; ++i) {
      if (!(std::fabs(z2[i] - 0.8775) < 1e-5))
        throw std::runtime_error("poly_scale=2.0 did not give zero coarse correction on exact child mode");
    }

    // With level 0 damping = 0.35 (max_lower > 16), 2*0.35 - 0.35^2 = 0.5775 and
    // S0^2 = 0.65^2 = 0.4225, so poly_scale = 3.0 gives K1 = (6 - 9) A1^{-1} = -3 A1^{-1},
    // making B0 * 1 = (0.5775 - 3 * 0.4225) * 1 = -0.69 * 1 (negative definite!),
    // which triggers the PCG rz guard and recovers cleanly via FP64 Jacobi.
    Engine eng3(diag_csr, ones, false, false, false, false, 1, 3.0f, 0, false, true);
    eng3.ls[0]->max_lower = MaxLower + 1;
    reset(eng3.ls, Int4, vals.p);
    Result r3 = eng3.solve(Int4);
    auto x3 = download(eng3.solution, n);
    if (r3.trigger != "rz" || !(host_relative_residual(diag_csr, ones, x3) <= 1e-8))
      throw std::runtime_error("poly_scale=3.0 did not trigger rz guard and recover cleanly");
  }
  puts("Terminal [1e-310] Cholesky solve and poly_scale>=2.0 guard + boundary tests passed.");
}
