#pragma once
#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct HostCSR {
  int n = 0, nnz = 0;
  std::vector<int> rp, ci;
  std::vector<double> val;
};

inline void assert_same_pattern(const HostCSR& A0, const HostCSR& As, const std::string& label, int step) {
  if (A0.n != As.n || A0.nnz != As.nnz || A0.rp != As.rp || A0.ci != As.ci) {
    fprintf(stderr, "FATAL: CSR symbolic pattern mismatch on %s step %d (nnz %d vs %d)!\n",
            label.c_str(), step, A0.nnz, As.nnz);
    exit(2);
  }
}

inline HostCSR load_mtx_csr(const std::string& path) {
  std::ifstream fin(path);
  if (!fin)
    throw std::runtime_error("Cannot open Matrix Market file: " + path);

  auto lower_ascii = [](std::string s) {
    for (char& ch : s)
      ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return s;
  };

  std::string line;
  if (!std::getline(fin, line))
    throw std::runtime_error("Empty Matrix Market file: " + path);
  if (line.rfind("%%MatrixMarket", 0) != 0 &&
      lower_ascii(line.substr(0, 14)) != "%%matrixmarket")
    throw std::runtime_error("Missing %%MatrixMarket banner in: " + path);

  std::string banner, object, format, field, symmetry, extra;
  {
    std::istringstream bss(line);
    if (!(bss >> banner >> object >> format >> field >> symmetry) ||
        (bss >> extra))
      throw std::runtime_error("Malformed %%MatrixMarket banner in: " + path);
  }
  banner = lower_ascii(banner);
  object = lower_ascii(object);
  format = lower_ascii(format);
  field = lower_ascii(field);
  symmetry = lower_ascii(symmetry);
  if (banner != "%%matrixmarket" || object != "matrix" ||
      format != "coordinate" || field != "real" ||
      (symmetry != "general" && symmetry != "symmetric")) {
    throw std::runtime_error(
        "Unsupported Matrix Market banner (require 'matrix coordinate real "
        "general|symmetric'): " + line);
  }
  const bool sym = (symmetry == "symmetric");

  bool have_dims = false;
  while (std::getline(fin, line)) {
    size_t first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos || line[first] == '%')
      continue;
    have_dims = true;
    break;
  }
  if (!have_dims)
    throw std::runtime_error("Missing Matrix Market dimension line in: " + path);

  long long nrows_ll = 0, ncols_ll = 0, nnz_in = -1;
  {
    std::istringstream iss(line);
    if (!(iss >> nrows_ll >> ncols_ll >> nnz_in) || (iss >> extra))
      throw std::runtime_error("Malformed Matrix Market dimension line in: " + path);
  }
  if (nrows_ll <= 0 || nrows_ll != ncols_ll || nrows_ll > INT_MAX)
    throw std::runtime_error("Matrix Market matrix must be square with nrows > 0: " + path);
  if (nnz_in < 0)
    throw std::runtime_error("Matrix Market nnz count must be non-negative: " + path);

  const int nrows = static_cast<int>(nrows_ll);
  const int ncols = static_cast<int>(ncols_ll);
  std::vector<std::vector<std::pair<int, double>>> adj(nrows);
  for (long long k = 0; k < nnz_in; ++k) {
    long long r_ll = 0, c_ll = 0;
    double v = 0.0;
    if (!(fin >> r_ll >> c_ll >> v))
      throw std::runtime_error("Matrix Market file ended early or had invalid entry at record " +
                               std::to_string(k) + " in: " + path);
    if (r_ll < 1 || r_ll > nrows || c_ll < 1 || c_ll > ncols)
      throw std::runtime_error("Matrix Market coordinate out of bounds at record " +
                               std::to_string(k) + " in: " + path);
    if (!std::isfinite(v))
      throw std::runtime_error("Matrix Market entry is nonfinite at record " +
                               std::to_string(k) + " in: " + path);
    const int r = static_cast<int>(r_ll - 1);
    const int c = static_cast<int>(c_ll - 1);
    if (sym && r < c)
      throw std::runtime_error("Matrix Market symmetric file contains upper-triangle entry (" +
                               std::to_string(r_ll) + ", " + std::to_string(c_ll) + ") at record " +
                               std::to_string(k) + " in: " + path);
    adj[r].push_back({c, v});
    if (sym && r != c)
      adj[c].push_back({r, v});
  }

  HostCSR A;
  A.n = nrows;
  A.rp.resize(nrows + 1, 0);
  for (int i = 0; i < nrows; ++i) {
    auto& row = adj[i];
    std::stable_sort(row.begin(), row.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    for (size_t k = 0; k < row.size();) {
      int c = row[k].first;
      double s = 0.0;
      while (k < row.size() && row[k].first == c) {
        s += row[k].second;
        k++;
      }
      if (!std::isfinite(s))
        throw std::runtime_error("Matrix Market summed entry overflowed in row " +
                                 std::to_string(i) + " of: " + path);
      if (A.ci.size() >= static_cast<size_t>(INT_MAX))
        throw std::runtime_error("Matrix Market assembled nnz exceeds INT_MAX in: " + path);
      A.ci.push_back(c);
      A.val.push_back(s);
    }
    A.rp[i + 1] = static_cast<int>(A.ci.size());
  }
  A.nnz = static_cast<int>(A.ci.size());

  // Verify explicit finite positive diagonal and structural symmetry on every row.
  for (int i = 0; i < nrows; ++i) {
    bool has_pos_diag = false;
    for (int k = A.rp[i]; k < A.rp[i + 1]; ++k) {
      const int c = A.ci[k];
      const double v = A.val[k];
      if (c == i) {
        has_pos_diag = std::isfinite(v) && (v > 0.0);
      } else {
        auto beg = A.ci.begin() + A.rp[c];
        auto end = A.ci.begin() + A.rp[c + 1];
        auto it = std::lower_bound(beg, end, i);
        if (it == end || *it != i)
          throw std::runtime_error("Matrix Market sparsity pattern is not structurally symmetric in: " + path);
        const int mirror_idx = static_cast<int>(it - A.ci.begin());
        if (A.val[mirror_idx] != v)
          throw std::runtime_error("Matrix Market matrix is not numerically symmetric (A[" +
                                   std::to_string(i) + "," + std::to_string(c) + "] != A[" +
                                   std::to_string(c) + "," + std::to_string(i) + "]) in: " + path);
      }
    }
    if (!has_pos_diag)
      throw std::runtime_error("Matrix Market row " + std::to_string(i) +
                               " is missing an explicit finite positive diagonal in: " + path);
  }
  return A;
}

inline HostCSR build_3d_structured(int L, int step = 0) {
  int N = L * L * L, B = std::max(1, L / 4);
  std::mt19937 rng(42); std::uniform_int_distribution<int> coin(0, 1);
  std::vector<double> patch(B * B * B);
  for (int i = 0; i < B * B * B; ++i) {
    double base = coin(rng) ? 1e4 : 1.0;
    double mod = (step == 0) ? 1.0 : exp(0.8 * sin(0.7 * (double)i + 1.3 * (double)step));
    patch[i] = base * mod;
  }
  auto pidx = [&](int x, int y, int z) {
    return (std::min(z / 4, B - 1) * B + std::min(y / 4, B - 1)) * B + std::min(x / 4, B - 1);
  };
  auto idx = [&](int x, int y, int z) { return (z * L + y) * L + x; };
  HostCSR A; A.n = N; A.rp.resize(N + 1, 0);
  A.ci.reserve((size_t)N * 7); A.val.reserve((size_t)N * 7);
  struct Ent { int c; double v; }; Ent buf[7];
  for (int z = 0; z < L; ++z)
    for (int y = 0; y < L; ++y)
      for (int x = 0; x < L; ++x) {
        int u = idx(x, y, z); double cu = patch[pidx(x, y, z)], diag = 1e-4; int cnt = 0;
        auto add = [&](int nx, int ny, int nz, double eps) {
          if (nx < 0 || nx >= L || ny < 0 || ny >= L || nz < 0 || nz >= L) return;
          double cv = patch[pidx(nx, ny, nz)];
          double w = eps * (2.0 * cu * cv) / (cu + cv);
          diag += w; buf[cnt++] = {idx(nx, ny, nz), -w};
        };
        add(x, y, z - 1, 1e-2); add(x, y - 1, z, 1.0); add(x - 1, y, z, 1e-3);
        int dp = cnt++;
        add(x + 1, y, z, 1e-3); add(x, y + 1, z, 1.0); add(x, y, z + 1, 1e-2);
        buf[dp] = {u, diag};
        for (int k = 0; k < cnt; ++k) { A.ci.push_back(buf[k].c); A.val.push_back(buf[k].v); }
        A.rp[u + 1] = (int)A.ci.size();
      }
  A.nnz = (int)A.ci.size();
  return A;
}

inline HostCSR build_p1_3d_tet_fem(int L, int step = 0) {
  int N = L * L * L;
  std::mt19937 rng(777);
  std::uniform_real_distribution<double> jit(-0.22, 0.22);
  std::vector<double> px(N), py(N), pz(N);
  auto idx = [&](int x, int y, int z) { return (z * L + y) * L + x; };
  for (int z = 0; z < L; ++z)
    for (int y = 0; y < L; ++y)
      for (int x = 0; x < L; ++x) {
        int u = idx(x, y, z);
        px[u] = (x + (x > 0 && x < L - 1 ? jit(rng) : 0.0)) / (double)(L - 1);
        py[u] = (y + (y > 0 && y < L - 1 ? jit(rng) : 0.0)) / (double)(L - 1);
        pz[u] = (z + (z > 0 && z < L - 1 ? jit(rng) : 0.0)) / (double)(L - 1);
      }

  int B = std::max(1, L / 4);
  std::mt19937 rng_c(99); std::uniform_int_distribution<int> coin(0, 1);
  std::vector<double> patch(B * B * B);
  for (int i = 0; i < B * B * B; ++i) {
    double base = coin(rng_c) ? 1e4 : 1.0;
    double mod = (step == 0) ? 1.0 : exp(0.8 * sin(0.7 * (double)i + 1.3 * (double)step));
    patch[i] = base * mod;
  }

  std::vector<std::vector<std::pair<int, double>>> adj(N);
  for (int i = 0; i < N; ++i) adj[i].reserve(24);

  auto assemble_tet = [&](int v0, int v1, int v2, int v3, double kc) {
    int vs[4] = {v0, v1, v2, v3};

    double x1 = px[v1] - px[v0], y1 = py[v1] - py[v0], z1 = pz[v1] - pz[v0];
    double x2 = px[v2] - px[v0], y2 = py[v2] - py[v0], z2 = pz[v2] - pz[v0];
    double x3 = px[v3] - px[v0], y3 = py[v3] - py[v0], z3 = pz[v3] - pz[v0];

    double c00 = y2 * z3 - z2 * y3, c01 = z2 * x3 - x2 * z3, c02 = x2 * y3 - y2 * x3;
    double c10 = y3 * z1 - z3 * y1, c11 = z3 * x1 - x3 * z1, c12 = x3 * y1 - y3 * x1;
    double c20 = y1 * z2 - z1 * y2, c21 = z1 * x2 - x1 * z2, c22 = x1 * y2 - y1 * x2;
    double detJ = x1 * c00 + y1 * c01 + z1 * c02;
    double vol = fabs(detJ) / 6.0;
    if (vol <= 1e-18) return;
    double inv_det = 1.0 / detJ;

    double gx[4], gy[4], gz[4];
    gx[1] = c00 * inv_det; gy[1] = c01 * inv_det; gz[1] = c02 * inv_det;
    gx[2] = c10 * inv_det; gy[2] = c11 * inv_det; gz[2] = c12 * inv_det;
    gx[3] = c20 * inv_det; gy[3] = c21 * inv_det; gz[3] = c22 * inv_det;
    gx[0] = -(gx[1] + gx[2] + gx[3]);
    gy[0] = -(gy[1] + gy[2] + gy[3]);
    gz[0] = -(gz[1] + gz[2] + gz[3]);

    const double kx = kc * 1e-3, ky = kc * 1.0, kz = kc * 1e-2;
    double mass_diag = 1e-4 * vol * 0.25;
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        double ke = vol * (kx * gx[a] * gx[b] + ky * gy[a] * gy[b] + kz * gz[a] * gz[b]);
        if (a == b) ke += mass_diag;
        adj[vs[a]].push_back({vs[b], ke});
      }
    }
  };

  const int tets_even[5][4] = {{0,1,2,4},{1,2,3,7},{1,4,5,7},{2,4,6,7},{1,2,4,7}};
  const int tets_odd[5][4]  = {{0,1,3,5},{0,2,3,6},{0,4,5,6},{3,5,6,7},{0,3,5,6}};

  for (int z = 0; z < L - 1; ++z)
    for (int y = 0; y < L - 1; ++y)
      for (int x = 0; x < L - 1; ++x) {
        double kc = patch[(std::min(z / 4, B - 1) * B + std::min(y / 4, B - 1)) * B + std::min(x / 4, B - 1)];
        int v[8] = {
          idx(x,y,z), idx(x+1,y,z), idx(x,y+1,z), idx(x+1,y+1,z),
          idx(x,y,z+1), idx(x+1,y,z+1), idx(x,y+1,z+1), idx(x+1,y+1,z+1)
        };
        const auto& tets = ((x + y + z) & 1) ? tets_odd : tets_even;
        for (int t = 0; t < 5; ++t)
          assemble_tet(v[tets[t][0]], v[tets[t][1]], v[tets[t][2]], v[tets[t][3]], kc);
      }

  std::vector<bool> is_bc(N, false);
  for (int z = 0; z < L; ++z)
    for (int y = 0; y < L; ++y)
      for (int x = 0; x < L; ++x)
        if (x == 0 || x == L - 1 || y == 0 || y == L - 1 || z == 0 || z == L - 1)
          is_bc[idx(x, y, z)] = true;

  HostCSR A; A.n = N; A.rp.resize(N + 1, 0);
  for (int i = 0; i < N; ++i) {
    if (is_bc[i]) {
      A.ci.push_back(i); A.val.push_back(1.0);
      A.rp[i + 1] = (int)A.ci.size();
      continue;
    }
    auto& row = adj[i];
    std::stable_sort(row.begin(), row.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
    for (size_t k = 0; k < row.size(); ) {
      int c = row[k].first; double s = 0.0;
      while (k < row.size() && row[k].first == c) { s += row[k].second; k++; }
      if (!is_bc[c]) { A.ci.push_back(c); A.val.push_back(s); }
    }
    A.rp[i + 1] = (int)A.ci.size();
  }
  A.nnz = (int)A.ci.size();
  return A;
}

inline HostCSR build_3d_elasticity(int L, int step = 0) {
  int n_nodes = L * L * L;
  int N = 3 * n_nodes;
  double E = 1.0, nu = 0.42;
  double mu_l = E / (2.0 * (1.0 + nu));
  double lam_l = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
  double Ke[24][24];
  memset(Ke, 0, sizeof(Ke));
  const double gp[2] = {-1.0 / sqrt(3.0), 1.0 / sqrt(3.0)};
  const int sx[8] = {-1, 1, -1, 1, -1, 1, -1, 1};
  const int sy[8] = {-1, -1, 1, 1, -1, -1, 1, 1};
  const int sz[8] = {-1, -1, -1, -1, 1, 1, 1, 1};
  for (int gx = 0; gx < 2; ++gx)
    for (int gy = 0; gy < 2; ++gy)
      for (int gz = 0; gz < 2; ++gz) {
        double xi = gp[gx], et = gp[gy], ze = gp[gz];
        double dN[8][3];
        for (int a = 0; a < 8; ++a) {
          dN[a][0] = 0.125 * sx[a] * (1.0 + sy[a] * et) * (1.0 + sz[a] * ze) * 2.0;
          dN[a][1] = 0.125 * sy[a] * (1.0 + sx[a] * xi) * (1.0 + sz[a] * ze) * 2.0;
          dN[a][2] = 0.125 * sz[a] * (1.0 + sx[a] * xi) * (1.0 + sy[a] * et) * 2.0;
        }
        double Bmat[6][24]; memset(Bmat, 0, sizeof(Bmat));
        for (int a = 0; a < 8; ++a) {
          Bmat[0][3*a + 0] = dN[a][0];
          Bmat[1][3*a + 1] = dN[a][1];
          Bmat[2][3*a + 2] = dN[a][2];
          Bmat[3][3*a + 0] = dN[a][1]; Bmat[3][3*a + 1] = dN[a][0];
          Bmat[4][3*a + 1] = dN[a][2]; Bmat[4][3*a + 2] = dN[a][1];
          Bmat[5][3*a + 0] = dN[a][2]; Bmat[5][3*a + 2] = dN[a][0];
        }
        double Dmat[6][6]; memset(Dmat, 0, sizeof(Dmat));
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) Dmat[i][j] = lam_l + (i == j ? 2.0 * mu_l : 0.0);
        Dmat[3][3] = Dmat[4][4] = Dmat[5][5] = mu_l;
        double DB[6][24];
        for (int i = 0; i < 6; ++i)
          for (int j = 0; j < 24; ++j) {
            double s = 0.0; for (int k = 0; k < 6; ++k) s += Dmat[i][k] * Bmat[k][j];
            DB[i][j] = s * 0.125;
          }
        for (int i = 0; i < 24; ++i)
          for (int j = 0; j < 24; ++j) {
            double s = 0.0; for (int k = 0; k < 6; ++k) s += Bmat[k][i] * DB[k][j];
            Ke[i][j] += s;
          }
      }

  auto idx = [&](int x, int y, int z) { return (z * L + y) * L + x; };
  std::vector<bool> is_bc(N, false);
  for (int y = 0; y < L; ++y)
    for (int x = 0; x < L; ++x)
      for (int d = 0; d < 3; ++d) is_bc[3 * idx(x, y, 0) + d] = true;

  std::vector<std::vector<std::pair<int, double>>> adj(N);
  for (int i = 0; i < N; ++i) adj[i].reserve(84);
  int B = std::max(1, L / 4);
  std::mt19937 rng(55); std::uniform_int_distribution<int> coin(0, 1);
  std::vector<double> patch(B * B * B);
  for (int i = 0; i < B * B * B; ++i) {
    double base = coin(rng) ? 100.0 : 1.0;
    double mod = (step == 0) ? 1.0 : exp(0.5 * sin(0.7 * (double)i + 1.1 * (double)step));
    patch[i] = base * mod;
  }

  for (int z = 0; z < L - 1; ++z)
    for (int y = 0; y < L - 1; ++y)
      for (int x = 0; x < L - 1; ++x) {
        double sc = patch[(std::min(z/4, B-1)*B + std::min(y/4, B-1))*B + std::min(x/4, B-1)];
        int nd[8] = {
          idx(x,y,z), idx(x+1,y,z), idx(x,y+1,z), idx(x+1,y+1,z),
          idx(x,y,z+1), idx(x+1,y,z+1), idx(x,y+1,z+1), idx(x+1,y+1,z+1)
        };
        for (int a = 0; a < 8; ++a)
          for (int b = 0; b < 8; ++b)
            for (int da = 0; da < 3; ++da)
              for (int db = 0; db < 3; ++db) {
                int r = 3*nd[a] + da, c = 3*nd[b] + db;
                if (!is_bc[r] && !is_bc[c]) {

                  adj[r].push_back({c, sc * Ke[3*a + da][3*b + db]});
                }
              }
      }

  HostCSR A; A.n = N; A.rp.resize(N + 1, 0);
  for (int i = 0; i < N; ++i) {
    if (is_bc[i]) {
      A.ci.push_back(i); A.val.push_back(1.0);
      A.rp[i + 1] = (int)A.ci.size();
      continue;
    }
    auto& row = adj[i];
    std::stable_sort(row.begin(), row.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
    for (size_t k = 0; k < row.size(); ) {
      int c = row[k].first; double s = 0.0;
      while (k < row.size() && row[k].first == c) { s += row[k].second; k++; }
      A.ci.push_back(c); A.val.push_back(s);
    }
    A.rp[i + 1] = (int)A.ci.size();
  }
  A.nnz = (int)A.ci.size();
  return A;
}
