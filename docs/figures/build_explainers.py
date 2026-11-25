"""Generate the BitFSAI README explainer SVGs (stdlib only).

Usage: python3 docs/figures/build_explainers.py [output_dir]
"""
import math
import re
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent

NAVY = "#102a43"
INK = "#1f2933"
MUTED = "#52606d"
FAINT = "#9aa5b1"
LINE = "#d9e2ec"
BG = "#ffffff"
PANEL = "#f5f7fa"
TEAL = "#0e8a7d"
TEAL_L = "#d8f1ed"
TEAL_D = "#0a5c53"
BLUE = "#1f4e79"
BLUE_L = "#dbe7f3"
SLATE = "#9fb3c8"
AMBER = "#c77700"
AMBER_L = "#fdf0d5"
RED = "#b42318"
RED_L = "#fde7e4"

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, 'Liberation Sans', sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rich(s, size):
    """Mini-markup: _{sub}, ^{sup}, *bold*, `mono`."""
    out = []
    pos = 0
    pat = re.compile(r"_\{([^}]*)\}|\^\{([^}]*)\}|\*([^*]+)\*|`([^`]+)`")
    small = round(size * 0.72, 1)
    for m in pat.finditer(s):
        out.append(esc(s[pos:m.start()]))
        if m.group(1) is not None:
            out.append(f'<tspan dy="{size*0.28:.1f}" font-size="{small}">{esc(m.group(1))}</tspan><tspan dy="{-size*0.28:.1f}">\u200b</tspan>')
        elif m.group(2) is not None:
            out.append(f'<tspan dy="{-size*0.38:.1f}" font-size="{small}">{esc(m.group(2))}</tspan><tspan dy="{size*0.38:.1f}">\u200b</tspan>')
        elif m.group(3) is not None:
            out.append(f'<tspan font-weight="700">{esc(m.group(3))}</tspan>')
        else:
            out.append(f'<tspan font-family="{MONO}">{esc(m.group(4))}</tspan>')
        pos = m.end()
    out.append(esc(s[pos:]))
    return "".join(out)


class SVG:
    def __init__(self, w, h, title, desc):
        self.w, self.h = w, h
        self.parts = []
        self.title, self.desc = title, desc

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=15, fill=INK, weight=400, anchor="start", family=SANS, italic=False):
        st = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{fill}" '
                 f'font-weight="{weight}" text-anchor="{anchor}"{st}>{rich(s, size)}</text>')

    def lines(self, x, y, items, size=15, lh=None, **kw):
        lh = lh or round(size * 1.4)
        for i, s in enumerate(items):
            self.text(x, y + i * lh, s, size=size, **kw)

    def rect(self, x, y, w, h, fill="none", stroke="none", rx=0, sw=1, dash=None, opacity=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' fill-opacity="{opacity}"' if opacity is not None else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d}{o}/>')

    def line(self, x1, y1, x2, y2, stroke=MUTED, sw=1.5, dash=None, arrow=False):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        a = ' marker-end="url(#arrow)"' if arrow else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}{a}/>')

    def path(self, d, stroke=MUTED, sw=1.5, fill="none", arrow=False, dash=None):
        a = ' marker-end="url(#arrow)"' if arrow else ""
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" fill="{fill}"{a}{ds}/>')

    def circle(self, cx, cy, r, fill="none", stroke="none", sw=1.5):
        self.add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def pill(self, x, y, label, fill, color, stroke=None, w=None, size=12):
        w = w or (len(label) * size * 0.66 + 20)
        self.rect(x, y, w, 24, fill=fill, stroke=stroke or fill, rx=12, sw=1.2)
        self.text(x + w / 2, y + 16.5, label, size=size, fill=color, weight=700, anchor="middle")
        return w

    def matrix(self, x, y, rows, size=15, colw=46, lh=22, color=INK, family=MONO):
        h = lh * len(rows) + 8
        ncol = len(rows[0])
        w = colw * ncol + 14
        b = 6
        self.path(f"M{x+b},{y} L{x},{y} L{x},{y+h} L{x+b},{y+h}", stroke=color, sw=1.4)
        self.path(f"M{x+w-b},{y} L{x+w},{y} L{x+w},{y+h} L{x+w-b},{y+h}", stroke=color, sw=1.4)
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                self.text(x + 7 + colw * c + colw / 2, y + 4 + lh * (r + 1) - 6, v, size=size,
                          fill=color, anchor="middle", family=family)
        return w, h

    def render(self):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}" role="img" aria-labelledby="t d">'
                f'<title id="t">{esc(self.title)}</title><desc id="d">{esc(self.desc)}</desc>'
                '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>'
                f'<rect x="1" y="1" width="{self.w-2}" height="{self.h-2}" rx="16" fill="{BG}" stroke="{LINE}" stroke-width="1.5"/>')
        return head + "\n" + "\n".join(self.parts) + "\n</svg>\n"


def header(s, title, sub):
    s.text(40, 54, title, size=25, fill=NAVY, weight=700)
    s.text(40, 82, sub, size=15.5, fill=MUTED)


# --------------------------------------------------------------------------
# 1. Bandwidth
# --------------------------------------------------------------------------
def fig_bandwidth():
    s = SVG(960, 470, "Bytes read per factor coupling",
            "Bytes per factor coupling: standard FP64 FSAI reads 12 bytes, mirrored FP16 reads 4 bytes, and BitFSAI reads 1 byte.")
    header(s, "Bytes read per factor coupling",
           "DRAM traffic a GPU smoother pulls for each coupling g_{ij} on every sweep")
    px = 46  # pixels per byte
    x0 = 330
    rows = [
        ("Standard FP64 FSAI", "separate lower-triangular factor", [(8, BLUE, "8 B FP64 value", "#fff"), (4, SLATE, "4 B int32 index", NAVY)], "12 B"),
        ("Mirrored FP16 factor", "both triangle slots, A's own indices", [(4, "#7fb3d5", "two 2 B halves", NAVY)], "4 B"),
        ("BitFSAI", "both triangle slots, A's own indices", [(1, TEAL, "", "#fff")], "1 B"),
    ]
    y = 128
    for name, sub, segs, total in rows:
        s.text(40, y + 22, name, size=17, fill=INK, weight=700)
        s.text(40, y + 44, sub, size=13.5, fill=MUTED)
        x = x0
        for b, col, lab, tc in segs:
            s.rect(x, y + 4, b * px, 44, fill=col, rx=4)
            if lab:
                s.text(x + b * px / 2, y + 31, lab, size=14, fill=tc, weight=600, anchor="middle")
            x += b * px
        s.text(x + 12, y + 32, total, size=19, fill=NAVY if total != "1 B" else TEAL_D, weight=700)
        if total == "1 B":
            s.text(x + 60, y + 32, "two 4-bit nibbles, 0 extra index bytes", size=14, fill=TEAL_D, weight=600)
        y += 76
    s.text(40, y + 6, "BitFSAI also stores 8 B per row for the FP32 diagonal and scale. Across 8 workloads, it uses 4.23x to 6.84x",
           size=13.5, fill=MUTED)
    s.text(40, y + 26, "less smoother storage than a standard FP64 factor on the same K <= 16 pattern, counting values and int32 indices.",
           size=13.5, fill=MUTED)
    # callout
    cy = y + 50
    s.rect(40, cy, 880, 66, fill=AMBER_L, stroke="#f3d9a4", rx=10)
    s.text(62, cy + 28, "One FP64 entry plus its int32 index pulls 12 bytes from DRAM for 2 floating-point operations.",
           size=14.5, fill=INK)
    s.text(62, cy + 50, "Arithmetic intensity sits under 0.17 FLOP/byte, two orders of magnitude below GPU compute throughput.",
           size=14.5, fill=INK)
    return s


# --------------------------------------------------------------------------
# 2. Storage layout
# --------------------------------------------------------------------------
PATTERN = {(0, 0), (0, 1), (0, 3), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (2, 4),
           (3, 0), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4)}


def fig_layout():
    s = SVG(960, 624, "BitFSAI storage layout",
            "Standard FSAI stores a separate lower-triangular factor with its own index arrays. BitFSAI stores an FP32 diagonal and row scale per row plus a 4-bit nibble mirrored in A's own CSR slots.")
    header(s, "Storing the factor inside A's own sparsity pattern",
           "Same 5x5 sparse SPD matrix A, two ways to lay out its FSAI smoother factor G")
    cs = 40
    # left panel
    lx, ly = 40, 112
    s.rect(lx, ly, 420, 346, fill=PANEL, stroke=LINE, rx=12)
    s.text(lx + 20, ly + 32, "Standard FSAI", size=18, fill=BLUE, weight=700)
    s.text(lx + 20, ly + 54, "separate lower-triangular factor G", size=14, fill=MUTED)
    gx, gy = lx + 30, ly + 76
    for i in range(5):
        for j in range(5):
            x, yy = gx + j * cs, gy + i * cs
            if (i, j) in PATTERN and j <= i:
                s.rect(x + 2, yy + 2, cs - 4, cs - 4, fill=BLUE, rx=4)
                s.text(x + cs / 2, yy + 26, "8 B", size=12, fill="#fff", weight=700, anchor="middle")
            else:
                s.rect(x + 2, yy + 2, cs - 4, cs - 4, fill="none", stroke=LINE, rx=4)
    ax = gx + 5 * cs + 22
    s.text(ax, gy + 14, "Extra arrays", size=14, fill=INK, weight=700)
    arrs = [("row_ptr", "int32, n+1", SLATE), ("col_idx", "int32, nnz(G)", SLATE), ("values", "FP64, nnz(G)", BLUE)]
    for k, (nm, ty, col) in enumerate(arrs):
        yy = gy + 30 + k * 52
        s.rect(ax, yy, 14, 34, fill=col, rx=3)
        s.text(ax + 22, yy + 15, nm, size=14, fill=INK, weight=700, family=MONO)
        s.text(ax + 22, yy + 32, ty, size=12.5, fill=MUTED, family=MONO)
    s.text(lx + 20, ly + 318, "A already has row_ptr and col_idx. This adds a second set.", size=13.5, fill=MUTED)
    # right panel
    rx0, ry = 500, 112
    s.rect(rx0, ry, 420, 346, fill=TEAL_L, stroke="#b4e2da", rx=12)
    s.text(rx0 + 20, ry + 32, "BitFSAI", size=18, fill=TEAL_D, weight=700)
    s.text(rx0 + 20, ry + 54, "nibbles written into A's own CSR slots", size=14, fill=TEAL_D)
    gx, gy = rx0 + 30, ry + 76
    for i in range(5):
        for j in range(5):
            x, yy = gx + j * cs, gy + i * cs
            if (i, j) not in PATTERN:
                s.rect(x + 2, yy + 2, cs - 4, cs - 4, fill="none", stroke="#b4e2da", rx=4)
            elif i == j:
                s.rect(x + 2, yy + 2, cs - 4, cs - 4, fill="#ffffff", stroke=TEAL, rx=4, sw=1.2)
                s.text(x + cs / 2, yy + 26, "d", size=14, fill=TEAL_D, weight=700, anchor="middle", italic=True)
            else:
                hl = (i, j) in {(3, 0), (0, 3)}
                s.rect(x + 2, yy + 2, cs - 4, cs - 4, fill=AMBER if hl else TEAL, rx=4)
                lab = "g₃₀" if hl else "4b"
                s.text(x + cs / 2, yy + 26, lab, size=12.5 if hl else 12, fill="#fff", weight=700, anchor="middle")
    ax = gx + 5 * cs + 22
    s.text(ax, gy + 14, "Stored fields", size=14, fill=INK, weight=700)
    arrs = [("d", "FP32 diagonal", "#ffffff"), ("s", "FP32 row scale", "#ffffff"), ("q", "4-bit, nnz(A)", TEAL)]
    for k, (nm, ty, col) in enumerate(arrs):
        yy = gy + 30 + k * 52
        s.rect(ax, yy, 14, 34, fill=col, stroke=TEAL, rx=3, sw=1.2)
        s.text(ax + 22, yy + 15, nm, size=14, fill=INK, weight=700, family=MONO)
        s.text(ax + 22, yy + 32, ty, size=12.5, fill=MUTED, family=MONO)
    s.text(rx0 + 20, ry + 302, "Reuses A's row_ptr and col_idx (*0 extra index bytes*)", size=13.5, fill=TEAL_D)
    s.text(rx0 + 20, ry + 324, "Setup mirrors each amber coupling into both slots.", size=13.5, fill=TEAL_D)
    # bottom: two-pass apply
    by = 490
    s.text(40, by, "Applying z = Gᵀ G r in two row-parallel passes (each off-diagonal decodes as s_{i} * nibble)", size=15, fill=INK, weight=700)
    s.rect(40, by + 14, 430, 92, fill=PANEL, stroke=LINE, rx=10)
    s.text(60, by + 42, "Pass 1: t = G r", size=15, fill=NAVY, weight=700)
    s.text(60, by + 68, "t_{i} = d_{i} r_{i} + s_{i} Σ_{j<i} q_{ij} r_{j}", size=15, fill=INK, family=MONO)
    s.text(60, by + 92, "each row reads its lower-triangle slots", size=13.5, fill=MUTED)
    s.rect(490, by + 14, 430, 92, fill=PANEL, stroke=LINE, rx=10)
    s.text(510, by + 42, "Pass 2: z = Gᵀ t", size=15, fill=NAVY, weight=700)
    s.text(510, by + 68, "z_{j} = d_{j} t_{j} + Σ_{i>j} s_{i} q_{ji} t_{i}", size=15, fill=INK, family=MONO)
    s.text(510, by + 92, "each row reads mirrored upper slots as a row gather", size=13.5, fill=MUTED)
    return s


# --------------------------------------------------------------------------
# 3. Why the diagonal stays in FP32
# --------------------------------------------------------------------------
def fig_split():
    s = SVG(960, 640, "Why the diagonal stays in FP32",
            "A 2 by 2 example proved in Lean. Diagonal-inclusive 4-bit rounding drives condition number above 321, while diagonal-split 4-bit recovers the exact factor.")
    header(s, "Why the diagonal stays in FP32",
           "A 2x2 counterexample proved in proofs/lean/Bitfsai/Core.lean")
    y0 = 108
    s.rect(40, y0, 880, 96, fill=PANEL, stroke=LINE, rx=12)
    s.text(62, y0 + 54, "A =", size=17, fill=INK, weight=700)
    s.matrix(100, y0 + 20, [["1", "0.04"], ["0.04", "0.0017"]], colw=64)
    s.text(266, y0 + 44, "SPD with a tiny second", size=14, fill=MUTED)
    s.text(266, y0 + 64, "pivot: 0.0017 - 0.04² = 0.0001", size=14, fill=MUTED)
    s.text(476, y0 + 54, "Exact", size=14, fill=MUTED)
    s.text(518, y0 + 54, "G =", size=17, fill=INK, weight=700)
    s.matrix(556, y0 + 20, [["1", "0"], ["-4", "100"]], colw=50)
    s.text(694, y0 + 54, "gives  G A Gᵀ = I,  κ = 1", size=16, fill=INK, weight=700)

    def card(x, color, light, border, title, sub, steps, gmat, pmat, verdict, verdict_sub):
        y = 228
        s.rect(x, y, 430, 300, fill=light, stroke=border, rx=12)
        s.text(x + 20, y + 32, title, size=18, fill=color, weight=700)
        s.text(x + 20, y + 54, sub, size=13.5, fill=MUTED)
        for k, st in enumerate(steps):
            s.text(x + 20, y + 86 + k * 24, st, size=14.5, fill=INK)
        my = y + 168
        s.text(x + 20, my + 34, "G =", size=15, fill=INK, weight=700)
        s.matrix(x + 54, my + 4, gmat, colw=42)
        s.text(x + 166, my + 34, "G A Gᵀ =", size=15, fill=INK, weight=700)
        s.matrix(x + 248, my + 4, pmat, colw=36)
        s.rect(x + 20, y + 236, 390, 48, fill="#ffffff", stroke=border, rx=8)
        s.text(x + 36, y + 267, verdict, size=21, fill=color, weight=800)
        s.text(x + 150, y + 266, verdict_sub, size=13.5, fill=MUTED)

    card(40, RED, RED_L, "#f5c2bb", "Diagonal-inclusive 4-bit",
         "one scale per row, shared with the diagonal",
         ["Row 2 is (-4, 100), so scale = 100/7 = 14.29", "-4 / 14.29 = -0.28, which *rounds to 0*", "The off-diagonal coupling vanishes"],
         [["1", "0"], ["0", "100"]], [["1", "4"], ["4", "17"]],
         "κ > 321", "vs. κ = 1 for the exact factor")
    card(490, TEAL_D, TEAL_L, "#b4e2da", "Diagonal-split 4-bit (BitFSAI)",
         "diagonal in FP32, off-diagonals get their own scale",
         ["Diagonal 100 stays in FP32", "Off-diagonal scale is 4/7 with code -7", "(-7)(4/7) = -4 *exactly*"],
         [["1", "0"], ["-4", "100"]], [["1", "0"], ["0", "1"]],
         "κ = 1", "exact factor recovered")
    yb = 548
    s.rect(40, yb, 880, 72, fill=BLUE_L, stroke="#b8cde3", rx=12)
    s.text(62, yb + 30, "For any matrix size, a lower-triangular G with positive diagonals makes GᵀG positive definite,",
           size=14.5, fill=INK)
    s.text(62, yb + 53, "regardless of how the 4-bit off-diagonals round. Round any diagonal entry to zero and GᵀG turns singular.",
           size=14.5, fill=INK)
    return s


# --------------------------------------------------------------------------
# 4. Re-projection
# --------------------------------------------------------------------------
def fig_reproj():
    s = SVG(960, 610, "Row re-projection after rounding",
            "Setup rounds off-diagonals to 4-bit nibbles, re-solves each row's diagonal and scale in closed form for unit energy, and checks the stored FP32 values.")
    header(s, "Re-projecting each row after rounding",
           "The split keeps GᵀG positive definite. Re-projection rescales each row in closed form during setup.")
    boxes = [
        ("1", "Exact row", ["Solve the small FSAI", "system on the row patch", "(<= 16 entries) in FP64"], BLUE_L, "#b8cde3", BLUE),
        ("2", "Round", ["Quantize off-diagonals", "to a 4-bit direction q", "(one nibble per slot)"], AMBER_L, "#f3d9a4", AMBER),
        ("3", "Re-solve (g, s)", ["Pick diagonal g and", "scale s inside span{e_{i}, q}", "for unit A-energy"], TEAL_L, "#b4e2da", TEAL_D),
        ("4", "Verify & store", ["Store g, s in FP32, q in 4b.", "Keep the row only if", "stored energy passes"], PANEL, LINE, NAVY),
    ]
    bw, bh, gap = 200, 128, 26
    x = 40
    y = 112
    for k, (num, title, body, fill, stroke, col) in enumerate(boxes):
        s.rect(x, y, bw, bh, fill=fill, stroke=stroke, rx=12)
        s.circle(x + 26, y + 28, 14, fill=col)
        s.text(x + 26, y + 33.5, num, size=14, fill="#fff", weight=700, anchor="middle")
        s.text(x + 48, y + 34, title, size=16, fill=col, weight=700)
        s.lines(x + 16, y + 66, body, size=13.5, lh=20, fill=INK)
        if k < 3:
            s.line(x + bw + 3, y + bh / 2, x + bw + gap - 3, y + bh / 2, arrow=True, sw=2)
        x += bw + gap
    # formula panel
    fy = 266
    s.rect(40, fy, 520, 196, fill="#ffffff", stroke="#b4e2da", rx=12, sw=1.5)
    s.text(60, fy + 32, "Step 3 in closed form", size=16, fill=TEAL_D, weight=700)
    s.text(60, fy + 62, "Row energy   E(s, g) = s² c_{11} + 2sg c_{12} + g² c_{22}", size=15, fill=INK, family=MONO)
    s.text(60, fy + 86, "where c_{11} = qᵀAq,  c_{12} = qᵀA e_{i},  c_{22} = a_{ii}", size=14, fill=MUTED, family=MONO)
    s.text(60, fy + 118, "g∗ = √(c_{11}/Δ),   s∗ = -c_{12}/√(c_{11}Δ),   Δ = c_{11}c_{22} - c_{12}²", size=15, fill=INK, family=MONO)
    s.text(60, fy + 150, "Real optimizer gives g* > 0, E = 1, and optimal", size=14, fill=INK)
    s.text(60, fy + 172, "Kaporin ratio over all (s, g).  `optimal_2x2_reprojection_properties`", size=13.5, fill=INK)
    # check panel
    s.rect(584, fy, 336, 196, fill=PANEL, stroke=LINE, rx=12)
    s.text(604, fy + 32, "Step 4 safety net", size=16, fill=NAVY, weight=700)
    s.text(604, fy + 60, "Kahan-compensated energy check", size=14, fill=INK)
    s.text(604, fy + 84, "|E - 1| + 8ε Σ|t_{ab}| <= 10⁻⁴", size=15, fill=INK, family=MONO)
    s.text(604, fy + 116, "Pass: keep the 4-bit row", size=14, fill=TEAL_D, weight=600)
    s.text(604, fy + 140, "Fail: fall back to diagonal row", size=14, fill=AMBER, weight=600)
    s.text(604, fy + 162, "Zero q: zero-direction fallback", size=14, fill=AMBER, weight=600)
    s.text(604, fy + 182, "Both q != 0 and q = 0 branches proved in Lean", size=12.5, fill=MUTED)
    # evidence
    ey = 486
    s.rect(40, ey, 880, 104, fill=BLUE_L, stroke="#b8cde3", rx=12)
    s.text(62, ey + 30, "Measured in NVIDIA AmgX (FP32 inner V-cycle, FP64 outer PCG)", size=15.5, fill=NAVY, weight=700)
    s.text(62, ey + 60, "bcsstk14 and bcsstk15 without re-projection", size=14.5, fill=INK)
    s.pill(380, ey + 42, "0/5 runs pass", RED_L, RED, stroke="#f5c2bb")
    s.text(518, ey + 60, "the PCG curvature guard rᵀz > 0 trips", size=14.5, fill=MUTED)
    s.text(62, ey + 88, "bcsstk14 and bcsstk15 with re-projection", size=14.5, fill=INK)
    s.pill(380, ey + 70, "5/5 runs pass", TEAL_L, TEAL_D, stroke="#b4e2da")
    s.text(518, ey + 88, "4.20x and 3.38x faster than JACOBI_L1", size=14.5, fill=MUTED)
    return s


# --------------------------------------------------------------------------
# 5. Multilevel
# --------------------------------------------------------------------------
def fig_multilevel():
    s = SVG(960, 640, "Compressing the full multigrid hierarchy",
            "V-cycle using a 4-bit BitFSAI smoother on every level, guarded FP8 E5M2 coarse operators with FP32 fallback, and a polynomial boost on every fourth level.")
    header(s, "Compressing the full multigrid hierarchy",
           "Standalone CUDA-Graph engine using 4-bit smoothing and guarded FP8 coarse operators")
    # V-cycle
    levels = 6
    top = 116
    dy = 58
    cx = 300
    widths = [440, 360, 290, 230, 180, 140]
    for j in range(levels):
        w = widths[j]
        y = top + j * dy
        boost = (j % 4 == 2)
        fill = BLUE_L if j == 0 else TEAL_L
        stroke = "#b8cde3" if j == 0 else "#b4e2da"
        s.rect(cx - w / 2, y, w, 40, fill=fill, stroke=stroke, rx=8)
        lab = "Level 0: fine grid, A in FP64" if j == 0 else (f"Level {j}: coarse A_{{c}}" if j < levels - 1 else f"Level {j}: coarsest")
        s.text(cx, y + 25.5, lab, size=14, fill=BLUE if j == 0 else TEAL_D, weight=700, anchor="middle")
        if boost:
            s.rect(cx + w / 2 + 12, y + 4, 96, 32, fill=AMBER_L, stroke="#f3d9a4", rx=16)
            s.text(cx + w / 2 + 60, y + 25, "2x boost", size=13, fill=AMBER, weight=700, anchor="middle")
    # down/up arrows
    for j in range(levels - 1):
        y1 = top + j * dy + 40
        y2 = top + (j + 1) * dy
        xl1 = cx - widths[j] / 2 + 24
        xl2 = cx - widths[j + 1] / 2 + 24
        xr1 = cx + widths[j] / 2 - 24
        xr2 = cx + widths[j + 1] / 2 - 24
        s.line(xl1, y1 + 2, xl2, y2 - 3, stroke=MUTED, sw=1.6, arrow=True)
        s.line(xr2, y2 - 2, xr1, y1 + 3, stroke=MUTED, sw=1.6, arrow=True)
    s.text(cx - 250, top + 2.6 * dy, "restrict", size=13, fill=MUTED, anchor="middle", italic=True)
    s.text(cx + 248, top + 3.6 * dy, "prolong", size=13, fill=MUTED, anchor="middle", italic=True)
    # legend under V
    ly = top + levels * dy + 16
    s.text(40, ly, "Every nonterminal level smooths with the 4-bit BitFSAI smoother.", size=13.5, fill=MUTED)
    s.text(40, ly + 20, "The engine captures the preconditioner in a CUDA Graph. Depth shown is illustrative.", size=13.5, fill=MUTED)
    # right column
    rx = 590
    s.rect(rx, 112, 330, 236, fill=PANEL, stroke=LINE, rx=12)
    s.text(rx + 18, 140, "FP8 E5M2 guard per coarse level", size=15.5, fill=NAVY, weight=700)
    checks = ["off-diagonals <= 0 (M-matrix)", "every connected component grounded", "magnitudes in normal E5M2 range", "[2⁻¹⁴, 57344]"]
    for k, c in enumerate(checks):
        yy = 170 + k * 24
        if k < 3:
            s.circle(rx + 26, yy - 5, 3.8, fill=TEAL)
        s.text(rx + 42, yy, c, size=14, fill=INK, family=MONO if k == 3 else SANS)
    s.text(rx + 20, 280, "Pass: repair diagonal row sums (keeps", size=13.5, fill=TEAL_D, weight=600)
    s.text(rx + 20, 299, "A·1 unchanged) and store values in FP8", size=13.5, fill=TEAL_D, weight=600)
    s.text(rx + 20, 326, "Fail: keep that level in FP32", size=13.5, fill=AMBER, weight=600)
    s.rect(rx, 362, 330, 128, fill=AMBER_L, stroke="#f3d9a4", rx=12)
    s.text(rx + 18, 390, "Sparse-depth boost (poly4@2)", size=15.5, fill=AMBER, weight=700)
    s.text(rx + 18, 416, "On every 4th level starting at level 2,", size=13.5, fill=INK)
    s.text(rx + 18, 436, "run the coarse correction twice:", size=13.5, fill=INK)
    s.text(rx + 18, 462, "K = 2t B - t² B A B,  t = 1.25", size=14, fill=INK, family=MONO)
    s.text(rx + 18, 481, "other levels stay a plain V-cycle", size=12.5, fill=MUTED)
    # results strip
    ry = 520
    s.rect(40, ry, 880, 102, fill=BLUE_L, stroke="#b8cde3", rx=12)
    s.text(62, ry + 28, "Measured on 192³ (7.08M unknowns, 10⁴ contrast), NVIDIA RTX Spark N1X", size=14.5, fill=NAVY, weight=700)
    stats = [("28 to 13", "PCG iterations"), ("1.53x", "faster GPU solve"), ("1.86x", "smaller coarse buffers"), ("3.51x", "less smoother + coarse storage*")]
    for k, (big, small) in enumerate(stats):
        x = 62 + k * 214
        s.text(x, ry + 66, big, size=25, fill=TEAL_D, weight=800)
        s.text(x, ry + 88, small, size=13, fill=INK)
    s.text(898, ry + 28, "*vs. FP64 factor (values + indices) + FP32 coarse", size=11.5, fill=MUTED, anchor="end")
    return s


# --------------------------------------------------------------------------
# 6. Bit ladder
# --------------------------------------------------------------------------
def fig_ladder():
    s = SVG(960, 640, "How low can the bits go",
            "Bits per stored weight from FP64 down to zero-storage operators, separating GPU kernels from proved theoretical limits.")
    header(s, "How low can the bits go?",
           "Bits per stored off-diagonal weight on a log scale. Sub-4-bit tiers are theory with no GPU kernel yet.")
    rows = [
        ("FP64", "64", 64, "base", "Standard factor weights"),
        ("FP16", "16", 16, "base", "Where prior GPU sparse smoothers stop"),
        ("FP8 E5M2", "8", 8, "gpu", "Guarded coarse-grid operators"),
        ("INT4", "4", 4, "gpu", "BitFSAI smoother (FP32 diagonal, 4-bit off-diagonals)"),
        ("Ternary", "1.58", 1.58, "proof", "Values {-1, 0, +1}, cut bound independent of contrast"),
        ("Sign", "1", 1, "proof", "Values {-1, +1}, fails where ternary converges"),
        ("Sub-bit", "< 1", 0.6, "proof", "Shared codebooks with packing and covering bounds"),
        ("Zero storage", "0", 0, "proof", "Rebuild coefficients from physical laws on the fly"),
    ]
    y = 112
    rh = 50
    bx = 210
    for name, bits, val, kind, desc in rows:
        s.text(40, y + 22, name, size=16, fill=INK, weight=700)
        s.text(40, y + 40, f"{bits} bit" if bits in ("1", "< 1") else f"{bits} bits", size=12.5, fill=MUTED)
        if val >= 1:
            w = 24 + 216 * math.log2(val) / 6
        else:
            w = 14 if val > 0 else 0
        col = {"base": SLATE, "gpu": TEAL, "proof": "none"}[kind]
        if val > 0:
            s.rect(bx, y + 8, w, 30, fill=col, stroke=(AMBER if kind == "proof" else col), rx=5, sw=1.6,
                   dash=("5 3" if kind == "proof" else None))
        else:
            s.circle(bx + 8, y + 23, 7, stroke=AMBER, sw=1.6)
        if kind == "gpu":
            s.pill(462, y + 11, "ON GPU", TEAL, "#fff", w=84)
        elif kind == "proof":
            s.pill(462, y + 11, "THEORY", "#ffffff", AMBER, stroke=AMBER, w=84)
        s.text(566, y + 28, desc, size=13.5, fill=INK)
        y += rh
    s.line(40, y + 2, 920, y + 2, stroke=LINE, sw=1)
    # separation example
    ey = y + 16
    s.rect(40, ey, 880, 104, fill=AMBER_L, stroke="#f3d9a4", rx=12)
    s.text(62, ey + 28, "1-bit sign vs. 1.58-bit ternary on a 3-node grounded patch (BitLimits.lean)", size=14.5, fill=AMBER, weight=700)
    s.text(62, ey + 54, "Target row is (1, t) at t = 1/5. Sign-only forces t to +1 or -1, while ternary can round t to 0.", size=14, fill=INK)
    s.text(62, ey + 78, "Sign-only hits local κ > 9 and diverges like 2/t as t goes to 0. Ternary stays at κ <= 3 and approaches 1.", size=13.8, fill=INK, weight=600)
    return s


figs = {
    "01-bandwidth.svg": fig_bandwidth,
    "02-storage-layout.svg": fig_layout,
    "03-why-split.svg": fig_split,
    "04-reprojection.svg": fig_reproj,
    "05-multilevel.svg": fig_multilevel,
    "06-bit-ladder.svg": fig_ladder,
}
OUT.mkdir(parents=True, exist_ok=True)
for name, fn in figs.items():
    (OUT / name).write_text(fn().render(), encoding="utf-8")
    print("wrote", OUT / name)
