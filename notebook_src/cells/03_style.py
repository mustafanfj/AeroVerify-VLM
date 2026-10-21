# ── Figure style: IEEE journal (text ≥ 7 pt at print size) ────────────────────
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDC, AXISC, SURFACE, BAND = "#e1e0d9", "#c3c2b7", "#ffffff", "#f6f5f1"
METHOD_COLORS  = {"E1": "#eb6834", "E2": "#1baf7a", "E3": "#eda100", "E4": "#2a78d6", "Lite": "#6250d6"}
METHOD_MARKERS = {"E1": "o", "E2": "s", "E3": "^", "E4": "D", "Lite": "P"}
METHOD_LABELS  = {"E1": "E1 Direct", "E2": "E2 Structured", "E3": "E3 Region", "E4": "E4 AeroVerify",
                  "Lite": "AeroVerify-Lite", "majority": "Most-common answer", "blind": "Question-only"}
BASELINE_COLOR = MUTED
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
POS, NEG, MID = "#2a78d6", "#e34948", "#f0efec"
DS_LABEL = {"floodnet": "FloodNet-VQA (UAV)", "rsvqa_lr": "RSVQA-LR (satellite, 10 m)"}
DS_SHORT = {"floodnet": "FloodNet", "rsvqa_lr": "RSVQA-LR"}

IEEE_1COL, IEEE_2COL = 3.5, 7.16
FS = 8                                            # base font size (pt) at print size
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": FS,
    "axes.titlesize": FS + 0.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 4,
    "axes.labelsize": FS, "axes.labelcolor": INK2,
    "xtick.labelsize": FS - 0.5, "ytick.labelsize": FS - 0.5, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXISC, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.fontsize": FS - 0.5, "legend.frameon": False, "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.2, "legend.handlelength": 1.4,
    "lines.linewidth": 1.4, "lines.markersize": 5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
PCT = mticker.FuncFormatter(lambda v, _: f"{v:g}%")

def savefig(fig, name, note=None):
    """Vector PDF (paper) + 600-dpi PNG (slides), then show inline."""
    if DRY_RUN:
        bb = fig.get_tightbbox(fig.canvas.get_renderer()); W, H = fig.get_size_inches()
        fig.text((bb.x0 + bb.width / 2) / W, (bb.y0 - 0.04) / H,
                 "DRY RUN · synthetic data + mock models — numbers are NOT results",
                 ha="center", va="top", fontsize=7, color=NEG)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.show()
    print(f"saved → {FIG_DIR / name}.pdf" + (f"   ({note})" if note else ""))

def panel_label(ax, text, x=-0.02, y=1.02):
    ax.text(x, y, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=FS + 1, fontweight="bold")
