%%stage analyze
# Fig. 1 — AeroVerify-VLM pipeline (drawn on a real example)
import textwrap
DS0 = DATASETS[0]
ex = EVD[(EVD.ds == DS0) & (EVD.role == "eval") & (EVD.pert == "clean") & (EVD.type == ("count_complex" if DS0 == "floodnet" else "count"))]
ex = ex.assign(_lab=ex.label.map(_to_int)).sort_values("_lab", ascending=False)        # a busy scene illustrates best
ex_q = ex.qid.iloc[min(2, len(ex) - 1)] if len(ex) else EV[EV.ds == DS0].qid.iloc[0]
ex_row = QI.loc[ex_q]; ex_img = get_image(DS0, ex_row.img)

fig = plt.figure(figsize=(IEEE_2COL, 2.75)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
def box(x, y, w, h, title, body, fc, ec, tc=INK, bold_title=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=0.8))
    ax.text(x + 1.2, y + h - 1.4, title, ha="left", va="top", fontsize=7.6, fontweight="bold" if bold_title else "normal", color=tc)
    if body: ax.text(x + 1.2, y + h - 3.9, body, ha="left", va="top", fontsize=7, color=tc if tc != INK else INK2, linespacing=1.15)
def arrow(x0, y0, x1, y1, c=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=c, shrinkA=0, shrinkB=0))

# input image with tile grid
thumb = ex_img.copy(); thumb.thumbnail((600, 600))
iax = fig.add_axes([0.012, 0.16, 0.19, 0.62]); iax.imshow(thumb); iax.set_xticks([]); iax.set_yticks([]); iax.grid(False)
for s in iax.spines.values(): s.set_color(AXISC)
tw, th = thumb.size
iax.axvline(tw / 2, color="white", lw=0.9); iax.axhline(th / 2, color="white", lw=0.9)
ax.text(1.2, 39.6, "Input", fontsize=8, fontweight="bold", va="top")
ax.text(1.2, 37.0, "UAV image + question", fontsize=7, color=INK2, va="top")
ax.text(1.2, 4.6, textwrap.fill("Q: " + ex_row.question, 34), fontsize=7, color=INK, va="top", style="italic")

lanes = [("Full view (E2)", "whole image, downsampled;\ntype-specific prompt"),
         ("Native tiles (E3)", "2×2 full-resolution tiles;\nused only if zoom gain ≥ 1"),
         ("Contrastive probe", "forced choice / decomposed\n(no yes–no tokens)"),
         ("Grounding", "detect instances, verify each\n(e.g. flooded or dry?)")]
y0s = [28.4, 19.6, 10.8, 2.0]
for (t, b), y in zip(lanes, y0s):
    box(30, y, 25, 7.4, t, b, "#eef4fc", BLUE_RAMP[2]); arrow(22.5, 20, 29.6, y + 3.7)
box(60.5, 11.5, 18.5, 17, "Reliability-\nweighted fusion", None, METHOD_COLORS["E4"], METHOD_COLORS["E4"], tc="white")
ax.text(69.75, 19.9, r"$w_s=\log\,\frac{(K-1)\,a_s}{1-a_s}$", ha="center", va="center", fontsize=8.5, color="white")
ax.text(69.75, 16.6, "$a_s$: DEV accuracy\nof source $s$", ha="center", va="center", fontsize=6.8, color="white")
ax.text(69.75, 13.0, "then calibrated per\nquestion type (DEV)", ha="center", va="center", fontsize=6.8, color="white", style="italic")
for y in y0s: arrow(55.6, y + 3.7, 60.2, 20)
box(84, 22, 15, 9.5, "Answer", "+ confidence\n+ evidence boxes", "#eef4fc", BLUE_RAMP[2])
box(84, 8.5, 15, 9.5, "Abstain", "confidence < τ\n(τ tuned on DEV)", BAND, AXISC)
arrow(79.3, 22, 83.7, 26.5, METHOD_COLORS["E4"]); arrow(79.3, 18, 83.7, 13.5, MUTED)
ax.text(30, 39.6, "Evidence sources", fontsize=8, fontweight="bold", va="top")
ax.text(60.5, 39.6, "Decision", fontsize=8, fontweight="bold", va="top")
ax.text(84, 39.6, "Output", fontsize=8, fontweight="bold", va="top")
ax.text(84, 6.4, "Lite: exit early\nwhen full view +\nprobe agree", fontsize=6.8, color=METHOD_COLORS["Lite"], va="top")
savefig(fig, "fig01_pipeline")
