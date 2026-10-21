%%stage analyze
# Fig. 3 — Main results: accuracy per question type with 95% CIs (full coverage)
MAIN_M = [("E1", "E1"), ("E2", "E2"), ("E3", "E3"), ("E4", "E4")]
nd, nm = len(DATASETS), len(MODELS)
heights = [len(TYPES[ds]) + 2 for ds in DATASETS]
fig, axes = plt.subplots(nd, nm, figsize=(IEEE_2COL, 0.42 * sum(heights) + 0.9), squeeze=False,
                         gridspec_kw=dict(height_ratios=heights, hspace=0.3, wspace=0.1, top=0.875), sharex=True)
for i, ds in enumerate(DATASETS):
    cats = ["OA", "AA"] + list(TYPES[ds]); yc = np.arange(len(cats))[::-1]; offs = np.linspace(0.27, -0.27, 4)
    for j, m in enumerate(MODELS):
        ax = axes[i, j]
        for (meth, key), off in zip(MAIN_M, offs):
            g = EV[(EV.model == m) & (EV.ds == ds) & (EV.method == meth)]
            s, ci = summarize(g), boot_ci(g)
            for y, c in zip(yc, cats):
                if c in ci and not np.isnan(ci[c][0]):
                    ax.plot(ci[c], [y + off] * 2, color=METHOD_COLORS[key], lw=1.1, alpha=0.5, solid_capstyle="round")
                ax.plot(s[c], y + off, METHOD_MARKERS[key], color=METHOD_COLORS[key], ms=4.8, mec="white", mew=0.6, zorder=3)
        for meth, mk in [("majority", "|"), ("blind", "x")]:
            s = summarize(EV[(EV.model == m) & (EV.ds == ds) & (EV.method == meth)])
            for y, c in zip(yc, cats):
                ax.plot(s[c], y, mk, color=BASELINE_COLOR, ms=6.5 if mk == "|" else 4.5, mew=1.1, zorder=2)
        for y in yc[::2]: ax.axhspan(y - 0.5, y + 0.5, color=BAND, zorder=0, lw=0)
        ax.axhline(yc[1] - 0.5, color=AXISC, lw=0.6)
        ax.set_yticks(yc, [TYPE_LABEL[c] for c in cats] if j == 0 else [""] * len(cats)); ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(cats) - 0.4); ax.set_xlim(-3, 102); ax.set_xticks([0, 25, 50, 75, 100]); ax.xaxis.set_major_formatter(PCT)
        if j > 0: ax.tick_params(axis="y", length=0)
        ax.set_title(f"{DS_SHORT[ds]} · {MODEL_LABEL[m]}", fontsize=8)
        if i == nd - 1: ax.set_xlabel("accuracy on EVAL questions")
h = [Line2D([], [], marker=METHOD_MARKERS[k], color=METHOD_COLORS[k], lw=1, ms=4.5,
            label=METHOD_LABELS[k] + (" (no abstention)" if k == "E4" else "")) for _, k in MAIN_M]
h += [Line2D([], [], marker="|", color=BASELINE_COLOR, lw=0, ms=7, mew=1.2, label="Most-common answer"),
      Line2D([], [], marker="x", color=BASELINE_COLOR, lw=0, ms=4.5, mew=1.2, label="Question-only")]
fig.legend(handles=h, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 0.995), fontsize=7.5)
savefig(fig, "fig03_main_results", "lines = 95% image-level bootstrap CI")
