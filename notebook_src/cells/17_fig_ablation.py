%%stage analyze
# Fig. 5 — E5 ablation: change vs full AeroVerify (accuracy at full coverage, and AURC)
ROWS = [("E4 −tiles", "− native tiles"), ("E4 −ground", "− grounding"), ("E4 −probe", "− contrastive probe"),
        ("E4 −weights", "− reliability weights"), ("E4 −gate", "− resolution gate"), ("Lite", "AeroVerify-Lite")]
COLS = [(ds, m) for ds in DATASETS for m in MODELS]
cmap = LinearSegmentedColormap.from_list("div", [NEG, "#f6c9c6", MID, "#c4daf6", POS])
fig, axes = plt.subplots(1, 2, figsize=(IEEE_2COL, 0.32 * len(ROWS) + 1.25), gridspec_kw=dict(wspace=0.06))
for ax, (metric, title, sign) in zip(axes, [("d_acc", "Δ accuracy (pp)  ·  higher is better", 1),
                                            ("d_AURC", "Δ AURC (pp)  ·  lower is better", -1)]):
    M = np.full((len(ROWS), len(COLS)), np.nan)
    for i, (v, _) in enumerate(ROWS):
        for j, (ds, m) in enumerate(COLS):
            r = ABL[(ABL.dataset == DS_SHORT[ds]) & (ABL.model == MODEL_LABEL[m]) & (ABL.variant == v)]
            gate_irrelevant = v == "E4 −gate" and EVD[(EVD.ds == ds)].gate.all()
            if len(r) and not gate_irrelevant: M[i, j] = r[metric].iloc[0]
    lim = max(2.0, np.nanmax(np.abs(M)) if np.isfinite(M).any() else 2.0)
    ax.imshow(sign * M, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(ROWS)):
        for j in range(len(COLS)):
            if np.isnan(M[i, j]):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc=BAND, ec="white", lw=1))
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color=MUTED)
            else:
                strong = abs(M[i, j]) > 0.6 * lim
                ax.text(j, i, f"{M[i, j]:+.1f}".replace("-", "−"), ha="center", va="center", fontsize=7.2,
                        color="white" if strong else INK, fontweight="bold" if strong else "normal")
    ax.set_xticks(range(len(COLS)), [f"{DS_SHORT[ds]}\n{MODEL_LABEL[m]}" for ds, m in COLS], fontsize=7)
    ax.set_yticks(range(len(ROWS)), [lab for _, lab in ROWS] if ax is axes[0] else [""] * len(ROWS))
    ax.tick_params(length=0); ax.grid(False)
    for s_ in ax.spines.values(): s_.set_visible(False)
    for x in np.arange(-0.5, len(COLS), 1): ax.axvline(x, color="white", lw=1.5)
    for y in np.arange(-0.5, len(ROWS), 1): ax.axhline(y, color="white", lw=1.5)
    ax.set_title(title, fontsize=8)
savefig(fig, "fig05_ablation", "red = removing the component hurts (it is useful), blue = removing it helps; n/a = gate always open on UAV imagery")
