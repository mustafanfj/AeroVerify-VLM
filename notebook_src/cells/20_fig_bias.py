%%stage analyze
# Fig. 8 — Answer bias on yes/no questions: predicted "yes" rate vs truth, and hallucinated "yes"
BM = [("blind", None), ("E1", "E1"), ("E2", "E2"), ("E3", "E3"), ("E4", "E4")]
fig, axes = plt.subplots(1, 2, figsize=(IEEE_2COL, 0.95 + 0.3 * len(BM) * len(MODELS) * len(DATASETS) / 2),
                         sharey=True, gridspec_kw=dict(wspace=0.08))
blocks = [(ds, m) for ds in DATASETS for m in MODELS]
for ax, (metric, title) in zip(axes, [("yes_rate", "(a) Share of “yes” answers"), ("FPR", "(b) Hallucinated “yes” (truth: “no”)")]):
    yy, ticks = 0.0, []
    for ds, m in blocks:
        ax.text(0, yy + 0.05, f"{DS_SHORT[ds]} · {MODEL_LABEL[m]}", transform=ax.get_yaxis_transform(), ha="left",
                va="bottom", fontsize=7.4, fontweight="bold")
        yy -= 0.75; top = yy
        for meth, key in BM:
            g = EV[(EV.model == m) & (EV.ds == ds) & (EV.method == meth)]
            v = summarize(g)[metric]; col = METHOD_COLORS[key] if key else BASELINE_COLOR
            ax.barh(yy, v, height=0.7, color=col)
            ax.annotate(f"{v:.0f}%", (v, yy), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7, color=INK2)
            ticks.append((yy, METHOD_LABELS[meth] + (" (no abst.)" if meth == "E4" else ""))); yy -= 1
        if metric == "yes_rate":
            tv = summarize(EV[(EV.model == m) & (EV.ds == ds) & (EV.method == "E2")])["true_yes_rate"]
            ax.plot([tv, tv], [yy + 0.5, top + 0.5], color=INK, lw=0.9, ls=(0, (3, 2)))
        yy -= 0.6
    ax.set_yticks([p for p, _ in ticks], [l for _, l in ticks]); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 112); ax.xaxis.set_major_formatter(PCT); ax.set_ylim(yy + 0.5, 0.9); ax.set_title(title, fontsize=8)
savefig(fig, "fig08_answer_bias", "dashed line in (a) = true share of “yes”; yes/no-type questions only")
