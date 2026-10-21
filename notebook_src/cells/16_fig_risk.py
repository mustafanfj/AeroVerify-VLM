%%stage analyze
# Fig. 4 — Selective prediction: risk–coverage curves
def rc_curve(g):
    cov, risk = rc_expected(g.conf.values, g.correct.values); return cov * 100, risk * 100
CURVES = [("E4", "AeroVerify (reliability-weighted)", METHOD_COLORS["E4"], "-"),
          ("E4 −weights", "Plurality vote, all sources (run-1 rule)", MUTED, "-"),
          ("SC", "Self-consistency (E2 + probe)", METHOD_COLORS["E2"], "-"),
          ("Type prior", "E2 + question-type prior", INK2, (0, (3, 1.5)))]
AU_NAMES = ["ours", "plurality", "SC", "type prior"]
fig, axes = plt.subplots(len(DATASETS), len(MODELS), figsize=(IEEE_2COL, 2.15 * len(DATASETS) + 0.4), squeeze=False,
                         gridspec_kw=dict(hspace=0.62, wspace=0.2, top=0.78))
for i, ds in enumerate(DATASETS):
    for j, m in enumerate(MODELS):
        ax = axes[i, j]; ymax = 0; au = []
        for meth, lab, col, ls in CURVES:
            g = EV[(EV.model == m) & (EV.ds == ds) & (EV.method == meth)]
            if g.empty: continue
            cov, risk = rc_curve(g); keep = cov >= 20; au.append(aurc(g))
            ax.plot(cov[keep], risk[keep], linestyle=ls, color=col, lw=1.6 if meth == "E4" else 1.1, label=lab)
            ymax = max(ymax, risk[keep].max())
            if meth == "E4":
                s = summarize(g, selective=True)
                ax.plot(s["coverage"], 100 - s["OA"], "o", ms=7, mfc="none", mec=col, mew=1.2, zorder=5, clip_on=False)
        for k in ["E1", "E2", "E3"]:
            g = EV[(EV.model == m) & (EV.ds == ds) & (EV.method == k)]; rv = (1 - g.correct.mean()) * 100; ymax = max(ymax, rv)
            ax.plot(100, rv, METHOD_MARKERS[k], color=METHOD_COLORS[k], ms=5, mec="white", mew=0.6, zorder=4, clip_on=False)
        ax.set_xlim(20, 101); ax.set_ylim(0, ymax * 1.08 + 1); ax.xaxis.set_major_formatter(PCT); ax.yaxis.set_major_formatter(PCT)
        ax.set_title(f"{DS_SHORT[ds]} · {MODEL_LABEL[m]}", fontsize=8, pad=13)
        ax.text(0.0, 1.015, "AURC↓  " + "  ·  ".join(f"{n} {a:.1f}" for n, a in zip(AU_NAMES, au)), transform=ax.transAxes,
                va="bottom", ha="left", fontsize=6.6, color=INK2)
        if i == len(DATASETS) - 1: ax.set_xlabel("coverage (share of questions answered)")
        if j == 0: ax.set_ylabel("risk (error on answered)")
pts = [Line2D([], [], color=c, lw=1.6 if m_ == "E4" else 1.1, linestyle=ls_, label=l) for m_, l, c, ls_ in CURVES]
pts += [Line2D([], [], marker="o", mfc="none", mec=METHOD_COLORS["E4"], lw=0, ms=6, label="operating point (τ from DEV)")]
pts += [Line2D([], [], marker=METHOD_MARKERS[k], color=METHOD_COLORS[k], lw=0, ms=5, label=METHOD_LABELS[k] + " (answers all)")
        for k in ["E1", "E2", "E3"]]
fig.legend(handles=pts, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.855), fontsize=7.2)
savefig(fig, "fig04_risk_coverage", "lower curve / lower AURC = better; all confidences calibrated per question type on DEV")
