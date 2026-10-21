%%stage analyze
# Fig. 7 — E7 robustness to UAV-specific degradations
if ROB.empty:
    print("no robustness runs (AV_ROBUSTNESS=0) — skipped")
else:
    PT = {"altitude": ("Altitude", "down-sample factor"), "motion_blur": ("Motion blur", "blur length (px)"),
          "haze": ("Haze", "haze blend"), "jpeg": ("JPEG compression", "quality")}
    RM = [("E1", "E1"), ("E2", "E2"), ("E3", "E3"), ("E4", "E4")]
    ds = next(iter(ROBUST_IMGS))
    fig, axes = plt.subplots(len(MODELS), len(PERTURBATIONS), figsize=(IEEE_2COL, 1.75 * len(MODELS) + 0.45),
                             squeeze=False, sharey="row", gridspec_kw=dict(hspace=0.55, wspace=0.17, top=0.86))
    for i, m in enumerate(MODELS):
        for j, (pert, levels) in enumerate(PERTURBATIONS.items()):
            ax = axes[i, j]; xs = list(range(len(levels) + 1))
            for meth, key in RM:
                d = ROB[(ROB.ds == ds) & (ROB.model == m) & (ROB.method == meth)]
                ys = [d[d.pert == "clean"].OA.mean()] + [d[(d.pert == pert) & np.isclose(d.level, lv)].OA.mean() for lv in levels]
                ax.plot(xs, ys, "-" + METHOD_MARKERS[key], color=METHOD_COLORS[key], lw=1.3, ms=4, mec="white", mew=0.5)
            ax.set_xticks(xs, ["clean"] + [f"{lv:g}" for lv in levels]); ax.set_xlim(-0.3, len(levels) + 0.3)
            ax.yaxis.set_major_formatter(PCT)
            if i == 0: ax.set_title(PT[pert][0], fontsize=8)
            ax.set_xlabel(PT[pert][1], fontsize=7, labelpad=1)
            if j == 0: ax.set_ylabel(f"{MODEL_LABEL[m]}\nOA")
    fig.legend(handles=[Line2D([], [], marker=METHOD_MARKERS[k], color=METHOD_COLORS[k], ms=4.5,
                               label=METHOD_LABELS[k] + (" (no abstention)" if k == "E4" else "")) for _, k in RM],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.88), fontsize=7.5)
    savefig(fig, "fig07_robustness", f"{DS_LABEL[ds]}, {len(ROBUST_IMGS[ds])} EVAL images per condition")
