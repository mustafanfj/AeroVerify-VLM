%%stage analyze
# Fig. 9 — Counting on UAV imagery: predicted vs true count
if "floodnet" not in DATASETS:
    print("FloodNet not selected — skipped")
else:
    pairs = [(m, meth) for m in MODELS for meth in ("E2", "E4")]
    fig, axes = plt.subplots(1, len(pairs), figsize=(IEEE_2COL, 2.05), squeeze=False, sharey=True, gridspec_kw=dict(wspace=0.08))
    cnt = EV[(EV.ds == "floodnet") & (EV.kind == "count")]
    vmax = float(cnt.label.astype(float).max()) + 2
    rng = np.random.default_rng(0)
    for ax, (m, meth) in zip(axes[0], pairs):
        g = cnt[(cnt.model == m) & (cnt.method == meth)].copy()
        g["y"] = g.label.astype(float); g["p"] = g.pred.map(_to_int).astype(float)
        for t, col, mk in [("count_simple", BLUE_RAMP[4], "o"), ("count_complex", METHOD_COLORS["E1"], "^")]:
            h = g[(g.type == t) & g.p.notna()]
            ax.scatter(h.y + rng.uniform(-0.25, 0.25, len(h)), h.p.clip(upper=vmax) + rng.uniform(-0.25, 0.25, len(h)),
                       s=9, marker=mk, color=col, alpha=0.65, lw=0, label=TYPE_LABEL[t])
        ax.plot([0, vmax], [0, vmax], color=INK, lw=0.7, ls=(0, (3, 2)))
        ax.set_xlim(-1, vmax); ax.set_ylim(-1, vmax); ax.set_aspect("equal", adjustable="box")
        n_clip = int((g.p > vmax).sum())
        if n_clip: ax.text(0.97, 0.03, f"{n_clip} above frame", transform=ax.transAxes, ha="right", fontsize=6.8, color=MUTED)
        mae = g.abs_err.mean(); exact = g.correct.mean() * 100
        ax.text(0.04, 0.96, f"MAE {mae:.2f}\nexact {exact:.0f}%", transform=ax.transAxes, va="top", fontsize=7.2,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        ax.set_title(f"{MODEL_LABEL[m]}\n{'E2 Structured' if meth == 'E2' else 'AeroVerify'}", fontsize=7.8, loc="center")
        ax.set_xlabel("true count")
    axes[0][0].set_ylabel("predicted count")
    hh, ll = axes[0][0].get_legend_handles_labels()
    fig.legend(hh, ll, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.16), fontsize=7.2, markerscale=1.4)
    savefig(fig, "fig09_counting", "FloodNet-VQA counting questions; dashed = perfect")
