%%stage analyze
# Fig. 6 — E6 accuracy vs latency (per question) with the Pareto front
EM = [("E1", "E1"), ("E2", "E2"), ("E3", "E3"), ("Lite", "Lite"), ("E4", "E4")]
fig, axes = plt.subplots(1, len(DATASETS), figsize=(IEEE_2COL, 2.45), squeeze=False, gridspec_kw=dict(wspace=0.22))
for ax, ds in zip(axes[0], DATASETS):
    for m, fill in zip(MODELS, ["full", "none"]):
        pts = []
        for meth, key in EM:
            g = EV[(EV.model == m) & (EV.ds == ds) & (EV.method == meth)]
            if g.empty: continue
            x, y = g.latency.mean(), g.correct.mean() * 100; pts.append((x, y))
            ax.plot(x, y, METHOD_MARKERS[key], color=METHOD_COLORS[key], ms=6.5, mec=METHOD_COLORS[key] if fill == "none" else "white",
                    mfc=METHOD_COLORS[key] if fill == "full" else "white", mew=1.1, zorder=3)
        pts.sort(); front, best = [], -1
        for x, y in pts:
            if y > best: front.append((x, y)); best = y
        ax.plot(*zip(*front), color=MUTED, lw=0.8, ls="-" if fill == "full" else (0, (3, 2)), zorder=1)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g} s"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter()); ax.yaxis.set_major_formatter(PCT)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo - 2, hi + 4)
    ax.set_xlabel("latency per question (log scale)"); ax.set_title(DS_LABEL[ds], fontsize=8)
axes[0][0].set_ylabel("overall accuracy (OA)")
h = [Line2D([], [], marker=METHOD_MARKERS[k], color=METHOD_COLORS[k], lw=0, ms=5.5, label=METHOD_LABELS[k]) for _, k in EM]
h += [Line2D([], [], marker="o", color=INK2, mfc=INK2, lw=0.8, ms=5, label=f"{MODEL_LABEL[MODELS[0]]} (filled)")]
if len(MODELS) > 1:
    h += [Line2D([], [], marker="o", color=INK2, mfc="white", lw=0.8, ls=(0, (3, 2)), ms=5, label=f"{MODEL_LABEL[MODELS[1]]} (hollow)")]
fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=7.2)
savefig(fig, "fig06_efficiency", "RTX 4090, batch 1; lines = Pareto front per model; calls per question in table_efficiency")
