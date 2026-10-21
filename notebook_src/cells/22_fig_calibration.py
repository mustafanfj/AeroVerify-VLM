%%stage analyze
# Fig. 10 — Is AeroVerify's confidence calibrated? Reliability diagram + ECE
def reliability(g, bins=6, min_n=5):
    e = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(g.conf, e) - 1, 0, bins - 1); out = []
    for b in range(bins):
        h = g[idx == b]
        if len(h): out.append((h.conf.mean(), h.correct.mean(), len(h)))
    ece = sum(n * abs(c - a) for c, a, n in out) / max(len(g), 1)
    out = [o for o in out if o[2] >= min_n]
    return out, ece * 100
fig, axes = plt.subplots(1, len(DATASETS), figsize=(IEEE_2COL, 2.4), squeeze=False, gridspec_kw=dict(wspace=0.18))
shades = [METHOD_COLORS["E4"], BLUE_RAMP[5]]; mks = ["o", "s"]
for ax, ds in zip(axes[0], DATASETS):
    ax.plot([0, 1], [0, 1], color=INK, lw=0.7, ls=(0, (3, 2)), label="perfect calibration")
    for m, c, mk in zip(MODELS, shades, mks):
        g = EV[(EV.model == m) & (EV.ds == ds) & (EV.method == "E4")]
        pts, ece = reliability(g)
        _, ece_raw = reliability(g.assign(conf=g.conf_raw))
        x, y, n = zip(*pts)
        ax.plot(x, y, "-", color=c, lw=1.2, label=f"{MODEL_LABEL[m]} (ECE {ece_raw:.1f} → {ece:.1f})")
        ax.scatter(x, y, s=[8 + 60 * k / max(n) for k in n], marker=mk, color=c, edgecolor="white", lw=0.5, zorder=3)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.set_xlabel("AeroVerify confidence (calibrated on DEV)"); ax.set_title(DS_LABEL[ds], fontsize=8); ax.legend(loc="lower right", fontsize=7)
axes[0][0].set_ylabel("accuracy")
savefig(fig, "fig10_calibration", "ECE (%) before → after type-conditional Platt scaling fit on DEV; marker area ∝ questions per bin")
