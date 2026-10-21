%%stage analyze
# Fig. 11 — What does AeroVerify abstain on?  Accuracy of E2 and AeroVerify on answered vs abstained questions
blocks = [(ds, m) for ds in DATASETS for m in MODELS]
fig, ax = plt.subplots(figsize=(IEEE_2COL, 2.3))
x = np.arange(len(blocks)); w = 0.2
bars = [("E2_on_answered", "E2 · questions AeroVerify answers", METHOD_COLORS["E2"], None),
        ("E4_on_answered", "AeroVerify · answered", METHOD_COLORS["E4"], None),
        ("E2_on_abstained", "E2 · questions AeroVerify abstains on", METHOD_COLORS["E2"], "////"),
        ("E4_on_abstained", "AeroVerify · abstained (if forced)", METHOD_COLORS["E4"], "////")]
for k, (col, lab, c, hatch) in enumerate(bars):
    vals = [ABST[(ABST.dataset == DS_SHORT[ds]) & (ABST.model == MODEL_LABEL[m])][col].iloc[0] for ds, m in blocks]
    ax.bar(x + (k - 1.5) * w, vals, width=w * 0.9, color=c if hatch is None else "white", edgecolor=c,
           hatch=hatch, lw=1.0, label=lab)
    for xi, v in zip(x + (k - 1.5) * w, vals):
        if v == v: ax.text(xi, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=6.8, color=INK2)
for xi, (ds, m) in zip(x, blocks):                                     # τ from DEV may answer everything
    if ABST[(ABST.dataset == DS_SHORT[ds]) & (ABST.model == MODEL_LABEL[m])].n_abstained.iloc[0] == 0:
        ax.text(xi + w, 4, "τ answers\nall questions", ha="center", va="bottom", fontsize=6.6, color=MUTED, style="italic")
ax.set_xticks(x, [f"{DS_SHORT[ds]} · {MODEL_LABEL[m]}\n({ABST[(ABST.dataset == DS_SHORT[ds]) & (ABST.model == MODEL_LABEL[m])].answered_pct.iloc[0]:.0f}% answered)"
                  for ds, m in blocks], fontsize=7.2)
ax.set_ylim(0, 105); ax.yaxis.set_major_formatter(PCT); ax.set_ylabel("accuracy"); ax.grid(axis="x", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize=7.2)
savefig(fig, "fig11_abstention", "abstained questions are those that even the best single prompt mostly gets wrong")
