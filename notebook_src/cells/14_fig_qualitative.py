%%stage analyze
# Fig. 2 — Evidence-grounded answers on real UAV images (primary model)
import textwrap
PRIMARY = "moondream2" if "moondream2" in MODELS else MODELS[0]
DSQ = "floodnet" if "floodnet" in DATASETS else DATASETS[0]
SRC_NAME = {"full": "Full view", "tiles": "Tiles", "probe": "Probe", "ground": "Grounding"}

CLEAN_REC = {}
with open(PRED_DIR / f"{PRIMARY}.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r["pert"] == "clean" and r["role"] == "eval" and r["ds"] == DSQ: CLEAN_REC[r["qid"]] = r

def _score(q):
    """Prefer examples that are easy to read: several sources right, and for counts a correct grounding source."""
    r, qi = CLEAN_REC.get(q), QI.loc[q]
    if r is None: return -99
    srcs = r["sources"]; ok = lambda s_: srcs.get(s_) is not None and str(srcs[s_]["ans"]) == str(qi.label)
    sc = sum(ok(s_) for s_ in SRC)
    if qi.kind == "count":
        sc += 3 * ok("ground") + (1 if 2 <= _to_int(qi.label) <= 12 else 0)
    return sc

def _pick_examples(m, ds, k=3):
    """One example per question type: prefer E2 wrong → AeroVerify right, then AeroVerify right, then anything;
    within a tier, the most legible example (see _score)."""
    e = EV[(EV.model == m) & (EV.ds == ds)].pivot_table(index="qid", columns="method", values="correct", aggfunc="first")
    e4 = EV[(EV.model == m) & (EV.ds == ds) & (EV.method == "E4")].set_index("qid")
    ans = e4.answered.reindex(e.index).astype(bool).values
    tiers = [e.index[e["E4"].astype(bool) & ~e["E2"].astype(bool) & ans], e.index[e["E4"].astype(bool) & ans], e.index]
    order = ["count_complex", "condition", "yesno", "count_simple"] if ds == "floodnet" else ["presence", "count", "comp", "rural_urban"]
    picks, used_imgs = [], set()
    for t in order:
        for tier in tiers:
            c = [q for q in tier if QI.loc[q, "type"] == t and QI.loc[q, "img"] not in used_imgs]
            if c:
                best = max(c, key=_score); picks.append(best); used_imgs.add(QI.loc[best, "img"]); break
        if len(picks) == k: break
    return picks[:k]

def _fmt_ans(a, kind):
    if a is None or (isinstance(a, float) and np.isnan(a)): return "–"
    if kind == "count_bin": return COUNT_BINS[BIN_KEYS.index(a)]
    return {MIXED: "partial", "mixed": "partial", NONFLOODED: "dry"}.get(a, str(a))

ex = _pick_examples(PRIMARY, DSQ)
fig = plt.figure(figsize=(IEEE_2COL, 3.75))
gs = fig.add_gridspec(2, len(ex), height_ratios=[1, 0.72], hspace=0.04, wspace=0.1)
recmap = {q: CLEAN_REC[q] for q in ex}
USED = set()
for j, qid in enumerate(ex):
    rec, qi = recmap[qid], QI.loc[qid]
    im = get_image(DSQ, qi.img); Wn, Hn = im.size; th = im.copy(); th.thumbnail((900, 900)); sx = th.size[0] / Wn
    ax = fig.add_subplot(gs[0, j]); ax.imshow(th); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s_ in ax.spines.values(): s_.set_visible(False)
    if rec["gate"] or DSQ == "rsvqa_lr":
        for n, (x0, y0, x1, y1) in tile_boxes(im.size).items():
            ax.add_patch(Rectangle((x0 * sx, y0 * sx), (x1 - x0) * sx, (y1 - y0) * sx, fill=False, ec="white", lw=0.8)); USED.add("tiles")
            v = rec["sources"]["tiles"]["per"].get(n)
            if v is not None:
                v = "|".join(map(str, v)) if isinstance(v, list) else {MIXED: "partial", "mixed": "partial", NONFLOODED: "dry"}.get(v, str(v))
                ax.text(x0 * sx + 6, y0 * sx + 6, v, fontsize=7, va="top", color=INK,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    g = rec["sources"].get("ground")
    if g and g.get("boxes"):
        labs = g.get("labels") or []
        for i, b in enumerate(next(iter(g["boxes"].values()))[:60]):
            lab = labs[i] if i < len(labs) else None
            col = NEG if lab == FLOODED else (POS if lab == NONFLOODED else "#ffd34d")   # yellow = detected, not verified
            USED.add({NEG: "flooded", POS: "dry"}.get(col, "detected"))
            ax.add_patch(Rectangle((b[0] * sx, b[1] * sx), (b[2] - b[0]) * sx, (b[3] - b[1]) * sx, fill=False, ec=col, lw=1.0))
    ax.set_title(textwrap.fill(qi.question, 33), fontsize=7.4, fontweight="normal", loc="center", color=INK, pad=3)
    ax = fig.add_subplot(gs[1, j]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    e4 = EV[(EV.model == PRIMARY) & (EV.method == "E4") & (EV.qid == qid)].iloc[0]
    lines = [("Truth", "", _fmt_ans(qi.label, qi.kind), INK, True)]
    for s_ in SRC:
        v = rec["sources"].get(s_)
        if v is None or (s_ == "tiles" and not rec["gate"]): continue
        a = v["ans"]; w = W.get((PRIMARY, DSQ, qi.route, s_), 0)
        agree = a is not None and str(a) == str(e4.pred)
        lines.append((SRC_NAME[s_], f"w={w:.1f}", _fmt_ans(a, qi.kind), METHOD_COLORS["E4"] if agree else MUTED, False))
    verdict = "abstain" if not e4.answered else _fmt_ans(e4.pred, qi.kind) + (" ✓" if e4.correct else " ✗")
    lines.append(("AeroVerify", f"p={e4.conf:.2f}", verdict, METHOD_COLORS["E4"], True))
    for i, (k_, w_, v_, c_, b_) in enumerate(lines):
        y = 0.97 - i * 0.16
        ax.text(0.02, y, k_, fontsize=7.2, color=INK2, va="top", fontweight="bold" if b_ else "normal")
        ax.text(0.43, y, w_, fontsize=7, color=MUTED, va="top", ha="left")
        ax.text(0.98, y, v_, fontsize=7.2, color=c_, va="top", ha="right", fontweight="bold" if b_ else "normal")
    ax.plot([0.02, 0.98], [0.97 - (len(lines) - 1) * 0.16 + 0.035] * 2, color=GRIDC, lw=0.6)
_tile = Rectangle((0, 0), 1, 1, fill=False, ec=MUTED, lw=1, label="2×2 native tiles (tile answer)")
handles = [h for key, h in [("tiles", _tile),
                            ("flooded", Line2D([], [], color=NEG, lw=1.2, label="detected · verified flooded")),
                            ("dry", Line2D([], [], color=POS, lw=1.2, label="detected · verified dry")),
                            ("detected", Line2D([], [], color="#ffd34d", lw=1.2, label="detected"))] if key in USED]
handles.append(Line2D([], [], color=METHOD_COLORS["E4"], lw=0, marker="s", ms=5, label="agrees with final answer"))
fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.01), fontsize=6.8)
savefig(fig, "fig02_qualitative", f"{MODEL_LABEL[PRIMARY]}; w = reliability weight, p = calibrated confidence, partial = partly flooded, dry = non flooded")
