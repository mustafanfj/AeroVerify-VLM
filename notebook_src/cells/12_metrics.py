%%stage analyze
# ── Metrics, bootstrap CIs, significance, LaTeX tables ────────────────────────
from scipy.stats import binomtest

def macro_f1(y, p, labels):
    f = []
    for c in labels:
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        if tp + fp + fn: f.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)) if f else np.nan

def summarize(g, selective=False):
    ds = g.ds.iloc[0]; h = g[g.answered] if selective else g; out = {}
    for t in TYPES[ds]:
        s = h[h.type == t]; out[t] = s.correct.mean() * 100 if len(s) else np.nan
    out["OA"] = h.correct.mean() * 100 if len(h) else np.nan
    out["AA"] = np.nanmean([out[t] for t in TYPES[ds]])
    cat = [t for t in TYPES[ds] if TYPES[ds][t] != "count"]
    out["F1"] = np.nanmean([macro_f1(h[h.type == t].label.values, h[h.type == t].pred.fillna("∅").values,
                                     sorted(h[h.type == t].label.unique())) * 100 for t in cat if (h.type == t).any()])
    out["MAE"] = h[h.kind == "count"].abs_err.mean() if (h.kind == "count").any() else np.nan
    out["coverage"] = g.answered.mean() * 100 if selective else 100.0
    yn = h[(h.kind == "yesno") & (h.label == "no")]
    out["FPR"] = (yn.pred == "yes").mean() * 100 if len(yn) else np.nan          # hallucinated "yes"
    yy = h[h.kind == "yesno"]
    out["yes_rate"] = (yy.pred == "yes").mean() * 100 if len(yy) else np.nan
    out["true_yes_rate"] = (yy.label == "yes").mean() * 100 if len(yy) else np.nan
    out["invalid"] = (~g.valid).mean() * 100; out["n"] = len(h)
    return out

def boot_ci(g, B=2000, selective=False, key="OA"):
    h = g[g.answered] if selective else g
    imgs = h.img_key.unique(); ix = h.img_key.map({i: k for k, i in enumerate(imgs)}).values
    rng = np.random.default_rng(SEED); Wb = np.zeros((B, len(imgs)))
    np.add.at(Wb, (np.repeat(np.arange(B), len(imgs)), rng.integers(0, len(imgs), (B, len(imgs))).ravel()), 1)
    res = {}
    for name in ["OA"] + list(TYPES[g.ds.iloc[0]]):
        mask = np.ones(len(h), bool) if name == "OA" else (h.type == name).values
        if not mask.any(): res[name] = (np.nan, np.nan); continue
        c = np.bincount(ix[mask], weights=h.correct.values[mask].astype(float), minlength=len(imgs))
        n = np.bincount(ix[mask], minlength=len(imgs)).astype(float)
        a = (Wb @ c) / np.maximum(Wb @ n, 1e-9) * 100; res[name] = tuple(np.percentile(a, [2.5, 97.5]))
    return res

def mcnemar(m, ds, a, b):
    A_ = EV[(EV.model == m) & (EV.ds == ds) & (EV.method == a)].set_index("qid").correct
    B_ = EV[(EV.model == m) & (EV.ds == ds) & (EV.method == b)].set_index("qid").correct.reindex(A_.index)
    w, l = int((~A_ & B_).sum()), int((A_ & ~B_).sum())
    return w, l, (B_.mean() - A_.mean()) * 100, (binomtest(min(w, l), w + l, 0.5).pvalue if w + l else 1.0)

MAIN = ["majority", "blind", "E1", "E2", "E3", "E4 −A", "Lite −A", "E4", "Lite"]
def sel_of(meth): return meth in ("E4", "Lite")
def base_of(meth): return {"E4 −A": "E4", "Lite −A": "Lite"}.get(meth, meth)

S, CI = {}, {}
for m in MODELS:
    for ds in DATASETS:
        for meth in MAIN:
            g = EV[(EV.model == m) & (EV.ds == ds) & (EV.method == base_of(meth))]
            if g.empty: continue
            S[(m, ds, meth)] = summarize(g, selective=sel_of(meth)); CI[(m, ds, meth)] = boot_ci(g, selective=sel_of(meth))

rows = []
for (m, ds, meth), s in S.items():
    r = dict(dataset=DS_SHORT[ds], model=MODEL_LABEL[m], method=meth)
    r.update({TYPE_LABEL[t]: s[t] for t in TYPES[ds]})
    r.update(OA=s["OA"], OA_CI="[{:.1f}, {:.1f}]".format(*CI[(m, ds, meth)]["OA"]), AA=s["AA"], F1=s["F1"], MAE=s["MAE"],
             coverage=s["coverage"], FPR_yes=s["FPR"], unparseable=s["invalid"])
    rows.append(r)
MAINTAB = pd.DataFrame(rows)
for ds in DATASETS:
    t = MAINTAB[MAINTAB.dataset == DS_SHORT[ds]].dropna(axis=1, how="all")
    t.to_csv(TAB_DIR / f"table_main_{ds}.csv", index=False)
    print(f"Main results — {DS_LABEL[ds]} (EVAL, accuracy %; selective rows = answered questions only)"); display(t.round(1))

sig = []
for m in MODELS:
    for ds in DATASETS:
        tests = [(a, b) for a, b in [("E2", "E4 −A"), ("E1", "E4 −A"), ("E3", "E4 −A"), ("majority", "E4 −A"),
                                     ("E2", "Lite −A"), ("E1", "E2")]]
        res = []
        for a, b in tests:
            w, l, d, p = mcnemar(m, ds, base_of(a) if a != "E4 −A" else "E4", "E4" if b == "E4 −A" else ("Lite" if b == "Lite −A" else b))
            res.append([f"{b} vs {a}", w, l, d, p])
        order = np.argsort([x[-1] for x in res]); run_max = 0
        for rank, i in enumerate(order):
            run_max = max(run_max, min(1.0, res[i][-1] * (len(res) - rank))); res[i].append(run_max)
        for x in res:
            sig.append(dict(dataset=DS_SHORT[ds], model=MODEL_LABEL[m], comparison=x[0], wins=x[1], losses=x[2],
                            delta_pp=x[3], p=x[4], p_holm=x[5], significant=x[5] < 0.05))
SIG = pd.DataFrame(sig); SIG.to_csv(TAB_DIR / "table_significance.csv", index=False)
print("Paired exact McNemar tests (Holm within dataset × model)"); display(SIG.round(4))

def rc_expected(conf, y):
    """Risk–coverage curve with ties broken uniformly at random (exact expectation, no ordering artefacts)."""
    conf = np.round(np.asarray(conf, float), 9); y = np.asarray(y, float); n = len(y)
    o = np.argsort(-conf, kind="stable"); c, yy = conf[o], y[o]
    start = np.r_[0, np.flatnonzero(np.diff(c)) + 1]; size = np.diff(np.r_[start, n])
    blk = np.add.reduceat(yy, start); before = np.r_[0.0, np.cumsum(blk)[:-1]]
    b = np.repeat(np.arange(len(start)), size); pos = np.arange(n) - start[b] + 1
    exp_cum = before[b] + blk[b] / size[b] * pos
    k = np.arange(1, n + 1); return k / n, 1 - exp_cum / k

def aurc(g):
    if len(g) == 0: return np.nan
    return rc_expected(g.conf.values, g.correct.values)[1].mean() * 100

abl = []
for m in MODELS:
    for ds in DATASETS:
        ref = EV[(EV.model == m) & (EV.ds == ds) & (EV.method == "E4")]
        for v in list(VARIANTS) + ["Lite"]:
            g = EV[(EV.model == m) & (EV.ds == ds) & (EV.method == v)]
            if g.empty: continue
            sf, ss = summarize(g), summarize(g, selective=True)
            abl.append(dict(dataset=DS_SHORT[ds], model=MODEL_LABEL[m], variant=v, acc_full=sf["OA"],
                            d_acc=sf["OA"] - summarize(ref)["OA"], AURC=aurc(g), d_AURC=aurc(g) - aurc(ref),
                            acc_answered=ss["OA"], coverage=ss["coverage"], calls=g.n_calls.mean(), latency=g.latency.mean()))
ABL = pd.DataFrame(abl); ABL.to_csv(TAB_DIR / "table_ablation.csv", index=False)
print("E5 ablation — accuracy at full coverage and AURC (lower = better)"); display(ABL.round(2))

eff = []
for m in MODELS:
    for ds in DATASETS:
        for meth in ["blind", "E1", "E2", "E3", "Lite", "E4"]:
            g = EV[(EV.model == m) & (EV.ds == ds) & (EV.method == meth)]
            if g.empty: continue
            eff.append(dict(dataset=DS_SHORT[ds], model=MODEL_LABEL[m], method=meth, OA=summarize(g)["OA"],
                            latency_s=g.latency.mean(), latency_p90=g.latency.quantile(0.9), calls=g.n_calls.mean(),
                            peak_mem_gb=g.mem.max() / 1024, weights_gb=META[m]["static_mem_mb"] / 1024))
EFF = pd.DataFrame(eff); EFF.to_csv(TAB_DIR / "table_efficiency.csv", index=False)
print("E6 efficiency (per question; OA at full coverage)"); display(EFF.round(3))

ROB = pd.DataFrame()
if (R.pert != "clean").any():
    rb = R[(R.role == "eval") & R.method.isin(["E1", "E2", "E3", "E4", "Lite"])]
    out = []
    for ds, imgs in ROBUST_IMGS.items():
        keys = {f"{ds}:{i}" for i in imgs}
        for (m, meth, pert, level), g in rb[rb.img_key.isin(keys)].groupby(["model", "method", "pert", "level"]):
            out.append(dict(ds=ds, model=m, method=meth, pert=pert, level=level, OA=summarize(g)["OA"],
                            OA_sel=summarize(g, selective=True)["OA"], coverage=g.answered.mean() * 100))
    ROB = pd.DataFrame(out)
    clean = ROB[ROB.pert == "clean"].set_index(["ds", "model", "method"]).OA
    ROB["drop_pp"] = [r.OA - clean.get((r.ds, r.model, r.method), np.nan) for r in ROB.itertuples()]
    ROB.to_csv(TAB_DIR / "table_robustness.csv", index=False)
    worst = {k: (min(v) if k in ("altitude", "jpeg") else max(v)) for k, v in PERTURBATIONS.items()}
    w = ROB[[worst.get(r.pert) == r.level for r in ROB.itertuples()]]
    print("E7 accuracy drop at the strongest perturbation (pp)")
    display(w.pivot_table(index=["model", "method"], columns="pert", values="drop_pp").round(1))

# what does AeroVerify abstain on?  (accuracy of E2 / AeroVerify on answered vs abstained questions)
abst = []
for m in MODELS:
    for ds in DATASETS:
        e4 = EV[(EV.model == m) & (EV.ds == ds) & (EV.method == "E4")].set_index("qid")
        e2 = EV[(EV.model == m) & (EV.ds == ds) & (EV.method == "E2")].set_index("qid").reindex(e4.index)
        a = e4.answered.astype(bool)
        abst.append(dict(dataset=DS_SHORT[ds], model=MODEL_LABEL[m], answered_pct=a.mean() * 100,
                         E2_on_answered=e2.correct[a].mean() * 100, E4_on_answered=e4.correct[a].mean() * 100,
                         E2_on_abstained=e2.correct[~a].mean() * 100, E4_on_abstained=e4.correct[~a].mean() * 100,
                         n_abstained=int((~a).sum()), AURC_E4=aurc(e4.reset_index()),
                         AURC_SC=aurc(EV[(EV.model == m) & (EV.ds == ds) & (EV.method == "SC")]),
                         AURC_plurality=aurc(EV[(EV.model == m) & (EV.ds == ds) & (EV.method == "E4 −weights")]),
                         AURC_type_prior=aurc(EV[(EV.model == m) & (EV.ds == ds) & (EV.method == "Type prior")])))
ABST = pd.DataFrame(abst); ABST.to_csv(TAB_DIR / "table_abstention.csv", index=False)

# Selective prediction: AeroVerify vs every confidence baseline (all calibrated on DEV the same way).
# ΔAURC = baseline − AeroVerify (positive = AeroVerify better), 95% CI from a paired image-level bootstrap.
SEL_METHODS = [("E4", "AeroVerify"), ("SC", "Self-consistency (E2+probe)"),
               ("E4 −weights", "Plurality vote (all sources)"), ("Type prior", "E2 + question-type prior")]
def _paired_aurc_ci(a, b, B=1000):
    a = a.sort_values("qid"); b = b.set_index("qid").loc[a.qid].reset_index()
    imgs = a.img_key.values; uniq, inv = np.unique(imgs, return_inverse=True)
    rows_of = [np.flatnonzero(inv == k) for k in range(len(uniq))]; rng = np.random.default_rng(SEED); out = []
    ca, ya, cb, yb = a.conf.values, a.correct.values, b.conf.values, b.correct.values
    for _ in range(B):
        ix = np.concatenate([rows_of[k] for k in rng.integers(0, len(uniq), len(uniq))])
        out.append(rc_expected(cb[ix], yb[ix])[1].mean() - rc_expected(ca[ix], ya[ix])[1].mean())
    return np.percentile(np.array(out) * 100, [2.5, 97.5])
sel_rows = []
for m in MODELS:
    for ds in DATASETS:
        ours = EV[(EV.model == m) & (EV.ds == ds) & (EV.method == "E4")]
        if ours.empty: continue
        for meth, lab in SEL_METHODS:
            g = EV[(EV.model == m) & (EV.ds == ds) & (EV.method == meth)]
            if g.empty: continue
            ss = summarize(g, selective=True); d = aurc(g) - aurc(ours)
            lo, hi = (np.nan, np.nan) if meth == "E4" else _paired_aurc_ci(ours, g, B=BOOT_B if "BOOT_B" in globals() else 1000)
            sel_rows.append(dict(dataset=DS_SHORT[ds], model=MODEL_LABEL[m], method=lab, AURC=aurc(g),
                                 dAURC_vs_ours=np.nan if meth == "E4" else d, ci_lo=lo, ci_hi=hi,
                                 acc_answered=ss["OA"], coverage=ss["coverage"], acc_full=summarize(g)["OA"]))
SELTAB = pd.DataFrame(sel_rows); SELTAB.to_csv(TAB_DIR / "table_selective.csv", index=False)
print("Selective prediction (EVAL). ΔAURC = baseline − AeroVerify, 95% paired bootstrap CI; >0 means AeroVerify better")
display(SELTAB.round(1))
L = [r"\begin{tabular}{lllrrrr}", r"\toprule",
     r"Data & Model & Confidence signal & AURC$\downarrow$ & $\Delta$AURC [95\% CI] & Acc.\ answered & Coverage \\", r"\midrule"]
for r in SELTAB.itertuples():
    dd = "--" if np.isnan(r.dAURC_vs_ours) else f"{r.dAURC_vs_ours:+.1f} [{r.ci_lo:+.1f}, {r.ci_hi:+.1f}]"
    nm = r"\textbf{" + r.method + "}" if r.method == "AeroVerify" else r.method
    L.append(f"{r.dataset} & {r.model} & {nm} & {r.AURC:.1f} & {dd} & {r.acc_answered:.1f} & {r.coverage:.1f}" + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]; (TAB_DIR / "table_selective.tex").write_text("\n".join(L))
print("Abstention analysis (EVAL): accuracy on answered vs abstained questions; AURC of three confidence signals")
display(ABST.round(1))

if "floodnet" in DATASETS:
    cc = EV[(EV.ds == "floodnet") & (EV.kind == "count") & EV.method.isin(["E1", "E2", "E3", "E4"])].copy()
    cc["err"] = (cc.pred.map(_to_int) - cc.label.map(_to_int)).abs()
    CNT = (cc.groupby(["model", "type", "method"]).err
           .agg(exact=lambda e: (e == 0).mean() * 100, within1=lambda e: (e <= 1).mean() * 100, MAE="mean").reset_index())
    CNT.to_csv(TAB_DIR / "table_counting.csv", index=False)
    print("FloodNet counting: exact %, within ±1 %, MAE"); display(CNT.round(2))
    L = [r"\begin{tabular}{lllrrr}", r"\toprule", r"Model & Question & Method & Exact (\%) & $\pm$1 (\%) & MAE$\downarrow$ \\", r"\midrule"]
    for r in CNT.itertuples():
        L.append(f"{MODEL_LABEL[r.model]} & {TYPE_LABEL[r.type]} & {METHOD_LABELS[r.method]} & {r.exact:.1f} & {r.within1:.1f} & {r.MAE:.2f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]; (TAB_DIR / "table_counting.tex").write_text("\n".join(L))

# per-source diagnostics (EVAL, clean): how often each evidence source answers, and how often it is right
src_rows = []
for (m, ds, t), g in EVD[(EVD.role == "eval") & (EVD.pert == "clean")].groupby(["model", "ds", "route"]):
    for s_ in ["e1"] + SRC:
        a = g[f"{s_}_ans"]; ok = a.map(_ok)
        if not ok.any() and s_ in ("tiles", "ground"): continue
        src_rows.append(dict(dataset=DS_SHORT[ds], model=MODEL_LABEL[m], type=t, source=s_, answered=ok.mean() * 100,
                             accuracy=(a == g.label).mean() * 100, weight=W.get((m, ds, t, s_), np.nan)))
SRCTAB = pd.DataFrame(src_rows); SRCTAB.to_csv(TAB_DIR / "table_sources.csv", index=False)
print("Evidence sources (EVAL): answered %, accuracy %, DEV reliability weight")
display(SRCTAB.pivot_table(index=["dataset", "model", "type"], columns="source", values="accuracy").round(1))

# ---------- LaTeX (booktabs) ----------
def _fmt(v, d=1): return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"
NAME_TEX = {"majority": "Most-common answer", "blind": "Question-only", "E1": "E1 Direct", "E2": "E2 Structured",
            "E3": "E3 Region", "E4 −A": r"\textbf{AeroVerify} (full cov.)", "Lite −A": r"AeroVerify-Lite (full cov.)",
            "E4": r"\textbf{AeroVerify} (selective)", "Lite": "AeroVerify-Lite (selective)"}
def latex_main(ds):
    ts = list(TYPES[ds]); cols = [TYPE_LABEL[t] for t in ts] + ["OA", "AA"] + (["MAE$\\downarrow$"] if ds == "floodnet" else []) + ["Cov."]
    L = [r"\begin{tabular}{ll" + "r" * len(cols) + "}", r"\toprule", "Model & Method & " + " & ".join(cols) + r" \\", r"\midrule"]
    for mi, m in enumerate(MODELS):
        block = [(meth, S[(m, ds, meth)]) for meth in MAIN if (m, ds, meth) in S]
        full_cov = [s for meth, s in block if not sel_of(meth)]
        best = {k: max(s[k] for s in full_cov if not np.isnan(s[k])) for k in ts + ["OA", "AA"]}
        for j, (meth, s) in enumerate(block):
            cells = []
            for k in ts + ["OA", "AA"]:
                txt = _fmt(s[k]); cells.append(r"\textbf{" + txt + "}" if (not sel_of(meth) and abs(s[k] - best[k]) < 1e-9) else txt)
            if ds == "floodnet": cells.append(_fmt(s["MAE"], 2))
            cells.append(_fmt(s["coverage"], 0))
            L.append((MODEL_LABEL[m] if j == 0 else "") + " & " + NAME_TEX[meth] + " & " + " & ".join(cells) + r" \\")
        if mi < len(MODELS) - 1: L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB_DIR / f"table_main_{ds}.tex").write_text("\n".join(L))

def latex_ablation():
    L = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Variant & $\Delta$Acc. (pp) & AURC$\downarrow$ & Calls & Latency (s) \\", r"\midrule"]
    for ds in DATASETS:
        for m in MODELS:
            L.append(r"\multicolumn{5}{l}{\textit{" + DS_SHORT[ds] + " / " + MODEL_LABEL[m] + r"}} \\")
            for r in ABL[(ABL.dataset == DS_SHORT[ds]) & (ABL.model == MODEL_LABEL[m])].itertuples():
                name = {"E4": r"AeroVerify (full)"}.get(r.variant, r.variant.replace("E4 −", r"\quad $-$ ").replace("Lite", "AeroVerify-Lite"))
                L.append(f"{name} & {r.d_acc:+.1f} & {r.AURC:.1f} & {r.calls:.1f} & {r.latency:.2f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB_DIR / "table_ablation.tex").write_text("\n".join(L))

for ds in DATASETS: latex_main(ds)
latex_ablation()
print("LaTeX tables →", sorted(p.name for p in TAB_DIR.glob("*.tex")))
