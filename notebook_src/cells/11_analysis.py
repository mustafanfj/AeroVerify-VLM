%%stage analyze
# ── Load evidence, learn reliability weights on DEV, fuse ─────────────────────
MODELS = [m for m in MODELS_TO_ANALYZE if (PRED_DIR / f"{m}.jsonl").exists()]
assert MODELS, f"No predictions in {PRED_DIR} — run the 'run' stage first."
META = {m: json.loads((PRED_DIR / f"{m}_meta.json").read_text()) for m in MODELS}
MODEL_LABEL = {m: MODEL_SPECS[m]["label"] for m in MODELS}
QI = SAMPLE.set_index("qid")
SRC = ["full", "tiles", "probe", "ground"]

def label_set(ds, t, q):
    if t == "condition":
        return list(COND_LABELS[cond_subtype(q)])
    return {"yesno": ["yes", "no"], "count_bin": BIN_KEYS, "cat": ["rural", "urban"]}.get(kind_of(ds, t))

def tol(y): return max(1, round(0.1 * y))
def _ok(x): return x is not None and not (isinstance(x, float) and np.isnan(x))

recs = []
for m in MODELS:
    with open(PRED_DIR / f"{m}.jsonl") as f:
        for line in f:
            r = json.loads(line); qi = QI.loc[r["qid"]]
            d = dict(model=m, ds=r["ds"], role=r["role"], pert=r["pert"], level=r["level"], qid=r["qid"],
                     img=r["img"], img_key=f"{r['ds']}:{r['img']}", type=qi.type, route=qi.route, kind=qi.kind,
                     label=qi.label, subgroup=qi.subgroup, small=bool(qi.small), relational=bool(qi.relational),
                     question=qi.question, gate=r["gate"])
            for s in ["e1", "blind"] + SRC:
                v = r.get(s) if s in ("e1", "blind") else r["sources"].get(s)
                d[f"{s}_ans"] = v["ans"] if v else None
                d[f"{s}_n"] = len(v["calls"]) if v else 0
                d[f"{s}_lat"] = sum(c["lat"] for c in v["calls"]) if v else 0.0
                d[f"{s}_mem"] = max([c["mem"] for c in v["calls"]], default=np.nan) if v else np.nan
            recs.append(d)
EVD = pd.DataFrame(recs)
EVD["valid_labels"] = [label_set(ds, t, q) for ds, t, q in zip(EVD.ds, EVD.route, EVD.question)]

def _hit(ans, y, kind):
    if ans is None or (isinstance(ans, float) and np.isnan(ans)): return np.nan
    if kind == "count":
        try: return float(abs(int(ans) - int(y)) <= tol(int(y)))
        except (TypeError, ValueError): return np.nan
    return float(ans == y)

# reliability weights per (model, dataset, type, source) from DEV, clean images
W = {}
dev = EVD[(EVD.role == "dev") & (EVD.pert == "clean")]
for (m, ds, t), g in dev.groupby(["model", "ds", "route"]):
    k = kind_of(ds, t); K = 2 if k == "yesno" else (len(BIN_KEYS) if k == "count_bin" else (3 if t == "condition" else 2))
    for s in SRC:
        h = np.array([_hit(a, y, k) for a, y in zip(g[f"{s}_ans"], g.label)], float); h = h[~np.isnan(h)]
        a = (h.sum() + 1) / (len(h) + 2)
        if len(h) == 0: W[(m, ds, t, s)] = 0.0
        else: W[(m, ds, t, s)] = float(np.clip(np.log((K - 1) * a / (1 - a)), 0, 4)) if k != "count" else float(a)
WEIGHTS = pd.Series(W).rename_axis(["model", "ds", "type", "source"]).unstack("source")[SRC]
print("Reliability weights learned on DEV (0 = ignored):"); display(WEIGHTS.round(2))

PRIORITY = {"count": ["ground", "tiles", "full", "probe"], "count_bin": ["tiles", "ground", "full", "probe"],
            "yesno": ["probe", "full", "tiles", "ground"], "cat": ["tiles", "full", "probe", "ground"]}

def fuse_row(r, use, weighted=True, gate=True):
    k = r.kind; prio = [s for s in PRIORITY[k] if s in use]
    votes = {}
    for s in prio:
        if s == "tiles" and gate and not r.gate: continue
        a = getattr(r, f"{s}_ans")
        if not _ok(a): continue
        if k != "count" and a not in r.valid_labels: continue
        votes[s] = a
    if not votes: return None, 0.0
    w = {s: (W.get((r.model, r.ds, r.route, s), 0.0) if weighted else 1.0) for s in votes}
    if k == "count":
        vals = sorted((int(votes[s]), max(w[s], 1e-3)) for s in votes)
        tot, acc = sum(x[1] for x in vals), 0.0
        for v_, w_ in vals:
            acc += w_
            if acc >= tot / 2: med = v_; break
        conf = sum(w[s] for s in votes if abs(int(votes[s]) - med) <= tol(med)) / max(sum(w.values()), 1e-9)
        return str(med), float(conf)
    if not weighted:                                                  # run-1 plurality rule
        c = Counter(votes.values()); top = max(c.values()); cands = {a for a, n in c.items() if n == top}
        a = next(votes[s] for s in prio if s in votes and votes[s] in cands)
        return a, top / len(votes)
    labels = r.valid_labels; sc = {l: sum(w[s] for s in votes if votes[s] == l) for l in labels}
    best = max(sc.values()); cands = [l for l in labels if sc[l] == best]
    a = cands[0] if len(cands) == 1 else next((votes[s] for s in prio if s in votes and votes[s] in cands), cands[0])
    z = np.array([sc[l] for l in labels]); p = np.exp(z - z.max()); p /= p.sum()
    return a, float(p[labels.index(a)])

VARIANTS = {   # name: (sources, reliability weights, resolution gate)
    "E4":          (SRC, True, True),
    "E4 −tiles":   (["full", "probe", "ground"], True, True),
    "E4 −ground":  (["full", "tiles", "probe"], True, True),
    "E4 −probe":   (["full", "tiles", "ground"], True, True),
    "E4 −weights": (SRC, False, True),
    "E4 −gate":    (SRC, True, False),
}

def cost(r, srcs, gate=True):
    ss = [s for s in srcs if not (s == "tiles" and gate and not r.gate)]
    mems = [getattr(r, f"{s}_mem") for s in ss]; mems = [x for x in mems if x == x]
    return (sum(getattr(r, f"{s}_n") for s in ss), sum(getattr(r, f"{s}_lat") for s in ss), max(mems) if mems else np.nan)

rows = []
for r in EVD.itertuples(index=False):
    base = dict(model=r.model, ds=r.ds, role=r.role, pert=r.pert, level=r.level, qid=r.qid, img_key=r.img_key,
                type=r.type, kind=r.kind, label=r.label, subgroup=r.subgroup)
    def add(method, pred, conf, n, lat, mem):
        rows.append(dict(base, method=method, pred=pred, conf=conf, n_calls=n, latency=lat, mem=mem))
    if r.pert == "clean":
        add("majority", MAJORITY.get((r.ds, r.type)), 1.0, 0, 0.0, np.nan)
        add("blind", r.blind_ans, 1.0, r.blind_n, r.blind_lat, r.blind_mem)
    add("E1", r.e1_ans, 1.0, r.e1_n, r.e1_lat, r.e1_mem)
    add("E2", r.full_ans, 1.0, r.full_n, r.full_lat, r.full_mem)
    add("E3", r.tiles_ans, 1.0, r.tiles_n, r.tiles_lat, r.tiles_mem)
    for v, (srcs, wtd, gate) in VARIANTS.items():
        p, c = fuse_row(r, srcs, wtd, gate); add(v, p, c, *cost(r, srcs, gate))
    p, c = fuse_row(r, ["full", "probe"], weighted=False); add("SC", p, c, *cost(r, ["full", "probe"]))   # self-consistency baseline
    # AeroVerify-Lite: stop after full + probe when they agree
    f, pr = r.full_ans, r.probe_ans
    agree = _ok(f) and _ok(pr) and (
        abs(int(f) - int(pr)) <= tol(int(f)) if r.kind == "count" else f == pr)
    if agree:
        p, c = fuse_row(r, ["full", "probe"]); add("Lite", p, c, *cost(r, ["full", "probe"]))
    else:
        p, c = fuse_row(r, SRC); add("Lite", p, c, *cost(r, SRC))
R = pd.DataFrame(rows)
R["valid"] = R.pred.notna()
R["correct"] = (R.pred == R.label) & R.valid

def _to_int(x):
    try: return int(x)
    except (TypeError, ValueError): return np.nan
R["abs_err"] = np.where(R.kind == "count", (R.pred.map(_to_int) - R.label.map(_to_int)).abs(), np.nan)

# ── Type-conditional confidence calibration (fit on DEV, clean images) ──────────
# Raw fused confidences are not comparable across question types (a 0.9 on a count question is far less
# reliable than a 0.9 on a condition question). We fit one Platt scaler per (model, dataset, question type,
# method) on DEV and apply it everywhere. Within a type it is monotone (ranking unchanged); across types it
# turns confidence into an estimate of P(correct). The SAME procedure is applied to every selective baseline.
SELECTIVE = ["E4", "Lite", "SC"] + [v for v in VARIANTS if v != "E4"]

def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4); return np.log(p / (1 - p))

def platt_fit(x, y, l2=1e-3):
    """Platt scaling with Platt's smoothed targets and a weak pull towards the identity map (robust on small DEV sets)."""
    from scipy.optimize import minimize
    x = np.asarray(x, float); n1 = y.sum(); n0 = len(y) - n1
    t = np.where(y > 0, (n1 + 1) / (n1 + 2), 1 / (n0 + 2))
    def nll(w):
        z = w[0] * x + w[1]
        return np.mean(np.logaddexp(0, z) - t * z) + l2 * ((w[0] - 1) ** 2 + w[1] ** 2)
    return minimize(nll, np.array([1.0, 0.0]), method="L-BFGS-B", bounds=[(0.0, None), (None, None)]).x   # slope ≥ 0: never reverses ranking

def calibrate(R, methods, min_n=8):
    R = R.copy(); R["conf_raw"] = R.conf.astype(float); CAL = {}
    fit = R[(R.role == "dev") & (R.pert == "clean") & R.method.isin(methods) & R.conf_raw.notna()]
    for (m, ds, meth), g in fit.groupby(["model", "ds", "method"]):
        w_all = platt_fit(_logit(g.conf_raw), g.correct.values.astype(float))
        for t in R[(R.model == m) & (R.ds == ds)].type.unique():
            h = g[g.type == t]
            CAL[(m, ds, meth, t)] = platt_fit(_logit(h.conf_raw), h.correct.values.astype(float)) if len(h) >= min_n else w_all
    for (m, ds, meth, t), w in CAL.items():
        sel = (R.model == m) & (R.ds == ds) & (R.method == meth) & (R.type == t) & R.conf_raw.notna()
        R.loc[sel, "conf"] = 1 / (1 + np.exp(-np.clip(w[0] * _logit(R.loc[sel, "conf_raw"]) + w[1], -30, 30)))
    return R, CAL

R, CAL = calibrate(R, SELECTIVE)

# Baseline: E2 answers ranked only by the DEV accuracy of their question type ("type prior").
# Any confidence signal worth publishing must beat this trivial one.
_pri = (R[(R.role == "dev") & (R.pert == "clean") & (R.method == "E2")]
        .groupby(["model", "ds", "type"]).correct.mean())
TP = R[R.method == "E2"].copy(); TP["method"] = "Type prior"
TP["conf"] = [_pri.get((m, d, t), 0.5) for m, d, t in zip(TP.model, TP.ds, TP.type)]; TP["conf_raw"] = TP.conf
R = pd.concat([R, TP], ignore_index=True)
SELECTIVE = SELECTIVE + ["Type prior"]

# abstention threshold τ per (model, dataset, variant) — DEV only
TAU = {}
for (m, ds, v), g in R[(R.role == "dev") & (R.pert == "clean") & R.method.isin(SELECTIVE)].groupby(["model", "ds", "method"]):
    best = (0.0, -1.0)
    for tau in np.unique(np.round(g.conf.values, 6)):
        a = g.conf >= tau
        if a.mean() < MIN_COVERAGE or a.sum() == 0: continue
        acc = g.correct[a].mean()
        if acc > best[1] + 1e-9: best = (float(tau), float(acc))
    TAU[(m, ds, v)] = best[0]
R["answered"] = True
for (m, ds, v), tau in TAU.items():
    sel = (R.model == m) & (R.ds == ds) & (R.method == v); R.loc[sel, "answered"] = R.loc[sel, "conf"] >= tau - 1e-9
EV = R[(R.role == "eval") & (R.pert == "clean")].copy()
R.to_csv(TAB_DIR / "all_predictions_long.csv", index=False)
print("τ (DEV):", {f"{m}/{DS_SHORT[ds]}/{v}": round(t, 3) for (m, ds, v), t in TAU.items() if v in ("E4", "Lite")})
print(f"{len(R):,} method-question rows; EVAL clean rows: {len(EV):,}")
