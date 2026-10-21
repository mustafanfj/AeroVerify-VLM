def _pick_lr(g, rng, slot4_rural):            # identical to run 1
    chosen = []
    for t in ("count", "presence", "comp"):
        c = g[g.type == t]
        if len(c): chosen.append(c.sample(1, random_state=int(rng.integers(1 << 30))).index[0])
    rest = g.drop(index=chosen)
    if slot4_rural and (rest.type == "rural_urban").any():
        chosen.append(rest[rest.type == "rural_urban"].index[0])
    else:
        pool = rest[rest.type.isin(["presence", "count"])]
        tagged = pool[[(SMALL_RE.search(_norm(q)) or REL_RE.search(_norm(q))) is not None for q in pool.question]]
        pool = tagged if len(tagged) else (pool if len(pool) else rest[rest.type != "rural_urban"])
        if len(pool): chosen.append(pool.sample(1, random_state=int(rng.integers(1 << 30))).index[0])
    return g.loc[chosen[:4]]

def _draw_lr(df_list, n, rng, role):
    df = pd.concat(df_list)
    ok = df.groupby("img").type.apply(lambda s: {"count", "presence", "comp"} <= set(s))
    imgs = [i for i in df.img.unique() if ok.get(i, False)]; rng.shuffle(imgs)
    first = df.drop_duplicates("img").set_index("img").split
    imgs = sorted(imgs, key=lambda i: 0 if first[i] == df_list[0].split.iloc[0] else 1)[:n]
    return pd.concat([_pick_lr(df[df.img == i], rng, k % 2 == 0) for k, i in enumerate(imgs)]).assign(role=role)

def sample_lr():
    v1 = Path(PROJECT_ROOT) / "outputs" / "sample_questions.csv"
    A = ALLQ["rsvqa_lr"]
    if v1.exists() and not DRY_RUN:                                   # reuse run-1 sample verbatim
        s = pd.read_csv(v1, dtype={"label": str, "answer_raw": str})
        s = pd.DataFrame(dict(qid="lr:" + s.qid.astype(str), role=s.role))
        print("RSVQA-LR: re-using the run-1 sample")
        return A.merge(s, on="qid")
    n = SAMPLE_SIZES["rsvqa_lr"]
    dev = _draw_lr([A[A.split == "train"]], n["dev"], np.random.default_rng(SEED + 1), "dev")
    ev = _draw_lr([A[A.split == "test"], A[A.split == "val"]], n["eval"], np.random.default_rng(SEED + 2), "eval")
    return pd.concat([dev, ev], ignore_index=True)

def sample_fn():
    A = ALLQ["floodnet"]; n = SAMPLE_SIZES["floodnet"]
    imgs = sorted(A.img.unique()); rng = np.random.default_rng(SEED + 3); rng.shuffle(imgs)
    if DRY_RUN: n = dict(dev=min(n["dev"], 30), eval=min(n["eval"], 60))
    dev, ev = set(imgs[:n["dev"]]), set(imgs[n["dev"]:n["dev"] + n["eval"]])
    s = A[A.img.isin(dev | ev)].copy(); s["role"] = np.where(s.img.isin(dev), "dev", "eval")
    return s

SAMPLE_PATH = OUT_DIR / "sample_v2.csv"
meta = dict(sizes=SAMPLE_SIZES, seed=SEED, dry=DRY_RUN, ds=sorted(DATASETS))
mp = OUT_DIR / "sample_v2_meta.json"
if SAMPLE_PATH.exists() and mp.exists() and json.loads(mp.read_text()) == json.loads(json.dumps(meta)):
    SAMPLE = pd.read_csv(SAMPLE_PATH, dtype={"label": str, "answer_raw": str, "img": str, "qid": str})
    print("loaded saved sample →", SAMPLE_PATH)
else:
    parts = [sample_lr() if ds == "rsvqa_lr" else sample_fn() for ds in DATASETS]
    SAMPLE = pd.concat(parts, ignore_index=True)
    SAMPLE.to_csv(SAMPLE_PATH, index=False); mp.write_text(json.dumps(meta))
    print("new sample saved →", SAMPLE_PATH)

if "floodnet" in DATASETS and not DRY_RUN and ("data" in STAGES or "run" in STAGES):
    fetch_floodnet_images(sorted(SAMPLE[SAMPLE.ds == "floodnet"].img.unique()))

SAMPLE["route"] = [route(q, ds) for q, ds in zip(SAMPLE.question, SAMPLE.ds)]
SAMPLE["kind"] = [kind_of(ds, t) for ds, t in zip(SAMPLE.ds, SAMPLE.type)]
tg = pd.DataFrame([question_tags(q, ds) for q, ds in zip(SAMPLE.question, SAMPLE.ds)])
SAMPLE[["small", "relational", "subgroup"]] = tg[["small", "relational", "subgroup"]].values
SAMPLE["img_key"] = SAMPLE.ds + ":" + SAMPLE.img.astype(str)
for ds in DATASETS:
    s = SAMPLE[SAMPLE.ds == ds]; acc = (s.route == s.type).mean()
    print(f"{DS_SHORT[ds]}: router agreement with dataset types {acc:.1%}")
    if acc < 0.98:
        display(s[s.route != s.type][["question", "type", "route"]].drop_duplicates("question").head(12))

# most-common answer per type, from training questions that are in neither sample
MAJORITY = {}
for ds in DATASETS:
    A = ALLQ[ds]
    pool = A[A.split == "train"] if ds == "rsvqa_lr" else A[~A.img.isin(SAMPLE[SAMPLE.ds == ds].img)]
    if pool.empty: pool = A
    for t, g in pool.groupby("type"):
        MAJORITY[(ds, t)] = g.label.value_counts().index[0]
# label space of condition questions = answers that occur in ≥ 5 % of the training pool (per sub-type)
if "floodnet" in DATASETS:
    A = ALLQ["floodnet"]; pool = A[~A.img.isin(SAMPLE[SAMPLE.ds == "floodnet"].img) & (A.type == "condition")]
    for sub in ("road", "image"):
        g = pool[[cond_subtype(q) == sub for q in pool.question]]
        if len(g):
            f = g.label.value_counts(normalize=True)
            COND_LABELS[sub] = [l for l in (MIXED, FLOODED, NONFLOODED) if f.get(l, 0) >= 0.05]
            print(f"condition/{sub}: training answer mix {f.round(3).to_dict()} → label space {COND_LABELS[sub]}")
print("most-common answers:", {f"{k[0]}/{k[1]}": v for k, v in MAJORITY.items()})

ROBUST_IMGS = {ds: list(SAMPLE[(SAMPLE.ds == ds) & (SAMPLE.role == "eval")].img.drop_duplicates()
                        [: (10 if DRY_RUN else ROBUST_N_IMAGES)]) for ds in ROBUST_DATASETS if ds in DATASETS}
display(SAMPLE.groupby(["ds", "role", "type"]).size().unstack(fill_value=0)
        .assign(images=SAMPLE.groupby(["ds", "role"]).img.nunique(), total=SAMPLE.groupby(["ds", "role"]).size()))
