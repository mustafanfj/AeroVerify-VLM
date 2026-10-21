# ── Datasets: RSVQA-LR (Zenodo) · FloodNet-VQA (Hugging Face mirror) · synthetic stand-ins ──
ZENODO = "https://zenodo.org/records/6344334/files/{}?download=1"

def _download(url, dest):
    if dest.exists() and dest.stat().st_size > 0: return
    tmp = dest.with_suffix(dest.suffix + ".part"); print("  downloading", dest.name, "…", flush=True)
    urllib.request.urlretrieve(url, tmp); tmp.rename(dest)

# ---------------- RSVQA-LR ----------------
def download_rsvqa_lr():
    d = data_dir("rsvqa_lr")
    for s in ("train", "val", "test"):
        for k in ("questions", "answers", "images"):
            _download(ZENODO.format(f"LR_split_{s}_{k}.json"), d / f"LR_split_{s}_{k}.json")
    _download(ZENODO.format("Images_LR.zip"), d / "Images_LR.zip")
    if not any((d / "Images_LR").glob("*.tif")):
        with zipfile.ZipFile(d / "Images_LR.zip") as z: z.extractall(d)

def load_rsvqa_lr():
    d, rows = data_dir("rsvqa_lr"), []
    for split in ("train", "val", "test"):
        Q = json.loads((d / f"LR_split_{split}_questions.json").read_text())["questions"]
        A = {a["id"]: a for a in json.loads((d / f"LR_split_{split}_answers.json").read_text())["answers"]
             if a.get("active", True)}
        for q in Q:
            if not q.get("active", True) or not q.get("answers_ids"): continue
            a = A.get(q["answers_ids"][0])
            if a is None: continue
            rows.append(dict(qid=f"lr:{q['id']}", img=str(q["img_id"]), split=split, type=q["type"],
                             question=q["question"], answer_raw=str(a["answer"])))
    return pd.DataFrame(rows)

# ---------------- FloodNet-VQA ----------------
def download_floodnet_annotations():
    d = data_dir("floodnet")
    if len(list((d / "train_image" / "ann").glob("*.json"))) > 1000: return
    from huggingface_hub import snapshot_download
    print("  downloading FloodNet-VQA annotations (≈1,450 small JSON files) …")
    snapshot_download(FLOODNET_HF, repo_type="dataset", allow_patterns=["train_image/ann/*"], local_dir=str(d))

def fetch_floodnet_images(names):
    """Download only the sampled images (≈5 MB each) instead of the 12 GB repository."""
    d = data_dir("floodnet")
    missing = [n for n in names if not (d / "train_image" / "img" / n).exists()]
    if not missing: return
    from huggingface_hub import hf_hub_download
    for n in tqdm(missing, desc="FloodNet images"):
        hf_hub_download(FLOODNET_HF, filename=f"train_image/img/{n}", repo_type="dataset", local_dir=str(d))

def _parse_tag(v):
    if isinstance(v, dict): return v
    try: return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return json.loads(v)

def load_floodnet():
    d, rows = data_dir("floodnet"), []
    for p in sorted((d / "train_image" / "ann").glob("*.json")):
        ann = json.loads(p.read_text()); img = p.name[:-len(".json")]
        for tag in ann.get("tags", []):
            if tag.get("name") != "question": continue
            v = _parse_tag(tag["value"])
            rows.append(dict(qid=f"fn:{v['Question_ID']}", img=img, split="train",
                             type=FLOODNET_TYPEMAP.get(v["Question_Type"], v["Question_Type"]),
                             question=v["Question"], answer_raw=str(v["Ground_Truth"]),
                             W=ann.get("size", {}).get("width"), H=ann.get("size", {}).get("height")))
    df = pd.DataFrame(rows)
    if df.qid.duplicated().any():                                  # make ids unique if the mirror repeats them
        df["qid"] = [f"fn:{i}" for i in range(len(df))]
    return df

# ---------------- synthetic stand-ins (DRY_RUN) ----------------
SYN_LR_CLASSES = {"residential building": (200, 70, 60), "commercial building": (235, 150, 60),
                  "water area": (40, 90, 170), "road": (150, 150, 150)}
SYN_PLURAL = {"residential building": "residential buildings", "commercial building": "commercial buildings",
              "water area": "water areas", "road": "roads", "building": "buildings"}

def _lr_size_class(area): return "small" if area < 30 else ("medium" if area < 100 else "large")
def syn_match(o, phrase):
    phrase = phrase.replace("buildings", "building").replace("areas", "area").replace("roads", "road")
    size = SIZE_RE.search(phrase)
    if size and _lr_size_class(o["area"]) != size.group(1): return False
    core = SIZE_RE.sub("", phrase).strip()
    return core == o["cls"] or (core == "building" and o["cls"].endswith("building"))

def build_synthetic_lr(n={"train": 24, "val": 12, "test": 22}):
    d, rng = data_dir("rsvqa_lr"), np.random.default_rng(SEED)
    (d / "Images_LR").mkdir(exist_ok=True); scenes, img_id, qid = {}, 0, 0
    nouns = ["building", "residential building", "commercial building", "water area", "road"]
    for split, k in n.items():
        Q, A = [], []
        for _ in range(k):
            urban = bool(rng.random() < 0.5); objs = []
            def add(cls, m, a, b):
                for _ in range(m):
                    w, h = rng.integers(a, b + 1), rng.integers(a, b + 1); x, y = rng.integers(0, 256 - w), rng.integers(0, 256 - h)
                    objs.append(dict(cls=cls, box=[int(x), int(y), int(x + w), int(y + h)], area=int(w * h)))
            add("residential building", int(rng.integers(15, 60) if urban else rng.integers(0, 12)), 2, 9)
            add("commercial building", int(rng.integers(3, 15) if urban else rng.integers(0, 3)), 4, 14)
            add("water area", int(rng.integers(0, 4)), 8, 36)
            add("road", int(rng.integers(3, 10) if urban else rng.integers(1, 4)), 3, 3)
            scenes[str(img_id)] = dict(urban=urban, objects=objs, W=256, H=256)
            img = np.array([120, 120, 115] if urban else [95, 125, 70], np.float32) + rng.normal(0, 9, (256, 256, 3))
            for o in objs:
                x0, y0, x1, y1 = o["box"]; img[y0:y1, x0:x1] = SYN_LR_CLASSES[o["cls"]]
            Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(d / "Images_LR" / f"{img_id}.tif")
            qs = []
            for _ in range(3):
                adj = rng.choice(["", "", "small ", "large "]); ph = f"{adj}{rng.choice(nouns)}".strip()
                if rng.random() < 0.3:
                    other = str(rng.choice(nouns)); q = f"Is there a {ph} next to a {other}?"
                    ans = any(syn_match(o, ph) for o in objs) and any(syn_match(o, other) for o in objs) and rng.random() < 0.6
                else:
                    q = rng.choice([f"Is there a {ph}?", f"Is a {ph} present?"]); ans = any(syn_match(o, ph) for o in objs)
                qs.append(("presence", q, "yes" if ans else "no"))
                pl = (rng.choice(["", "small "]) + SYN_PLURAL[str(rng.choice(nouns))]).strip()
                qs.append(("count", f"How many {pl} are there?", str(sum(syn_match(o, pl) for o in objs))))
                a_, b_ = rng.choice(nouns, 2, replace=False)
                ca, cb = sum(syn_match(o, a_) for o in objs), sum(syn_match(o, b_) for o in objs)
                f = rng.integers(0, 3)
                q, ans = [(f"Are there more {SYN_PLURAL[a_]} than {SYN_PLURAL[b_]}?", ca > cb),
                          (f"Are there less {SYN_PLURAL[a_]} than {SYN_PLURAL[b_]}?", ca < cb),
                          (f"Is the number of {SYN_PLURAL[a_]} equal to the number of {SYN_PLURAL[b_]}?", ca == cb)][f]
                qs.append(("comp", q, "yes" if ans else "no"))
            qs.append(("rural_urban", "Is it a rural or an urban area?", "urban" if urban else "rural"))
            for t, q, a in qs:
                Q.append(dict(id=qid, img_id=img_id, type=t, question=q, answers_ids=[qid], active=True))
                A.append(dict(id=qid, question_id=qid, answer=a, active=True)); qid += 1
            img_id += 1
        (d / f"LR_split_{split}_questions.json").write_text(json.dumps({"questions": Q}))
        (d / f"LR_split_{split}_answers.json").write_text(json.dumps({"answers": A}))
    (d / "scenes.json").write_text(json.dumps(scenes))

def build_synthetic_floodnet(n_images=110, W=1600, H=1200):
    """Small UAV-like scenes in the exact Supervisely/HF format of the real mirror."""
    d, rng = data_dir("floodnet"), np.random.default_rng(SEED + 7)
    (d / "train_image" / "ann").mkdir(parents=True, exist_ok=True); (d / "train_image" / "img").mkdir(parents=True, exist_ok=True)
    scenes, qid = {}, 0
    for i in range(n_images):
        name = f"{7000 + i}.JPG"; flooded_scene = rng.random() < 0.55
        water = []
        if flooded_scene:
            for _ in range(rng.integers(1, 4)):
                w, h = rng.integers(300, 900), rng.integers(250, 700)
                x, y = rng.integers(0, W - w), rng.integers(0, H - h); water.append([int(x), int(y), int(x + w), int(y + h)])
        def wet(b):
            return any(not (b[2] < w[0] or b[0] > w[2] or b[3] < w[1] or b[1] > w[3]) for w in water)
        roads = []
        if rng.random() < 0.8:
            y = int(rng.integers(150, H - 150)); roads.append([0, y, W, y + 34])
            if rng.random() < 0.5:
                x = int(rng.integers(150, W - 150)); roads.append([x, 0, x + 34, H])
        road_segs = []
        for r in roads:                                              # split roads into segments for flood state
            if r[2] - r[0] > r[3] - r[1]:
                for x in range(0, W, 200): road_segs.append([x, r[1], min(W, x + 200), r[3]])
            else:
                for y in range(0, H, 200): road_segs.append([r[0], y, r[2], min(H, y + 200)])
        bld = []
        for _ in range(int(rng.integers(0, 22))):
            s = int(rng.integers(22, 70)); x, y = int(rng.integers(0, W - s)), int(rng.integers(0, H - s))
            bld.append(dict(box=[x, y, x + s, y + s], area=s * s))
        for b in bld: b["flooded"] = wet(b["box"])
        segs = [dict(box=s, flooded=wet(s)) for s in road_segs]
        scenes[name] = dict(W=W, H=H, water=water, buildings=bld, roads=segs)
        img = np.zeros((H, W, 3), np.float32) + np.array([86, 118, 70]) + rng.normal(0, 8, (H, W, 3))
        for s in segs:
            x0, y0, x1, y1 = s["box"]; img[y0:y1, x0:x1] = (140, 138, 132)
        for w in water:
            x0, y0, x1, y1 = w; img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * 0.35 + np.array([110, 96, 70]) * 0.65
        for b in bld:
            x0, y0, x1, y1 = b["box"]; img[y0:y1, x0:x1] = (178, 70, 60) if not b["flooded"] else (150, 85, 70)
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(d / "train_image" / "img" / name, quality=88)
        nb_f = sum(b["flooded"] for b in bld); nb = len(bld)
        qs = [("What is the overall condition of the given image?", FLOODED if water else "non flooded", "Condition_Recognition")]
        if segs:
            fl = {s["flooded"] for s in segs}
            rc = MIXED if len(fl) == 2 else (FLOODED if True in fl else "non flooded")
            qs.append(("What is the condition of the road in this image?", rc.replace(",", ", ") if rng.random() < 0.3 else rc,
                       "Condition_Recognition"))
            if rng.random() < 0.5: qs.append(("Is the entire road flooded?", "Yes" if fl == {True} else "No", "Yes_No"))
            else:                  qs.append(("Is the entire road non flooded?", "Yes" if fl == {False} else "No", "Yes_No"))
        qs.append(("How many buildings are in the image?", nb, "Simple_Counting"))
        if rng.random() < 0.5: qs.append(("How many flooded buildings are in the image?", nb_f, "Complex_Counting"))
        else:                  qs.append(("How many buildings are non flooded in this image?", nb - nb_f, "Complex_Counting"))
        tags = []
        for q, a, t in qs:
            tags.append(dict(id=qid, tagId=1, name="question",
                             value=repr({"Question_ID": str(qid), "Question": q, "Ground_Truth": a, "Question_Type": t})))
            qid += 1
        (d / "train_image" / "ann" / f"{name}.json").write_text(json.dumps(
            dict(description="", tags=tags, size=dict(height=H, width=W), objects=[])))
    (d / "scenes.json").write_text(json.dumps(scenes))

ALLQ = {}
for ds in DATASETS:
    if "data" in STAGES:
        if DRY_RUN:
            if not (data_dir(ds) / "scenes.json").exists():
                (build_synthetic_lr if ds == "rsvqa_lr" else build_synthetic_floodnet)()
        else:
            (download_rsvqa_lr if ds == "rsvqa_lr" else download_floodnet_annotations)()
    df = load_rsvqa_lr() if ds == "rsvqa_lr" else load_floodnet()
    df["ds"] = ds
    df["label"] = [gt_label(a, ds, t) for a, t in zip(df.answer_raw, df.type)]
    df = df[df.label.notna() & df.type.isin(TYPES[ds])]
    ALLQ[ds] = df.reset_index(drop=True)
    print(f"{DS_LABEL[ds]:28s}: {df.img.nunique():6d} images {len(df):7d} questions | "
          + ", ".join(f"{k}={v}" for k, v in df.type.value_counts().items()))
if "floodnet" in ALLQ:
    print("FloodNet condition labels:", dict(ALLQ["floodnet"][ALLQ["floodnet"].type == "condition"].label.value_counts()))
