# ── VLM answerers, shared grounding module, persistent call cache ─────────────
def _infer(fn):
    return torch.inference_mode()(fn) if torch is not None else fn

class CallCache:
    """One JSON line per model call (prompt → raw output, latency, peak memory). Resumable."""
    def __init__(self, name):
        self.path = CACHE_DIR / f"calls_{name}.jsonl"; self.db = {}
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    try: r = json.loads(line); self.db[r["k"]] = r
                    except json.JSONDecodeError: pass
        self._fh = open(self.path, "a"); self.hits = self.misses = 0
    @staticmethod
    def key(*p): return hashlib.sha1("§".join(map(str, p)).encode()).hexdigest()[:20]
    def get(self, k):
        r = self.db.get(k)
        if r is not None: self.hits += 1
        return r
    def put(self, k, rec):
        rec = dict(rec, k=k); self.db[k] = rec; self.misses += 1
        self._fh.write(json.dumps(rec) + "\n"); self._fh.flush(); return rec

def _timed(fn):
    if HAS_CUDA: torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter(); out = fn()
    if HAS_CUDA: torch.cuda.synchronize()
    return out, time.perf_counter() - t0, (torch.cuda.max_memory_allocated() / 2**20 if HAS_CUDA else float("nan"))

class Answerer:
    name = "base"
    def __init__(self, cache): self.cache = cache
    def timed(self, fn): return _timed(fn)
    def ask(self, view, prompt, intent=None):
        k = CallCache.key(self.name, "query", view.key, prompt)
        r = self.cache.get(k)
        if r is None:
            out, lat, mem = self.timed(lambda: self._query(view, prompt, intent))
            r = self.cache.put(k, dict(kind="query", view=view.key, prompt=prompt, out=out, lat=lat, mem=mem))
        return r
    def static_memory_mb(self): return torch.cuda.memory_allocated() / 2**20 if HAS_CUDA else float("nan")

class Grounder:
    """Open-vocabulary grounding shared by every answerer (Moondream2 detect head)."""
    name = "grounder"
    def __init__(self, cache, model=None): self.cache, self.model = cache, model
    def timed(self, fn): return _timed(fn)
    def detect(self, view, phrase):
        k = CallCache.key(self.name, "detect", view.key, phrase, DET_MAX_OBJECTS)
        r = self.cache.get(k)
        if r is None:
            boxes, lat, mem = self.timed(lambda: self._detect(view, phrase))
            r = self.cache.put(k, dict(kind="detect", view=view.key, prompt=phrase, out=boxes, lat=lat, mem=mem))
        return r
    @_infer
    def _detect(self, view, phrase):
        objs = self.model.detect(view.pil, phrase, settings={"max_objects": DET_MAX_OBJECTS, "variant": None})["objects"]
        x0, y0 = view.box[:2]; w, h = view.native
        return [[x0 + o["x_min"] * w, y0 + o["y_min"] * h, x0 + o["x_max"] * w, y0 + o["y_max"] * h] for o in objs]

def _load_moondream(spec):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(spec["hf_id"], revision=spec["revision"], trust_remote_code=True,
                                                torch_dtype=torch.bfloat16, device_map={"": "cuda"}).eval()

class Moondream2(Answerer):
    name = "moondream2"
    def __init__(self, cache, spec):
        super().__init__(cache); self.model = _load_moondream(spec)
        self.settings = {"temperature": 0.0, "top_p": 0.3, "max_tokens": MAX_NEW_TOKENS, "variant": None}
    @_infer
    def _query(self, view, prompt, intent=None):
        torch.manual_seed(SEED)
        try:
            out = self.model.query(view.pil, prompt, settings=self.settings)
        except (RuntimeError, ZeroDivisionError, ValueError):
            if self.settings["temperature"] != 0.0: raise
            self.settings["temperature"] = 1e-4; out = self.model.query(view.pil, prompt, settings=self.settings)
        return str(out["answer"]).strip()

class SmolVLM(Answerer):
    name = "smolvlm"
    def __init__(self, cache, spec):
        super().__init__(cache)
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoVLM
        self.processor = AutoProcessor.from_pretrained(spec["hf_id"], revision=spec["revision"],
                                                       size={"longest_edge": spec["longest_edge"]})
        self.model = AutoVLM.from_pretrained(spec["hf_id"], revision=spec["revision"], torch_dtype=torch.bfloat16,
                                             _attn_implementation="sdpa").to("cuda").eval()
    @_infer
    def _query(self, view, prompt, intent=None):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
        inp = self.processor(text=text, images=[view.pil], return_tensors="pt")
        inp = {k: (v.to("cuda", dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to("cuda")) for k, v in inp.items()}
        out = self.model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return self.processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

# ---------------- mock models for DRY_RUN (simulate run-1 behaviour, incl. yes-bias) ----------------
class _MockWorld:
    def __init__(self):
        self.sc = {ds: json.loads((data_dir(ds) / "scenes.json").read_text()) for ds in DATASETS}
    def scene(self, v): return self.sc[v.ds][str(v.img)]
    @staticmethod
    def quality(v):
        q = 1.0
        if v.pert == "altitude": q *= v.level ** 0.8
        if v.pert == "motion_blur": q *= math.exp(-v.level / 60)
        if v.pert == "haze": q *= (1 - v.level) ** 0.8
        if v.pert == "jpeg": q *= min(1, 0.55 + v.level / 60)
        return q
    def eff_area(self, v, area):
        s = min(1.0, MODEL_SIDE / max(v.native)) * self.quality(v)
        return area * s * s
    def inside(self, v, box):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        return v.box[0] <= cx < v.box[2] and v.box[1] <= cy < v.box[3]
    def seen(self, v, area, rng, a0=250):
        e = self.eff_area(v, area)
        return rng.random() < 1 / (1 + math.exp(-(math.log(e + 1e-6) - math.log(a0)) * 1.6))
    def objects(self, v, phrase, rng, attr=None):
        if v.key == "blank": return []
        s = self.scene(v); out = []
        if v.ds == "rsvqa_lr":
            for o in s["objects"]:
                if self.inside(v, o["box"]) and syn_match(o, phrase) and self.seen(v, o["area"], rng, a0=60): out.append(o)
            return out
        p = phrase.lower()
        pool = s["buildings"] if ("build" in p or "house" in p or "roof" in p) else (s["roads"] if "road" in p else [])
        for o in pool:
            area = o.get("area") or (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1])
            if not self.inside(v, o["box"]) or not self.seen(v, area, rng): continue
            if attr is not None:
                fl = o["flooded"] if rng.random() < 0.85 * self.quality(v) + 0.1 else not o["flooded"]
                if (attr == FLOODED) != fl: continue
            out.append(o)
        return out

class MockAnswerer(Answerer):
    def __init__(self, cache, name, skill, yes_bias):
        super().__init__(cache); self.name, self.skill, self.yes_bias = name, skill, yes_bias; self.W = _MockWorld()
    def timed(self, fn):
        out = fn(); r = np.random.default_rng(len(self.cache.db))
        return out, (0.09 if self.name == "moondream2" else 0.14) * r.uniform(0.85, 1.25), self.static_memory_mb() + r.uniform(200, 600)
    def static_memory_mb(self): return {"moondream2": 3680.0, "smolvlm": 4300.0}.get(self.name, 4000.0)
    def _rng(self, v, p): return np.random.default_rng(int(CallCache.key(self.name, v.key, p), 16) % 2**32)
    def _yn(self, truth, rng, style):
        bias = self.yes_bias if style == "yesno" else self.yes_bias * 0.25
        a = truth
        if not truth and rng.random() < bias: a = True
        if rng.random() < 0.08 / self.skill: a = not a
        if style == "forced": return rng.choice(["present", "Present."]) if a else rng.choice(["absent", "Absent."])
        if style == "truefalse": return "true" if a else "false"
        return rng.choice(["Yes.", "yes", "Yes, it is."]) if a else rng.choice(["No.", "no", "No, it is not."])
    def _query(self, v, prompt, it=None):
        rng, it = self._rng(v, prompt), (it or {"k": "free"}); k = it["k"]; W = self.W
        if k == "count":
            n = len(W.objects(v, it["target"], rng, it.get("attr")))
            n = max(0, int(round(n * rng.uniform(0.75, 1.15))))
            return rng.choice([f"{n}", f"There are {n}.", f"{n} {it['target']}"]) if rng.random() > 0.02 else "several"
        if k == "presence":
            truth = len(W.objects(v, it["target"], rng)) > 0
            return self._yn(truth, rng, it.get("style", "yesno"))
        if k == "comp":
            p = it["parts"]; na, nb = len(W.objects(v, p["a"], rng)), len(W.objects(v, p["b"], rng))
            t = na > nb if p["op"] == "more" else (na < nb if p["op"] == "less" else na == nb)
            return self._yn(t, rng, it.get("style", "yesno"))
        if k == "ru":
            urban = W.scene(v)["urban"] if v.key != "blank" else rng.random() < 0.4
            if rng.random() > 0.8 * W.quality(v) + 0.1: urban = not urban
            st = it.get("style", "word")
            if st == "city": return "city" if urban else "countryside"
            return ("urban" if urban else "rural") if rng.random() > 0.2 else f"It is a {'urban' if urban else 'rural'} area."
        if k in ("cond", "fn_yesno", "road_tile", "box_flooded"):
            if v.key == "blank":
                st = rng.choice([FLOODED, NONFLOODED, "mixed"])
            else:
                s = W.scene(v)
                if k == "box_flooded" or it.get("obj", "road") == "image":
                    pool = s["buildings"] if k == "box_flooded" else [dict(box=w, flooded=True) for w in s["water"]]
                    hit = [o for o in pool if W.inside(v, o["box"])]
                    if k == "box_flooded":
                        fl = bool(hit and hit[0]["flooded"])
                        if rng.random() > 0.85 * W.quality(v) + 0.1: fl = not fl
                        return ("flooded" if fl else "dry") if rng.random() > 0.15 else self._yn(fl, rng, "yesno")
                    st = FLOODED if hit else NONFLOODED
                else:
                    segs = [o for o in s["roads"] if W.inside(v, o["box"])]
                    if not segs: st = "noroad"
                    else:
                        fl = {o["flooded"] if rng.random() < 0.85 * W.quality(v) + 0.1 else not o["flooded"] for o in segs}
                        st = "mixed" if len(fl) == 2 else (FLOODED if True in fl else NONFLOODED)
                if rng.random() < 0.12: st = rng.choice([FLOODED, NONFLOODED, "mixed"])
            if k == "road_tile":
                return "no road" if st == "noroad" else {"mixed": "partially flooded"}.get(st, st)
            if st == "noroad": st = NONFLOODED
            if k == "fn_yesno":
                p = it["parts"]; truth = st == (FLOODED if p["polarity"] == FLOODED else NONFLOODED)
                return self._yn(truth, rng, it.get("style", "yesno"))
            if it.get("style") == "amount":
                return {FLOODED: "all", NONFLOODED: "none", "mixed": "some"}[st]
            return {"mixed": rng.choice(["partially flooded", "flooded, non flooded"])}.get(st, st)
        return rng.choice(["yes", "no", "3", "flooded"])

class MockGrounder(Grounder):
    name = "grounder"
    def __init__(self, cache): super().__init__(cache); self.W = _MockWorld()
    def timed(self, fn):
        out = fn(); r = np.random.default_rng(len(self.cache.db) + 7)
        return out, 0.21 * r.uniform(0.85, 1.2), 4200 + r.uniform(0, 300)
    def _detect(self, v, phrase):
        rng = np.random.default_rng(int(CallCache.key("det", v.key, phrase), 16) % 2**32)
        return [list(map(float, o["box"])) for o in self.W.objects(v, phrase, rng)]

def load_models(name):
    cache = CallCache(name)
    if DRY_RUN:
        a = MockAnswerer(cache, name, skill=1.0 if name == "moondream2" else 0.8,
                         yes_bias=0.6 if name == "moondream2" else 0.25)
        return a, MockGrounder(cache)
    spec = MODEL_SPECS[name]; print(f"loading answerer {spec['hf_id']} @ {spec['revision']} …")
    a = {"moondream2": Moondream2, "smolvlm": SmolVLM}[name](cache, spec)
    g = Grounder(cache, a.model if name == "moondream2" else _load_moondream(GROUNDER))
    return a, g

def free_models(a, g):
    a.cache._fh.close(); del a, g; gc.collect()
    if HAS_CUDA: torch.cuda.empty_cache()
