# ── Methods: prompts and evidence sources ─────────────────────────────────────
E1_SUFFIX = "Answer the question using a single word or phrase."      # standard short-answer VQA prompt
CTX  = {"floodnet": "This is a nadir UAV (drone) image of a neighbourhood.",
        "rsvqa_lr": "This is an overhead remote-sensing image seen from above."}
TCTX = {"floodnet": "This is one zoomed-in region of a UAV (drone) image.",
        "rsvqa_lr": "This is a zoomed-in region of an overhead remote-sensing image."}
FMT = {"yesno": "Answer with only one word: yes or no.", "num": "Answer with a single number only.",
       "ru": "Answer with only one word: rural or urban."}

def _plural(w): return w if w.endswith("s") else w + "s"
def fn_count_target(q):
    a = fn_attr(q); o = _plural(fn_object(q))
    return f"{a} {o}" if a else o

def _rec(r): return dict(out=r["out"] if r["kind"] == "query" else len(r["out"]), lat=r["lat"], mem=r["mem"])
def _yes(c): return "yes" if c else "no"
def _cmp(na, nb, op): return _yes(na > nb if op == "more" else (na < nb if op == "less" else na == nb))

# ---------- E2: structured single prompt on the full image ----------
def prompt_structured(ds, q, t):
    c = CTX[ds]
    if ds == "rsvqa_lr":
        return {"presence": f"{c} Look carefully at the whole image, including small objects. Question: {q} {FMT['yesno']}",
                "comp": f"{c} Compare how many of each kind of object are visible. Question: {q} {FMT['yesno']}",
                "count": f"{c} Count every separate instance carefully. Question: {q} {FMT['num']}",
                "rural_urban": f"{c} Judge from the density of buildings and roads and from the land use. "
                               f"Question: Is this a rural or an urban area? {FMT['ru']}"}[t]
    if t.startswith("count"):
        return f"{c} Count every separate {fn_count_target(q)[:-1]} carefully, including partly visible ones. Question: {q} {FMT['num']}"
    if t == "condition":
        opts = "flooded, non flooded" + (", partially flooded" if MIXED in COND_LABELS[cond_subtype(q)] else "")
        return f"{c} Question: {q} Answer with exactly one of: {opts}."
    return f"{c} Question: {q} {FMT['yesno']}"

def intent_for(ds, q, t, style="yesno"):
    if ds == "rsvqa_lr":
        if t == "presence": return dict(k="presence", target=presence_target(q) or "building", style=style)
        if t == "count": return dict(k="count", target=count_target(q) or "building")
        if t == "comp": return dict(k="comp", parts=comp_parts(q) or dict(op="more", a="building", b="road"), style=style)
        return dict(k="ru")
    if t.startswith("count"): return dict(k="count", target=_plural(fn_object(q)), attr=fn_attr(q))
    if t == "condition": return dict(k="cond", obj=cond_subtype(q))
    return dict(k="fn_yesno", parts=fn_yesno_parts(q) or dict(obj="road", polarity=FLOODED, quant="entire"), style=style)

def src_full(A, ds, q, t, fv):
    r = A.ask(fv, prompt_structured(ds, q, t), intent_for(ds, q, t))
    return dict(ans=normalize_answer(r["out"], ds, t, q), raw=r["out"], calls=[_rec(r)])

# ---------- tiles (E3; in E4 only when the resolution gate is open) ----------
ROAD_TILE = "Look at the road in this region. Reply with exactly one of: no road, flooded, non flooded, partially flooded."
def _road_states(A, tv):
    calls, per = [], {}
    for n, v in tv.items():
        r = A.ask(v, f"{TCTX['floodnet']} {ROAD_TILE}", dict(k="road_tile")); calls.append(_rec(r))
        s = str(r["out"]).lower()
        per[n] = None if re.search(r"\bno road\b|\bnone\b|\bnot visible\b", s) else parse_condition(s)
    return per, calls

def _agg_states(states):
    st = {s for s in states if s}
    if not st: return None
    if "mixed" in st or {FLOODED, NONFLOODED} <= st: return "mixed"
    return st.pop()

def src_tiles(A, ds, q, t, tv):
    tc, calls, per = TCTX[ds], [], {}
    if ds == "rsvqa_lr":
        if t == "presence":
            for n, v in tv.items():
                r = A.ask(v, f"{tc} Question: {q} {FMT['yesno']}", intent_for(ds, q, t)); calls.append(_rec(r)); per[n] = parse_yesno(r["out"])
            val = [a for a in per.values() if a]; ans = _yes("yes" in val) if val else None
        elif t == "count":
            tg = count_target(q)
            for n, v in tv.items():
                r = A.ask(v, f"{tc} How many {tg} are visible in this region? {FMT['num']}", intent_for(ds, q, t))
                calls.append(_rec(r)); per[n] = parse_count(r["out"])
            val = [c for c in per.values() if c is not None]; ans = count_bin(sum(val)) if val else None
        elif t == "comp":
            p = comp_parts(q); sa = sb = 0; ok = False
            if p:
                for n, v in tv.items():
                    ra = A.ask(v, f"{tc} How many {p['a']} are visible in this region? {FMT['num']}", dict(k="count", target=p["a"]))
                    rb = A.ask(v, f"{tc} How many {p['b']} are visible in this region? {FMT['num']}", dict(k="count", target=p["b"]))
                    calls += [_rec(ra), _rec(rb)]; ca, cb = parse_count(ra["out"]), parse_count(rb["out"])
                    per[n] = [ca, cb]; ok |= (ca is not None or cb is not None); sa += ca or 0; sb += cb or 0
            ans = _cmp(sa, sb, p["op"]) if ok else None
        else:
            for n, v in tv.items():
                r = A.ask(v, f"{tc} Is this region rural or urban? {FMT['ru']}", dict(k="ru")); calls.append(_rec(r)); per[n] = parse_rural_urban(r["out"])
            val = [a for a in per.values() if a]; ans = None if not val else ("urban" if 2 * val.count("urban") >= len(val) else "rural")
        return dict(ans=ans, per=per, calls=calls)
    # FloodNet (native-resolution tiles)
    if t.startswith("count"):
        tg = fn_count_target(q)
        for n, v in tv.items():
            r = A.ask(v, f"{tc} How many {tg} are visible in this region? {FMT['num']}", intent_for(ds, q, t))
            calls.append(_rec(r)); per[n] = parse_count(r["out"])
        val = [c for c in per.values() if c is not None]; ans = str(sum(val)) if val else None
    elif t == "condition" and cond_subtype(q) == "image":
        for n, v in tv.items():
            r = A.ask(v, f"{tc} Is there flood water in this region? Reply with exactly one word: flooded or dry.", dict(k="cond", obj="image"))
            calls.append(_rec(r)); per[n] = parse_condition(r["out"])
        val = [a for a in per.values() if a]                            # majority of tiles (OR over-reports flooding)
        ans = None if not val else (FLOODED if 2 * sum(a in (FLOODED, "mixed") for a in val) >= len(val) else NONFLOODED)
    else:
        per, calls = _road_states(A, tv); agg = _agg_states(per.values())
        if t == "condition":
            if MIXED in COND_LABELS["road"]:
                ans = cond_to_label(agg, "road")
            else:                                                       # binary label space → majority of road tiles
                vals = [s_ for s_ in per.values() if s_]
                ans = None if not vals else (FLOODED if 2 * sum(v in (FLOODED, "mixed") for v in vals) >= len(vals) else NONFLOODED)
        else:
            p = fn_yesno_parts(q); vals = [s for s in per.values() if s]
            if p and vals:
                if p["quant"] == "entire": ans = _yes(all(s == p["polarity"] for s in vals))
                else: ans = _yes(any(s in (p["polarity"], "mixed") for s in vals))
            else: ans = None
    return dict(ans=ans, per=per, calls=calls)

# ---------- contrastive / decomposed probe (avoids yes/no tokens where possible) ----------
def _swap_comp(p):
    a, b, op = p["a"], p["b"], p["op"]
    return {"more": f"Are there fewer {b} than {a}?", "less": f"Are there more {b} than {a}?"}.get(
        op, f"Is the number of {b} equal to the number of {a}?")

def _amount_probe(A, fv, obj):
    what = "the road" if obj == "road" else "the scene"
    p = f"{CTX['floodnet']} How much of {what} is covered by flood water? Reply with exactly one word: none, some, or all."
    r = A.ask(fv, p, dict(k="cond", obj=obj, style="amount")); return parse_amount(r["out"]), r, p

def src_probe(A, ds, q, t, fv):
    c = CTX[ds]
    if ds == "rsvqa_lr":
        if t == "presence":
            tg = presence_target(q)
            p = (f"{c} Is any {tg} visible in this image? Reply with exactly one word: present or absent." if tg else
                 f"{c} Question: {q} Reply with exactly one word: present or absent.")
            r = A.ask(fv, p, intent_for(ds, q, t, style="forced")); ans = parse_yesno(r["out"])
        elif t == "count":
            tg = count_target(q); p = f"{c} What is the number of {tg} in this image? {FMT['num']}"
            r = A.ask(fv, p, intent_for(ds, q, t)); ans = normalize_answer(r["out"], ds, t)
        elif t == "comp":
            parts = comp_parts(q)
            p = f"{c} {_swap_comp(parts)} {FMT['yesno']}" if parts else f"{c} Question: {q} Reply with exactly one word: true or false."
            it = dict(k="comp", parts=dict(op={"more": "less", "less": "more"}.get(parts["op"], "equal"), a=parts["b"], b=parts["a"])) if parts else None
            r = A.ask(fv, p, it); ans = parse_yesno(r["out"])
        else:
            p = f"{c} Does this area look more like a city or the countryside? Reply with exactly one word: city or countryside."
            r = A.ask(fv, p, dict(k="ru", style="city")); ans = parse_rural_urban(r["out"])
        return dict(ans=ans, raw=r["out"], prompt=p, calls=[_rec(r)])
    if t.startswith("count"):
        p = f"{c} What is the total number of {fn_count_target(q)} in this image? Reply with a number only."
        r = A.ask(fv, p, intent_for(ds, q, t)); ans = normalize_answer(r["out"], ds, t)
    elif t == "condition":
        sub = cond_subtype(q); a, r, p = _amount_probe(A, fv, sub); ans = cond_to_label(a, sub)
    else:
        parts = fn_yesno_parts(q)
        if parts:
            a, r, p = _amount_probe(A, fv, "road" if parts["obj"] == "road" else "image")
            if a is None: ans = None
            elif parts["quant"] == "entire": ans = _yes(a == parts["polarity"])
            else: ans = _yes(a in (parts["polarity"], "mixed"))
        else:
            p = f"{c} Question: {q} Reply with exactly one word: true or false."
            r = A.ask(fv, p, intent_for(ds, q, t, style="truefalse")); ans = parse_yesno(r["out"])
    return dict(ans=ans, raw=r["out"], prompt=p, calls=[_rec(r)])

# ---------- grounding (shared detector + instance-level verification) ----------
def _merge(boxes, iou=0.5):
    keep = []
    for b in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        if all(_iou(b, k) < iou for k in keep): keep.append(b)
    return keep

def _iou(a, b):
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0

def src_ground(A, G, ds, q, t, fv, tv, gate):
    views = list(tv.values()) if gate else [fv]
    if ds == "rsvqa_lr":
        if t == "rural_urban": return None
        calls, boxes, ans = [], {}, None
        if t == "presence":
            tg = presence_target(q); subj = relational_subject(tg); ph = subj or tg
            if ph:
                r = G.detect(fv, ph); calls.append(_rec(r)); boxes[ph] = r["out"]; n = len(r["out"])
                ans = ("no" if n == 0 else None) if subj else _yes(n > 0)
        elif t == "count":
            tg = count_target(q)
            if tg: r = G.detect(fv, tg); calls.append(_rec(r)); boxes[tg] = r["out"]; ans = count_bin(len(r["out"]))
        else:
            p = comp_parts(q)
            if p:
                ra, rb = G.detect(fv, p["a"]), G.detect(fv, p["b"]); calls += [_rec(ra), _rec(rb)]
                boxes[p["a"]], boxes[p["b"]] = ra["out"], rb["out"]; ans = _cmp(len(ra["out"]), len(rb["out"]), p["op"])
        return dict(ans=ans, boxes=boxes, calls=calls)
    if not t.startswith("count"): return None
    obj = fn_object(q); calls, allb = [], []
    for v in views:
        r = G.detect(v, obj); calls.append(_rec(r)); allb += r["out"]
    allb = _merge(allb); n = len(allb); attr = fn_attr(q)
    if t == "count_simple" or attr is None:
        return dict(ans=str(n), boxes={obj: allb}, labels=None, calls=calls)
    chosen = sorted(allb, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:MAX_BOXES]; labels = []
    for k, b in enumerate(chosen):                                   # instance-level verification
        bv = box_view(fv, b, k)
        r = A.ask(bv, f"This close-up is from a UAV (drone) image. Is this {obj} standing in or surrounded "
                      f"by flood water? Reply with exactly one word: flooded or dry.", dict(k="box_flooded"))
        calls.append(_rec(r)); labels.append(parse_instance(r["out"]))
    if chosen and any(labels):
        frac = np.mean([l == attr for l in labels if l]); est = int(round(frac * n))
    else:
        est = 0 if n == 0 else None
    return dict(ans=None if est is None else str(est), boxes={obj: chosen}, labels=labels, n_det=n, calls=calls)

def answer_question(A, G, row, fv, tv_native, tv_up, gate, with_blind=False):
    ds, q, t = row["ds"], row["question"], row["route"]
    r1 = A.ask(fv, f"{q}\n{E1_SUFFIX}", intent_for(ds, q, t))
    tv = tv_native if ds == "floodnet" else tv_up                    # E3 baseline tiles (always computed)
    rec = dict(qid=row["qid"], gate=bool(gate),
               e1=dict(ans=normalize_answer(r1["out"], ds, t, q), raw=r1["out"], calls=[_rec(r1)]),
               sources=dict(full=src_full(A, ds, q, t, fv), tiles=src_tiles(A, ds, q, t, tv),
                            probe=src_probe(A, ds, q, t, fv), ground=src_ground(A, G, ds, q, t, fv, tv_native, gate)))
    if with_blind:
        rb = A.ask(BLANK_VIEW, prompt_structured(ds, q, t), intent_for(ds, q, t))
        rec["blind"] = dict(ans=normalize_answer(rb["out"], ds, t, q), raw=rb["out"], calls=[_rec(rb)])
    return rec
