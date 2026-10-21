# ── Question understanding & answer normalisation (both datasets) ─────────────
# Every free-text answer is mapped to the dataset's fixed labels; unmappable → invalid (scored wrong).
BIN_KEYS = ["0", "1-10", "11-100", "101-1000", ">1000"]
COUNT_BINS = ["0", "1–10", "11–100", "101–1000", ">1000"]
FLOODED, NONFLOODED, MIXED = "flooded", "non flooded", "flooded,non flooded"

# type → answer kind  (yesno · cat · count_bin · count)
TYPES = {
    "rsvqa_lr": {"presence": "yesno", "comp": "yesno", "count": "count_bin", "rural_urban": "cat"},
    "floodnet": {"yesno": "yesno", "condition": "cat", "count_simple": "count", "count_complex": "count"},
}
TYPE_LABEL = {"presence": "Presence", "comp": "Comparison", "count": "Count", "rural_urban": "Rural/Urban",
              "yesno": "Yes/No", "condition": "Condition", "count_simple": "Simple count",
              "count_complex": "Complex count", "OA": "Overall (OA)", "AA": "Average (AA)"}
FLOODNET_TYPEMAP = {"Simple_Counting": "count_simple", "Complex_Counting": "count_complex",
                    "Condition_Recognition": "condition", "Yes_No": "yesno"}

REL_RE   = re.compile(r"\b(?:at the |on the |to the )?(top of|bottom of|left of|right of|next to)\b")
SMALL_RE = re.compile(r"\bsmall\b")
SIZE_RE  = re.compile(r"\b(small|medium|large)\b")
_TAIL    = r"(?:\s+(?:in|on|of) (?:the|this|given) (?:image|picture|area|region|scene))?\s*\??\s*$"

def _norm(q): return re.sub(r"\s+", " ", str(q).strip().lower())
def _strip_article(s): return re.sub(r"^(?:a|an|the|some|any)\s+", "", s.strip())

# ---------- RSVQA-LR (as run 1, plus "is a X present") ----------
def route_lr(q):
    s = _norm(q)
    if "rural" in s or "urban" in s: return "rural_urban"
    if re.match(r"^(how many|what is the number of|what is the amount of|what is the count of)\b", s): return "count"
    if re.search(r"\b(more|less|fewer|greater|smaller|larger|bigger|higher|lower)\b.*\bthan\b", s) \
       or re.search(r"\bequal\b|\bas many\b|\bsame number\b", s): return "comp"
    return "presence"

def presence_target(q):
    s = _norm(q)
    m = re.match(r"^(?:is|are) there (.+?)" + _TAIL, s) or \
        re.match(r"^(?:is|are) (.+?) (?:present|visible)" + _TAIL, s) or \
        re.match(r"^(?:can you see|do you see|does the image (?:contain|show)) (.+?)" + _TAIL, s)
    return _strip_article(m.group(1)) if m else None

def count_target(q):
    s = _norm(q)
    m = re.match(r"^(?:how many|what is the number of|what is the amount of|what is the count of) (.+?)"
                 r"(?:\s+(?:are|is) there|\s+(?:are|is) present|\s+(?:are|is)|\s+(?:in|on) (?:the|this|given) "
                 r"(?:image|picture|scene))*\s*\??\s*$", s)
    return _strip_article(m.group(1)) if m else None

def comp_parts(q):
    s = _norm(q)
    m = re.match(r"^(?:is|are) there (more|less|fewer) (.+?) than (.+?)" + _TAIL, s)
    if m:
        return dict(op="more" if m.group(1) == "more" else "less", a=_strip_article(m.group(2)), b=_strip_article(m.group(3)))
    m = re.match(r"^is the number of (.+?) (equal to|the same as|greater than|larger than|bigger than|higher than|"
                 r"more than|less than|smaller than|lower than|fewer than) the number of (.+?)" + _TAIL, s)
    if m:
        rel = m.group(2)
        op = "equal" if rel in ("equal to", "the same as") else \
             "more" if rel in ("greater than", "larger than", "bigger than", "higher than", "more than") else "less"
        return dict(op=op, a=_strip_article(m.group(1)), b=_strip_article(m.group(3)))
    m = re.match(r"^are there as many (.+?) as (.+?)" + _TAIL, s)
    return dict(op="equal", a=_strip_article(m.group(1)), b=_strip_article(m.group(2))) if m else None

def relational_subject(target):
    if not target: return None
    m = REL_RE.search(target)
    return _strip_article(target[:m.start()].strip()) if m else None

# ---------- FloodNet-VQA ----------
FN_NOUNS = ["building", "road", "vehicle", "car", "tree", "pool", "house"]
def route_fn(q):
    s = _norm(q)
    if s.startswith("how many") or "number of" in s:
        return "count_complex" if re.search(r"\b(non[ -]?flooded|not flooded|flooded)\b", s) else "count_simple"
    if "condition" in s: return "condition"
    return "yesno"

def fn_attr(q):
    s = _norm(q)
    if re.search(r"\bnon[ -]?flooded\b|\bnot flooded\b", s): return NONFLOODED
    if re.search(r"\bflooded\b", s): return FLOODED
    return None

def fn_object(q):
    s = _norm(q)
    for n in FN_NOUNS:
        if re.search(rf"\b{n}s?\b", s):
            return {"car": "vehicle", "house": "building"}.get(n, n)
    return "building"

def fn_yesno_parts(q):
    """'Is the entire road (non) flooded?' → {'obj','polarity','quant'}; None if another form."""
    s = _norm(q)
    m = re.match(r"^(?:is|are) (?:the )?(entire|whole|all(?: of)?(?: the)?|any|some(?: of)?(?: the)?)?\s*(\w+?)s? "
                 r"(?:area |areas |surface )?(non[ -]?flooded|not flooded|flooded)\s*\??$", s)
    if not m:
        m2 = re.match(r"^(?:is|are) there (?:any |a )?(non[ -]?flooded|flooded) (\w+?)s?\s*\??$", s)
        if m2:
            return dict(obj=m2.group(2), polarity=NONFLOODED if "non" in m2.group(1) else FLOODED, quant="any")
        return None
    quant = "any" if (m.group(1) or "").startswith(("any", "some")) else "entire"
    pol = NONFLOODED if re.match(r"non|not", m.group(3)) else FLOODED
    return dict(obj=m.group(2), polarity=pol, quant=quant)

def cond_subtype(q):
    return "road" if "road" in _norm(q) else "image"

# ---------- generic ----------
def route(q, ds): return route_lr(q) if ds == "rsvqa_lr" else route_fn(q)
def kind_of(ds, t): return TYPES[ds][t]

def question_tags(q, ds):
    s = _norm(q)
    if ds == "rsvqa_lr":
        sg = "spatial" if REL_RE.search(s) else ("small-object" if SMALL_RE.search(s) else "other")
    else:
        sg = "attribute" if fn_attr(q) else "plain"
    return dict(small=bool(SMALL_RE.search(s)), relational=bool(REL_RE.search(s)), subgroup=sg)

_NUM_WORDS = {"zero": 0, "no": 0, "none": 0, "one": 1, "a single": 1, "single": 1, "two": 2, "a couple of": 2,
              "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
              "eleven": 11, "twelve": 12, "a dozen": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
              "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
              "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100, "a hundred": 100, "thousand": 1000}
_NUM_WORD_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _NUM_WORDS), key=len, reverse=True)) + r")\b")

def count_bin(n):
    if n is None: return None
    n = int(n)
    return "0" if n <= 0 else "1-10" if n <= 10 else "11-100" if n <= 100 else "101-1000" if n <= 1000 else ">1000"

def parse_count(text):
    s = str(text).lower()
    m = re.search(r"\d[\d,]*(?:\.\d+)?", s)
    if m:
        try: return int(round(float(m.group(0).replace(",", ""))))
        except ValueError: pass
    m = _NUM_WORD_RE.search(s)
    if m: return _NUM_WORDS[m.group(1)]
    if re.search(r"\b(not any|nothing|there are no|there is no)\b", s): return 0
    return None

def parse_yesno(text):
    s = str(text).lower()
    m = re.search(r"\b(yes|no|true|false|yeah|yep|nope|correct|incorrect|present|absent)\b", s)
    if m:
        return "yes" if m.group(1) in ("yes", "true", "yeah", "yep", "correct", "present") else "no"
    if re.search(r"\b(there (?:is|are) no|not|none|cannot see|can't see)\b", s): return "no"
    if re.search(r"\bthere (?:is|are)\b|\bi can see\b", s): return "yes"
    return None

_RU = {"rural": "rural", "countryside": "rural", "agricultural": "rural", "farmland": "rural", "village": "rural",
       "urban": "urban", "city": "urban", "town": "urban", "suburban": "urban", "residential": "urban"}
def parse_rural_urban(text):
    s = str(text).lower()
    hits = [(m.start(), _RU[m.group(1)]) for m in re.finditer(r"\b(" + "|".join(_RU) + r")\b", s)]
    return min(hits)[1] if hits else None

def parse_condition(text):
    """→ flooded | non flooded | mixed | None"""
    s = str(text).lower().replace("-", " ")
    if re.search(r"\b(partial|partially|partly|some of|mixed|both|half)\b", s): return "mixed"
    if re.search(r"\b(non flooded|not flooded|unflooded|dry|no flood\w*|none|good|normal|fine|clear|intact|passable|undamaged)\b", s):
        return NONFLOODED
    if re.search(r"\b(flooded|flood|under water|underwater|inundated|submerged|waterlogged|impassable|all of it|all)\b", s):
        return FLOODED
    return None

def parse_instance(text):
    """instance check 'flooded or dry' → flooded / non flooded (falls back to yes/no)."""
    c = parse_condition(text)
    if c in (FLOODED, NONFLOODED): return c
    yn = parse_yesno(text)
    return None if yn is None else (FLOODED if yn == "yes" else NONFLOODED)

def parse_amount(text):
    """'none / some / all' probe → condition label"""
    s = str(text).lower()
    m = re.search(r"\b(none|nothing|no part|some|part|partly|partially|half|most|all|entire|entirely|whole)\b", s)
    if not m: return parse_condition(text)
    w = m.group(1)
    return NONFLOODED if w in ("none", "nothing", "no part") else FLOODED if w in ("all", "entire", "entirely", "whole") else "mixed"

# answer space per condition sub-type; re-estimated from the training pool in the sampling cell
COND_LABELS = {"road": [MIXED, FLOODED, NONFLOODED], "image": [FLOODED, NONFLOODED]}

def cond_to_label(c, subtype):
    if c is None: return None
    if c == "mixed": return MIXED if MIXED in COND_LABELS[subtype] else FLOODED
    return c

def normalize_answer(text, ds, t, q=None):
    if text is None: return None
    k = kind_of(ds, t)
    if k == "yesno": return parse_yesno(text)
    if k == "count_bin": return count_bin(parse_count(text))
    if k == "count":
        n = parse_count(text); return None if n is None else str(max(0, n))
    if t == "rural_urban": return parse_rural_urban(text)
    if t == "condition": return cond_to_label(parse_condition(text), cond_subtype(q or ""))
    raise ValueError((ds, t))

_GT_BINS = {"0": "0", "between 1 and 10": "1-10", "between 11 and 100": "11-100",
            "between 101 and 1000": "101-1000", "more than 1000": ">1000"}
def gt_label(answer, ds, t):
    a = str(answer).strip().lower()
    if ds == "rsvqa_lr":
        return (_GT_BINS.get(a) or count_bin(parse_count(a))) if t == "count" else a
    if t in ("count_simple", "count_complex"):
        n = parse_count(a); return None if n is None else str(n)
    if t == "condition":
        a = re.sub(r"\s+", " ", a.replace("-", " ").replace("_", " "))
        return re.sub(r"\s*,\s*", ",", a)
    return a

# ── self-tests ───────────────────────────────────────────────────────────────
assert route_lr("Is it a rural or an urban area?") == "rural_urban" and route_lr("How many roads are there?") == "count"
assert route_lr("Is the number of roads equal to the number of water areas?") == "comp"
assert presence_target("Is a residential building present?") == "residential building"
assert count_target("How many buildings are in the image?") == "buildings"
assert count_target("How many large buildings are there?") == "large buildings"
assert comp_parts("Are there less roads than buildings?") == dict(op="less", a="roads", b="buildings")
assert route_fn("How many buildings are in the image?") == "count_simple"
assert route_fn("How many buildings are non flooded in this image?") == "count_complex"
assert route_fn("What is the condition of the road in this image?") == "condition"
assert route_fn("Is the entire road non flooded?") == "yesno"
assert fn_attr("How many buildings are non flooded in this image?") == NONFLOODED
assert fn_attr("How many flooded buildings are there?") == FLOODED and fn_object("How many flooded buildings?") == "building"
assert fn_yesno_parts("Is the entire road non flooded?") == dict(obj="road", polarity=NONFLOODED, quant="entire")
assert fn_yesno_parts("Is the entire road flooded?")["polarity"] == FLOODED
assert parse_condition("non-flooded") == NONFLOODED and parse_condition("Flooded.") == FLOODED
assert parse_condition("partially flooded") == "mixed" and parse_amount("Some of it") == "mixed"
assert normalize_answer("The road is partly flooded", "floodnet", "condition", "condition of the road?") == MIXED
assert normalize_answer("partly flooded", "floodnet", "condition", "overall condition of the image?") == FLOODED
assert gt_label("flooded, non-flooded", "floodnet", "condition") == MIXED and gt_label(7, "floodnet", "count_simple") == "7"
assert normalize_answer("There are 12 buildings.", "floodnet", "count_simple") == "12"
assert parse_condition("Good") == NONFLOODED and parse_condition("Waterlogged") == FLOODED
assert parse_instance("Dry.") == NONFLOODED and parse_instance("yes") == FLOODED
assert parse_yesno("present") == "yes" and parse_yesno("absent") == "no"
assert gt_label("17", "rsvqa_lr", "count") == "11-100"
print("question parser self-tests passed ✓")
