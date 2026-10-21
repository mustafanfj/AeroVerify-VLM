import json, matplotlib, os, sys, contextlib, io
matplotlib.use("Agg"); import matplotlib.pyplot as plt; plt.show = lambda *a, **k: plt.close("all")
nb = json.load(open(sys.argv[1])); ns = {"display": lambda x: None, "__name__": "__main__"}; n = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    src = "".join(c["source"])
    if src.startswith("%%stage"):
        st, src = src.split("\n", 1)
        if st.split()[1] not in ns["STAGES"]: continue
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf): exec(compile(src, f"cell{i}", "exec"), ns)
    except Exception:
        print(buf.getvalue()[-2000:]); raise
    n += 1
print("executed", n, "code cells")
