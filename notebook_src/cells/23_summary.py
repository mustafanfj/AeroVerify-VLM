%%stage analyze
# ── Headline numbers + file index (also written to outputs_v2/summary.txt) ────
import contextlib, io
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    for ds in DATASETS:
        print(f"\n{DS_LABEL[ds]}")
        for m in MODELS:
            s2, s4, s4s, sl = S[(m, ds, "E2")], S[(m, ds, "E4 −A")], S[(m, ds, "E4")], S[(m, ds, "Lite −A")]
            t = SIG[(SIG.dataset == DS_SHORT[ds]) & (SIG.model == MODEL_LABEL[m]) & (SIG.comparison == "E4 −A vs E2")].iloc[0]
            lat = {k: EV[(EV.model == m) & (EV.ds == ds) & (EV.method == k)].latency.mean() for k in ("E2", "E4", "Lite")}
            print(f"  {MODEL_LABEL[m]:10s} OA  E2 {s2['OA']:.1f} → AeroVerify {s4['OA']:.1f} ({t.delta_pp:+.1f} pp, Holm p={t.p_holm:.2g})"
                  f" → {s4s['OA']:.1f} on {s4s['coverage']:.0f}% answered | Lite {sl['OA']:.1f}"
                  f" | hallucinated yes {s2['FPR']:.0f}% → {s4['FPR']:.0f}%"
                  f" | latency {lat['E2']:.2f}s → {lat['E4']:.2f}s (Lite {lat['Lite']:.2f}s)")
    print("\nSelective prediction (all confidences calibrated per question type on DEV). ΔAURC = baseline − ours, 95% CI:")
    print(SELTAB.round(1).to_string(index=False))
    print("\nEvidence-source accuracy on EVAL (%):")
    print(SRCTAB.pivot_table(index=["dataset", "model", "type"], columns="source", values="accuracy").round(1).to_string())
    print("\nAnswered (%):")
    print(SRCTAB.pivot_table(index=["dataset", "model", "type"], columns="source", values="answered").round(0).to_string())
    print("\nCondition label space:", COND_LABELS)
print(_buf.getvalue()); (OUT_DIR / "summary.txt").write_text(_buf.getvalue())
print("\nFigures:", FIG_DIR); [print("  ", p.name) for p in sorted(FIG_DIR.glob("*.pdf"))]
print("Tables :", TAB_DIR); [print("  ", p.name) for p in sorted(TAB_DIR.glob("*")) if p.suffix in (".tex", ".csv")]
if DRY_RUN: print("\n⚠ DRY RUN — synthetic data and mock models; these numbers only test the pipeline.")
