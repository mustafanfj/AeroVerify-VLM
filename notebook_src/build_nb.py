import json, glob, os
MD = {
 "01_params.py": "## 1 · Parameters\nEdit here or override with `AV_*` environment variables. `AV_DRY_RUN=1` runs everything on synthetic data with mock models.",
 "02_setup.py": "## 2 · Environment",
 "04_qparse.py": "## 3 · Question routing and answer normalisation\nRule-based routers (no labels used) and parsers that map free text to each dataset's fixed answers. Self-tests run at the end of the cell.",
 "05_data.py": "## 4 · Datasets\n* **FloodNet-VQA**: the Hugging Face mirror of the official challenge data (`takara-ai/FloodNet_2021-Track_2_Dataset_HF`). Only the annotations (~1,450 small JSON files) and the **sampled** images (~5 MB each) are downloaded, not the 12 GB repository.\n* **RSVQA-LR**: downloaded from Zenodo (record 6344334).",
 "07_images.py": "## 5 · Images, native tiles, resolution gate, UAV perturbations",
 "08_models.py": "## 6 · Models\nAnswerers: **Moondream2** (pinned `2025-06-21`) and **SmolVLM-Instruct**, both with greedy decoding. One **shared grounding module** (the Moondream2 detect head) serves both, so the comparison between answerers is fair. Every call is cached with its latency and peak memory.",
 "09_methods.py": "## 7 · Methods and evidence sources\n* **E1:** the question plus the standard short-answer suffix.\n* **E2:** a type-specific prompt on the full image.\n* **E3:** 2×2 tiles: native resolution on FloodNet; upsampled crops on RSVQA-LR (the run-1 baseline).\n* **Probe:** forced-choice or decomposed questions, e.g. *\"how much of the road is flooded: none / some / all\"* or *\"present / absent\"*.\n* **Grounding:** instances are detected per tile, and for attribute counts each instance is verified (*\"flooded or dry?\"*).",
 "13_fig_pipeline.py": "### Fig. 1 · Method overview (drawn on a real EVAL image)",
 "14_fig_qualitative.py": "### Fig. 2 · Qualitative evidence\nEach column shows one EVAL question (picked automatically: E2 wrong, AeroVerify right): tile answers, detected instances coloured by their verification, the vote of each source with its DEV reliability weight *w*, and the final confidence *p*.",
 "15_fig_main.py": "### Fig. 3 · Main results (full coverage, 95% image-level bootstrap CIs)",
 "16_fig_risk.py": "### Fig. 4 · Selective prediction (risk–coverage)",
 "17_fig_ablation.py": "### Fig. 5 · E5 ablation",
 "18_fig_efficiency.py": "### Fig. 6 · E6 accuracy–latency trade-off",
 "19_fig_robustness.py": "### Fig. 7 · E7 robustness to UAV degradations",
 "20_fig_bias.py": "### Fig. 8 · Yes-bias and hallucinated \"yes\"",
 "21_fig_counting.py": "### Fig. 9 · Counting on UAV imagery",
 "22_fig_calibration.py": "### Fig. 10 · Confidence calibration",
 "23_summary.py": "## Summary",
}
def lines(src):
    src = src.rstrip("\n") + "\n"; out = src.splitlines(keepends=True); out[-1] = out[-1].rstrip("\n"); return out
cells = []
for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells", "*"))):
    b = os.path.basename(f); src = open(f).read()
    if b in MD: cells.append({"cell_type": "markdown", "metadata": {}, "source": lines(MD[b])})
    if b.endswith(".md"): cells.append({"cell_type": "markdown", "metadata": {}, "source": lines(src)})
    else: cells.append({"cell_type": "code", "execution_count": None, "metadata": {"tags": ["parameters"]} if b == "01_params.py" else {},
                        "outputs": [], "source": lines(src)})
for i, c in enumerate(cells): c["id"] = f"cell-{i:02d}"
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python (aeroverify_vlm)", "language": "python", "name": "aeroverify-vlm"},
      "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 5}
HERE = os.path.dirname(os.path.abspath(__file__)); out = os.path.join(HERE, "..", "notebooks", "aeroverify_vlm_v2.ipynb")
json.dump(nb, open(out, "w"), indent=1, ensure_ascii=False)
print(len(cells), "cells")
