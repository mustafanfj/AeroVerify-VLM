# AeroVerify-VLM v2: Evidence-Grounded, Reliability-Weighted VQA for UAV Imagery

**Version 2.** Redesigned after run 1 on RSVQA-LR. Each change addresses a measured failure:

| Run-1 finding | v2 design |
|---|---|
| 10 m satellite tiles: both VLMs ≈ most-common answer; crops add no information | **FloodNet-VQA (real UAV, 4000×3000 px)** is the primary benchmark. RSVQA-LR is kept as the *low-resolution limit* case |
| Upscaled crops hurt SmolVLM | **Resolution-gated native tiles**: zoom only when tiles carry more pixels than the model's input |
| Detector evidence helped Moondream2 most; SmolVLM had none | **Shared grounding module** for both VLMs, plus **grounded instance-level counting** |
| Moondream2 says "yes" to 92–97 % of presence questions; correlated sources cannot outvote a shared bias | **Contrastive / decomposed probes** that avoid yes/no tokens |
| Plurality voting gives 3–4 confidence levels | **Reliability-weighted fusion** (weights from DEV accuracy), giving a posterior confidence and smooth risk–coverage |
| E1 penalised by unparseable free text | E1 uses the standard short-answer VQA suffix. **E2 is the headline baseline** |
| No cost control | **AeroVerify-Lite**: an early-exit cascade |

**Experiments:** E1 direct · E2 structured · E3 region · **E4 AeroVerify** (and Lite) · E5 ablations (computed from cached evidence) · E6 efficiency · E7 UAV-specific robustness (altitude, motion blur, haze, JPEG).

**Outputs:** `outputs_v2/figures/*.pdf` (IEEE widths, ≥7 pt text, embedded fonts) and `outputs_v2/tables/*.tex` (booktabs, paste into LaTeX).

> Run headless with `code/run_experiments.sh` (see `README.md`). Settings can be overridden with `AV_*` environment variables.
