## Analysis
**AeroVerify decision rule.** Each evidence source *s* (full image, native tiles, probe, grounding) gets a reliability weight from its DEV accuracy *a<sub>s</sub>* for that question type: *w<sub>s</sub>* = log[(K−1)·a<sub>s</sub>/(1−a<sub>s</sub>)], clipped at 0, so a source no better than chance is ignored. This is the optimal weighted vote for independent voters (Nitzan & Paroush, 1982).
* **Categorical answers:** the posterior is p(ℓ) ∝ exp Σ<sub>s</sub> w<sub>s</sub>·1[v<sub>s</sub>=ℓ]. The answer is the arg-max and the **confidence** is its posterior probability.
* **Counts (FloodNet):** the answer is the weighted median of the source counts. The confidence is the weight share within ±max(1, 10 %) of that median.
* **Tiles** take part only when the **resolution gate** is open (tile side ≥ model input). AeroVerify **abstains** below τ, with τ tuned on DEV.
* **AeroVerify-Lite** asks the full image and the probe first. If they agree it stops; otherwise it gathers the remaining evidence.

All variants and ablations are recomputed here from the stored evidence.
