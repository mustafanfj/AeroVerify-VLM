## Run inference
For every question the run stores the answers of **all evidence sources** with their call costs: E1, full image (= E2), tiles (= E3), contrastive probe, grounding, and the question-only baseline. AeroVerify, AeroVerify-Lite and every E5 ablation are then computed **after** the run from this evidence, so they need no extra GPU time. Re-running resumes from the call cache.
