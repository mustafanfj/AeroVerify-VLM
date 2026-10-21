### Sampling
* **FloodNet-VQA:** the answers of the official val/test splits were never released, so (as in prior work) DEV and EVAL images are drawn **disjointly from the 1,448 annotated training images**. Every question of a sampled image is used.
* **RSVQA-LR:** the **same sample as run 1** (40 train images for DEV, 160 test/val images for EVAL), so the two runs are directly comparable.
* DEV is used only for reliability weights and thresholds. Every reported number comes from EVAL. The most-common-answer baseline uses training questions that are in neither sample.
