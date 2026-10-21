def run_model(name):
    A, G = load_models(name)
    meta = dict(model=name, static_mem_mb=A.static_memory_mb())
    if HAS_CUDA and not DRY_RUN:                                     # warm-up (not cached)
        r0 = SAMPLE.iloc[0]; w = full_view(r0.ds, r0.img)
        for _ in range(2): A._query(w, "Describe this image in one word.")
    out_path, n = PRED_DIR / f"{name}.jsonl", 0
    with open(out_path, "w") as fh:
        def run_group(ds, img, g, pert, level, blind):
            nonlocal n
            fv = full_view(ds, img, pert, level)
            gate = zoom_gain(fv.native) >= GATE_MIN_ZOOM
            tv_native = tile_views(ds, img, pert, level) if ds == "floodnet" else {}
            tv_up = tile_views(ds, img, pert, level, upscale_to=LR_CROP_SIZE) if ds == "rsvqa_lr" else {}
            for _, row in g.iterrows():
                rec = answer_question(A, G, row, fv, tv_native, tv_up, gate, with_blind=blind)
                rec.update(model=name, ds=ds, img=str(img), role=row["role"], pert=pert, level=level)
                fh.write(json.dumps(rec) + "\n"); n += 1
        for ds in DATASETS:
            S = SAMPLE[SAMPLE.ds == ds]
            for img, g in tqdm(list(S.groupby("img", sort=False)), desc=f"{name} · {DS_SHORT[ds]} clean"):
                run_group(ds, img, g, "clean", 0.0, True)
        if RUN_ROBUSTNESS:
            for ds, imgs in ROBUST_IMGS.items():
                S = SAMPLE[(SAMPLE.ds == ds) & SAMPLE.img.isin(imgs)]
                for kind, levels in PERTURBATIONS.items():
                    for level in levels:
                        for img, g in tqdm(list(S.groupby("img", sort=False)), desc=f"{name} · {kind} {level}", leave=False):
                            run_group(ds, img, g, kind, level, False)
    meta.update(records=n, new_calls=A.cache.misses, cache_hits=A.cache.hits)
    (PRED_DIR / f"{name}_meta.json").write_text(json.dumps(meta))
    print(f"{name}: {n} records · {A.cache.misses} new calls · {A.cache.hits} cache hits → {out_path}")
    free_models(A, G)

if "run" in STAGES:
    for m in MODELS_TO_RUN:
        t0 = time.time(); run_model(m); print(f"   {m} took {(time.time() - t0) / 60:.1f} min")
else:
    print("run stage not selected")
