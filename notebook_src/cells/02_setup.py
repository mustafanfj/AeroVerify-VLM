# ── Environment ───────────────────────────────────────────────────────────────
import sys, re, time, math, random, hashlib, zipfile, gc, ast, urllib.request
from pathlib import Path
from dataclasses import dataclass
from collections import Counter, defaultdict, OrderedDict

ROOT = Path(PROJECT_ROOT) / ("dry_run_v2" if DRY_RUN else "")
OUT_DIR   = ROOT / "outputs_v2"
FIG_DIR, TAB_DIR = OUT_DIR / "figures", OUT_DIR / "tables"
CACHE_DIR, PRED_DIR = OUT_DIR / "cache", OUT_DIR / "predictions"
for d in (FIG_DIR, TAB_DIR, CACHE_DIR, PRED_DIR):
    d.mkdir(parents=True, exist_ok=True)
def data_dir(ds):
    p = ROOT / "data" / (f"synthetic_{ds}" if DRY_RUN else ds); p.mkdir(parents=True, exist_ok=True); return p

os.environ.setdefault("HF_HOME", str(Path(PROJECT_ROOT) / ".cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(Path(PROJECT_ROOT) / ".cache" / "torch"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
Image.MAX_IMAGE_PIXELS = None
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **k): return x
random.seed(SEED); np.random.seed(SEED)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
try:
    import torch
    torch.manual_seed(SEED); HAS_CUDA = torch.cuda.is_available()
except ImportError:
    torch, HAS_CUDA = None, False

# %%stage <name>: run a cell only when that stage is selected
try:
    from IPython.core.magic import register_cell_magic
    @register_cell_magic
    def stage(line, cell):
        if line.strip() in STAGES:
            get_ipython().run_cell(cell).raise_error()
        else:
            print(f"(stage '{line.strip()}' not selected — cell skipped)")
except (ImportError, NameError):
    pass

print("Python :", sys.version.split()[0], "|", sys.executable)
print("Outputs:", OUT_DIR)
if torch is not None:
    print("PyTorch:", torch.__version__, "| CUDA:", HAS_CUDA, "|", torch.cuda.get_device_name(0) if HAS_CUDA else "-")
if not DRY_RUN and "run" in STAGES and not HAS_CUDA:
    raise RuntimeError("No CUDA GPU visible — select the project .venv kernel or set AV_DRY_RUN=1")
