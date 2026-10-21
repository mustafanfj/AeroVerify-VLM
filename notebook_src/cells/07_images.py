# ── Images, native tiles, resolution gate, UAV perturbations ──────────────────
try:
    import tifffile
except ImportError:
    tifffile = None
import io

class _LRU(OrderedDict):
    def __init__(self, n): super().__init__(); self.n = n
    def get_or(self, k, fn):
        if k in self: self.move_to_end(k); return self[k]
        v = fn(); self[k] = v
        if len(self) > self.n: self.popitem(last=False)
        return v
_RAW, _PERT = _LRU(6), _LRU(4)

def _read(ds, img):
    if ds == "rsvqa_lr":
        p = data_dir(ds) / "Images_LR" / f"{img}.tif"
        if not p.exists(): p = next(data_dir(ds).rglob(f"{img}.tif"))
        arr = tifffile.imread(p) if tifffile is not None else np.array(Image.open(p))
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4): arr = np.moveaxis(arr, 0, -1)
        arr = arr[..., :3]
        if arr.dtype != np.uint8:
            lo, hi = np.percentile(arr, (2, 98)); arr = np.clip((arr - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    return Image.open(data_dir(ds) / "train_image" / "img" / img).convert("RGB")

def load_image(ds, img): return _RAW.get_or((ds, img), lambda: _read(ds, img))

def _motion_blur(im, L):
    a = np.asarray(im, np.float32); L = int(max(1, L)); pad = L // 2
    c = np.cumsum(np.pad(a, ((0, 0), (pad + 1, L - pad - 1), (0, 0)), mode="edge"), axis=1)
    out = (c[:, L:, :] - c[:, :-L, :]) / L
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

def perturb(im, kind, level):
    if kind in (None, "clean"): return im
    w, h = im.size
    if kind == "altitude":
        return im.resize((max(8, int(w * level)), max(8, int(h * level))), Image.BICUBIC).resize((w, h), Image.BICUBIC)
    if kind == "motion_blur": return _motion_blur(im, level * w / 4000)     # length defined at 4000-px width
    if kind == "haze":
        a = np.asarray(im, np.float32); A = np.array([214, 216, 220], np.float32)
        return Image.fromarray(np.clip(a * (1 - level) + A * level, 0, 255).astype(np.uint8))
    if kind == "jpeg":
        b = io.BytesIO(); im.save(b, "JPEG", quality=int(level)); b.seek(0); return Image.open(b).convert("RGB")
    raise ValueError(kind)

def get_image(ds, img, pert="clean", level=0.0):
    return _PERT.get_or((ds, img, pert, level), lambda: perturb(load_image(ds, img), pert, level))

@dataclass
class View:
    pil: Image.Image
    key: str
    ds: str = ""
    img: str = ""
    box: tuple = (0, 0, 1, 1)       # region in native pixel coordinates
    native: tuple = (1, 1)          # native size of the region (px)
    pert: str = "clean"
    level: float = 0.0

def full_view(ds, img, pert="clean", level=0.0):
    im = get_image(ds, img, pert, level)
    return View(im, f"{ds}|{img}|{pert}{level}|full", ds, img, (0, 0) + im.size, im.size, pert, level)

def tile_boxes(size, grid=GRID):
    w, h = size
    return {f"r{r}c{c}": (c * w // grid, r * h // grid, (c + 1) * w // grid, (r + 1) * h // grid)
            for r in range(grid) for c in range(grid)}

def zoom_gain(size, grid=GRID):
    """Native pixels per model pixel inside one tile. >1 → tiles reveal detail the full view cannot."""
    return (max(size) / grid) / MODEL_SIDE

def tile_views(ds, img, pert="clean", level=0.0, upscale_to=None):
    im = get_image(ds, img, pert, level); out = {}
    for n, b in tile_boxes(im.size).items():
        c = im.crop(b)
        if upscale_to: c = c.resize((upscale_to, upscale_to), Image.LANCZOS)
        out[n] = View(c, f"{ds}|{img}|{pert}{level}|{n}" + (f"@{upscale_to}" if upscale_to else ""),
                      ds, img, b, (b[2] - b[0], b[3] - b[1]), pert, level)
    return out

def box_view(fv, box, k, margin=0.35, min_side=96):
    """Crop one detected instance (native coords) with context margin."""
    W, H = fv.pil.size; x0, y0, x1, y1 = box
    mw, mh = max((x1 - x0) * margin, (min_side - (x1 - x0)) / 2, 0), max((y1 - y0) * margin, (min_side - (y1 - y0)) / 2, 0)
    b = (int(max(0, x0 - mw)), int(max(0, y0 - mh)), int(min(W, x1 + mw)), int(min(H, y1 + mh)))
    return View(fv.pil.crop(b), f"{fv.key}|box{k}:{b}", fv.ds, fv.img, b, (b[2] - b[0], b[3] - b[1]), fv.pert, fv.level)

BLANK_VIEW = View(Image.new("RGB", (512, 512), (128, 128, 128)), "blank", "", "", (0, 0, 512, 512), (512, 512))
