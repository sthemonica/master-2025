# audit_viewer.py
import os, time, csv
from pathlib import Path
import numpy as np
import cv2
import imageio.v2 as iio
from PIL import Image
from skimage import measure

# ================= CONFIG =================
# Ajuste estes caminhos:
ROOT = r"C:\Users\sthem\OneDrive\Documentos\GitHub\master-2025\3-Datasets\img-lcm"
IMG_DIR      = Path(ROOT) / "img_dir" / "val"
MASK_RAW_DIR = Path(ROOT) / "img_dir_masks" / "val_masks" / "raw"
NII_DIR      = Path(ROOT) / "annot_dir_cintia" / "val_segmentadas"
OUT_DIR      = Path(ROOT) / "auditoria"

DILATE_PX_DEFAULT = 2    # 0 desliga; aumente se houver pequeno desalinhamento
WINDOW_NAME = "Validação NIfTI ⇄ Células (OpenCV) — a:aprovar  x:reprovar  n:sem_inf  s:snapshot  q:sair"
# ==========================================

OUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_CSV = OUT_DIR / "audit_masks.csv"
SNAP_DIR  = OUT_DIR / "overlays"
SNAP_DIR.mkdir(exist_ok=True)

# ---- helpers de IO ----
def load_img_rgb(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"), dtype=np.uint8)

def load_mask_ids(p: Path) -> np.ndarray:
    m = iio.imread(p)
    if m.ndim == 3: m = m[..., 0]
    return m.astype(np.int32)

def resize_bin_cv(bin_u8: np.ndarray, out_hw):
    H, W = out_hw
    return cv2.resize(bin_u8.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

# ---- leitura robusta de NIfTI + alinhamento ----
CAND_TRANSF = ["id","rot90","rot180","rot270","transpose","flipud","fliplr"]
def apply_transform(name, a):
    if name == "id":        return a
    if name == "rot90":     return np.rot90(a, 1)
    if name == "rot180":    return np.rot90(a, 2)
    if name == "rot270":    return np.rot90(a, 3)
    if name == "transpose": return a.T
    if name == "flipud":    return np.flipud(a)
    if name == "fliplr":    return np.fliplr(a)
    return a

def best_infected_map(nii_path: Path, target_shape, cell_union, force_slice=None, force_transform=None):
    """Tenta usar nibabel; se falhar, retorna mapa vazio. Faz busca de slice/transform por maior interseção."""
    try:
        import nibabel as nib
        nif = nib.load(str(nii_path))
        arr = np.asanyarray(nif.dataobj)
        if arr.dtype.fields is not None:  # dtype estruturado
            arr = arr.view(np.uint8).reshape(arr.shape + (arr.dtype.itemsize,)).max(axis=-1)
        arr = arr.astype(np.float32, copy=False)
    except Exception as e:
        print(f"[WARN] nibabel falhou em {nii_path.name}: {e}")
        H, W = target_shape
        return np.zeros((H, W), np.uint8), 0, "id", 0, "failed"

    if arr.ndim == 2:
        arr = arr[..., None]
    H, W = target_shape
    k_range = [force_slice] if force_slice is not None else range(arr.shape[-1])
    transforms = [force_transform] if force_transform else CAND_TRANSF

    best = (-1, 0, "id", None)  # score, k, name, res
    for k in k_range:
        sl = (arr[..., k] > 0).astype(np.uint8)
        if sl.shape != (H, W):
            sl = resize_bin_cv(sl, (H, W))
        for name in transforms:
            cand = apply_transform(name, sl)
            if cand.shape != (H, W):
                cand = resize_bin_cv(cand, (H, W))
            inter = int(np.count_nonzero((cand == 1) & (cell_union == 1)))
            if inter > best[0]:
                best = (inter, k, name, cand.astype(np.uint8))

    score, k, name, res = best
    status = "empty" if (res is None or res.max() == 0) else "ok"
    if res is None:
        res = np.zeros((H, W), np.uint8)
    return res, int(k), str(name), int(score), status

# ---- Audit CSV ----
def append_audit_row(stem, slice_v, tr_v, dil_v, decision):
    header = ["stem","slice","transform","dilate_px","decision","timestamp"]
    exists = AUDIT_CSV.exists()
    with open(AUDIT_CSV, "a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=header)
        if not exists: wr.writeheader()
        wr.writerow({
            "stem": stem,
            "slice": slice_v,
            "transform": tr_v,
            "dilate_px": dil_v,
            "decision": decision,
            "timestamp": int(time.time())
        })

# ---- main viewer ----
def stack3(img, union, overlay, scale=0.6):
    """Monta 3 painéis lado a lado, com títulos."""
    def put_title(im, text):
        pad = 40
        canvas = np.ones((im.shape[0]+pad, im.shape[1], 3), dtype=np.uint8)*255
        canvas[pad:pad+im.shape[0], :im.shape[1]] = im
        cv2.putText(canvas, text, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
        return canvas
    union_rgb = (np.stack([union*255]*3, axis=-1)).astype(np.uint8)
    left  = put_title(img, "Imagem (.bmp)")
    mid   = put_title(union_rgb, "Células (union)")
    right = put_title(overlay, "NIfTI alinhado (vermelho) + contornos")
    out = np.concatenate([left, mid, right], axis=1)
    if scale != 1.0:
        out = cv2.resize(out, (int(out.shape[1]*scale), int(out.shape[0]*scale)))
    return out

def draw_cell_contours(rgb, union_mask):
    cnts, _ = cv2.findContours(union_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        cv2.drawContours(rgb, [c], -1, (255,255,255), 1)
    return rgb

def main():
    stems = sorted([p.stem for p in IMG_DIR.glob("*.bmp")])
    if not stems:
        print("Nenhuma .bmp encontrada em", IMG_DIR)
        return

    idx = 0
    force_slice = None   # mude para um int após validar (ganha velocidade)
    force_tr    = None   # mude para "transpose" etc. após validar
    dilate_px   = DILATE_PX_DEFAULT

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1600, 600)

    while True:
        stem = stems[idx]
        img_path  = IMG_DIR / f"{stem}.bmp"
        mask_path = next(iter(MASK_RAW_DIR.glob(f"{stem}*raw*.png")), MASK_RAW_DIR / f"{stem}_masks_raw.png")
        nii_path  = NII_DIR / f"{stem}.nii.gz"

        img  = load_img_rgb(img_path)
        mask = load_mask_ids(mask_path)
        H, W = mask.shape

        union = (mask > 0).astype(np.uint8)
        infected, k_auto, tr_auto, score, status = best_infected_map(
            nii_path, (H, W), union,
            force_slice=force_slice, force_transform=force_tr
        )
        k_show  = force_slice if force_slice is not None else k_auto
        tr_show = force_tr if force_tr is not None else tr_auto

        if dilate_px and dilate_px > 0:
            ksz = 2*dilate_px + 1
            infected = cv2.dilate(infected, np.ones((ksz, ksz), np.uint8))

        # overlay vermelho
        overlay = img.copy().astype(np.float32)
        overlay[..., 2] = np.maximum(overlay[..., 2], infected.astype(np.float32)*255.0)  # canal R em BGR? cv2 usa BGR
        overlay = overlay.astype(np.uint8)
        overlay = draw_cell_contours(overlay, union)

        panel = stack3(img, union*255, overlay, scale=0.75)
        # legenda inferior
        legend = f"{stem}  |  slice={k_show}  transf={tr_show}  inter={score}  dilate={dilate_px}  status={status}  |  ←/→ navega  ↑/↓ slice  t transf  d dilata  a ok  x ruim  n sem_inf  s snapshot  q sair"
        cv2.putText(panel, legend, (10, panel.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20,20,20), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, panel)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):  # q ou ESC
            break
        elif key == 81:  # left
            idx = max(0, idx-1)
        elif key == 83:  # right
            idx = min(len(stems)-1, idx+1)
        elif key == 82:  # up (slice++)
            force_slice = 0 if force_slice is None else force_slice + 1
        elif key == 84:  # down (slice--)
            force_slice = 0 if force_slice is None else max(0, force_slice - 1)
        elif key == ord('t'):
            if force_tr is None:
                force_tr = CAND_TRANSF[0]
            else:
                i = CAND_TRANSF.index(force_tr)
                force_tr = CAND_TRANSF[(i+1) % len(CAND_TRANSF)]
        elif key == ord('d'):
            dilate_px = 0 if dilate_px else DILATE_PX_DEFAULT
        elif key in (ord('a'), ord('x'), ord('n'), ord('s')):
            if key == ord('s'):
                snap_path = SNAP_DIR / f"{stem}_slice{k_show}_tr{tr_show}_d{dilate_px}.png"
                cv2.imwrite(str(snap_path), panel)
                print("PNG salvo em:", snap_path)
            else:
                decision = {ord('a'):'approve', ord('x'):'reject', ord('n'):'no_infection'}[key]
                append_audit_row(stem, k_show, tr_show, dilate_px, decision)
                print(f"[{decision}] {stem}  (slice={k_show}  transf={tr_show}  dilate={dilate_px})")
                idx = min(len(stems)-1, idx+1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
