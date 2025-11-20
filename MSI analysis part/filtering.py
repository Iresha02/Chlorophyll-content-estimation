# ============================================
# Adaptive Wiener Filtering (3x3 window)
# ============================================


from pathlib import Path
import cv2
import numpy as np


# ============ Core Wiener filter ============

def cast_like(arr: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Cast 'arr' to the dtype and valid range of 'ref'.
    Keeps integer inputs in their original bit depth; keeps float as float.
    """
    if np.issubdtype(ref.dtype, np.integer):
        info = np.iinfo(ref.dtype)
        return np.clip(arr, info.min, info.max).astype(ref.dtype)
    return arr.astype(ref.dtype)


def wiener_adaptive_conv(image: np.ndarray, w: int = 1, eps: float = 1e-12) -> np.ndarray:
    """
    Adaptive Wiener filter using local stats (3x3 if w=1).
    Implements:
      μ = local mean
      σ² = local variance
      v² = mean of σ² over image
      P_w* = μ + ((σ² - v²)/(σ²+eps)) * (P - μ)
    """
    if image.ndim != 2:
        raise ValueError("Wiener filter expects a 2D (monochrome) image.")

    P = image.astype(np.float32, copy=False)

    win = 2 * w + 1         # 3 when w=1
    N = float(win * win)    # 9 when w=1
    kernel = np.full((win, win), 1.0 / N, np.float32)

    # Local mean and second moment
    mu  = cv2.filter2D(P,    ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)
    E2  = cv2.filter2D(P*P,  ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)
    s2  = np.maximum(E2 - mu*mu, 0.0)   # local variance (clamp tiny negatives)

    v2  = float(np.mean(s2))            # global white-noise variance estimate
    gain = (s2 - v2) / (s2 + eps)
    Pw   = mu + gain * (P - mu)

    return cast_like(Pw, image)


# ============ Batch utilities ============

IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def ensure_gray2d(img: np.ndarray, src: Path) -> np.ndarray:
    """Guarantee a 2D grayscale array."""
    if img is None:
        raise ValueError(f"Failed to read image: {src}")
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 1:
        return img[..., 0]
    # Safety: if something is 3-channel, convert to 1-channel
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def batch_wiener_filter_tree_dc(
    root_in: Path,
    root_out: Path,
    w: int = 1,
    overwrite: bool = True,
    add_suffix: bool = False,
    enforce_100x100: bool = False
) -> None:
    """
    Walk root_in (Cropped_DC), apply Wiener to every image, and save to root_out,
    mirroring the directory structure.

    Parameters
    ----------
    root_in   : Path  e.g., r"...\\Cropped_DC"
    root_out  : Path  e.g., r"...\\Filtered" (or 'Cropped_DC_Wiener')
    w         : int   m = 1 -> 3×3 kernel (as per journal)
    overwrite : bool  overwrite existing files if True
    add_suffix: bool  if True, write 'name_wiener.ext'; else keep same name
    enforce_100x100 : bool  if True, assert images are exactly 100x100
    """
    root_in = Path(root_in)
    root_out = Path(root_out)
    count_in, count_done, count_skipped = 0, 0, 0

    for src in root_in.rglob("*"):
        if not is_image(src):
            continue
        count_in += 1

        rel = src.relative_to(root_in)
        dst_name = (rel.stem + "_wiener" + rel.suffix) if add_suffix else rel.name
        dst = (root_out / rel.parent / dst_name)
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not overwrite and dst.exists():
            count_skipped += 1
            continue

        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        img = ensure_gray2d(img, src)

        if enforce_100x100 and img.shape != (100, 100):
            raise ValueError(f"Image is {img.shape}, expected 100x100: {src}")

        Pw = wiener_adaptive_conv(img, w=w)

        if not cv2.imwrite(str(dst), Pw):
            raise IOError(f"Failed to write: {dst}")
        count_done += 1

    print(f"[Wiener batch] scanned: {count_in}, filtered: {count_done}, skipped: {count_skipped}")


# ============ Run ============

if __name__ == "__main__":
    # --- EDIT THESE TWO PATHS ---
    in_root  = Path(r"C:\Users\irana\OneDrive\Desktop\Chlorophyll MSI\Preprocessed_Chlorophyll MSI")
    out_root = Path(r"C:\Users\irana\OneDrive\Desktop\Chlorophyll MSI\Filtered")
    # ----------------------------

    # m=1 => 3x3 window (as per the turmeric journal)
    batch_wiener_filter_tree_dc(
        root_in=in_root,
        root_out=out_root,
        w=1,
        overwrite=True,        # set False to keep existing outputs
        add_suffix=False,      # set True to write *_wiener.tif in the mirrored tree
        enforce_100x100=False  # set True if all images must be 100x100
    )
