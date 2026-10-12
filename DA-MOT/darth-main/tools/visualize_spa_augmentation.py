"""Visualize SPA low-frequency amplitude masking on a single image.

Applies the exact same SPAFrequencyTransform used in the paper
(darth_spa_pipeline_mild_4domain_5e.py):
    mode='low_freq_mask', low_ratio=0.10, keep_prob=0.90

Produces:
    <output_dir>/original.jpg
    <output_dir>/spa_aug_1.jpg
    <output_dir>/spa_aug_2.jpg
    <output_dir>/spa_compare.jpg  (side-by-side: original | aug1 | aug2)

Usage:
    python tools/visualize_spa_augmentation.py \
        --img /path/to/image.jpg \
        [--out-dir work_dirs/spa_vis] \
        [--low-ratio 0.10] \
        [--keep-prob 0.90] \
        [--seed 42]
"""
import argparse
import os
import sys

import cv2
import numpy as np


# ── core SPA logic (copied from darth/datasets/pipelines/transforms.py) ──────

def _low_band_mask(H, W, low_ratio):
    band = np.zeros((H, W), dtype=np.float32)
    h_half = int(low_ratio * H) // 2
    w_half = int(low_ratio * W) // 2
    if h_half < 1 or w_half < 1:
        return band
    cy, cx = H // 2, W // 2
    band[max(cy - h_half, 0):min(cy + h_half, H),
         max(cx - w_half, 0):min(cx + w_half, W)] = 1.0
    return band


def _symmetrize(mask):
    """Conjugate-symmetric mask so IFFT stays real (mirrors SPA paper)."""
    return np.minimum(mask, mask[::-1, ::-1].copy())


def spa_high_freq_noise(img_bgr, noise_ratio=0.15, seed=None):
    """Apply SPA high-frequency noise injection (LearnablePatchErasing simplified).

    Convex combination: (1 - gamma) * img + gamma * noise
    noise is uniform-random in [0, 255], geometry-preserving.

    Args:
        img_bgr (np.ndarray): HxWx3 uint8 BGR.
        noise_ratio (float): gamma, strength of noise. Paper default 0.03~0.15.
        seed (int | None): optional RNG seed.

    Returns:
        np.ndarray: HxWx3 uint8 BGR.
    """
    rng = np.random.default_rng(seed)
    img = img_bgr.astype(np.float32)
    noise = rng.uniform(0.0, 255.0, size=img.shape).astype(np.float32)
    out = (1.0 - noise_ratio) * img + noise_ratio * noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def spa_low_freq_mask(img_bgr, low_ratio=0.10, keep_prob=0.90, seed=None):
    """Apply one SPA low-frequency amplitude mask to a BGR uint8 image.

    Args:
        img_bgr (np.ndarray): HxWx3 uint8 BGR (as loaded by cv2.imread).
        low_ratio (float): fraction of H/W treated as low-frequency band.
        keep_prob (float): Bernoulli keep probability inside the low band.
        seed (int | None): optional RNG seed for reproducibility.

    Returns:
        np.ndarray: HxWx3 uint8 BGR augmented image.
    """
    rng = np.random.default_rng(seed)
    img = img_bgr.astype(np.float32)
    H, W, C = img.shape
    x = img.transpose(2, 0, 1)          # C, H, W

    fft = np.fft.fft2(x, axes=(-2, -1))
    amp = np.abs(fft)
    pha = np.angle(fft)
    amp = np.fft.fftshift(amp, axes=(-2, -1))

    low = _low_band_mask(H, W, low_ratio)
    high = 1.0 - low
    rand_keep = (rng.random((H, W)) < keep_prob).astype(np.float32)
    mask2d = low * rand_keep + high
    mask2d = _symmetrize(mask2d)
    amp = amp * mask2d[None, :, :]

    amp = np.fft.ifftshift(amp, axes=(-2, -1))
    real = np.cos(pha) * amp
    imag = np.sin(pha) * amp
    out = np.fft.ifft2(real + 1j * imag, axes=(-2, -1)).real
    out = out.transpose(1, 2, 0)        # H, W, C

    if not np.isfinite(out).all():
        out = img                        # fallback
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


# ── freq-band highlight (for optional debug) ──────────────────────────────────

def amp_spectrum_vis(img_bgr, low_ratio=0.10):
    """Return a log-amplitude spectrum image with the low-freq band outlined."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fft = np.fft.fft2(gray)
    amp = np.abs(np.fft.fftshift(fft))
    amp_log = np.log1p(amp)
    amp_norm = cv2.normalize(amp_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis = cv2.cvtColor(amp_norm, cv2.COLOR_GRAY2BGR)

    H, W = gray.shape
    h_half = int(low_ratio * H) // 2
    w_half = int(low_ratio * W) // 2
    cy, cx = H // 2, W // 2
    pt1 = (max(cx - w_half, 0), max(cy - h_half, 0))
    pt2 = (min(cx + w_half, W - 1), min(cy + h_half, H - 1))
    cv2.rectangle(vis, pt1, pt2, (0, 255, 0), 2)
    return vis


# ── comparison grid ───────────────────────────────────────────────────────────

def make_compare(orig, aug1, aug2, labels=('Original', 'SPA aug #1', 'SPA aug #2'),
                 font_scale=1.0, thickness=2):
    H, W = orig.shape[:2]
    pad = 8
    label_h = 40
    total_h = H + label_h + pad * 2
    total_w = W * 3 + pad * 4
    canvas = np.full((total_h, total_w, 3), 40, dtype=np.uint8)

    for i, (img, label) in enumerate(zip([orig, aug1, aug2], labels)):
        x0 = pad + i * (W + pad)
        y0 = label_h
        canvas[y0:y0 + H, x0:x0 + W] = img
        cv2.putText(canvas, label, (x0, label_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (220, 220, 220), thickness, cv2.LINE_AA)

    return canvas


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Visualize SPA low-freq augmentation (pipeline_mild params)')
    p.add_argument('--img', default=(
        '/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_overcast/'
        '005cdef0-180a776c/005cdef0-180a776c-0000001.jpg'),
        help='Input image path')
    p.add_argument('--out-dir', default='work_dirs/spa_vis',
        help='Output directory')
    p.add_argument('--low-ratio', type=float, default=0.20,
        help='Low-freq band ratio (paper: 0.10, stronger: 0.20+)')
    p.add_argument('--keep-prob', type=float, default=0.50,
        help='Bernoulli keep prob inside low-freq band (paper: 0.90, stronger: 0.50-)')
    p.add_argument('--seed', type=int, default=None,
        help='RNG seed (optional; two augmentations always differ)')
    return p.parse_args()


def main():
    args = parse_args()

    img_path = args.img
    if not os.path.isfile(img_path):
        sys.exit(f'[ERROR] Image not found: {img_path}')

    os.makedirs(args.out_dir, exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
        sys.exit(f'[ERROR] cv2.imread failed: {img_path}')

    H, W = img.shape[:2]
    print(f'[INFO] Input  : {img_path}  ({W}x{H})')
    print(f'[INFO] Params : low_ratio={args.low_ratio}  keep_prob={args.keep_prob}')

    seed1 = args.seed if args.seed is not None else 0

    # aug1: low-frequency amplitude masking (global appearance / illumination)
    aug_low = spa_low_freq_mask(img, args.low_ratio, args.keep_prob, seed=seed1)
    # aug2: high-frequency noise injection (texture-level perturbation)
    aug_high = spa_high_freq_noise(img, noise_ratio=0.15, seed=seed1)

    out_orig    = os.path.join(args.out_dir, 'original.jpg')
    out_low     = os.path.join(args.out_dir, 'spa_aug_low_freq.jpg')
    out_high    = os.path.join(args.out_dir, 'spa_aug_high_freq.jpg')
    out_compare = os.path.join(args.out_dir, 'spa_compare.jpg')

    cv2.imwrite(out_orig,    img,      [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(out_low,     aug_low,  [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(out_high,    aug_high, [cv2.IMWRITE_JPEG_QUALITY, 95])

    compare = make_compare(
        img, aug_low, aug_high,
        labels=('Original', 'SPA: low-freq mask', 'SPA: high-freq noise'))
    cv2.imwrite(out_compare, compare, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f'[INFO] Saved  : {out_orig}')
    print(f'[INFO] Saved  : {out_low}   (low-freq amplitude masking)')
    print(f'[INFO] Saved  : {out_high}  (high-freq noise injection)')
    print(f'[INFO] Saved  : {out_compare}  (side-by-side comparison)')


if __name__ == '__main__':
    main()
