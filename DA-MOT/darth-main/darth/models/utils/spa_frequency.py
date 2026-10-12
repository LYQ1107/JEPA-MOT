"""SPA-style geometry-preserving frequency augmentation.

Direct port of `Prompt.masking` from SPA (ICML 2025),
`SPA-main/tta_library/spa.py`. Operates on already-normalized GPU tensors
with `torch.fft` — exactly as the original paper does.

Key property: perturbs Fourier amplitude (low-frequency band only) while
keeping the phase intact, preserving geometric structure. Bboxes, labels and
match_indices remain valid and do NOT need to be transformed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SPAFrequencyAugmentor(nn.Module):
    """Low-frequency amplitude masking, ported from SPA `Prompt.masking`.

    Differences from the original:
    - `low_mask` is built dynamically per call (DARTH images are not fixed
      size), rather than pre-allocated in `__init__`.
    - `high_frequency_noise_injection` is a separate optional path using the
      `LearnablePatchErasing` idea but with a fixed gamma instead of a
      learnable alpha (we do not need adversarial updates for MOT distillation).

    Args:
        low_ratio   (float): Fraction of H and W (centered) treated as the
                             low-frequency band. Original uses prompt_alpha=0.2.
        keep_prob   (float): Bernoulli keep prob inside the low band.
                             keep_prob = 1 - freq_mask_ratio.
                             Original ViT: freq_mask_ratio=0.2 → keep_prob=0.8.
        noise_ratio (float): gamma for the high-frequency noise convex combo.
                             Original ViT: noise_ratio=0.4.
        apply_noise (bool):  Whether to apply high-frequency noise.
    """

    def __init__(
        self,
        low_ratio=0.2,
        keep_prob=0.8,
        noise_ratio=0.0,
        apply_noise=False,
        clamp=None,       # kept for backward compat, unused (no clamping per paper)
    ):
        super().__init__()
        self.low_ratio = float(low_ratio)
        self.keep_prob = float(keep_prob)
        self.noise_ratio = float(noise_ratio)
        self.apply_noise = bool(apply_noise)

    # ------------------------------------------------------------------
    # Helpers — direct port from SPA `Prompt`
    # ------------------------------------------------------------------

    @staticmethod
    def _build_low_mask(H, W, low_ratio, device, dtype):
        """Build [1,1,H,W] mask with 1 inside centered low-frequency band.

        Equivalent to SPA's F.pad approach but dynamic for variable H/W.
        """
        promptH = max(int(H * low_ratio), 1)
        promptW = max(int(W * low_ratio), 1)
        paddingH = (H - promptH) // 2
        paddingW = (W - promptW) // 2
        # ones patch of shape [1,1,promptH,promptW], pad to [1,1,H,W]
        init = torch.ones(1, 1, promptH, promptW, device=device, dtype=dtype)
        low_mask = F.pad(
            init,
            [paddingW, W - paddingW - promptW,
             paddingH, H - paddingH - promptH],
            mode='constant', value=0.0
        ).contiguous()
        return low_mask  # [1,1,H,W]

    @staticmethod
    def _convert_symmetry_mask(mask):
        """Make mask conjugate-symmetric so IFFT output stays real.

        Direct port of SPA `Prompt.__convert_symmertry_mask`.
        mask shape: [B, C, H, W] (fftshifted amplitude mask).
        """
        startH = 1 - (mask.shape[-2] % 2)
        startW = 1 - (mask.shape[-1] % 2)
        sub_mask = mask[..., startH:, startW:]
        assert sub_mask.shape[-1] % 2 == 1 and sub_mask.shape[-2] % 2 == 1
        maskH = sub_mask.shape[-2] // 2
        maskW = sub_mask.shape[-1] // 2
        sub_mask[..., -maskH:, :] = 1
        sub_mask[..., maskH, :maskW] = 1
        sub_mask = sub_mask * torch.flip(sub_mask, dims=[-1, -2])
        mask[..., startH:, startW:] = sub_mask
        return mask

    @staticmethod
    def _ifft(amp, pha, H, W):
        """Recompose complex spectrum and invert. Port of SPA `Prompt.iFFT`."""
        real = torch.cos(pha) * amp
        imag = torch.sin(pha) * amp
        fft = torch.complex(real, imag)
        return torch.fft.ifft2(fft, dim=(-2, -1), s=[H, W]).real

    # ------------------------------------------------------------------
    # Core augmentations
    # ------------------------------------------------------------------

    def low_frequency_amplitude_mask(self, x):
        """Port of SPA `Prompt.masking` / `_random_mask_on_low_frequency_only`.

        Args:
            x (Tensor): [B, C, H, W] normalized GPU tensor.
        Returns:
            Tensor: same shape/dtype/device.
        """
        B, C, H, W = x.shape
        fft = torch.fft.fft2(x.clone(), dim=(-2, -1))
        amp, pha = torch.abs(fft), torch.angle(fft)
        amp = torch.fft.fftshift(amp)

        low_mask = self._build_low_mask(H, W, self.low_ratio, x.device, amp.dtype)
        high_mask = 1.0 - low_mask

        # random keep low frequency, keep all high frequency
        rand_keep = torch.empty_like(amp).bernoulli_(self.keep_prob)
        mask = low_mask * rand_keep + high_mask
        mask = self._convert_symmetry_mask(mask)   # ← conjugate symmetry
        amp = amp * mask

        amp = torch.fft.ifftshift(amp)
        return self._ifft(amp, pha, H, W)

    def high_frequency_noise_injection(self, x):
        """Port of SPA `LearnablePatchErasing` with fixed gamma.

        Convex combination: (1 - gamma) * x + gamma * noise  (SPA eq.)
        noise drawn from N(0,1), no patch structure (simplified).
        """
        if not self.apply_noise or self.noise_ratio <= 0:
            return x
        noise = torch.randn_like(x)
        return (1.0 - self.noise_ratio) * x + self.noise_ratio * noise

    @torch.no_grad()
    def forward(self, x):
        """Apply SPA augmentation to a normalized GPU tensor.

        Falls back to the original tensor on NaN/Inf or shape mismatch.
        """
        if x is None or x.numel() == 0:
            return x
        in_dtype = x.dtype
        xf = x.float()
        out = self.low_frequency_amplitude_mask(xf)
        if self.apply_noise and self.noise_ratio > 0:
            out = self.high_frequency_noise_injection(out)
        if not torch.isfinite(out).all() or out.shape != x.shape:
            return x
        return out.to(in_dtype)
