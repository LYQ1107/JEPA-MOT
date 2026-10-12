"""Official V-JEPA 2.1 frozen encoder loading and token extraction.

This module deliberately uses the encoder from the official Meta repository
and never constructs or calls the JEPA predictor for downstream tracking.
The checkpoint is loaded locally so that the repository's hub helper does not
use a machine-local download mirror.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parent
VJEPA_REPO = REPO_ROOT / "vendor" / "vjepa2"
if str(VJEPA_REPO) not in sys.path:
    sys.path.insert(0, str(VJEPA_REPO))


MODEL_SPECS = {
    "vjepa2_1_vit_base_384": {
        "arch": "vit_base",
        "checkpoint_key": "ema_encoder",
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "n_output_distillation": 1,
        "checkpoint_file": "vjepa2_1_vitb_dist_vitG_384.pt",
    },
    "vjepa2_1_vit_large_384": {
        "arch": "vit_large",
        "checkpoint_key": "ema_encoder",
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "n_output_distillation": 1,
        "checkpoint_file": "vjepa2_1_vitl_dist_vitG_384.pt",
    },
    "vjepa2_1_vit_giant_384": {
        "arch": "vit_giant_xformers",
        "checkpoint_key": "target_encoder",
        "embed_dim": 1408,
        "depth": 40,
        "num_heads": 22,
        "n_output_distillation": 4,
        "checkpoint_file": "vjepa2_1_vitg_384.pt",
    },
    "vjepa2_1_vit_gigantic_384": {
        "arch": "vit_gigantic_xformers",
        "checkpoint_key": "target_encoder",
        "embed_dim": 1664,
        "depth": 48,
        "num_heads": 26,
        "n_output_distillation": 4,
        "checkpoint_file": "vjepa2_1_vitG_384.pt",
    },
}


def _clean_backbone_key(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Match the key cleanup used by the official hub loader."""

    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "").replace("backbone.", "")
        cleaned[key] = value
    return cleaned


def _make_encoder(
    model_name: str,
    *,
    img_size: int = 384,
    num_frames: int = 16,
    patch_size: int = 16,
    tubelet_size: int = 2,
    n_output_distillation: int | None = None,
    out_layers: list[int] | None = None,
):
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unsupported official V-JEPA 2.1 model: {model_name}")

    from app.vjepa_2_1.models import vision_transformer as vit_encoder

    spec = MODEL_SPECS[model_name]
    kwargs = dict(
        patch_size=patch_size,
        img_size=(img_size, img_size),
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
        n_output_distillation=(
            spec["n_output_distillation"]
            if n_output_distillation is None
            else int(n_output_distillation)
        ),
    )
    if out_layers is not None:
        kwargs["out_layers"] = list(out_layers)

    encoder = vit_encoder.__dict__[spec["arch"]](**kwargs)
    return encoder, spec


def load_official_vjepa21_encoder(
    checkpoint: str | Path,
    *,
    model_name: str = "vjepa2_1_vit_large_384",
    img_size: int = 384,
    num_frames: int = 16,
    patch_size: int = 16,
    tubelet_size: int = 2,
    n_output_distillation: int | None = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype | None = None,
):
    """Load a local official V-JEPA 2.1 encoder checkpoint.

    The public ViT-B/L checkpoints use ``ema_encoder``.  The full ViT-g/G
    checkpoints use ``target_encoder`` in the official hub implementation.
    Only the encoder is materialized; the predictor is intentionally omitted.
    """

    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing V-JEPA 2.1 checkpoint: {checkpoint}")

    encoder, spec = _make_encoder(
        model_name,
        img_size=img_size,
        num_frames=num_frames,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        n_output_distillation=n_output_distillation,
    )

    payload = torch.load(str(checkpoint), map_location="cpu")
    checkpoint_key = spec["checkpoint_key"]
    if checkpoint_key not in payload:
        available = ", ".join(sorted(str(k) for k in payload.keys()))
        raise KeyError(
            f"Checkpoint {checkpoint} has no {checkpoint_key!r}; available keys: {available}"
        )
    state_dict = _clean_backbone_key(payload[checkpoint_key])
    load_msg = encoder.load_state_dict(state_dict, strict=True)
    if load_msg.missing_keys or load_msg.unexpected_keys:
        raise RuntimeError(f"Unexpected non-strict load result: {load_msg}")
    del state_dict
    del payload
    gc.collect()

    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder = encoder.to(device)
    if dtype is not None:
        encoder = encoder.to(dtype=dtype)
    encoder.eval()
    return encoder


def load_official_vjepa21_predictor(
    checkpoint: str | Path,
    *,
    embed_dim: int = 1024,
    img_size: int = 384,
    num_frames: int = 16,
    patch_size: int = 16,
    tubelet_size: int = 2,
    device: str | torch.device = "cuda",
    dtype: torch.dtype | None = None,
):
    """Load the official V-JEPA 2.1 predictor paired with ViT-L.

    The public ViT-L distillation checkpoint contains a 12-layer predictor
    whose output is projected to the ViT-G teacher width (1664).  This helper
    loads that predictor from the same local checkpoint as the encoder; it
    does not train or alter either JEPA component.
    """

    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing V-JEPA 2.1 checkpoint: {checkpoint}")

    from app.vjepa_2_1.models import predictor as vit_predictor

    predictor = vit_predictor.vit_predictor(
        img_size=(img_size, img_size),
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        embed_dim=int(embed_dim),
        predictor_embed_dim=384,
        teacher_embed_dim=1664,
        depth=12,
        num_heads=12,
        num_mask_tokens=8,
        use_mask_tokens=True,
        zero_init_mask_tokens=True,
        n_output_distillation=1,
        # The public checkpoint also contains predictor_proj_context.  Keep
        # the official all-token interface for an exact strict load; callers
        # may still consume only the masked-target output.
        return_all_tokens=True,
        use_rope=True,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
        modality_embedding=True,
    )
    payload = torch.load(str(checkpoint), map_location="cpu")
    if "predictor" not in payload:
        available = ", ".join(sorted(str(k) for k in payload.keys()))
        raise KeyError(
            f"Checkpoint {checkpoint} has no 'predictor'; available keys: {available}"
        )
    state_dict = _clean_backbone_key(payload["predictor"])
    load_msg = predictor.load_state_dict(state_dict, strict=True)
    if load_msg.missing_keys or load_msg.unexpected_keys:
        raise RuntimeError(f"Unexpected non-strict predictor load result: {load_msg}")
    del state_dict
    del payload

    predictor.eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    predictor = predictor.to(device)
    if dtype is not None:
        predictor = predictor.to(dtype=dtype)
    predictor.eval()
    return predictor


def forward_predictor_tokens(
    encoder,
    predictor,
    video: torch.Tensor,
    masks_x: torch.Tensor,
    masks_y: torch.Tensor,
    *,
    autocast_dtype: torch.dtype | None = torch.bfloat16,
):
    """Predict masked target tokens with the official frozen JEPA pair."""

    if video.ndim != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got {tuple(video.shape)}")
    device_type = video.device.type
    use_autocast = device_type == "cuda" and autocast_dtype is not None
    with torch.inference_mode():
        if use_autocast:
            with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                context = encoder(video, masks=[masks_x])
                predicted, _ = predictor(
                    context, [masks_x], [masks_y], mod="video"
                )
        else:
            context = encoder(video, masks=[masks_x])
            predicted, _ = predictor(context, [masks_x], [masks_y], mod="video")
    return predicted


def encoder_metadata(
    encoder,
    *,
    model_name: str,
    checkpoint: str | Path,
    num_frames: int,
    img_size: int,
    patch_size: int,
    tubelet_size: int,
) -> dict[str, Any]:
    """Return serializable architecture metadata for cache provenance."""

    return {
        "model_name": model_name,
        "checkpoint": str(Path(checkpoint).resolve()),
        "embed_dim": int(encoder.embed_dim),
        "depth": int(encoder.get_num_layers()),
        "hierarchical_layers": [int(x) for x in encoder.hierarchical_layers],
        "img_size": int(img_size),
        "num_frames": int(num_frames),
        "patch_size": int(patch_size),
        "tubelet_size": int(tubelet_size),
        "n_output_distillation": int(
            len(encoder.out_layers_distillation)
            if hasattr(encoder, "out_layers_distillation")
            else 1
        ),
        "encoder_only": True,
        "predictor_used": False,
        "cls_token": False,
    }


def forward_tokens(
    encoder,
    video: torch.Tensor,
    *,
    autocast_dtype: torch.dtype | None = torch.bfloat16,
    hierarchical: bool = False,
):
    """Run frozen encoder inference on [B,C,T,H,W] and return dense tokens.

    ``hierarchical=True`` exposes the official encoder's four normalized
    distillation-layer outputs concatenated along the channel dimension.  It
    is still encoder-only inference: no predictor is constructed or called.
    """

    if video.ndim != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got {tuple(video.shape)}")
    device_type = video.device.type
    use_autocast = device_type == "cuda" and autocast_dtype is not None
    previous_hierarchical = bool(getattr(encoder, "return_hierarchical", False))
    if hierarchical:
        encoder.return_hierarchical = True
    with torch.inference_mode():
        try:
            if use_autocast:
                with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                    output = encoder(video)
            else:
                output = encoder(video)
        finally:
            encoder.return_hierarchical = previous_hierarchical
    return output
