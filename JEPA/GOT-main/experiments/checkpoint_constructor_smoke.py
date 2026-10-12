#!/usr/bin/env python3
"""Read-only constructor/load smoke for the public GOT-JEPA 378 checkpoint.

This intentionally lives next to the experiment audit and does not modify the
tracker or copy the 1.7 GB checkpoint into the repository.  The checkpoint was
saved with a user-local ``ltr.admin.local`` module and the original constructor
tries to download external DINO/CoTracker weights.  We therefore provide
minimal import shims and instantiate those two architectures without external
weights, then load the corresponding weights from the GOT-JEPA archive itself.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import traceback
import types
import zipfile
from pathlib import Path


def _sha256(path: str, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _install_pytracking_shims(repo: Path) -> None:
    """Avoid importing pytracking's optional GUI/evaluation dependencies."""
    # ``tompnet_PT.py`` imports pytracking.utils.plotting.  Importing the
    # top-level package would additionally require visdom/jpeg4py and several
    # optional packages, none of which are needed for a constructor smoke.
    p_root = repo / "pytracking"
    pkg = types.ModuleType("pytracking")
    pkg.__path__ = [str(p_root)]
    # ltr.data.loader only needs these names while importing processing_utils.
    pkg.TensorDict = type("TensorDict", (dict,), {})
    pkg.TensorList = list
    sys.modules["pytracking"] = pkg
    utils = types.ModuleType("pytracking.utils")
    utils.__path__ = [str(p_root / "utils")]
    sys.modules["pytracking.utils"] = utils
    plotting_path = p_root / "utils" / "plotting.py"
    spec = importlib.util.spec_from_file_location(
        "pytracking.utils.plotting", str(plotting_path)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {plotting_path}")
    plotting = importlib.util.module_from_spec(spec)
    sys.modules["pytracking.utils.plotting"] = plotting
    spec.loader.exec_module(plotting)
    # image_loader imports jpeg4py even though this smoke never reads images.
    sys.modules.setdefault("jpeg4py", types.ModuleType("jpeg4py"))


def _install_local_module_shim() -> None:
    # The public archive pickles Settings.env as ltr.admin.local.EnvironmentSettings.
    # A no-op class is sufficient to unpickle the already-materialized settings.
    m = types.ModuleType("ltr.admin.local")
    m.EnvironmentSettings = type("EnvironmentSettings", (), {})
    sys.modules["ltr.admin.local"] = m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo", required=True, help="GOT-main/pytracking")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ckpt_path = Path(args.checkpoint).resolve()
    repo = Path(args.repo).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "started",
        "source": {
            "provider": "Google Drive",
            "drive_file_id": "1P8e_2nKIaVKcxUkj59hgGVZMD54gKpdn",
            "drive_url": "https://drive.google.com/file/d/1P8e_2nKIaVKcxUkj59hgGVZMD54gKpdn/view",
        },
        "checkpoint": str(ckpt_path),
        "checkpoint_size_bytes": ckpt_path.stat().st_size,
        "checkpoint_sha256": _sha256(str(ckpt_path)),
        "python": sys.version,
        "platform": platform.platform(),
    }
    # The archive is a modern torch zip serialization container.  Record the
    # root/member metadata without extracting the 1.7 GB payload.
    try:
        with zipfile.ZipFile(str(ckpt_path)) as zf:
            names = zf.namelist()
            result["archive_format"] = "zip"
            result["archive_member_count"] = len(names)
            result["archive_roots"] = sorted({n.split("/", 1)[0] for n in names})
    except Exception as exc:
        result["archive_inspection_error"] = f"{type(exc).__name__}: {exc}"
    t0 = time.time()
    try:
        import torch

        result["torch"] = torch.__version__
        result["torch_cuda"] = torch.version.cuda
        _install_local_module_shim()
        sys.path.insert(0, str(repo))
        # Load once to inspect metadata and retain the state dict for the load
        # check.  map_location=cpu avoids consuming a GPU or altering runtime.
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        result["top_level_keys"] = list(checkpoint.keys())
        result["epoch"] = checkpoint.get("epoch")
        result["actor_type"] = checkpoint.get("actor_type")
        result["net_type"] = checkpoint.get("net_type")
        result["net_tensor_count"] = len(checkpoint["net"])
        result["constructor"] = getattr(checkpoint["constructor"], "__dict__", {})

        _install_pytracking_shims(repo)
        # Import after the shims.  The source module's default backbone calls
        # torch.hub with pretrained=True and the CoTracker helper expects a
        # separate file.  Build both architectures locally and inject the
        # archive weights below.
        import ltr.models.tracking.tompnet_PT as tompnet_pt

        dino_repo = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
        if not dino_repo.exists():
            raise FileNotFoundError(f"DINOv2 hub source is absent: {dino_repo}")

        def dino_architecture(model_name="dinov2_vitl14"):
            return torch.hub.load(
                str(dino_repo), model_name, source="local", pretrained=False
            )

        tompnet_pt.backbones.dinov2 = dino_architecture

        # Build CoTracker2 without its external checkpoint and load the subset
        # carried inside GOT_JEPA_378.tar.  The predictor wrapper is assembled
        # without invoking its checkpoint-reading __init__.
        from cotracker.models.core.cotracker.cotracker import CoTracker2
        from ltr.cotracker2.co_tracker.cotracker.predictor import CoTrackerPredictor

        def cotracker_architecture(checkpoint_path=None, device="cuda"):
            predictor = CoTrackerPredictor.__new__(CoTrackerPredictor)
            torch.nn.Module.__init__(predictor)
            predictor.support_grid_size = 6
            predictor.model = CoTracker2(stride=4, window_len=8, add_space_attn=True)
            predictor.interp_shape = predictor.model.model_resolution
            predictor.model.eval()
            sub = {
                k[len("cotracker_model.model.") :]: v
                for k, v in checkpoint["net"].items()
                if k.startswith("cotracker_model.model.")
            }
            missing, unexpected = predictor.model.load_state_dict(sub, strict=False)
            result["cotracker_load_missing"] = list(missing)
            result["cotracker_load_unexpected"] = list(unexpected)
            return predictor

        tompnet_pt.backbones.cotracker_predictor = cotracker_architecture

        constr = checkpoint["constructor"]
        # Use the serialized constructor after replacing only its two external
        # resource calls.  This verifies the actual ToMPnet_PT architecture.
        net = constr.get()
        missing, unexpected = net.load_state_dict(checkpoint["net"], strict=False)
        result["load_missing_count"] = len(missing)
        result["load_missing"] = list(missing)
        result["load_unexpected_count"] = len(unexpected)
        result["load_unexpected"] = list(unexpected)
        result["constructed_type"] = type(net).__name__
        result["constructed_parameters"] = sum(p.numel() for p in net.parameters())

        # Minimal functional check of the two serialized JEPA heads.  This does
        # not run a full image tracker; it confirms their tensor API and finite
        # output after loading.
        net.eval()
        with torch.no_grad():
            x = torch.randn(1, 256, 27, 27)
            y_cls = net.JEPA_predictor_cls(x)
            y_breg = net.JEPA_predictor_breg(x)
        result["jepa_cls_shape"] = list(y_cls.shape)
        result["jepa_breg_shape"] = list(y_breg.shape)
        result["jepa_outputs_finite"] = bool(torch.isfinite(y_cls).all() and torch.isfinite(y_breg).all())
        result["status"] = "complete"
    except Exception as exc:  # Preserve an actionable failure artifact.
        result["status"] = "failure"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    result["elapsed_seconds"] = time.time() - t0
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
