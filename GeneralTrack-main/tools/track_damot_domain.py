import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from archive_paths import rebase_archive_path, resolve_archive_path

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

from generaltrack_damot_utils import GeneralTrackInputError, load_prepared_metadata, read_json, repo_path, safe_video_name
from yolox.evaluators import MOTEvaluator
from yolox.tracker.jepa_metric_adapter import apply_adapter, load_adapter


class PreparedGeneralTrackDataset(torch.utils.data.Dataset):
    def __init__(self, prepared_dir, image_root, resize, jepa_cache=None, jepa_adapter=None,
                 jepa_map_cache=None, jepa_adapter_alpha=None, jepa_feature_dim=None):
        self.prepared_dir = Path(prepared_dir)
        self.annotation = read_json(self.prepared_dir / "annotations" / "val.json")
        self.image_root = rebase_archive_path(image_root)
        self.images = sorted(
            self.annotation["images"],
            key=lambda x: (int(x["video_id"]), int(x["frame_id"]), int(x["id"])),
        )
        annotation_image_ids = {int(x["id"]) for x in self.images}
        self.video_names = {
            int(v["id"]): safe_video_name(v.get("file_name", v.get("name", v["id"])))
            for v in self.annotation.get("videos", [])
        }
        self.resize = tuple(resize)
        self.jepa_cache_path = Path(jepa_cache).resolve() if jepa_cache else None
        self.jepa_adapter_path = Path(jepa_adapter).resolve() if jepa_adapter else None
        self.jepa_map_cache_path = Path(jepa_map_cache).resolve() if jepa_map_cache else None
        self.jepa_adapter_alpha = (None if jepa_adapter_alpha is None else float(jepa_adapter_alpha))
        self.jepa_feature_dim = (None if jepa_feature_dim is None else int(jepa_feature_dim))
        self.jepa_adapter = None
        if self.jepa_adapter_path is not None:
            if not self.jepa_adapter_path.exists():
                raise GeneralTrackInputError("Missing JEPA adapter: {}".format(self.jepa_adapter_path))
            self.jepa_adapter = load_adapter(self.jepa_adapter_path)
        self.jepa_features = None
        self.jepa_offsets = None
        self.jepa_image_index = {}
        if self.jepa_cache_path is not None:
            if not self.jepa_cache_path.exists():
                raise GeneralTrackInputError("Missing JEPA cache: {}".format(self.jepa_cache_path))
            with np.load(str(self.jepa_cache_path), allow_pickle=False) as cache:
                self.jepa_features = np.asarray(cache["features"])
                self.jepa_offsets = np.asarray(cache["offsets"], dtype=np.int64)
                cache_image_ids = np.asarray(cache["image_ids"], dtype=np.int64)
            if self.jepa_offsets.shape != (len(cache_image_ids) + 1,):
                raise GeneralTrackInputError("Invalid JEPA cache offsets: {}".format(self.jepa_cache_path))
            self.jepa_image_index = {int(image_id): idx for idx, image_id in enumerate(cache_image_ids.tolist())}
            if set(self.jepa_image_index) != annotation_image_ids:
                raise GeneralTrackInputError(
                    "JEPA cache image-id set differs from prepared annotation: cache={} annotation={} cache_path={}".format(
                        len(self.jepa_image_index), len(annotation_image_ids), self.jepa_cache_path))
            if self.jepa_features.ndim != 2:
                raise GeneralTrackInputError(
                    "JEPA cache must contain [N,D] features: {}".format(self.jepa_cache_path))
            cache_dim = int(self.jepa_features.shape[1])
            if self.jepa_feature_dim is not None and self.jepa_feature_dim != cache_dim:
                raise GeneralTrackInputError(
                    "JEPA feature dimension mismatch: requested={} cache={} path={}".format(
                        self.jepa_feature_dim, cache_dim, self.jepa_cache_path))
            self.jepa_feature_dim = cache_dim
        self.jepa_map_features = None
        self.jepa_map_offsets = None
        self.jepa_map_image_index = {}
        if self.jepa_map_cache_path is not None:
            if not self.jepa_map_cache_path.exists():
                raise GeneralTrackInputError("Missing JEPA map cache: {}".format(self.jepa_map_cache_path))
            with np.load(str(self.jepa_map_cache_path), allow_pickle=False) as cache:
                self.jepa_map_features = np.asarray(cache["features"])
                self.jepa_map_offsets = np.asarray(cache["offsets"], dtype=np.int64)
                cache_image_ids = np.asarray(cache["image_ids"], dtype=np.int64)
            if self.jepa_map_offsets.shape != (len(cache_image_ids) + 1,):
                raise GeneralTrackInputError("Invalid JEPA map cache offsets: {}".format(self.jepa_map_cache_path))
            self.jepa_map_image_index = {int(image_id): idx for idx, image_id in enumerate(cache_image_ids.tolist())}
            if set(self.jepa_map_image_index) != annotation_image_ids:
                raise GeneralTrackInputError(
                    "JEPA map cache image-id set differs from prepared annotation: cache={} annotation={} cache_path={}".format(
                        len(self.jepa_map_image_index), len(annotation_image_ids), self.jepa_map_cache_path))
            if (self.jepa_map_features.ndim != 4 or
                    tuple(self.jepa_map_features.shape[1:]) != (256, 27, 27)):
                raise GeneralTrackInputError(
                    "JEPA map cache must contain [N,256,27,27] features: {}".format(self.jepa_map_cache_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_info = self.images[index]
        img_path = resolve_archive_path(str(self.image_root / image_info["file_name"]))
        img = cv2.imread(str(img_path))
        if img is None:
            raise GeneralTrackInputError("Could not read image: {}".format(img_path))

        img_raw = Image.open(img_path).convert("RGB")
        img_raw = torch.from_numpy(np.asarray(img_raw).astype(np.uint8)).permute(2, 0, 1).float()
        img_raw = transforms.Resize(self.resize)(img_raw)

        detections = np.asarray(image_info.get("detection", []), dtype=np.float32)
        if detections.size == 0:
            detections = np.zeros((0, 6), dtype=np.float32)
        detections = detections.reshape(-1, 6)
        if self.jepa_cache_path is not None:
            cache_index = self.jepa_image_index.get(int(image_info["id"]))
            if cache_index is None:
                raise GeneralTrackInputError("Image id {} missing from JEPA cache {}".format(
                    image_info["id"], self.jepa_cache_path))
            start = int(self.jepa_offsets[cache_index])
            end = int(self.jepa_offsets[cache_index + 1])
            features = np.asarray(self.jepa_features[start:end], dtype=np.float32)
            if len(features) != len(detections):
                raise GeneralTrackInputError(
                    "JEPA/detection row mismatch image_id={} detections={} features={} cache={}".format(
                        image_info["id"], len(detections), len(features), self.jepa_cache_path))
            if len(detections):
                if self.jepa_adapter is not None:
                    alpha = (self.jepa_adapter[1] if self.jepa_adapter_alpha is None
                             else self.jepa_adapter_alpha)
                    features = apply_adapter(features, self.jepa_adapter[0], alpha)
                detections = np.concatenate([detections, features], axis=1)

        map_batch = None
        if self.jepa_map_cache_path is not None:
            cache_index = self.jepa_map_image_index.get(int(image_info["id"]))
            if cache_index is None:
                raise GeneralTrackInputError("Image id {} missing from JEPA map cache {}".format(
                    image_info["id"], self.jepa_map_cache_path))
            start = int(self.jepa_map_offsets[cache_index])
            end = int(self.jepa_map_offsets[cache_index + 1])
            map_batch = np.asarray(self.jepa_map_features[start:end], dtype=np.float32)
            if len(map_batch) != len(image_info.get("detection", [])):
                raise GeneralTrackInputError(
                    "JEPA map/detection row mismatch image_id={} detections={} maps={} cache={}".format(
                        image_info["id"], len(detections[:, :6]), len(map_batch), self.jepa_map_cache_path))

        target = np.zeros((0, 6), dtype=np.float32)
        video_name = self.video_names.get(int(image_info["video_id"]), str(image_info["video_id"]))
        runtime_file_name = "{}/{}".format(video_name, Path(image_info["file_name"]).name)
        info_imgs = (
            int(image_info["height"]),
            int(image_info["width"]),
            int(image_info["frame_id"]),
            int(image_info["video_id"]),
            runtime_file_name,
        )
        if map_batch is not None:
            return (img, target, info_imgs, np.asarray([int(image_info["id"])]), img_raw,
                    detections, torch.from_numpy(map_batch.copy()))
        return img, target, info_imgs, np.asarray([int(image_info["id"])]), img_raw, detections


def make_parser():
    parser = argparse.ArgumentParser("Run GeneralTrack on one DAMOT/DARTH domain")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--track_thresh", type=float, default=0.35)
    parser.add_argument("--det_thresh", type=float, default=0.45)
    parser.add_argument("--track_buffer", type=int, default=50)
    parser.add_argument("--match_thresh", type=float, default=0.9)
    parser.add_argument("--matchlost_thresh", type=float, default=0.9)
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6)
    parser.add_argument("--min_box_area", type=float, default=100)
    parser.add_argument("--mot20", action="store_true", default=False)
    parser.add_argument("--byte", action="store_true", default=True)
    parser.add_argument("--resize", type=int, nargs="+", default=[720, 1280])
    parser.add_argument("--restore_ckpt", default="checkpoints/generaltrack_bdd.pth")
    parser.add_argument("--mixed_precision", action="store_true", default=False)
    parser.add_argument("--wdecay", type=float, default=0.00005)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--spatial_scale", type=float, default=1 / 8)
    parser.add_argument("--roialign_size", type=int, nargs="+", default=[2, 2])
    parser.add_argument("--corr_radius", type=int, default=4)
    parser.add_argument("--corr_levels", type=int, default=4)
    parser.add_argument("--jepa-cache", default=None,
                        help="Optional frozen [N,D] per-detection JEPA cache")
    parser.add_argument("--jepa-feature-dim", type=int, default=None,
                        help="Optional explicit feature dimension; otherwise infer it from --jepa-cache")
    parser.add_argument("--jepa-map-cache", default=None,
                        help="Optional frozen [N,256,27,27] ToMP head map cache for official filter mode")
    parser.add_argument("--jepa-adapter", default=None,
                        help="Optional source-only 256-D metric adapter checkpoint")
    parser.add_argument("--jepa-adapter-alpha", type=float, default=None,
                        help="Optional deployment blend override for the metric adapter")
    parser.add_argument("--jepa-pair-adapter", default=None,
                        help="Optional source-only JEPA/geometry pair adapter checkpoint")
    parser.add_argument("--jepa-mode", choices=["none", "lost", "lost_iou_gate", "lost_fallback", "lost_top2", "lost_consensus", "lost_consensus_score", "lost_relative_swap", "high_fallback", "high_fallback_pair", "high_fallback_low", "high_fallback_swap", "high_jepa_reassign", "high_fallback_box", "high_fallback_duplicate", "high_fallback_duplicate_geom", "high_fallback_duplicate_geom_jepa", "high_fallback_duplicate_geom_jepa_birthconfirm", "high_fallback_duplicate_geom_jepa_consensus", "high_fallback_duplicate_geom_jepa_lostfallback", "high_fallback_duplicate_geom_jepa_box", "high_fallback_duplicate_geom_jepa_swap", "high_fallback_duplicate_geom_jepa_reassign", "high_fallback_duplicate_geom_jepa_unique", "high_fallback_duplicate_geom_jepa_expand", "high_fallback_duplicate_geom_jepa_iou_tiebreak", "high_fallback_birth_confirm", "high_fallback_lost_consensus", "high_fallback_lost_fallback", "high_fallback_dormant", "high_consensus", "high_selective", "high_selective_lost_consensus", "high_relative", "high_relative_consensus", "pair_relative_consensus", "high_selective_spatial", "high_selective_spatial_lost_consensus", "high_jepa_iou", "pair_high", "pair_high_ambiguous", "birth_confirm", "high_top2", "low_fallback", "ambiguous", "all", "temporal", "temporal_ambiguous", "official_filter"], default="none")
    parser.add_argument("--jepa-lambda", type=float, default=0.35)
    parser.add_argument("--jepa-max-gap", type=int, default=8)
    parser.add_argument("--jepa-ambiguity-margin", type=float, default=0.08)
    parser.add_argument("--jepa-relative-margin", type=float, default=0.03,
                        help="Within-row JEPA top-1/top-2 margin for high_relative mode")
    parser.add_argument("--jepa-history", type=int, default=4)
    parser.add_argument("--jepa-history-aggregation", choices=["mean", "last", "max", "recent_weighted", "last_mean", "momentum"], default="mean",
                        help="Causal JEPA query aggregation over accepted track history")
    parser.add_argument("--jepa-momentum-beta", type=float, default=0.25,
                        help="Fixed causal feature-space extrapolation coefficient for momentum queries")
    parser.add_argument("--jepa-low-sim", type=float, default=0.85)
    parser.add_argument("--jepa-low-iou", type=float, default=0.05)
    parser.add_argument("--jepa-lost-min-sim", type=float, default=0.90)
    parser.add_argument("--jepa-veto-max-score", type=float, default=None,
                        help="Detector-score ceiling for lost_consensus_score mode")
    parser.add_argument("--jepa-high-min-sim", type=float, default=0.90)
    parser.add_argument("--jepa-high-iou", type=float, default=0.05,
                        help="Predicted-box IoU gate for high_selective_spatial mode")
    parser.add_argument("--jepa-box-min-sim", type=float, default=0.90,
                        help="JEPA similarity floor for accepting a high-score box update")
    parser.add_argument("--jepa-box-min-iou", type=float, default=0.05,
                        help="Predicted-box IoU floor for accepting a high-score box update")
    parser.add_argument("--jepa-duplicate-min-sim", type=float, default=0.90,
                        help="JEPA similarity floor for suppressing a duplicate birth")
    parser.add_argument("--jepa-duplicate-iou", type=float, default=0.30,
                        help="IoU floor for suppressing a duplicate birth")
    parser.add_argument("--jepa-primary-weight", type=float, default=0.50,
                        help="JEPA weight in high_jepa_iou hybrid cost")
    parser.add_argument("--jepa-consensus-min-gap", type=int, default=1)
    parser.add_argument("--jepa-lost-iou", type=float, default=0.10)
    parser.add_argument("--jepa-fallback-iou", type=float, default=0.05)
    parser.add_argument("--jepa-fallback-margin", type=float, default=0.02)
    parser.add_argument("--jepa-birth-score", type=float, default=0.60,
                        help="Borderline birth score below which output waits for JEPA confirmation")
    parser.add_argument("--jepa-birth-min-sim", type=float, default=0.90,
                        help="Source-calibrated JEPA similarity required to confirm a borderline birth")
    parser.add_argument("--jepa-dormant-min-sim", type=float, default=0.90,
                        help="Source-calibrated JEPA similarity required to reuse an expired track ID")
    parser.add_argument("--jepa-dormant-margin", type=float, default=0.03,
                        help="Minimum gallery cosine margin for an expired-ID claim")
    parser.add_argument("--jepa-dormant-max-gap", type=int, default=200,
                        help="Maximum causal gap for a dormant-gallery ID claim")
    parser.add_argument("--jepa-dormant-max-size", type=int, default=512,
                        help="Maximum number of expired tracks retained in the causal gallery")
    parser.add_argument("--jepa-temporal-adapter", default=None,
                        help="Optional source-only causal temporal predictor checkpoint")
    parser.add_argument("--jepa-official-checkpoint",
                        default="/data3/testdata/vranlee/GOT_JEPA_checkpoints/GOT_JEPA_378.tar",
                        help="Official GOT-JEPA checkpoint for official_filter mode")
    parser.add_argument("--jepa-official-batch", type=int, default=8)
    parser.add_argument("--jepa-official-temperature", type=float, default=1.0)
    parser.add_argument("--jepa-official-scope", choices=["lost", "high", "both"], default="lost")
    return parser


def _limit_prepared_videos(prepared_dir, max_videos):
    if not max_videos:
        return prepared_dir
    prepared_dir = Path(prepared_dir)
    data = read_json(prepared_dir / "annotations" / "val.json")
    keep = {v["id"] for v in sorted(data["videos"], key=lambda v: int(v["id"]))[:max_videos]}
    data["videos"] = [v for v in data["videos"] if v["id"] in keep]
    data["images"] = [i for i in data["images"] if i["video_id"] in keep]
    keep_images = {i["id"] for i in data["images"]}
    data["annotations"] = [a for a in data.get("annotations", []) if a.get("image_id") in keep_images]
    tmp_dir = repo_path(Path("outputs") / "generaltrack_tmp" / prepared_dir.parent.name / prepared_dir.name)
    (tmp_dir / "annotations").mkdir(parents=True, exist_ok=True)
    from generaltrack_damot_utils import write_json

    write_json(tmp_dir / "annotations" / "val.json", data)
    metadata = load_prepared_metadata(prepared_dir)
    write_json(tmp_dir / "metadata.json", metadata)
    return tmp_dir


def main():
    args = make_parser().parse_args()
    if not torch.cuda.is_available():
        raise GeneralTrackInputError("GeneralTrack relation model requires CUDA for tracking.")
    args.restore_ckpt = str(repo_path(args.restore_ckpt))
    if not Path(args.restore_ckpt).exists():
        raise GeneralTrackInputError("Missing GeneralTrack checkpoint: {}".format(args.restore_ckpt))
    if len(args.resize) != 2:
        raise GeneralTrackInputError("--resize must contain height width")
    args.roialign_size = tuple(args.roialign_size)

    prepared_dir = _limit_prepared_videos(repo_path(args.prepared_dir), args.max_videos)
    metadata = load_prepared_metadata(prepared_dir)
    if not metadata:
        metadata = load_prepared_metadata(repo_path(args.prepared_dir))
    image_root = metadata.get("image_root")
    if not image_root:
        raise GeneralTrackInputError("metadata.json must contain image_root")

    result_folder = repo_path(args.output_dir) / "track_results"
    result_folder.mkdir(parents=True, exist_ok=True)
    if args.jepa_mode != "none" and args.jepa_mode != "official_filter" and not args.jepa_cache:
        raise GeneralTrackInputError("--jepa-mode requires --jepa-cache")
    if args.jepa_mode in ("temporal", "temporal_ambiguous") and not args.jepa_temporal_adapter:
        raise GeneralTrackInputError("temporal JEPA modes require --jepa-temporal-adapter")
    if args.jepa_mode in ("pair_high", "pair_high_ambiguous", "pair_relative_consensus", "high_fallback_pair") and not args.jepa_pair_adapter:
        raise GeneralTrackInputError("{} mode requires --jepa-pair-adapter".format(args.jepa_mode))
    if args.jepa_mode == "official_filter" and not args.jepa_map_cache:
        raise GeneralTrackInputError("official_filter mode requires --jepa-map-cache")
    dataset = PreparedGeneralTrackDataset(prepared_dir, image_root, args.resize, args.jepa_cache,
                                          args.jepa_adapter, args.jepa_map_cache,
                                          args.jepa_adapter_alpha, args.jepa_feature_dim)
    if dataset.jepa_feature_dim is not None:
        # The association adapter uses the dimension of the loaded cache.  A
        # missing value keeps the historical 256-D GOT-JEPA default for the
        # no-cache/no-op path.
        args.jepa_feature_dim = int(dataset.jepa_feature_dim)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    evaluator = MOTEvaluator(
        args=args,
        dataloader=loader,
        img_size=tuple(args.resize),
        confthre=0.01,
        nmsthre=0.7,
        num_classes=1,
    )
    evaluator.evaluate_bdd100k(
        model=None,
        distributed=False,
        half=args.fp16,
        trt_file=None,
        decoder=None,
        test_size=tuple(args.resize),
        result_folder=str(result_folder),
    )
    print("Wrote GeneralTrack results: {}".format(result_folder))


if __name__ == "__main__":
    main()
