import csv
import fcntl
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from archive_paths import rebase_archive_path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "damot_generaltrack_datasets.yaml"
TRACK_FIELDS = [
    "method",
    "protocol",
    "dataset",
    "source_domain",
    "target_domain",
    "num_videos",
    "num_frames",
    "HOTA",
    "DetA",
    "AssA",
    "MOTA",
    "IDF1",
    "IDSW",
    "FP",
    "FN",
    "det_source",
]


class GeneralTrackInputError(RuntimeError):
    pass


def repo_path(path):
    path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return rebase_archive_path(path)


def load_dataset_config(config_path=DEFAULT_CONFIG):
    config_path = repo_path(config_path)
    text = config_path.read_text()
    try:
        import yaml

        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


def get_dataset_cfg(config, dataset):
    datasets = config.get("datasets", {})
    if dataset not in datasets:
        raise GeneralTrackInputError(
            "Unknown dataset '{}'. Available: {}".format(dataset, ", ".join(sorted(datasets)))
        )
    return datasets[dataset]


def list_domains(dataset_cfg, domains):
    if not domains or domains == "all":
        return list(dataset_cfg.get("targets", []))
    if isinstance(domains, str):
        domains = [d for d in domains.replace(",", " ").split() if d]
    unknown = [d for d in domains if d not in dataset_cfg.get("targets", [])]
    if unknown:
        raise GeneralTrackInputError("Unknown domain(s): {}".format(", ".join(unknown)))
    return domains


def _format_pattern(pattern, dataset, domain):
    return str(pattern).format(dataset=dataset, domain=domain)


def resolve_image_root(dataset_cfg, dataset, domain):
    return repo_path(_format_pattern(dataset_cfg.get("image_root", ""), dataset, domain))


def _existing_glob(pattern):
    pattern = str(repo_path(pattern))
    matches = sorted(glob.glob(pattern))
    return Path(matches[0]) if matches else None


def resolve_annotation(dataset_cfg, dataset, domain):
    patterns = dataset_cfg.get("target_ann_patterns", {}).get(domain)
    if patterns and not isinstance(patterns, list):
        patterns = [patterns]
    for pattern in patterns or []:
        candidate = _existing_glob(_format_pattern(pattern, dataset, domain))
        if candidate:
            return candidate

    ann_root = repo_path(dataset_cfg.get("annotation_root", ""))
    if ann_root.exists():
        candidates = sorted(ann_root.rglob("*.json"))
        fuzzy = [
            p
            for p in candidates
            if domain.lower() in p.name.lower() and "cocofmt" in p.name.lower()
        ]
        if fuzzy:
            return fuzzy[0]
    return None


def resolve_detection(dataset_cfg, dataset, domain, det_source="source_detector"):
    roots = dataset_cfg.get("detection_roots", {})
    root = roots.get(det_source, dataset_cfg.get("detection_root", ""))
    if not root:
        return None
    root = repo_path(root)
    patterns = dataset_cfg.get("detection_patterns", [])
    for pattern in patterns:
        candidate = _existing_glob(root / _format_pattern(pattern, dataset, domain))
        if candidate:
            return candidate
    if root.exists():
        fuzzy = sorted(
            p
            for p in root.rglob("*")
            if p.is_file()
            and domain.lower() in p.name.lower()
            and p.suffix.lower() in {".json", ".jsonl", ".txt", ".csv"}
        )
        if fuzzy:
            return fuzzy[0]
    return None


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def append_or_replace_summary(summary_path, row):
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = summary_path.with_suffix(summary_path.suffix + ".lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        _append_or_replace_summary_unlocked(summary_path, row)
        fcntl.flock(lock_file, fcntl.LOCK_UN)


def _append_or_replace_summary_unlocked(summary_path, row):
    rows = []
    if summary_path.exists():
        with open(summary_path, newline="") as f:
            rows = list(csv.DictReader(f))
    key = (row["method"], row["dataset"], row["target_domain"], row["det_source"])
    rows = [
        r
        for r in rows
        if (r.get("method"), r.get("dataset"), r.get("target_domain"), r.get("det_source"))
        != key
    ]
    rows.append(row)
    write_csv(summary_path, rows, TRACK_FIELDS)


def normalize_track_id(ann):
    for key in ("track_id", "instance_id", "mot_instance_id"):
        if key in ann:
            return int(ann[key]), key
    return int(ann.get("id", 0)), "id"


def bbox_xywh_to_xyxy(bbox):
    x, y, w, h = [float(v) for v in bbox[:4]]
    return [x, y, x + w, y + h]


def bbox_xyxy_to_xywh(bbox):
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return [x1, y1, x2 - x1, y2 - y1]


def infer_frame_offset(images):
    frame_ids = [int(img.get("frame_id", 0)) for img in images if "frame_id" in img]
    return 1 if frame_ids and min(frame_ids) == 0 else 0


def video_name_map(data):
    videos = {}
    for video in data.get("videos", []):
        name = video.get("file_name", video.get("name", str(video.get("id"))))
        videos[int(video["id"])] = name
    return videos


def normalize_cocomot_annotations(data, image_root, max_videos=None, drop_missing_images=False):
    videos_in = sorted(data.get("videos", []), key=lambda v: int(v["id"]))
    if max_videos:
        keep_original_video_ids = {int(v["id"]) for v in videos_in[:max_videos]}
    else:
        keep_original_video_ids = {int(v["id"]) for v in videos_in}
    video_id_map = {
        old_id: new_id + 1 for new_id, old_id in enumerate(sorted(keep_original_video_ids))
    }
    frame_offset = infer_frame_offset(data.get("images", []))

    videos = []
    for video in videos_in:
        old_id = int(video["id"])
        if old_id not in video_id_map:
            continue
        new_video = dict(video)
        new_video["source_video_id"] = old_id
        new_video["id"] = video_id_map[old_id]
        if "file_name" not in new_video:
            new_video["file_name"] = new_video.get("name", str(old_id))
        videos.append(new_video)

    images = []
    kept_image_ids = set()
    missing_images = []
    for image in sorted(
        data.get("images", []),
        key=lambda x: (video_id_map.get(int(x.get("video_id", -1)), 10**9), int(x.get("frame_id", 0)), int(x["id"])),
    ):
        old_vid = int(image.get("video_id", -1))
        if old_vid not in video_id_map:
            continue
        image_out = dict(image)
        image_out["source_video_id"] = old_vid
        image_out["video_id"] = video_id_map[old_vid]
        source_frame_id = int(image_out.get("frame_id", 0))
        image_out["source_frame_id"] = source_frame_id
        image_out["frame_id"] = source_frame_id + frame_offset
        image_out.setdefault("detection", [])
        image_missing = bool(image_root and not (Path(image_root) / image_out["file_name"]).exists())
        if image_missing:
            missing_images.append(image_out["file_name"])
            if drop_missing_images:
                continue
        images.append(image_out)
        kept_image_ids.add(int(image_out["id"]))

    field_counts = defaultdict(int)
    annotations = []
    for ann in data.get("annotations", []):
        image_id = int(ann.get("image_id", -1))
        if image_id not in kept_image_ids:
            continue
        ann_out = dict(ann)
        tid, field = normalize_track_id(ann)
        field_counts[field] += 1
        ann_out["track_id"] = tid
        if "area" not in ann_out and "bbox" in ann_out:
            ann_out["area"] = float(ann_out["bbox"][2]) * float(ann_out["bbox"][3])
        annotations.append(ann_out)

    return {
        "videos": videos,
        "images": images,
        "annotations": annotations,
        "categories": data.get("categories", [{"id": 1, "name": "object"}]),
    }, {
        "frame_id_offset": frame_offset,
        "track_id_fields": dict(field_counts),
        "missing_images": missing_images,
    }


def _det_array_from_record(det):
    bbox = det.get("bbox", det.get("box", det.get("box2d")))
    if isinstance(bbox, dict):
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    else:
        x1, y1, x2, y2 = bbox_xywh_to_xyxy(bbox)
    score = float(det.get("score", det.get("confidence", 1.0)))
    category_id = int(det.get("category_id", det.get("class", det.get("cls", 1))))
    tracker_cls = max(category_id - 1, 0)
    return [float(x1), float(y1), float(x2), float(y2), score, tracker_cls]


def load_detection_image_mapping(det_path):
    det_path = Path(det_path)
    mapping_path = det_path.parent / "image_mapping.json"
    if not mapping_path.exists():
        return {}, None
    mapping = read_json(mapping_path)
    if not isinstance(mapping, list):
        return {}, None
    image_id_to_file_name = {}
    covered_file_names = set()
    for item in mapping:
        if not isinstance(item, dict) or "image_id" not in item or "file_name" not in item:
            continue
        file_name = item["file_name"]
        image_id_to_file_name[int(item["image_id"])] = file_name
        covered_file_names.add(file_name)
    return image_id_to_file_name, covered_file_names


def load_json_detections(det_path):
    data = read_json(det_path)
    by_image_id = defaultdict(list)
    by_file_name = defaultdict(list)
    image_id_to_file_name, covered_file_names = load_detection_image_mapping(det_path)

    if isinstance(data, list):
        records = data
    elif "detections" in data:
        records = data["detections"]
    elif "annotations" in data:
        records = data["annotations"]
    elif "images" in data:
        for image in data["images"]:
            if "file_name" in image:
                covered_file_names = covered_file_names or set()
                covered_file_names.add(image["file_name"])
            for det in image.get("detection", []):
                by_image_id[int(image["id"])].append([float(x) for x in det])
        return by_image_id, by_file_name, covered_file_names
    else:
        raise GeneralTrackInputError("Unsupported detection JSON structure: {}".format(det_path))

    for det in records:
        arr = _det_array_from_record(det)
        if "image_id" in det:
            image_id = int(det["image_id"])
            file_name = image_id_to_file_name.get(image_id)
            if file_name:
                by_file_name[file_name].append(arr)
            else:
                by_image_id[image_id].append(arr)
        elif "file_name" in det:
            by_file_name[det["file_name"]].append(arr)
            covered_file_names = covered_file_names or set()
            covered_file_names.add(det["file_name"])
        else:
            raise GeneralTrackInputError(
                "Detection record lacks image_id/file_name in {}".format(det_path)
            )
    return by_image_id, by_file_name, covered_file_names


def _read_numeric_txt(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.replace(",", " ")
            parts = [p for p in line.split() if p]
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
    return rows


def load_txt_detections(det_path, images):
    by_video_frame = defaultdict(list)
    image_video_names = {}
    for image in images:
        image_video_names[int(image["video_id"])] = image["file_name"].split("/")[0]

    files = []
    det_path = Path(det_path)
    if det_path.is_dir():
        files = sorted(
            p for p in det_path.rglob("*") if p.suffix.lower() in {".txt", ".csv"}
        )
    else:
        files = [det_path]

    for file_path in files:
        video_hint = file_path.stem
        if video_hint in {"det", "dets", "yolox_dets"}:
            video_hint = file_path.parent.parent.name if file_path.parent.name == "det" else file_path.parent.name
        rows = _read_numeric_txt(file_path)
        for row in rows:
            if len(row) < 6:
                continue
            frame = int(row[0])
            x, y, w, h = row[2:6]
            score = row[6] if len(row) > 6 else 1.0
            cls = int(row[7]) if len(row) > 7 and row[7] >= 0 else 1
            by_video_frame[(video_hint, frame)].append([x, y, x + w, y + h, score, cls])

    by_image_id = defaultdict(list)
    by_file_name = defaultdict(list)
    for image in images:
        video_name = image["file_name"].split("/")[0]
        frame = int(image["frame_id"])
        detections = by_video_frame.get((video_name, frame), [])
        if detections:
            by_image_id[int(image["id"])].extend(detections)
    return by_image_id, by_file_name, None


def attach_detections(prepared, det_path):
    suffix = Path(det_path).suffix.lower()
    if suffix == ".json":
        by_image_id, by_file_name, covered_file_names = load_json_detections(det_path)
    elif suffix in {".txt", ".csv"} or Path(det_path).is_dir():
        by_image_id, by_file_name, covered_file_names = load_txt_detections(det_path, prepared["images"])
    else:
        raise GeneralTrackInputError("Unsupported detection input: {}".format(det_path))

    num_dets = 0
    missing_frames = []
    for image in prepared["images"]:
        detections = by_image_id.get(int(image["id"]), []) or by_file_name.get(image["file_name"], [])
        image["detection"] = detections
        if covered_file_names is not None:
            image["detection_source_covered"] = image["file_name"] in covered_file_names
        num_dets += len(detections)
        if not detections:
            missing_frames.append(image["file_name"])
    return {
        "num_detections": num_dets,
        "missing_detections": missing_frames,
        "covered_file_names": covered_file_names,
        "num_covered_frames": len(covered_file_names) if covered_file_names is not None else None,
    }


def keep_detection_mapped_frames(prepared, covered_file_names):
    if covered_file_names is None:
        return {
            "enabled": False,
            "dropped_unmapped_frames": [],
            "num_detections": sum(len(img.get("detection", [])) for img in prepared["images"]),
            "missing_detections": [img["file_name"] for img in prepared["images"] if not img.get("detection")],
        }

    keep_images = [img for img in prepared["images"] if img["file_name"] in covered_file_names]
    keep_image_ids = {int(img["id"]) for img in keep_images}
    keep_video_ids = {int(img["video_id"]) for img in keep_images}
    dropped = [img["file_name"] for img in prepared["images"] if img["file_name"] not in covered_file_names]

    prepared["images"] = keep_images
    prepared["annotations"] = [
        ann for ann in prepared["annotations"] if int(ann.get("image_id", -1)) in keep_image_ids
    ]
    prepared["videos"] = [
        video for video in prepared["videos"] if int(video.get("id", -1)) in keep_video_ids
    ]

    return {
        "enabled": True,
        "dropped_unmapped_frames": dropped,
        "num_detections": sum(len(img.get("detection", [])) for img in prepared["images"]),
        "missing_detections": [img["file_name"] for img in prepared["images"] if not img.get("detection")],
    }


def load_prepared_metadata(prepared_dir):
    prepared_dir = Path(prepared_dir)
    metadata_path = prepared_dir / "metadata.json"
    if metadata_path.exists():
        return read_json(metadata_path)
    return {}


def category_names(data):
    return [c.get("name", str(c.get("id"))) for c in data.get("categories", [])]


def safe_video_name(name):
    return str(name).strip("/").replace("/", "__")


def read_mot_result_file(path):
    rows = []
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return rows
    for row in _read_numeric_txt(path):
        if len(row) < 6:
            continue
        rows.append(
            {
                "frame_id": int(row[0]),
                "track_id": int(row[1]),
                "bbox": [float(row[2]), float(row[3]), float(row[4]), float(row[5])],
                "score": float(row[6]) if len(row) > 6 else 1.0,
                "category_id": int(row[7]) if len(row) > 7 and row[7] >= 0 else 1,
            }
        )
    return rows


def box_iou_matrix(gt_boxes, tracker_boxes):
    gt_boxes = np.asarray(gt_boxes, dtype=float)
    tracker_boxes = np.asarray(tracker_boxes, dtype=float)
    if len(gt_boxes) == 0 or len(tracker_boxes) == 0:
        return np.zeros((len(gt_boxes), len(tracker_boxes)))
    gt = gt_boxes.copy()
    tr = tracker_boxes.copy()
    gt[:, 2] = gt[:, 0] + gt[:, 2]
    gt[:, 3] = gt[:, 1] + gt[:, 3]
    tr[:, 2] = tr[:, 0] + tr[:, 2]
    tr[:, 3] = tr[:, 1] + tr[:, 3]
    ix1 = np.maximum(gt[:, None, 0], tr[None, :, 0])
    iy1 = np.maximum(gt[:, None, 1], tr[None, :, 1])
    ix2 = np.minimum(gt[:, None, 2], tr[None, :, 2])
    iy2 = np.minimum(gt[:, None, 3], tr[None, :, 3])
    iw = np.maximum(ix2 - ix1, 0)
    ih = np.maximum(iy2 - iy1, 0)
    inter = iw * ih
    gt_area = np.maximum(gt[:, 2] - gt[:, 0], 0) * np.maximum(gt[:, 3] - gt[:, 1], 0)
    tr_area = np.maximum(tr[:, 2] - tr[:, 0], 0) * np.maximum(tr[:, 3] - tr[:, 1], 0)
    union = gt_area[:, None] + tr_area[None, :] - inter
    return inter / np.maximum(union, 1e-12)
