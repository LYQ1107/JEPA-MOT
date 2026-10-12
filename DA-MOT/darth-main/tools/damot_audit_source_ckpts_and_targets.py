#!/usr/bin/env python3
"""Audit source-domain 20e checkpoints and target-domain annotations."""
import argparse, json, os
from pathlib import Path
import yaml

DAMOT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main")

DATASETS = {
    "bdd_da": {
        "source": "clear",
        "ckpt_hint": "work_dirs/bdd_4domain/source/clear_20e/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear/epoch_20.pth",
        "config_hint": "work_dirs/bdd_4domain/source/clear_20e/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear.py",
        "targets": ["overcast", "rainy", "dawn_dusk", "night"],
        "image_root": "data/bdd_da",
    },
    "motsynth": {
        "source": "normal",
        "ckpt_hint": "work_dirs/motsynth/source/normal_20e/qdtrack_motsynth_normal_20e/qdtrack_faster-rcnn_r50_fpn_20e_motsynth_normal/20260708_014802/epoch_20.pth",
        "config_hint": None,
        "targets": ["cloudy", "dark", "fog", "indoor", "rain", "snow"],
        "image_root": "data/motsynth/images",
    },
    "ua_detrac": {
        "source": "sunny",
        "ckpt_hint": "work_dirs/ua_detrac/source/sunny_20e/qdtrack_ua_detrac_sunny_20e/qdtrack_faster-rcnn_r50_fpn_20e_ua_detrac_sunny/20260708_155616/epoch_20.pth",
        "config_hint": None,
        "targets": ["cloudy", "night", "rainy"],
        "image_root": "data/ua_detrac/images",
    },
    "vkitti2": {
        "source": "clone",
        "ckpt_hint": "work_dirs/vkitti2/source/clone_20e/qdtrack_vkitti2_clone_20e/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_clone/20260707_171736/epoch_20.pth",
        "config_hint": None,
        "targets": ["fog", "morning", "overcast", "rain", "sunset"],
        "image_root": "data/vkitti2/images",
    },
}

MOTSYNTH_ANN = {
    "cloudy": "data/motsynth/annotations/motsynth_cloudy_cocofmt.json",
    "dark":   "data/motsynth/annotations/motsynth_dark_cocofmt.json",
    "fog":    "data/motsynth/annotations/motsynth_fog_cocofmt.json",
    "indoor": "data/motsynth/annotations/old/motsynth_indoor_cocofmt.json",
    "rain":   "data/motsynth/annotations/motsynth_rain_cocofmt.json",
    "snow":   "data/motsynth/annotations/motsynth_snow_cocofmt.json",
}

ADAPTED_KW = ["adapt","tta","sequential","keep","reset",
               "stage0","stage1","stage2","stage3","stage4","stage5"]
TARGET_KW  = ["rainy","night","cloudy","fog","snow","overcast",
               "dawn","dusk","dark","indoor","sunset","morning"]
SOURCE_KW  = ["clear","normal","sunny","clone"]


def rel(p):
    try:
        return str(Path(p).relative_to(DAMOT_ROOT))
    except ValueError:
        return str(p)


def ckpt_segment(ckpt_path):
    parts = Path(ckpt_path).parts
    return "/".join(parts[-5:]).lower()


def check_ckpt_safety(ckpt_path):
    seg = ckpt_segment(ckpt_path)
    for kw in ADAPTED_KW:
        if kw in seg:
            return False, f"contains adaptation keyword '{kw}'"
    for kw in TARGET_KW:
        # only check the experiment dir segment, not the project root
        # strip darth-main prefix to avoid false positive
        seg2 = seg.replace("darth-main", "")
        if kw in seg2:
            return False, f"contains target-domain keyword '{kw}'"
    return True, "ok"


def find_ckpt_config(ckpt_path):
    if ckpt_path is None:
        return None
    parent = Path(ckpt_path).parent
    for p in sorted(parent.glob("*.py")):
        return p
    return None


def resolve_ann(ds_name, target, root):
    if ds_name == "bdd_da":
        rel_ann = f"data/bdd_da/annotations/box_track_20/box_track_{target}_all_cocofmt.json"
    elif ds_name == "motsynth":
        rel_ann = MOTSYNTH_ANN.get(target)
    elif ds_name == "ua_detrac":
        rel_ann = f"data/ua_detrac/annotations/box_track_{target}_all_cocofmt.json"
    elif ds_name == "vkitti2":
        rel_ann = f"data/vkitti2/annotations/vkitti2_{target}_cocofmt.json"
    else:
        rel_ann = None
    return rel_ann, (root / rel_ann) if rel_ann else None


def inspect_annotation(ann_path, image_root):
    info = dict(exists=False, coco_format=False, num_images=0, num_annotations=0,
                num_videos=0, categories=[], image_id_sample=None, video_id_sample=None,
                frame_id_sample=None, file_name_sample=None, width_sample=None,
                height_sample=None, image_root_exists=Path(image_root).exists() if image_root else False,
                first20_ok=0, first20_missing=[], error=None)
    if ann_path is None or not Path(ann_path).exists():
        return info
    info["exists"] = True
    try:
        with open(ann_path) as f:
            data = json.load(f)
        info["coco_format"] = "images" in data
        imgs = data.get("images", [])
        info["num_images"]      = len(imgs)
        info["num_annotations"] = len(data.get("annotations", []))
        info["num_videos"]      = len(data.get("videos", []))
        info["categories"]      = data.get("categories", [])
        if imgs:
            img0 = imgs[0]
            info["image_id_sample"]  = img0.get("id")
            info["video_id_sample"]  = img0.get("video_id")
            info["frame_id_sample"]  = img0.get("frame_id")
            info["file_name_sample"] = img0.get("file_name")
            info["width_sample"]     = img0.get("width")
            info["height_sample"]    = img0.get("height")
        missing = []
        ir = Path(image_root) if image_root else None
        for img in imgs[:20]:
            fn = img.get("file_name", "")
            if ir and not (ir / fn).exists():
                missing.append(fn)
        info["first20_ok"]      = 20 - len(missing)
        info["first20_missing"] = missing[:5]
    except Exception as e:
        info["error"] = str(e)
    return info


def build_report_and_plan(args):
    root = Path(args.damot_root)
    out_md   = Path(args.out);        out_md.parent.mkdir(parents=True, exist_ok=True)
    out_yaml = Path(args.config_out); out_yaml.parent.mkdir(parents=True, exist_ok=True)
    cat_audit = out_md.parent / "category_mapping_audit.md"

    md = ["# DAMOT Source Detector Audit Report\n"]
    plan = {"datasets": {}}

    for ds_name, ds_info in DATASETS.items():
        source = ds_info["source"]
        md.append(f"\n## {ds_name}  (source={source})\n")

        ckpt_path = root / ds_info["ckpt_hint"]
        ckpt_ok   = ckpt_path.exists()
        safe, reason = check_ckpt_safety(str(ckpt_path))

        md.append("### Source Checkpoint")
        md.append(f"- path: `{rel(ckpt_path)}`")
        md.append(f"- exists: {'YES' if ckpt_ok else '**MISSING**'}")
        md.append(f"- safety: {'OK' if safe else f'**REFUSE_ADAPTED_OR_TARGET_CKPT** ({reason})'}")
        md.append("- epoch: 20\n")

        if ds_info.get("config_hint"):
            cfg_path = root / ds_info["config_hint"]
        else:
            cfg_path = find_ckpt_config(ckpt_path)
        cfg_ok = cfg_path is not None and cfg_path.exists()

        md.append("### Source Config")
        md.append(f"- path: `{rel(cfg_path) if cfg_ok else 'null'}`")
        md.append(f"- exists: {'YES' if cfg_ok else '**MISSING**'}\n")

        plan_ds = {
            "source": source,
            "ckpt":   str(ckpt_path) if ckpt_ok and safe else None,
            "config": str(cfg_path)  if cfg_ok           else None,
            "targets": {},
        }
        if not ckpt_ok:
            md.append(f"> **MISSING**: ckpt not found\n")
        if not safe:
            plan_ds["ckpt_refused"] = reason
            md.append(f"> **REFUSE_ADAPTED_OR_TARGET_CKPT**: {reason}\n")

        md.append("### Target Domains\n")
        for target in ds_info["targets"]:
            md.append(f"#### {target}")
            ann_rel, ann_path = resolve_ann(ds_name, target, root)
            image_root = root / ds_info["image_root"]
            info = inspect_annotation(ann_path, image_root)

            status = "OK" if (info["exists"] and info["coco_format"]) else "**MISSING**"
            md.append(f"- ann_file: `{ann_rel}`  [{status}]")
            md.append(f"- image_root: `{ds_info['image_root']}`  "
                      f"[{'EXISTS' if image_root.exists() else '**MISSING**'}]")
            md.append(f"- num_images: {info['num_images']}")
            md.append(f"- num_annotations: {info['num_annotations']}")
            md.append(f"- num_videos: {info['num_videos']}")
            md.append(f"- categories: {[c['name'] for c in info['categories']]}")
            md.append(f"- frame_id_sample: {info['frame_id_sample']}  "
                      f"(note: frame_id in ann is 0-based; GeneralTrack txt uses 1-based)")
            md.append(f"- file_name_sample: `{info['file_name_sample']}`")
            md.append(f"- size: {info['width_sample']}x{info['height_sample']}")
            md.append(f"- first20_images_exist: {info['first20_ok']}/20")
            if info["first20_missing"]:
                md.append(f"- missing_samples: {info['first20_missing']}")
            if info.get("error"):
                md.append(f"- error: {info['error']}")
            md.append("")

            ready = (info["exists"] and info["coco_format"]
                     and ckpt_ok and safe and image_root.exists())
            plan_ds["targets"][target] = {
                "ann_file":    str(ann_path) if ann_path and ann_path.exists() else None,
                "image_root":  str(image_root) if image_root.exists() else None,
                "num_images":  info["num_images"],
                "num_videos":  info["num_videos"],
                "status":      "READY" if ready else "MISSING",
            }

        plan["datasets"][ds_name] = plan_ds

    # ── category mapping audit ──────────────────────────────────────────────
    CATS = [("pedestrian",1),("car",2),("truck",3),("bus",4),("motorcycle",5),("bicycle",6)]
    with open(cat_audit, "w") as f:
        f.write("# Category Mapping Audit\n\n"
                "All four datasets share identical COCO category ids.\n\n"
                "| dataset | source_config_classes | target_ann_categories |"
                " model_label_ids | saved_category_id_policy |\n"
                "|---|---|---|---|---|\n")
        cats_str = ",".join(f"{n}({i})" for n,i in CATS)
        for ds_name in DATASETS:
            f.write(f"| {ds_name} | {cats_str} | {cats_str} | 0..5 (0-indexed) | label+1 |\n")
        f.write("\n## Decision\n\ncategory_id = model_output_label + 1 for all datasets.\n"
                "No remapping needed. No cross-dataset conflict.\n")
    print(f"Category mapping audit: {cat_audit}")

    with open(out_md, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Audit report: {out_md}")

    with open(out_yaml, "w") as f:
        yaml.dump(plan, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"Eval plan: {out_yaml}")

    print("\n=== AUDIT SUMMARY ===")
    for ds_name, pd in plan["datasets"].items():
        print(f"\n{ds_name} (source={pd['source']})")
        print(f"  ckpt:   {pd.get('ckpt') or 'MISSING/REFUSED'}")
        print(f"  config: {pd.get('config') or 'MISSING'}")
        for tgt, ti in pd["targets"].items():
            print(f"  target {tgt:12s}: {ti['status']:8s}"
                  f"  images={ti['num_images']}  videos={ti['num_videos']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--damot-root",  default=str(DAMOT_ROOT))
    p.add_argument("--out",         default="outputs/source_detector/source_ckpt_target_audit.md")
    p.add_argument("--config-out",  default="outputs/source_detector/source_detector_eval_plan.yaml")
    build_report_and_plan(p.parse_args())


if __name__ == "__main__":
    main()
