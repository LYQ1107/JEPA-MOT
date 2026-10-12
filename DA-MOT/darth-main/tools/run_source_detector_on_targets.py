#!/usr/bin/env python3
"""
Run source-domain 20e baseline checkpoint directly on target domains.
No training, no adaptation.
Outputs: detections_coco.json, generaltrack_dets/<video>/det/yolox_dets.txt,
         image_mapping.json, metadata.json, run.log, test_stdout.log
Detection source: bbox_coco.json written by test.py --format-only (authoritative).
"""
import argparse, csv, json, logging, os, shutil, subprocess, sys, time
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent

ADAPTED_KW = ["adapt","tta","sequential","keep","reset",
              "stage0","stage1","stage2","stage3","stage4","stage5"]
TARGET_KW  = ["rainy","night","cloudy","fog","snow","overcast",
              "dawn","dusk","dark","indoor","sunset","morning"]

# 17 target domains (indoor excluded per user request)
EVAL_CONFIGS = {
    ("bdd_da",   "overcast"):  "configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_overcast.py",
    ("bdd_da",   "rainy"):     "configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_rainy.py",
    ("bdd_da",   "dawn_dusk"): "configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_dawn_dusk.py",
    ("bdd_da",   "night"):     "configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_night.py",
    ("motsynth", "cloudy"):    "configs/mot/qdtrack/motsynth/qdtrack_faster-rcnn_r50_fpn_20e_motsynth_cloudy_eval.py",
    ("motsynth", "dark"):      "configs/mot/qdtrack/motsynth/qdtrack_faster-rcnn_r50_fpn_20e_motsynth_dark_eval.py",
    ("motsynth", "fog"):       "configs/mot/qdtrack/motsynth/qdtrack_faster-rcnn_r50_fpn_20e_motsynth_fog_eval.py",
    ("motsynth", "rain"):      "configs/mot/qdtrack/motsynth/qdtrack_faster-rcnn_r50_fpn_20e_motsynth_rain_eval.py",
    ("motsynth", "snow"):      "configs/mot/qdtrack/motsynth/qdtrack_faster-rcnn_r50_fpn_20e_motsynth_snow_eval.py",
    ("ua_detrac","cloudy"):    "configs/mot/qdtrack/ua_detrac/qdtrack_faster-rcnn_r50_fpn_20e_ua_detrac_cloudy_eval.py",
    ("ua_detrac","night"):     "configs/mot/qdtrack/ua_detrac/qdtrack_faster-rcnn_r50_fpn_20e_ua_detrac_night_eval.py",
    ("ua_detrac","rainy"):     "configs/mot/qdtrack/ua_detrac/qdtrack_faster-rcnn_r50_fpn_20e_ua_detrac_rainy_eval.py",
    ("vkitti2",  "fog"):       "configs/mot/qdtrack/vkitti2/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_fog_eval.py",
    ("vkitti2",  "morning"):   "configs/mot/qdtrack/vkitti2/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_morning_eval.py",
    ("vkitti2",  "overcast"):  "configs/mot/qdtrack/vkitti2/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_overcast_eval.py",
    ("vkitti2",  "rain"):      "configs/mot/qdtrack/vkitti2/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_rain_eval.py",
    ("vkitti2",  "sunset"):    "configs/mot/qdtrack/vkitti2/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_sunset_eval.py",
}

# domains explicitly excluded from the plan (still in ann but we skip them)
EXCLUDED_DOMAINS = {("motsynth", "indoor")}


# ── safety ──────────────────────────────────────────────────────────────────

def ckpt_segment(p):
    parts = Path(p).parts
    return "/".join(parts[-5:]).lower()

def check_ckpt(ckpt_path, allow_adapted=False):
    if allow_adapted:
        return True, "allowed"
    seg = ckpt_segment(ckpt_path).replace("darth-main", "")
    for kw in ADAPTED_KW:
        if kw in seg: return False, f"adaptation keyword '{kw}'"
    for kw in TARGET_KW:
        if kw in seg: return False, f"target-domain keyword '{kw}'"
    return True, "ok"


# ── annotation helpers ──────────────────────────────────────────────────────

def build_image_mapping(ann_data):
    vid_id2name = {v["id"]: v["name"] for v in ann_data.get("videos", [])}
    return [{
        "image_id":   img["id"],
        "video_id":   img.get("video_id"),
        "video_name": vid_id2name.get(img.get("video_id"), str(img.get("video_id"))),
        "frame_id":   img.get("frame_id"),
        "file_name":  img.get("file_name"),
        "width":      img.get("width"),
        "height":     img.get("height"),
    } for img in ann_data.get("images", [])]


# ── output helpers ──────────────────────────────────────────────────────────

def load_bbox_coco_json(tmp_work, score_thr):
    """Read bbox_coco.json written by test.py (authoritative detection result)."""
    candidates = sorted(tmp_work.glob("**/bbox_coco.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None, "bbox_coco.json not found under tmp_work"
    src = candidates[0]
    raw = json.load(open(src))
    if not isinstance(raw, list):
        return None, f"bbox_coco.json is not a list (got {type(raw)})"
    if score_thr > 0:
        raw = [d for d in raw if d.get("score", 0) >= score_thr]
    return raw, None


def write_generaltrack_txts(coco_dets, image_mapping, out_dir):
    """Write per-video yolox_dets.txt (frame_id 1-based, xywh format)."""
    from collections import defaultdict
    id2info  = {m["image_id"]: m for m in image_mapping}
    vid_dets = defaultdict(list)
    for det in coco_dets:
        info = id2info.get(det["image_id"])
        if info: vid_dets[info["video_name"]].append((info, det))

    all_videos = {m["video_name"] for m in image_mapping}
    written, empty = {}, []
    for vid_name in sorted(all_videos):
        det_dir  = out_dir / "generaltrack_dets" / vid_name / "det"
        det_dir.mkdir(parents=True, exist_ok=True)
        txt_path = det_dir / "yolox_dets.txt"
        lines = []
        for info, det in sorted(vid_dets.get(vid_name, []),
                                key=lambda x: (x[0]["frame_id"], -x[1]["score"])):
            frame_id = (info["frame_id"] or 0) + 1   # 0-based → 1-based
            x, y, w, h = det["bbox"]
            lines.append(f"{frame_id},-1,{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                         f"{det['score']:.6f},{det['category_id']}")
        txt_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        if not lines: empty.append(vid_name)
        written[vid_name] = str(txt_path)
    return written, empty


def mirror_to_generaltrack(src_dir, gt_root, dataset, domain):
    gt_dir = Path(gt_root) / dataset / domain
    gt_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["detections_coco.json", "image_mapping.json", "metadata.json"]:
        src = src_dir / fname
        dst = gt_dir / fname
        if src.exists():
            if dst.exists() or dst.is_symlink(): dst.unlink()
            try:    dst.symlink_to(src)
            except Exception: shutil.copy2(str(src), str(dst))
    src_gt = src_dir / "generaltrack_dets"
    dst_gt = gt_dir  / "generaltrack_dets"
    if src_gt.exists():
        if dst_gt.is_symlink() or dst_gt.exists():
            if dst_gt.is_symlink():
                dst_gt.unlink()
            else:
                shutil.rmtree(str(dst_gt))
        try:    dst_gt.symlink_to(src_gt)
        except Exception: shutil.copytree(str(src_gt), str(dst_gt))


# ── per-domain runner ────────────────────────────────────────────────────────

def run_one_domain(ds_name, domain, plan_entry, target_info, args, log, summary_rows):
    t0 = time.time()
    ckpt       = plan_entry.get("ckpt")
    config     = plan_entry.get("config")
    source     = plan_entry.get("source", "unknown")
    ann_file   = target_info.get("ann_file")
    image_root = target_info.get("image_root")

    out_dir = Path(args.output_root) / ds_name / domain
    out_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(str(out_dir / "run.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    def fail(reason, status="FAILED"):
        log.error(f"[{ds_name}/{domain}] {status}: {reason}")
        summary_rows.append(dict(
            dataset=ds_name, source_domain=source, target_domain=domain,
            ckpt=ckpt or "", ann_file=ann_file or "", image_root=image_root or "",
            num_videos=0, num_images=0, num_detections=0, score_thr=args.score_thr,
            detections_coco_json="", generaltrack_det_root="",
            status=status, error_message=reason))
        log.removeHandler(fh)

    # ── exclusion check ──────────────────────────────────────────────────────
    if (ds_name, domain) in EXCLUDED_DOMAINS:
        return fail(f"excluded domain ({ds_name},{domain})", "EXCLUDED")

    log.info(f"{'='*60}")
    log.info(f"[{ds_name}/{domain}] START  ckpt={ckpt}")
    log.info(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')}")

    # ── safety checks ────────────────────────────────────────────────────────
    if not ckpt or not Path(ckpt).exists():
        return fail(f"ckpt missing: {ckpt}", "MISSING_CKPT")
    safe, reason = check_ckpt(ckpt, getattr(args, "allow_adapted_ckpt", False))
    if not safe:
        return fail(f"REFUSE_ADAPTED_OR_TARGET_CKPT  ckpt={ckpt}  reason={reason}",
                    "REFUSE_ADAPTED_OR_TARGET_CKPT")

    eval_cfg_rel = EVAL_CONFIGS.get((ds_name, domain))
    if not eval_cfg_rel:
        return fail(f"no eval config for ({ds_name},{domain})", "MISSING_CONFIG")
    eval_cfg = ROOT / eval_cfg_rel
    if not eval_cfg.exists():
        return fail(f"eval config file missing: {eval_cfg}", "MISSING_CONFIG")

    if not ann_file or not Path(ann_file).exists():
        return fail(f"annotation missing: {ann_file}", "MISSING_ANNOTATION")

    # ── check if already done (bbox_coco.json exists from a previous run) ───
    coco_json  = out_dir / "detections_coco.json"
    tmp_work   = out_dir / "tmp_test_work"
    tmp_dir    = out_dir / "tmp_collect"
    existing_bbox = list(tmp_work.glob("**/bbox_coco.json")) if tmp_work.exists() else []

    # ── load annotation ──────────────────────────────────────────────────────
    ann_data    = json.load(open(ann_file))
    img_mapping = build_image_mapping(ann_data)
    num_images  = len(img_mapping)
    num_videos  = len(ann_data.get("videos", []))

    if getattr(args, "max_videos", None):
        vids = [v["name"] for v in ann_data.get("videos", [])][:args.max_videos]
        img_mapping = [m for m in img_mapping if m["video_name"] in set(vids)]
        num_images, num_videos = len(img_mapping), min(num_videos, args.max_videos)

    with open(out_dir / "image_mapping.json", "w") as f:
        json.dump(img_mapping, f)

    # ── skip inference if bbox_coco.json already present ────────────────────
    if existing_bbox and not getattr(args, "force", False):
        log.info(f"  found existing bbox_coco.json, skipping inference: {existing_bbox[0]}")
        ret_code = 0
    else:
        tmp_work.mkdir(exist_ok=True)
        tmp_dir.mkdir(exist_ok=True)
        python  = sys.executable
        test_py = str(ROOT / "tools/run/test.py")
        gpus    = args.gpus

        if args.launcher == "pytorch":
            cmd = [python, "-m", "torch.distributed.launch",
                   f"--nproc_per_node={gpus}",
                   "--master_addr=127.0.0.1",
                   f"--master_port={args.master_port}",
                   test_py, str(eval_cfg),
                   "--checkpoint", ckpt,
                   "--launcher", "pytorch",
                   "--work-dir", str(tmp_work),
                   "--tmpdir", str(tmp_dir),
                   "--format-only",
                   "--fuse-conv-bn",
                   "--cfg-options",
                   f"data.workers_per_gpu={args.workers}",
                   "cudnn_benchmark=True"]
        else:
            cmd = [python, test_py, str(eval_cfg),
                   "--checkpoint", ckpt,
                   "--launcher", "none",
                   "--work-dir", str(tmp_work),
                   "--format-only",
                   "--fuse-conv-bn",
                   "--cfg-options",
                   f"data.workers_per_gpu={args.workers}",
                   "cudnn_benchmark=True"]

        log.info(f"  CMD: {' '.join(cmd)}")
        ret = subprocess.run(cmd, cwd=str(ROOT), env=os.environ.copy(),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out_dir / "test_stdout.log").write_text(ret.stdout.decode(errors="replace"))
        ret_code = ret.returncode
        log.info(f"  test.py exit_code={ret_code}")

    # ── read detections from bbox_coco.json ──────────────────────────────────
    coco_dets, err = load_bbox_coco_json(tmp_work, args.score_thr)
    if coco_dets is None:
        return fail(f"bbox_coco.json not found or invalid: {err}",
                    "FAILED" if ret_code == 0 else "DETECTION_FAILED")

    # validate: every detection image_id must exist in our mapping
    valid_ids  = {m["image_id"] for m in img_mapping}
    coco_dets  = [d for d in coco_dets if d.get("image_id") in valid_ids]
    num_dets   = len(coco_dets)

    with open(coco_json, "w") as f:
        json.dump(coco_dets, f)

    written, empty = write_generaltrack_txts(coco_dets, img_mapping, out_dir)
    gt_root_dir = out_dir / "generaltrack_dets"
    log.info(f"  dets={num_dets}  videos={len(written)}  empty_videos={len(empty)}")

    meta = dict(
        dataset=ds_name, source_domain=source, target_domain=domain,
        ckpt=ckpt, eval_config=str(eval_cfg), ann_file=ann_file,
        image_root=image_root, num_images=num_images, num_videos=num_videos,
        num_detections=num_dets, score_thr=args.score_thr,
        mode="direct_source_detector_no_adaptation",
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        empty_videos=empty, elapsed_sec=round(time.time() - t0, 1))
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    gt_out_root = Path(args.generaltrack_output_root)
    if gt_out_root.parent.exists():
        mirror_to_generaltrack(out_dir, gt_out_root, ds_name, domain)

    det_status = "DETECTION_SAVED" if ret_code == 0 else "DETECTION_SAVED_EVAL_WARN"
    if ret_code != 0:
        log.warning(f"  test.py exited {ret_code} but bbox_coco.json was produced")

    log.info(f"[{ds_name}/{domain}] {det_status}  dets={num_dets}  "
             f"elapsed={round(time.time()-t0,1)}s")
    summary_rows.append(dict(
        dataset=ds_name, source_domain=source, target_domain=domain,
        ckpt=ckpt, ann_file=ann_file, image_root=image_root,
        num_videos=num_videos, num_images=num_images, num_detections=num_dets,
        score_thr=args.score_thr,
        detections_coco_json=str(coco_json),
        generaltrack_det_root=str(gt_root_dir),
        status=det_status, error_message=""))
    log.removeHandler(fh)


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plan",       required=True)
    p.add_argument("--dataset",    default="all")
    p.add_argument("--domains",    default="all")
    p.add_argument("--output-root",  default="outputs/source_detector")
    p.add_argument("--generaltrack-output-root",
                   default="/data1/LWR/vranlee/SERVER_ONLY/avis/GeneralTrack-main/outputs/source_detector")
    p.add_argument("--max-videos",   type=int, default=None)
    p.add_argument("--gpus",         type=int, default=4)
    p.add_argument("--launcher",     default="pytorch", choices=["pytorch", "none"])
    p.add_argument("--master-port",  type=int, default=29510)
    p.add_argument("--workers",      type=int, default=4)
    p.add_argument("--score-thr",    type=float, default=0.001)
    p.add_argument("--force",        action="store_true",
                   help="re-run inference even if bbox_coco.json already exists")
    p.add_argument("--allow-adapted-ckpt", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(ROOT)

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,8,9"
        print("[WARNING] CUDA_VISIBLE_DEVICES not set, defaulting to 6,7,8,9")

    log = logging.getLogger("src_det")
    log.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(ch)

    with open(args.plan) as f:
        plan = yaml.safe_load(f)

    datasets_cfg = plan.get("datasets", {})
    want_ds  = (list(datasets_cfg.keys()) if args.dataset == "all"
                else [d.strip() for d in args.dataset.replace(",", " ").split()])
    want_dom = (None if args.domains == "all"
                else {d.strip() for d in args.domains.replace(",", " ").split()})

    summary_rows = []
    Path(args.output_root).mkdir(parents=True, exist_ok=True)

    for ds_name in want_ds:
        if ds_name not in datasets_cfg:
            log.warning(f"Dataset '{ds_name}' not in plan, skipping"); continue
        plan_ds = datasets_cfg[ds_name]
        for domain, tinfo in plan_ds.get("targets", {}).items():
            if (ds_name, domain) in EXCLUDED_DOMAINS: continue
            if want_dom and domain not in want_dom:   continue
            run_one_domain(ds_name, domain, plan_ds, tinfo, args, log, summary_rows)

    summary_csv = Path(args.output_root) / "source_detector_detection_summary.csv"
    fields = ["dataset","source_domain","target_domain","ckpt","ann_file","image_root",
              "num_videos","num_images","num_detections","score_thr",
              "detections_coco_json","generaltrack_det_root","status","error_message"]
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(summary_rows)
    log.info(f"Summary CSV: {summary_csv}")

    print("\n=== SOURCE DETECTOR SUMMARY ===")
    for r in summary_rows:
        print(f"  {r['dataset']:10s} {r['target_domain']:12s} "
              f"{r['status']:35s} dets={r['num_detections']}")


if __name__ == "__main__":
    main()
