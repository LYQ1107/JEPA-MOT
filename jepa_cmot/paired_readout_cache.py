"""Matched causal windows for official encoder and masked predictor ROI readouts."""
import argparse
from collections import defaultdict, deque
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from build_generaltrack_vjepa21_cache import (
    _letterbox_rgb, _normalize_frame, _temporal_maps, _pool_temporal_box_features,
)
from build_generaltrack_vjepa21_dense_predictor_cache import _predictor_masks
from vjepa21_encoder import (
    load_official_vjepa21_encoder, load_official_vjepa21_predictor,
    forward_tokens, forward_predictor_tokens,
)
from .train_source import sha256

ROOT = Path(__file__).resolve().parents[1]


def masked_forecast(encoder, predictor, video, masks_x, masks_y):
    # Explicitly keep target pixels out of the patch projection as well as
    # target tokens out of context attention.
    context_video = video.clone()
    context_video[:, :, -2:] = 0
    return forward_predictor_tokens(encoder, predictor, context_video, masks_x, masks_y,
                                    autocast_dtype=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--source-domain', required=True)
    parser.add_argument('--observed-domain')
    parser.add_argument('--source-gt', action='store_true')
    parser.add_argument('--window', type=int, default=4)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--engineering-smoke', action='store_true')
    parser.add_argument('--check-masked-target-invariance', action='store_true')
    args = parser.parse_args()
    if args.window < 4 or args.window % 2:
        raise ValueError('Need at least one context and one masked target tubelet')
    torch.set_num_threads(4)
    # Use FP32 deterministic execution for the perturbation probe. FP16
    # repeats varied, so their differences cannot establish target leakage.
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    annotation = json.loads(args.annotation.read_text())
    by_image = defaultdict(list)
    if args.source_gt:
        split = json.loads((ROOT/'docs/JEPA_CMOT/VIDEO_SPLITS.json').read_text())
        allowed = set(split['datasets'][args.dataset][args.source_domain]['source_train'])
        if not {v['name'] for v in annotation['videos']}.issubset(allowed):
            raise ValueError('Source readout includes a video outside the frozen fitting split')
        for a in annotation['annotations']:
            if not a.get('iscrowd'):
                by_image[a['image_id']].append(a)
    elif annotation.get('annotations') or annotation.get('gt_read'):
        raise ValueError('Detection readouts must not consume GT')
    images = sorted(annotation['images'], key=lambda i: (i['video_id'], i['frame_id'], i['id']))
    if args.max_frames:
        images = images[:args.max_frames]
    checkpoint = ROOT/'JEPA/checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt'
    device = torch.device('cuda')
    encoder = load_official_vjepa21_encoder(checkpoint, model_name='vjepa2_1_vit_large_384',
        img_size=384, num_frames=args.window, patch_size=16, tubelet_size=2, device=device)
    predictor = load_official_vjepa21_predictor(checkpoint, embed_dim=int(encoder.embed_dim),
        img_size=384, num_frames=args.window, patch_size=16, tubelet_size=2, device=device)
    masks_x, masks_y = _predictor_masks(args.window, 384, 16, device)
    buffer = deque(maxlen=args.window)
    previous = None
    outputs = {'encoder': [], 'predictor': []}
    offsets = [0]
    identities = defaultdict(list)
    probe = None
    resets = 0
    start = time.perf_counter()
    for index, image in enumerate(images):
        segment = (image['video_id'], image.get('segment_id', 0))
        if previous is None or segment != previous[0]:
            buffer.clear()
            resets += 1
        elif image['frame_id'] != previous[1] + 1:
            raise ValueError('Window crosses a native frame gap without a segment reset')
        previous = (segment, image['frame_id'])
        bgr = cv2.imread(image['file_name'])
        if bgr is None:
            raise FileNotFoundError(image['file_name'])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        canvas, scale, left, top = _letterbox_rgb(rgb, 384)
        buffer.append(_normalize_frame(canvas))
        # Missing past frames have no current pixels. Repeating the first image
        # would expose masked target appearance through the context at startup.
        frames = [torch.zeros_like(buffer[0])] * (args.window-len(buffer)) + list(buffer)
        video = torch.stack(frames, dim=1).unsqueeze(0).to(device)
        rows = by_image[image['id']] if args.source_gt else image['detection']
        if args.source_gt:
            detections = np.asarray([[a['bbox'][0], a['bbox'][1], a['bbox'][0]+a['bbox'][2],
                a['bbox'][1]+a['bbox'][3], 1., a['category_id']-1] for a in rows], np.float32).reshape(-1, 6)
        else:
            detections = np.asarray(rows, np.float32).reshape(-1, 6)
        with torch.inference_mode():
            encoded = forward_tokens(encoder, video, autocast_dtype=None)
            predicted = masked_forecast(encoder, predictor, video, masks_x, masks_y)
            if args.check_masked_target_invariance and probe is None and len(buffer) == args.window:
                changed = video.clone()
                changed[:, :, -2:] = 0
                masked_change = masked_forecast(encoder, predictor, changed, masks_x, masks_y)
                legacy = forward_predictor_tokens(encoder, predictor, video, masks_x, masks_y,
                                                  autocast_dtype=None)
                repeated = masked_forecast(encoder, predictor, video, masks_x, masks_y)
                changed = video.clone()
                changed[:, :, :-2] = 0
                past_change = masked_forecast(encoder, predictor, changed, masks_x, masks_y)
                probe = dict(masked_target_max_abs_delta=float((predicted-masked_change).abs().max()),
                             same_input_repeat_max_abs_delta=float((predicted-repeated).abs().max()),
                             past_context_max_abs_delta=float((predicted-past_change).abs().max()),
                             token_mask_only_vs_explicit_pixel_mask_max_abs_delta=float((legacy-predicted).abs().max()))
                print('MASKED_TARGET_INVARIANCE_PROBE', probe, flush=True)
                if (probe['masked_target_max_abs_delta'] > 1e-6
                    or probe['same_input_repeat_max_abs_delta'] > 1e-6
                    or probe['past_context_max_abs_delta'] < 1e-5):
                    raise ValueError('Official masked predictor does not satisfy the causal readout probe')
            for name, tokens, count in [('encoder', encoded, args.window), ('predictor', predicted, 2)]:
                pooled = _pool_temporal_box_features(_temporal_maps(tokens, count, 384, 16), detections,
                    temporal_pool='last', image_width=image['width'], image_height=image['height'],
                    scale=scale, pad_left=left, pad_top=top, patch_size=16, layer_count=1, roi_pool='box_mean')
                if not args.source_gt:
                    pooled[detections[:, 4] < .35] = 0
                outputs[name].append(np.asarray(pooled, np.float32))
        offsets.append(offsets[-1]+len(detections))
        if args.source_gt:
            identities['image_ids'].extend([image['id']]*len(rows))
            identities['frame_ids'].extend([image['frame_id']]*len(rows))
            identities['video_ids'].extend([image['video_id']]*len(rows))
            identities['track_ids'].extend([a['instance_id'] for a in rows])
            identities['segment_ids'].extend([image.get('segment_id', 0)]*len(rows))
            identities['boxes'].extend(detections[:, :4].tolist())
        else:
            identities['image_ids'].append(image['id'])
        if (index+1) % 128 == 0:
            print('PAIRED_READOUT_PROGRESS', index+1, len(images), offsets[-1], flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    weights_sha = sha256(checkpoint)
    for name in outputs:
        output = args.out/('features_'+name+'_w'+str(args.window)+'.npz')
        arrays = dict(features=np.concatenate(outputs[name]), offsets=np.asarray(offsets),
                      **{k: np.asarray(v, np.float32 if k == 'boxes' else np.int64) for k, v in identities.items()})
        if args.source_gt and name == 'predictor':
            arrays['candidate_ranking_features'] = np.concatenate(outputs['encoder'])
        np.savez_compressed(output, **arrays)
        metadata = dict(status='complete', roi_mode='dense_'+name+'_w'+str(args.window), roi_context=1.,
            feature_shape=list(arrays['features'].shape), checkpoint_sha256=weights_sha,
            annotation_sha256=sha256(args.annotation), dataset=args.dataset, source_domain=args.source_domain,
            observed_video_domain=args.observed_domain, gt_read=args.source_gt, target_domains_read=[],
            feature_score_min=.35, causal=True, future_frames_read=False, window=args.window,
            startup_padding='normalized zero, no repeated current pixels in past context',
            predictor_target='final two-frame pixels zeroed before patch projection, tokens masked before attention',
            predictor_context_max_offset=-2, readout_includes_current_pixels=name == 'encoder',
            engineering_smoke_only=bool(args.max_frames or args.engineering_smoke), frames=len(images), video_segment_resets=resets,
            history_may_cross_segments=False, physical_input='same native causal window for both readouts',
            source_frame_sampling=annotation.get('frame_sampling'), masked_target_invariance_probe=probe,
            source_negative_pool_reference='paired encoder last-feature cosine for both readouts' if args.source_gt else None,
            code_sha256=sha256(Path(__file__)),
            frozen_encoder_parameters=sum(p.numel() for p in encoder.parameters()),
            frozen_official_predictor_parameters=sum(p.numel() for p in predictor.parameters()),
            precision='FP32, math SDPA, deterministic algorithms, cuBLAS workspace :4096:8',
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            timing_scope='combined paired extraction/write/hash, excludes model loading; not per-readout or live FPS',
            seconds=time.perf_counter()-start)
        output.with_suffix('.npz.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print('PAIRED_READOUT_COMPLETE', name, metadata['feature_shape'], probe, flush=True)


if __name__ == '__main__':
    main()
