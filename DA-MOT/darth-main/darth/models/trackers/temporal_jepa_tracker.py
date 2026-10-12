"""A causal, track-level JEPA extension of the installed QDTracker.

The detector, QDTrack feature extraction, NMS, ID allocation, and lifecycle
are copied from the installed ``mmtrack`` QuasiDenseTracker.  The only
association change is an explicitly gated additive score for a track latent
predicted from observations strictly before the current frame.
"""

from __future__ import absolute_import, division, print_function

import copy
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from mmdet.core import bbox_overlaps

from mmtrack.models.builder import TRACKERS
from mmtrack.models.trackers import QuasiDenseTracker

from ..jepa import TOKEN_META_DIM, TemporalJEPAAdapter, temporal_jepa_loss


def _as_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


@TRACKERS.register_module()
class TemporalJEPATracker(QuasiDenseTracker):
    """QDTracker with a causal track-history JEPA association adapter.

    ``jepa_enabled=False`` is an intentional exact-baseline path and calls
    the installed tracker implementation directly.  Online adaptation is
    opt-in and updates only ``self.jepa``; detector and QDTracker lifecycle
    state remain frozen.
    """

    def __init__(self,
                 jepa_enabled=False,
                 jepa_checkpoint=None,
                 jepa_embed_dim=256,
                 jepa_history_len=4,
                 jepa_meta_dim=TOKEN_META_DIM,
                 jepa_hidden_dim=128,
                 jepa_latent_dim=None,
                 jepa_lambda=0.20,
                 jepa_gate_min_quality=0.55,
                 jepa_gate_max_uncertainty=1.5,
                 jepa_gate_native_margin=0.10,
                 jepa_instability_cosine=0.65,
                 jepa_online=False,
                 jepa_adaptation_interval=10,
                 jepa_delay=1,
                 jepa_updates_per_window=1,
                 jepa_batch_size=8,
                 jepa_learning_rate=1e-4,
                 jepa_ema=0.99,
                 jepa_use_teacher=True,
                 jepa_use_replay=True,
                 jepa_target_replay_size=512,
                 jepa_source_replay=None,
                 jepa_source_anchor_weight=1.0,
                 jepa_seed=17,
                 jepa_pseudo_det_thr=0.80,
                 jepa_pseudo_match_thr=0.60,
                 jepa_pseudo_max_gap=6,
                 jepa_log_path=None,
                 jepa_checkpoint_dir=None,
                 jepa_domain='unknown',
                 jepa_frozen_representation=False,
                 jepa_random_control=False,
                 **kwargs):
        # QuasiDenseTracker.__init__ invokes self.reset(), so all fields read
        # by reset must exist before calling super().
        self._jepa_constructing = True
        self._jepa_history = {}
        self._jepa_pending = []
        self._jepa_target_replay = []
        self._jepa_ready_since_update = []
        self._jepa_window_stats = self._new_window_stats()
        self._jepa_reset_count = 0
        self._jepa_adapt_step = 0
        self._jepa_seen_target = 0
        self._jepa_rng = random.Random(int(jepa_seed))
        self._jepa_optimizer = None
        self._jepa_device = None
        self._jepa_source_replay_cache = None
        self.__dict__['_jepa_source_anchor'] = None
        self._jepa_last_adapt_frame = -1

        super(TemporalJEPATracker, self).__init__(**kwargs)

        self.jepa_enabled = bool(jepa_enabled)
        self.jepa_checkpoint = jepa_checkpoint
        self.jepa_lambda = float(jepa_lambda)
        self.jepa_gate_min_quality = float(jepa_gate_min_quality)
        self.jepa_gate_max_uncertainty = float(jepa_gate_max_uncertainty)
        self.jepa_gate_native_margin = float(jepa_gate_native_margin)
        self.jepa_instability_cosine = float(jepa_instability_cosine)
        self.jepa_online = bool(jepa_online)
        self.jepa_adaptation_interval = max(1, int(jepa_adaptation_interval))
        self.jepa_delay = max(1, int(jepa_delay))
        self.jepa_updates_per_window = max(1, int(jepa_updates_per_window))
        self.jepa_batch_size = max(1, int(jepa_batch_size))
        self.jepa_learning_rate = float(jepa_learning_rate)
        self.jepa_ema = float(jepa_ema)
        self.jepa_use_teacher = bool(jepa_use_teacher)
        self.jepa_use_replay = bool(jepa_use_replay)
        self.jepa_target_replay_size = max(1, int(jepa_target_replay_size))
        self.jepa_source_replay = jepa_source_replay
        self.jepa_source_anchor_weight = max(
            0.0, float(jepa_source_anchor_weight))
        self.jepa_seed = int(jepa_seed)
        self.jepa_pseudo_det_thr = float(jepa_pseudo_det_thr)
        self.jepa_pseudo_match_thr = float(jepa_pseudo_match_thr)
        self.jepa_pseudo_max_gap = max(0, int(jepa_pseudo_max_gap))
        self.jepa_log_path = jepa_log_path
        self.jepa_checkpoint_dir = jepa_checkpoint_dir
        self.jepa_domain = str(jepa_domain)
        self.jepa_frozen_representation = bool(jepa_frozen_representation)
        self.jepa_random_control = bool(jepa_random_control)
        self._jepa_constructing = False

        self.jepa = TemporalJEPAAdapter(
            embed_dim=int(jepa_embed_dim),
            history_len=int(jepa_history_len),
            meta_dim=int(jepa_meta_dim),
            hidden_dim=int(jepa_hidden_dim),
            latent_dim=jepa_latent_dim)
        if jepa_checkpoint is not None:
            self._load_jepa_checkpoint(jepa_checkpoint)
        self.jepa.eval()
        # Keep a frozen source-domain function for continual source-anchor
        # distillation.  Bypass nn.Module registration so these weights do
        # not become part of the trainable adapter/checkpoint state.
        source_anchor = copy.deepcopy(self.jepa)
        for parameter in source_anchor.parameters():
            parameter.requires_grad = False
        source_anchor.eval()
        self.__dict__['_jepa_source_anchor'] = source_anchor

        self._write_event({
            'event': 'tracker_init',
            'jepa_enabled': self.jepa_enabled,
            'jepa_online': self.jepa_online,
            'use_teacher': self.jepa_use_teacher,
            'use_replay': self.jepa_use_replay,
            'source_anchor_weight': self.jepa_source_anchor_weight,
            'random_control': self.jepa_random_control,
            'frozen_representation': self.jepa_frozen_representation,
            'gt_used_online': False,
            'future_used_online': False,
            'domain': self.jepa_domain,
        })

    @staticmethod
    def _new_window_stats():
        return dict(
            frames=0,
            gated_pairs=0,
            candidate_pairs=0,
            base_matches=0,
            final_matches=0,
            association_changes=0,
            pseudo_pairs=0,
            pseudo_confidence_sum=0.0,
            association_elapsed_ms=0.0)

    def _write_event(self, event):
        """Append one compact JSONL event without overwriting old evidence."""
        if not getattr(self, 'jepa_log_path', None):
            return
        log_path = self.jepa_log_path
        parent = os.path.dirname(os.path.abspath(log_path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        payload = dict(event)
        payload.setdefault('domain', getattr(self, 'jepa_domain', 'unknown'))
        payload.setdefault('gt_used_online', False)
        payload.setdefault('future_used_online', False)
        with open(log_path, 'a') as handle:
            handle.write(json.dumps(payload, sort_keys=True) + '\n')

    def _load_jepa_checkpoint(self, checkpoint_path):
        if not os.path.isfile(checkpoint_path):
            raise IOError('JEPA checkpoint does not exist: {}'.format(
                checkpoint_path))
        payload = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(payload, dict):
            state = payload.get('state_dict', payload.get('adapter', payload))
        else:
            state = payload
        if not isinstance(state, dict):
            raise TypeError('unsupported JEPA checkpoint format')
        cleaned = {}
        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            new_key = key
            for prefix in ('module.', 'jepa.', 'adapter.'):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            cleaned[new_key] = value
        missing, unexpected = self.jepa.load_state_dict(cleaned, strict=False)
        self._write_event({
            'event': 'checkpoint_loaded',
            'checkpoint': os.path.abspath(checkpoint_path),
            'missing_keys': list(missing),
            'unexpected_keys': list(unexpected),
        })

    def reset(self):
        """Reset per-sequence tracking state, retaining continual JEPA state."""
        if not getattr(self, '_jepa_constructing', False):
            if self._jepa_window_stats.get('frames', 0):
                self._write_event({
                    'event': 'sequence_summary',
                    'sequence_index': self._jepa_reset_count,
                    'window_stats': dict(self._jepa_window_stats),
                })
            if self._jepa_pending:
                self._write_event({
                    'event': 'sequence_reset_dropped_pending',
                    'count': len(self._jepa_pending),
                    'reason': 'pending_pairs_cannot_cross_sequence_boundary',
                })
            self._jepa_reset_count += 1
            self._write_event({
                'event': 'sequence_reset',
                'sequence_index': self._jepa_reset_count,
            })
        super(TemporalJEPATracker, self).reset()
        self._jepa_history = {}
        self._jepa_pending = []
        self._jepa_ready_since_update = []
        self._jepa_window_stats = self._new_window_stats()
        self._jepa_last_adapt_frame = -1

    def _ensure_device(self, device):
        if self._jepa_device != device:
            self.jepa.to(device)
            if self._jepa_source_anchor is not None:
                self._jepa_source_anchor.to(device)
            self._jepa_device = device
            self._jepa_optimizer = None
        if self.jepa_online and self._jepa_optimizer is None:
            parameters = [p for p in self.jepa.parameters() if p.requires_grad]
            self._jepa_optimizer = torch.optim.Adam(
                parameters, lr=self.jepa_learning_rate)

    @staticmethod
    def _geometry(bbox, img_meta):
        h, w = img_meta.get('img_shape', img_meta.get('ori_shape'))[:2]
        h = max(float(h), 1.0)
        w = max(float(w), 1.0)
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        width = max(x2 - x1, 0.0)
        height = max(y2 - y1, 0.0)
        return torch.tensor(
            [(x1 + x2) / (2.0 * w), (y1 + y2) / (2.0 * h),
             width / w, height / h],
            dtype=torch.float32)

    def _make_observation_meta(self, bbox, score, frame_id, img_meta,
                               previous, first_frame):
        geometry = self._geometry(bbox, img_meta)
        if previous is None:
            velocity = torch.zeros(4, dtype=torch.float32)
            gap = 0.0
        else:
            delta = max(int(frame_id) - int(previous['frame_id']), 1)
            velocity = (geometry - previous['meta'][:4]) / float(delta)
            gap = float(max(int(frame_id) - int(previous['frame_id']) - 1, 0))
        age = float(max(int(frame_id) - int(first_frame) + 1, 1))
        visibility_proxy = 1.0 / (1.0 + gap)
        meta = torch.cat([
            geometry,
            velocity,
            torch.tensor([float(score), age, gap, visibility_proxy])
        ])
        return meta, geometry

    def _append_observation(self, track_id, bbox, embed, frame_id, img_meta):
        track_id = int(track_id)
        history = self._jepa_history.setdefault(track_id, [])
        previous = history[-1] if history else None
        first_frame = history[0]['frame_id'] if history else frame_id
        meta, geometry = self._make_observation_meta(
            bbox, bbox[-1], frame_id, img_meta, previous, first_frame)
        history.append({
            'frame_id': int(frame_id),
            'bbox': bbox.detach().cpu().float(),
            'geometry': geometry,
            'meta': meta,
            'embed': embed.detach().cpu().float(),
        })
        del history[:-self.jepa.history_len]

    @staticmethod
    def _history_batch_from_records(records, history_len, embed_dim, device,
                                    meta_dim=TOKEN_META_DIM):
        records = list(records[-history_len:])
        embeds = torch.zeros(
            history_len, embed_dim, dtype=torch.float32, device=device)
        if records:
            meta_dim = int(records[-1]['meta'].numel())
        metas = torch.zeros(
            history_len, meta_dim, dtype=torch.float32, device=device)
        mask = torch.zeros(history_len, dtype=torch.float32, device=device)
        start = history_len - len(records)
        for offset, record in enumerate(records):
            slot = start + offset
            embeds[slot] = record['embed'].to(device=device)
            metas[slot] = record['meta'].to(device=device)
            mask[slot] = 1.0
        return embeds, metas, mask

    def _history_batch(self, track_ids, device):
        valid_ids = []
        batches = []
        for track_id in track_ids:
            track_id = int(track_id)
            records = self._jepa_history.get(track_id, [])
            if not records:
                continue
            valid_ids.append(track_id)
            batches.append(
                self._history_batch_from_records(
                    records, self.jepa.history_len, self.jepa.embed_dim,
                    device, self.jepa.meta_dim))
        if not batches:
            return [], None
        return valid_ids, dict(
            history_embeds=torch.stack([item[0] for item in batches]),
            history_meta=torch.stack([item[1] for item in batches]),
            history_mask=torch.stack([item[2] for item in batches]))

    @staticmethod
    def _native_scores(embeds, memo_embeds, labels, memo_labels, with_cats,
                       match_metric):
        if match_metric == 'bisoftmax':
            similarities = torch.mm(embeds, memo_embeds.t())
            d2t_scores = similarities.softmax(dim=1)
            t2d_scores = similarities.softmax(dim=0)
            scores = (d2t_scores + t2d_scores) / 2
        elif match_metric == 'softmax':
            similarities = torch.mm(embeds, memo_embeds.t())
            scores = similarities.softmax(dim=1)
        elif match_metric == 'cosine':
            scores = torch.mm(
                F.normalize(embeds, p=2, dim=1),
                F.normalize(memo_embeds, p=2, dim=1).t())
        else:
            raise NotImplementedError
        if with_cats:
            cat_same = labels.view(-1, 1) == memo_labels.view(1, -1)
            scores = scores * cat_same.float().to(scores.device)
        return scores

    def _assign_matches(self, scores, bboxes, memo_ids):
        """Use the installed greedy matching rule on a score copy."""
        scores = scores.clone()
        ids = torch.full((bboxes.size(0), ), -1, dtype=torch.long)
        for i in range(bboxes.size(0)):
            conf, memo_ind = torch.max(scores[i, :], dim=0)
            track_id = memo_ids[memo_ind]
            if conf > self.match_score_thr:
                if track_id > -1:
                    if bboxes[i, -1] > self.obj_score_thr:
                        ids[i] = track_id
                        scores[:i, memo_ind] = 0
                        scores[i + 1:, memo_ind] = 0
                    elif conf > self.nms_conf_thr:
                        ids[i] = -2
        return ids

    def _gate_and_predict(self, bboxes, labels, embeds, frame_id, memo_ids,
                          memo_embeds, native_scores, img_metas):
        del img_metas  # kept in the signature to make the protocol explicit
        predicted_ids, batch = self._history_batch(
            [int(track_id) for track_id in memo_ids if int(track_id) > -1],
            embeds.device)
        gate = torch.zeros_like(native_scores, dtype=torch.bool)
        if not predicted_ids:
            return torch.zeros_like(native_scores), gate, {}

        with torch.no_grad():
            if self.jepa_frozen_representation:
                latest = torch.stack([
                    self._jepa_history[track_id][-1]['embed'].to(
                        embeds.device)
                    for track_id in predicted_ids])
                predicted = self.jepa.encode_target(latest)
                quality = torch.ones(len(predicted_ids), device=embeds.device)
                uncertainty = torch.zeros(
                    len(predicted_ids), device=embeds.device)
            else:
                outputs = self.jepa(
                    batch['history_embeds'], batch['history_meta'],
                    batch['history_mask'])
                predicted = F.normalize(
                    outputs['latent'], p=2, dim=-1, eps=1e-6)
                quality = outputs['quality']
                uncertainty = outputs['uncertainty']
            if self.jepa_random_control:
                # Explicit random/frozen control: same gate and cost path,
                # but no learned temporal signal is used.
                generator = torch.Generator(device=embeds.device)
                generator.manual_seed(self.jepa_seed + int(frame_id))
                predicted = F.normalize(
                    torch.randn(
                        predicted.shape, device=embeds.device,
                        generator=generator),
                    p=2, dim=-1, eps=1e-6)
                quality = torch.ones_like(quality)
                uncertainty = torch.zeros_like(uncertainty)
            if self.jepa_use_teacher:
                candidates = self.jepa.encode_target(embeds)
            else:
                candidates = self.jepa.encode_context(embeds)
            candidates = F.normalize(candidates, p=2, dim=-1, eps=1e-6)
            raw_jepa_scores = (torch.mm(candidates, predicted.t()) + 1.0) / 2.0

        id_to_prediction = {}
        for row, track_id in enumerate(predicted_ids):
            id_to_prediction[track_id] = dict(
                row=row,
                quality=_as_float(quality[row]),
                uncertainty=_as_float(uncertainty[row]),
                history_len=len(self._jepa_history.get(track_id, [])))
            track = self.tracks.get(track_id)
            if track is None:
                continue
            last_frame = int(track['last_frame'])
            gap = int(frame_id) - last_frame
            reappearance = gap > 1
            records = self._jepa_history.get(track_id, [])
            unstable = False
            if len(records) >= 2:
                previous_embed = records[-2]['embed'].to(embeds.device)
                latest_embed = records[-1]['embed'].to(embeds.device)
                similarity = F.cosine_similarity(
                    previous_embed[None], latest_embed[None]).item()
                unstable = similarity < self.jepa_instability_cosine
            id_mask = (memo_ids == track_id).to(native_scores.device)
            native_max = native_scores[:, id_mask]
            native_uncertain = (native_max.numel() == 0 or
                                float(native_max.max()) <
                                self.match_score_thr +
                                self.jepa_gate_native_margin)
            eligible = (self.jepa_random_control or
                        (float(quality[row]) >= self.jepa_gate_min_quality and
                         float(uncertainty[row]) <=
                         self.jepa_gate_max_uncertainty))
            should_gate = eligible and (reappearance or unstable or
                                        native_uncertain)
            if should_gate:
                col = torch.nonzero(
                    memo_ids == track_id, as_tuple=False).view(-1)
                if col.numel():
                    gate[:, col] = True

        # ``native_scores`` also contains backdrop columns.  Map the JEPA
        # rows back to the full memo layout and leave backdrop columns at
        # zero; they must never receive a predicted track latent.
        jepa_scores = torch.zeros_like(native_scores)
        for row, track_id in enumerate(predicted_ids):
            columns = torch.nonzero(
                memo_ids == track_id, as_tuple=False).view(-1)
            if columns.numel():
                jepa_scores[:, columns.to(jepa_scores.device)] = \
                    raw_jepa_scores[:, row:row + 1]
        jepa_scores = jepa_scores * gate.float()
        return jepa_scores, gate, id_to_prediction

    def _new_track_ids(self, ids, bboxes):
        # This matches QuasiDenseTracker exactly, including the CPU boolean
        # mask used by the upstream implementation.
        new_inds = (ids == -1) & (bboxes[:, 4] > self.init_score_thr).cpu()
        num_news = new_inds.sum()
        ids[new_inds] = torch.arange(
            self.num_tracks,
            self.num_tracks + num_news,
            dtype=torch.long)
        self.num_tracks += num_news
        return ids

    def _sample_from_records(self, records, target_embed, target_bbox,
                             target_score, frame_id, img_meta, quality):
        if not records:
            return None
        embeds, metas, mask = self._history_batch_from_records(
            records, self.jepa.history_len, self.jepa.embed_dim, 'cpu',
            self.jepa.meta_dim)
        previous = records[-1]
        target_geometry = self._geometry(target_bbox, img_meta)
        delta = max(int(frame_id) - int(previous['frame_id']), 1)
        target_motion = (target_geometry - previous['meta'][:4]) / float(delta)
        return dict(
            history_embeds=embeds,
            history_meta=metas,
            history_mask=mask,
            target_embed=target_embed.detach().cpu().float(),
            target_motion=target_motion.float(),
            quality_target=torch.tensor(float(quality), dtype=torch.float32),
            ready_frame=int(frame_id) + self.jepa_delay,
            source=False,
            created_frame=int(frame_id),
            target_score=float(target_score),
        )

    def _reservoir_add(self, sample):
        self._jepa_seen_target += 1
        if len(self._jepa_target_replay) < self.jepa_target_replay_size:
            self._jepa_target_replay.append(sample)
            return
        slot = self._jepa_rng.randrange(self._jepa_seen_target)
        if slot < self.jepa_target_replay_size:
            self._jepa_target_replay[slot] = sample

    def _load_source_replay(self):
        if self._jepa_source_replay_cache is not None:
            return self._jepa_source_replay_cache
        self._jepa_source_replay_cache = []
        if not self.jepa_source_replay:
            return self._jepa_source_replay_cache
        if not os.path.isfile(self.jepa_source_replay):
            raise IOError('source replay does not exist: {}'.format(
                self.jepa_source_replay))
        import numpy as np
        arrays = np.load(self.jepa_source_replay, allow_pickle=False)
        required = ('history_embeds', 'history_meta', 'history_mask',
                    'target_embed', 'target_motion')
        for key in required:
            if key not in arrays:
                raise KeyError('source replay misses {}'.format(key))
        count = int(arrays['target_embed'].shape[0])
        for index in range(count):
            quality = 1.0
            if 'quality_target' in arrays:
                quality = float(arrays['quality_target'][index])
            self._jepa_source_replay_cache.append({
                'history_embeds': torch.from_numpy(
                    arrays['history_embeds'][index]).float(),
                'history_meta': torch.from_numpy(
                    arrays['history_meta'][index]).float(),
                'history_mask': torch.from_numpy(
                    arrays['history_mask'][index]).float(),
                'target_embed': torch.from_numpy(
                    arrays['target_embed'][index]).float(),
                'target_motion': torch.from_numpy(
                    arrays['target_motion'][index]).float(),
                'quality_target': torch.tensor(
                    quality, dtype=torch.float32),
                'source': True,
                'created_frame': -1,
                'target_score': 1.0,
            })
        return self._jepa_source_replay_cache

    @staticmethod
    def _to_device_batch(samples, device):
        return dict(
            history_embeds=torch.stack(
                [sample['history_embeds'] for sample in samples]).to(device),
            history_meta=torch.stack(
                [sample['history_meta'] for sample in samples]).to(device),
            history_mask=torch.stack(
                [sample['history_mask'] for sample in samples]).to(device),
            target_embed=torch.stack(
                [sample['target_embed'] for sample in samples]).to(device),
            target_motion=torch.stack(
                [sample['target_motion'] for sample in samples]).to(device),
            quality_target=torch.stack(
                [sample['quality_target'] for sample in samples]).to(device))

    def _source_anchor_distillation(self, source_samples, device):
        """Keep online outputs close to the frozen source adapter function."""
        if (not source_samples or self.jepa_source_anchor_weight <= 0.0 or
                self._jepa_source_anchor is None):
            return torch.zeros((), device=device)
        batch = self._to_device_batch(source_samples, device)
        outputs = self.jepa(
            batch['history_embeds'], batch['history_meta'],
            batch['history_mask'])
        with torch.no_grad():
            anchor_outputs = self._jepa_source_anchor(
                batch['history_embeds'], batch['history_meta'],
                batch['history_mask'])
            anchor_latent = F.normalize(
                anchor_outputs['latent'], p=2, dim=-1, eps=1e-6)
            anchor_quality = anchor_outputs['quality']
            anchor_uncertainty = anchor_outputs['uncertainty'].clamp(
                min=1e-3, max=10.0)
            anchor_motion = anchor_outputs['motion']
        predicted = F.normalize(outputs['latent'], p=2, dim=-1, eps=1e-6)
        current_uncertainty = outputs['uncertainty'].clamp(
            min=1e-3, max=10.0)
        return (
            F.mse_loss(predicted, anchor_latent) +
            F.mse_loss(outputs['quality'], anchor_quality) +
            F.mse_loss(torch.log(current_uncertainty),
                       torch.log(anchor_uncertainty)) +
            F.smooth_l1_loss(outputs['motion'], anchor_motion))

    def _adapt_if_due(self, frame_id):
        if not self.jepa_online:
            return
        ready = []
        pending = []
        for sample in self._jepa_pending:
            if int(sample['ready_frame']) <= int(frame_id):
                ready.append(sample)
            else:
                pending.append(sample)
        self._jepa_pending = pending
        for sample in ready:
            if self.jepa_use_replay:
                self._reservoir_add(sample)
            self._jepa_ready_since_update.append(sample)
        if not self._jepa_ready_since_update:
            return
        if (self._jepa_last_adapt_frame >= 0 and
                int(frame_id) - self._jepa_last_adapt_frame <
                self.jepa_adaptation_interval):
            return

        current = list(self._jepa_ready_since_update)
        source_samples = []
        if self.jepa_use_replay:
            target_pool = list(self._jepa_target_replay)
            source_pool = list(self._load_source_replay())
            # Deterministic domain-balanced sampling.  Current delayed pairs
            # are included first, so every update can react to new evidence.
            target_samples = current[:self.jepa_batch_size]
            current_sample_ids = set(id(sample) for sample in target_samples)
            for sample in target_pool:
                if len(target_samples) >= self.jepa_batch_size:
                    break
                if id(sample) not in current_sample_ids:
                    target_samples.append(sample)
                    current_sample_ids.add(id(sample))
            # Cycle through the complete source pool rather than training on
            # the same prefix at every window.  The cycle is deterministic
            # under the run seed and gives the source anchor coverage.
            if source_pool:
                source_start = ((self._jepa_adapt_step *
                                 self.jepa_batch_size) % len(source_pool))
                source_samples = [
                    source_pool[(source_start + offset) % len(source_pool)]
                    for offset in range(min(self.jepa_batch_size,
                                            len(source_pool)))]
            samples = source_samples + target_samples
        else:
            samples = current[:self.jepa_batch_size]
        self._jepa_ready_since_update = []
        if not samples:
            return

        # The tracker is called from an outer torch.no_grad() context by
        # mmtrack.  Online learning intentionally re-enables gradients here.
        device = self._jepa_device
        self._ensure_device(device)
        self.jepa.train()
        loss_values = []
        update_ok = True
        with torch.enable_grad():
            for _ in range(self.jepa_updates_per_window):
                batch = self._to_device_batch_batch(samples, device)
                self._jepa_optimizer.zero_grad()
                losses = temporal_jepa_loss(
                    self.jepa, batch, use_teacher=self.jepa_use_teacher)
                source_anchor_loss = self._source_anchor_distillation(
                    source_samples, device)
                total = (losses['total'] +
                         self.jepa_source_anchor_weight * source_anchor_loss)
                if not bool(torch.isfinite(total).item()):
                    update_ok = False
                    break
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.jepa.parameters() if p.requires_grad],
                    max_norm=5.0)
                self._jepa_optimizer.step()
                if self.jepa_use_teacher:
                    self.jepa.update_teacher(self.jepa_ema)
                loss_values.append({
                    key: _as_float(value)
                    for key, value in losses.items()
                    if key != 'predictions'
                })
                loss_values[-1]['source_anchor'] = _as_float(
                    source_anchor_loss)
        self.jepa.eval()
        self._jepa_last_adapt_frame = int(frame_id)
        self._jepa_adapt_step += 1
        checkpoint_path = self._save_window_checkpoint(frame_id)
        pseudo_count = len(current)
        pseudo_conf = (sum(sample['quality_target'].item() for sample in current) /
                       max(pseudo_count, 1))
        event = dict(
            event='adaptation_window',
            window=self._jepa_adapt_step,
            frame=int(frame_id),
            sequence_index=self._jepa_reset_count,
            pseudo_count=pseudo_count,
            pseudo_confidence=float(pseudo_conf),
            loss=loss_values,
            update_ok=update_ok,
            checkpoint=checkpoint_path,
            source_buffer_size=len(self._load_source_replay()),
            target_buffer_size=len(self._jepa_target_replay),
            samples_used=len(samples),
            use_teacher=self.jepa_use_teacher,
            use_replay=self.jepa_use_replay,
            source_anchor_weight=self.jepa_source_anchor_weight,
            random_control=self.jepa_random_control,
            gt_used_online=False,
            future_used_online=False,
            window_stats=dict(self._jepa_window_stats),
        )
        self._write_event(event)
        self._jepa_window_stats = self._new_window_stats()

    # Kept as a separate method to make accidental changes to batching easy
    # to spot in code review and protocol tests.
    def _to_device_batch_batch(self, samples, device):
        return self._to_device_batch(samples, device)

    def _save_window_checkpoint(self, frame_id):
        if not self.jepa_checkpoint_dir:
            return None
        directory = os.path.abspath(self.jepa_checkpoint_dir)
        if not os.path.isdir(directory):
            os.makedirs(directory)
        base = os.path.join(
            directory, 'window_{:06d}.pth'.format(self._jepa_adapt_step))
        path = base
        suffix = 1
        while os.path.exists(path):
            path = base[:-4] + '_r{}.pth'.format(suffix)
            suffix += 1
        state = {
            key: value.detach().cpu()
            for key, value in self.jepa.state_dict().items()
        }
        payload = dict(
            state_dict=state,
            frame=int(frame_id),
            window=self._jepa_adapt_step,
            config=dict(
                embed_dim=self.jepa.embed_dim,
                history_len=self.jepa.history_len,
                meta_dim=self.jepa.meta_dim,
                latent_dim=self.jepa.latent_dim,
                use_teacher=self.jepa_use_teacher,
                use_replay=self.jepa_use_replay,
                source_anchor_weight=self.jepa_source_anchor_weight,
                frozen_representation=self.jepa_frozen_representation),
            provenance=dict(
                gt_used_online=False,
                future_used_online=False,
                target_source='delayed_high_confidence_association',
                detector_frozen=True,
                lifecycle_frozen=True),
        )
        if self._jepa_optimizer is not None:
            payload['optimizer_state_dict'] = self._jepa_optimizer.state_dict()
        torch.save(payload, path)
        return path

    def _queue_pseudo_pairs(self, ids, bboxes, embeds, base_scores, memo_ids,
                            frame_id, img_metas):
        if not self.jepa_online:
            return
        for index, track_id_tensor in enumerate(ids):
            track_id = int(track_id_tensor)
            if track_id < 0:
                continue
            records = self._jepa_history.get(track_id, [])
            if not records:
                continue
            if float(bboxes[index, -1]) < self.jepa_pseudo_det_thr:
                continue
            memo_matches = torch.nonzero(
                memo_ids == track_id, as_tuple=False).view(-1)
            if memo_matches.numel() == 0:
                continue
            native_conf = float(base_scores[index, memo_matches[0]])
            if native_conf < self.jepa_pseudo_match_thr:
                continue
            gap = int(frame_id) - int(records[-1]['frame_id'])
            if gap < 1 or gap > self.jepa_pseudo_max_gap + 1:
                continue
            quality = min(1.0, max(0.0, 0.5 * float(bboxes[index, -1]) +
                                   0.5 * native_conf))
            sample = self._sample_from_records(
                copy.deepcopy(records), embeds[index], bboxes[index],
                bboxes[index, -1], frame_id, img_metas[0], quality)
            if sample is not None:
                self._jepa_pending.append(sample)
                self._jepa_window_stats['pseudo_pairs'] += 1
                self._jepa_window_stats['pseudo_confidence_sum'] += quality

    def track(self, img_metas, feats, model, bboxes, labels, frame_id):
        """Run QD association and an optional causal JEPA score correction."""
        if not self.jepa_enabled:
            # Exact baseline path: no feature extraction, state mutation, or
            # numerical change is introduced when JEPA is disabled.
            return super(TemporalJEPATracker, self).track(
                img_metas, feats, model, bboxes, labels, frame_id)

        frame_start = time.time()
        self._adapt_if_due(frame_id)
        if bboxes.shape[0] == 0:
            ids = torch.zeros_like(labels)
            self._jepa_window_stats['frames'] += 1
            return bboxes, labels, ids

        self._ensure_device(bboxes.device)
        track_bboxes = bboxes[:, :-1] * torch.tensor(
            img_metas[0]['scale_factor']).to(bboxes.device)
        track_feats = model.track_head.extract_bbox_feats(
            feats, [track_bboxes])
        _, inds = bboxes[:, -1].sort(descending=True)
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        embeds = track_feats[inds, :]

        valids = bboxes.new_ones((bboxes.size(0)))
        ious = bbox_overlaps(bboxes[:, :-1], bboxes[:, :-1])
        for i in range(1, bboxes.size(0)):
            thr = (self.nms_backdrop_iou_thr
                   if bboxes[i, -1] < self.obj_score_thr
                   else self.nms_class_iou_thr)
            if (ious[i, :i] > thr).any():
                valids[i] = 0
        valids = valids == 1
        bboxes = bboxes[valids, :]
        labels = labels[valids]
        embeds = embeds[valids, :]

        ids = torch.full((bboxes.size(0), ), -1, dtype=torch.long)
        if bboxes.size(0) > 0 and not self.empty:
            (memo_bboxes, memo_labels, memo_embeds, memo_ids,
             memo_vs) = self.memo
            del memo_bboxes, memo_vs
            native_scores = self._native_scores(
                embeds, memo_embeds, labels, memo_labels, self.with_cats,
                self.match_metric)
            base_ids = self._assign_matches(
                native_scores, bboxes, memo_ids)
            jepa_scores, gate, prediction_info = self._gate_and_predict(
                bboxes, labels, embeds, frame_id, memo_ids, memo_embeds,
                native_scores, img_metas)
            del prediction_info
            combined_scores = native_scores + self.jepa_lambda * jepa_scores
            ids = self._assign_matches(combined_scores, bboxes, memo_ids)
            ids = self._new_track_ids(ids, bboxes)
            self._jepa_window_stats['candidate_pairs'] += int(
                gate.numel())
            self._jepa_window_stats['gated_pairs'] += int(gate.sum())
            self._jepa_window_stats['base_matches'] += int(
                (base_ids > -1).sum())
            self._jepa_window_stats['final_matches'] += int(
                (ids > -1).sum())
            self._jepa_window_stats['association_changes'] += int(
                ((base_ids > -1) & (ids != base_ids)).sum())
            self._queue_pseudo_pairs(
                ids, bboxes, embeds, native_scores, memo_ids, frame_id,
                img_metas)
        else:
            ids = self._new_track_ids(ids, bboxes)

        # Current observations are appended only after prediction, matching,
        # and pseudo-pair creation.  Thus they cannot affect their own JEPA
        # prediction or gate.
        self.update(ids, bboxes, embeds, labels, frame_id)
        for index, track_id_tensor in enumerate(ids):
            track_id = int(track_id_tensor)
            if track_id > -1:
                self._append_observation(
                    track_id, bboxes[index], embeds[index], frame_id,
                    img_metas[0])
        active = set(int(track_id) for track_id in self.tracks.keys())
        self._jepa_history = {
            track_id: records
            for track_id, records in self._jepa_history.items()
            if track_id in active
        }
        self._jepa_window_stats['frames'] += 1
        self._jepa_window_stats['association_elapsed_ms'] += (
            time.time() - frame_start) * 1000.0
        return bboxes, labels, ids


__all__ = ['TemporalJEPATracker']
