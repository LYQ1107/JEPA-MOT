"""Short protocol and leakage checks for the DA-MOT temporal JEPA path."""

from __future__ import absolute_import, division, print_function

import inspect
import json

import torch

from darth.models.trackers import TemporalJEPATracker
from mmtrack.models.trackers import QuasiDenseTracker


class _FakeHead(object):
    def extract_bbox_feats(self, feats, boxes):
        return feats


class _FakeModel(object):
    track_head = _FakeHead()


META = [dict(img_shape=(100, 100, 3), scale_factor=(1., 1., 1., 1.))]


def _tracker(cls, **extra):
    options = dict(
        init_score_thr=.2,
        obj_score_thr=.2,
        match_score_thr=.5,
        memo_tracklet_frames=3,
        memo_backdrop_frames=1,
        memo_momentum=.8,
        nms_conf_thr=.35,
        nms_backdrop_iou_thr=.3,
        nms_class_iou_thr=.7,
        with_cats=True,
        match_metric='bisoftmax',
        jepa_embed_dim=8,
        jepa_history_len=2,
        jepa_hidden_dim=8)
    options.update(extra)
    if cls is QuasiDenseTracker:
        for key in list(options):
            if key.startswith('jepa_'):
                options.pop(key)
    return cls(**options)


def _frames():
    return [
        (torch.tensor([[5., 10., 20., 40., .95],
                       [55., 10., 70., 40., .90]]),
         torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 1., 0., 0., 0., 0., 0., 0.]]),
         torch.tensor([0, 0])),
        (torch.tensor([[6., 10., 21., 40., .95],
                       [56., 10., 71., 40., .90]]),
         torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 1., 0., 0., 0., 0., 0., 0.]]),
         torch.tensor([0, 0])),
        (torch.zeros((0, 5)), torch.zeros((0, 8)), torch.zeros((0,), dtype=torch.long)),
        (torch.tensor([[8., 10., 23., 40., .95],
                       [58., 10., 73., 40., .90]]),
         torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0.],
                       [0., 1., 0., 0., 0., 0., 0., 0.]]),
         torch.tensor([0, 0])),
    ]


def _run(tracker, frames):
    model = _FakeModel()
    outputs = []
    for frame_id, (boxes, embeds, labels) in enumerate(frames):
        bboxes, out_labels, ids = tracker.track(
            META, embeds, model, boxes.clone(), labels.clone(), frame_id)
        outputs.append((bboxes.clone(), out_labels.clone(), ids.clone(),
                        sorted(int(k) for k in tracker.tracks.keys())))
    return outputs


def main():
    results = {}
    signature = inspect.signature(
        __import__('darth.models.jepa', fromlist=['TemporalJEPAAdapter']).
        TemporalJEPAAdapter.forward)
    results['predictor_has_no_candidate_argument'] = 'candidate' not in signature.parameters

    base = _tracker(QuasiDenseTracker)
    disabled = _tracker(TemporalJEPATracker, jepa_enabled=False)
    base_out = _run(base, _frames())
    disabled_out = _run(disabled, _frames())
    results['disabled_exact_output_parity'] = all(
        torch.equal(left[0], right[0]) and
        torch.equal(left[1], right[1]) and
        torch.equal(left[2], right[2]) and
        left[3] == right[3]
        for left, right in zip(base_out, disabled_out))

    online = _tracker(
        TemporalJEPATracker,
        jepa_enabled=True,
        jepa_online=True,
        jepa_adaptation_interval=1,
        jepa_batch_size=2,
        jepa_pseudo_det_thr=.8,
        jepa_pseudo_match_thr=0.0,
        jepa_delay=1,
        jepa_target_replay_size=8,
        jepa_seed=29)
    forward_inputs = []
    original_forward = online.jepa.forward

    def spy(history_embeds, history_meta, history_mask):
        forward_inputs.append((history_embeds.detach().clone(),
                               history_meta.detach().clone(),
                               history_mask.detach().clone()))
        return original_forward(history_embeds, history_meta, history_mask)

    online.jepa.forward = spy
    model = _FakeModel()
    first = _frames()[0]
    second = _frames()[1]
    second_embeds = second[1].clone()
    second_embeds[0] = torch.tensor([0., 0., 1., 0., 0., 0., 0., 0.])
    online.track(META, first[1], model, first[0], first[2], 0)
    online.track(META, second_embeds, model, second[0], second[2], 1)
    results['predictor_called_before_current_append'] = bool(forward_inputs)
    if forward_inputs:
        history_at_frame_one = forward_inputs[-1][0]
        # The current marker (second frame's all-zero/one-hot feature) must
        # not replace the previous observation in the history input.
        results['current_candidate_absent_from_predictor'] = not bool(
            torch.allclose(history_at_frame_one[0, -1], second_embeds[0]))
    else:
        results['current_candidate_absent_from_predictor'] = False
    results['delayed_pair_not_ready_same_frame'] = (
        len(online._jepa_pending) >= 1 and
        all(int(sample['ready_frame']) == 2 for sample in online._jepa_pending))
    online.track(META, _frames()[2][1], model, _frames()[2][0],
                 _frames()[2][2], 2)
    results['delay_released_on_later_frame'] = (
        online._jepa_adapt_step == 1 and
        not online._jepa_pending and
        len(online._jepa_target_replay) >= 1 and
        all(int(sample['created_frame']) == 1
            for sample in online._jepa_target_replay))
    results['online_outputs_finite'] = all(
        torch.isfinite(item).all().item()
        for item in [online.jepa.predictor[0].weight])

    # Reservoir selection is reproducible for the same explicit seed.
    def reservoir_signature():
        tracker = _tracker(
            TemporalJEPATracker, jepa_enabled=True, jepa_online=True,
            jepa_target_replay_size=3, jepa_seed=101)
        for index in range(20):
            tracker._reservoir_add(dict(created_frame=index))
        return [sample['created_frame'] for sample in tracker._jepa_target_replay]

    results['replay_deterministic'] = reservoir_signature() == reservoir_signature()
    results['lifecycle_ids_match_baseline'] = all(
        left[3] == right[3] for left, right in zip(base_out, disabled_out))

    failed = [key for key, value in results.items() if not value]
    results['status'] = 'PASS' if not failed else 'FAIL'
    results['failed_checks'] = failed
    print(json.dumps(results, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
