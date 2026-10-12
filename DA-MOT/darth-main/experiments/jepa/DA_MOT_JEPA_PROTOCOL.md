# DA-MOT track-level JEPA integration

Status: implementation and the controlled full-target runs are complete.  The
implementation passes the causal/leakage/parity checks.  The full result is
reported as provisional/mixed: association metrics improve modestly, while
DetA/FP/FN/Frag move slightly in the opposite direction and there is no
multi-domain-order study yet.

## Scientific boundary

The controlled baseline is the existing source-clear QDTrack checkpoint with
the installed `mmtrack` `QuasiDenseTracker`.  The JEPA tracker keeps the same
detector, track-head feature extraction, NMS, greedy matching threshold, ID
allocation, and `QuasiDenseTracker.update` lifecycle.  It adds only a gated
score term for selected track/memo pairs.

```text
current image
    -> frozen Faster R-CNN detector
    -> QDTrack head: candidate boxes, labels, appearance features
    -> upstream QD NMS
                         +-------------------------------+
past observations H<t -->| student context + predictor   |--> predicted latent,
                         | (no current candidate input)  |    quality, uncertainty
                         +-------------------------------+
candidate feature ------> frozen/EMA target encoder       |  (after prediction)
                                      |                    v
                         native QD score + gated lambda * JEPA score
                                      |
                         greedy association -> inherited update/lifecycle
                                      |
              high-confidence assigned pair -> delay >= 1 -> replay
                                      |
                         adapter-only update + EMA target teacher
                         + frozen-source output distillation
```

`misses > 0` is never called an occlusion label.  The adapter receives a
missing-gap and visibility proxy computed from observed track timestamps;
these are observable proxies that can also represent a detector miss, blur,
fast motion, or leaving the image.  There is no explicit GT visibility score
in this first implementation.

The predictor API is intentionally only
`forward(history_embeds, history_meta, history_mask)`.  A current candidate is
encoded separately at association time and cannot affect its own prediction.
Online events and adapter checkpoints include `gt_used_online=false` and
`future_used_online=false`.  Ground-truth labels and later observations are
used only for offline source-cache/training/replay construction (and the
source-only retention audit); the precomputed source replay is the permitted
source-domain input to online updates.  Target annotations are evaluator-only
and are never read by target tracking.

## Implemented files

- `darth/models/jepa/temporal_adapter.py`: student predictor, frozen/EMA
  target encoder, quality/uncertainty/motion heads, masked JEPA latent loss,
  hard-negative ranking, covariance and source-replay-compatible losses.
- `darth/models/trackers/temporal_jepa_tracker.py`: registered real QDTrack
  tracker, causal history, gate, delayed pseudo-pairs, deterministic
  domain-balanced reservoir, adapter-only online updates, JSONL/checkpoint
  provenance.
- `tools/jepa/build_track_feature_cache.py`: source-labelled real track-head
  feature cache; never used by target inference.
- `tools/jepa/train_temporal_adapter.py`: sequence-disjoint source training,
  mask/drop/noise/identity-mix corruption, and source replay export.
- `tools/jepa/validate_protocol.py`: parity, leakage, delay, finite-output,
  lifecycle and replay checks.
- `tools/jepa/evaluate_adapter.py`: source-replay retention/forgetting audit;
  it never reads target annotations.
- `tools/jepa/make_cocovid_prefix.py`: non-destructive evaluator-only prefix
  selection.
- `configs/jepa/temporal_jepa_bdd_sample.py` and
  `configs/jepa/temporal_jepa_bdd_night.py`: reproducible templates.

## Reproduction commands

Use the DA-MOT environment and run from `darth-main`:

```bash
export PYTHONPATH=/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main
export PYTHON=/home/lwr/anaconda3/envs/DARTH/bin/python

$PYTHON tools/jepa/validate_protocol.py

$PYTHON tools/jepa/make_cocovid_prefix.py \
  data/bdd_da/annotations/box_track_20/box_track_night_all_cocofmt.json \
  --output /data2/usr_for_deadline/damot_jepa_20260831/target_night_prefix_3x30.json \
  --max-videos 3 --max-frames 30

$PYTHON tools/jepa/build_track_feature_cache.py \
  configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear.py \
  --checkpoint work_dirs/bdd_4domain/source/clear_20e/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear/epoch_20.pth \
  --output /data2/usr_for_deadline/damot_jepa_20260831/source_sample/source_track_features_v1.npz \
  --device cuda:0 --max-videos 2 --max-frames-per-video 6

$PYTHON tools/jepa/train_temporal_adapter.py \
  /data2/usr_for_deadline/damot_jepa_20260831/source_sample/source_track_features_v1.npz \
  --output /data2/usr_for_deadline/damot_jepa_20260831/source_sample/temporal_jepa_source_v2.pth \
  --replay-output /data2/usr_for_deadline/damot_jepa_20260831/source_sample/source_replay_v2.npz \
  --device cuda:0 --history-len 4 --hidden-dim 128 --epochs 5 \
  --batch-size 32 --learning-rate 1e-3

# Full source-only baseline.
$PYTHON tools/run/test.py \
  configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear.py \
  --checkpoint work_dirs/bdd_4domain/source/clear_20e/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear/epoch_20.pth \
  --eval bbox track --work-dir work_dirs/jepa_damot_20260831 \
  --exp-name baseline_night_full --version v1 \
  --cfg-options data.test.ann_file=data/bdd_da/annotations/box_track_20/box_track_night_all_cocofmt.json \
    data.test.img_prefix=/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_night/ \
    data.workers_per_gpu=0

# Corrected full continual seed 17.  For seed 23, substitute the seed-23
# adapter/replay and unique output paths, and add model.tracker.jepa_seed=23.
$PYTHON tools/run/test.py configs/jepa/temporal_jepa_bdd_night.py \
  --checkpoint work_dirs/bdd_4domain/source/clear_20e/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear/epoch_20.pth \
  --eval bbox track --work-dir work_dirs/jepa_damot_20260831 \
  --exp-name continual_night_full_anchor --version v1 \
  --cfg-options data.workers_per_gpu=0 \
    model.tracker.jepa_checkpoint=/data2/usr_for_deadline/damot_jepa_20260831/source_sample/temporal_jepa_source_v2.pth \
    model.tracker.jepa_online=True model.tracker.jepa_adaptation_interval=10 \
    model.tracker.jepa_batch_size=8 model.tracker.jepa_delay=1 \
    model.tracker.jepa_source_replay=/data2/usr_for_deadline/damot_jepa_20260831/source_sample/source_replay_v2.npz \
    model.tracker.jepa_source_anchor_weight=1.0 \
    model.tracker.jepa_log_path=/data2/usr_for_deadline/damot_jepa_20260831/night_full/continual_anchor_v1/events.jsonl \
    model.tracker.jepa_checkpoint_dir=/data2/usr_for_deadline/damot_jepa_20260831/night_full/continual_anchor_v1/checkpoints

# Source retention audit (run once before and once after continual tracking).
$PYTHON tools/jepa/evaluate_adapter.py \
  --adapter /data2/usr_for_deadline/damot_jepa_20260831/source_sample/temporal_jepa_source_v2.pth \
  --replay /data2/usr_for_deadline/damot_jepa_20260831/source_sample/source_replay_v2.npz \
  --device cpu --batch-size 198
```

All tracking rows below use the same source checkpoint and target prefix.  A
unique `--work-dir/--exp-name/--version` must be used for every row.  The
continual row additionally sets `jepa_online=True`, `jepa_delay=1`,
`jepa_source_replay=.../source_replay_v2.npz`, and unique
`jepa_log_path`/`jepa_checkpoint_dir` values.  The no-teacher and no-replay
rows change only `jepa_use_teacher=False` and `jepa_use_replay=False`,
respectively.  The random row sets `jepa_random_control=True` and
`jepa_online=False`.

## Prefix matrix (3 sequences × 30 frames)

Metrics are the existing DA-MOT CLEAR/TrackEval outputs.  `Frag` is the
reporting name for the evaluator's `FM` field.  The raw tracked-detection
count is fixed across the full baseline and continual outputs; official FP/FN
can still differ by one because the evaluator's frame-level matching is
affected by association/track output ordering.

| row | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN | Frag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A source-only QDTrack | 34.456 | 26.294 | 46.889 | 33.411 | 6.535 | 39 | 192 | 527 | 40 |
| B frozen last-observation representation control | 35.504 | 26.106 | 49.586 | 35.431 | 7.891 | 28 | 192 | 527 | 43 |
| C offline temporal JEPA, frozen at target | 35.310 | 26.175 | 49.174 | 35.742 | 6.782 | 37 | 192 | 527 | 43 |
| D continual JEPA + teacher + replay | 35.310 | 26.175 | 49.174 | 35.742 | 6.782 | 37 | 192 | 527 | 43 |
| E continual without teacher | 35.151 | 26.142 | 48.799 | 35.120 | 6.535 | 39 | 192 | 527 | 43 |
| F continual without replay | 35.310 | 26.175 | 49.174 | 35.742 | 6.782 | 37 | 192 | 527 | 43 |
| G random control | 35.170 | 26.288 | 48.636 | 35.431 | 6.782 | 37 | 192 | 527 | 43 |

These rows do not establish a final improvement: they are one seed on a short
prefix, and the random/control rows are close enough that the temporal
mechanism is not isolated yet.  The useful result at this stage is that the
real association path, delayed online protocol, and metrics all execute
without target GT input.  The full target run is kept separate from this
prefix table.

The D/E/F rows above were rerun with the corrected source-anchor mechanism;
B/C/G are controls whose behavior is unchanged by that correction.  D's
corrected seed-17 result is reproduced with a different source-training and
online seed below.

## Full night target

The full target contains 36 sequences and 7286 frames.  The baseline and
corrected continual row use the same detector checkpoint, input order, QDTrack
head, NMS, and lifecycle.  The pre-fix continual row is retained as failure
evidence because it exposed source-anchor drift.

| row | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN | Frag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A source-only QDTrack | 22.197 | 16.367 | 30.646 | 23.671 | 5.248 | 2284 | 12022 | 61887 | 2772 |
| D continual, pre-fix (no source-anchor distillation) | 22.675 | 16.355 | 31.955 | 24.245 | 5.607 | 1993 | 12023 | 61888 | 2776 |
| D continual, corrected source replay + anchor | 22.613 | 16.350 | 31.798 | 24.194 | 5.596 | 2002 | 12023 | 61888 | 2777 |

For the corrected full row, delta from A is HOTA `+0.416`, DetA `-0.018`,
AssA `+1.153`, IDF1 `+0.523`, MOTA `+0.348`, IDSW `-282`, FP `+1`, FN `+1`,
and Frag `+5`.  This supports a modest association signal, not a clean
end-to-end win.  The pre-fix row is not promoted despite its slightly higher
HOTA because its source retention failed.

## Runtime and memory

- Full source-only inference: 642 seconds for 7286 frames, approximately
  11.35 FPS.
- Corrected full continual seed 17: 764 seconds, approximately 9.54 FPS;
  seed 23: 755 seconds, approximately 9.65 FPS.  The reported wall times
  include normal tracking and online update work; official metric accumulation
  is reported separately by the evaluator.
- Each corrected full run completed 617 adaptation windows, with 5856/5857
  accepted delayed pseudo-pairs, mean confidence about 0.936, a 198-sample
  source pool, and a target reservoir capped at 512 samples.  The sum of
  per-window association telemetry was 214.5 seconds for seed 17 and 209.8
  seconds for seed 23.
- Source cache extraction recorded a peak CUDA allocation of 653,413,376
  bytes.  The original `tools/run/test.py` did not record
  `torch.cuda.max_memory_allocated()` for full tracking, so no unsupported
  full-run peak-memory number is claimed here.

## Seed check and source retention

The corrected continual row was run on both the three-sequence prefix and the
full target with two independent source-training/online seeds.

| corrected D run | HOTA | IDF1 | MOTA | IDSW | FP | FN | Frag |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefix, seed 17 | 35.310 | 35.742 | 6.782 | 37 | 192 | 527 | 43 |
| prefix, seed 23 | 34.945 | 34.654 | 6.412 | 40 | 192 | 527 | 43 |
| full, seed 17 | 22.613 | 24.194 | 5.596 | 2002 | 12023 | 61888 | 2777 |
| full, seed 23 | 22.578 | 24.187 | 5.574 | 2020 | 12023 | 61888 | 2777 |

The source-replay anchor loss (198 source temporal samples; lower is better
for this fixed objective) was recomputed with one batch of all 198 samples for
every state.  This fixed batch size matters because the ranking term is
batch-dependent.

| adapter state | total loss | uncertainty component |
|---|---:|---:|
| source adapter v2 before online run | -0.1243 | -1.9434 |
| source adapter seed-23 before online run | -0.1106 | -1.8365 |
| pre-fix full continual final | 2.7487 | 26.9213 |
| corrected prefix final | -0.1303 | -1.9497 |
| corrected full continual final | -0.1737 | -2.1611 |
| corrected full seed-23 final | -0.1495 | -1.9588 |

The pre-fix failure came from repeatedly sampling the same source prefix and
allowing uncertainty to become poorly calibrated on other source positions.
The corrected tracker cycles over the source pool and distills the frozen
source function on each online update.  The source replay evaluation uses
source-labelled targets by design; target GT is not used in this audit.

## Provenance and audit

- Source checkpoint SHA256:
  `be78a6f608d11b86515c1248134a9590a6f44f1f26cb7bc256d5ffb0de614207`.
- Baseline config SHA256:
  `99de178b1d727c69d8911ddaec942ad35d20454ed6ec31efc5833f2b32b49d49`.
- Source cache v1: 253 labelled observations, 256-D QDTrack features, two
  source sample videos and 12 frames; peak CUDA allocation 653,413,376 bytes.
- Source adapter v2 SHA256:
  `2e83908b6e6ad8dfdf9961db838f037c2c4e5a3683c649af51b8cdad4d83ff2c`.
- Source adapter seed-23 SHA256:
  `57352c7f2e4457122052a9de9bd53f98d06bb2224cd7bfc1775b440219d851f8`.
- Source replay v2 SHA256:
  `2cdf669403d603b464cdbc87017313236813d20cfea4741c787252bfd5f99801`.
- Corrected full seed-17 final checkpoint SHA256:
  `97bf25f14760d929c24e8c97b8548683e8098ee3cf97df06127ac52dc1391fb7`.
- Corrected full seed-23 final checkpoint SHA256:
  `eaff17f32295e49f5b67c06c25e51f75a8c4b77129941411eff5bf0b7fd09ade`.
- Target prefix annotation SHA256:
  `5c6946ad8c1805bfc0ddf0f356c2d64346ad2c020774751e13c5a31406960c5d`.
- Final tracker implementation SHA256:
  `f79ae80160f6401f28f153b21ea554dd32b3b0a0a7343b6b8e7ea007f0d75559`.
- Protocol validator SHA256:
  `42923a62c16b3a14bd730c4f9bc2097a152da7b25a9fa67bd90ae9db27e2ef23`.
- Prefix annotation is a new evaluator subset; its target GT is not loaded by
  the test pipeline.  Target annotations are read only by `dataset.evaluate`
  after inference.
- The current DARTH target-adaptation checkpoints are not in the controlled
  no-GT table: their training configs load target annotations and the local
  DARTH adapter consumes GT-derived training information.  They are historical
  domain-adaptation evidence, not a valid no-target-GT comparator here.

The first JEPA real-model smoke exposed a candidate/memo shape mismatch when
backdrop columns were present; the fix maps JEPA scores back to the full memo
and leaves backdrop columns zero.  The failed run was kept in the prior
`frozen_smoke/v1` log directory.  A first protocol-script failure was only a
test fixture passing JEPA kwargs to the upstream tracker; it was corrected and
`validate_protocol.py` now reports `PASS`.

## Limitations and next audit

The evidence is still bounded: one source sample cache, one target domain, and
a 3×30 prefix for the ablations.  The two-seed full check is directionally
consistent for HOTA, AssA, IDF1, MOTA, and IDSW, with a small DetA/FP/FN/Frag
trade-off.  Domain-order robustness and cross-target generalization remain
untested.  Therefore the implementation/protocol is GO for continued
research, but the scientific claim is provisional rather than a final
improvement claim.
