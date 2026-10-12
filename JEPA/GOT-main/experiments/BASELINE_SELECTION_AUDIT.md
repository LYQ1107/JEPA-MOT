# Baseline selection audit

## Decision

Use the local HybridSORT **OC-SORT public-detection adapter** as the primary
baseline. It is a mature online tracker with Kalman prediction, observation-
centric velocity association and re-association, and its `update_public`
interface consumes the same fixed detector rows that the JEPA adapter will
consume. The detector and lifecycle therefore remain outside the JEPA change.

Source: `/data1/LWR/vranlee/SERVER_ONLY/avis/masa/HybridSORT/trackers/ocsort_tracker/ocsort.py`, SHA-256
`6d0875ef82540483cd48c383065b8181bb18b8d59d33b42db3510eb88e33e401` at the
formal run. Environment: `HybridSORT` (Python 3.8.0, NumPy 1.23.4, SciPy
1.10.1, OpenCV 3.4.18).

## Cross-control

The local canonical ByteTrack implementation was run with the same MOT17
public detections, sequence prefix and thresholds. On five sequences × 120
frames, official TrackEval gave:

| tracker | HOTA mean | HOTA@0.5 | AssA@0.5 | DetA@0.5 | IDF1 | MOTA | IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|
| OC-SORT | .600354 | .647286 | .829144 | .505315 | .656208 | .503406 | 16 |
| ByteTrack | .574768 | .623742 | .793535 | .490280 | .615385 | .480702 | 106 |

OC-SORT is retained; ByteTrack remains a reproducibility control, not a
competing tuned result.

## Fairness checks

- Both trackers read the same `det/det.txt` rows and frame range; no GT box or
  track ID is passed to the tracker.
- TrackEval staging uses one GT copy and one `seqinfo.ini` per sequence, with
  all rows truncated to 120 frames. Duplicate `(frame,id)` rows are removed
  only for the OC-SORT adapter's initialization padding.
- The tracker never reads evaluation GT. GT is used only by TrackEval after
  output files are complete.
- `DetA` can change only if a tracker changes lifecycle/emission; the JEPA
  implementation must report this explicitly and must not alter detector rows.

## Scope limitation

This is a five-sequence, 120-frame sanity evaluation, not a full MOT17
leaderboard score. The next experiment adds JEPA in a new script and directory
without editing HybridSORT production files.
