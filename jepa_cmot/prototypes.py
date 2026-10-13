"""Frozen past-only mean and lifetime EMA controls; no trainable identity head."""
import numpy as np

from .causal_tracker import CausalHistoryTracker


def unit(value):
    value = np.asarray(value, np.float32)
    return value / max(float(np.linalg.norm(value)), 1e-8)


def ema_step(previous, observation, beta=.25):
    observation = unit(observation)
    return observation.copy() if previous is None else unit((1-beta)*previous+beta*observation)


class PrototypeHistoryTracker(CausalHistoryTracker):
    def __init__(self, args, learner=None, frame_rate=30):
        if learner is not None:
            raise ValueError('Prototype comparison has no learned identity head')
        self._ema = {}
        self._ema_frames = {}
        super().__init__(args, learner, frame_rate)

    def _commit_feature(self, track, det, allow=True):
        super()._commit_feature(track, det, allow)
        tid = int(track.track_id)
        if not allow or getattr(det, 'jepa_feature', None) is None:
            return
        if self._ema_frames.get(tid) == self.frame_id:
            return
        self._ema[tid] = ema_step(self._ema.get(tid), det.jepa_feature)
        self._ema_frames[tid] = self.frame_id

    def prepare_queries(self, width, height):
        already_prepared = self._prepared_frame == self.frame_id+1
        super().prepare_queries(width, height)
        if already_prepared:
            return
        for tid in self._queries:
            if self.args.query_aggregation == 'mean4':
                self._queries[tid] = unit(self._packed[tid][0].sum(0))
            elif self.args.query_aggregation == 'ema':
                self._queries[tid] = self._ema[tid].copy()
            else:
                raise ValueError('Unknown frozen prototype aggregation')

    def update(self, *args, **kwargs):
        result = super().update(*args, **kwargs)
        alive = set(self._observations)
        self._ema = {tid: value for tid, value in self._ema.items() if tid in alive}
        self._ema_frames = {tid: value for tid, value in self._ema_frames.items() if tid in alive}
        return result
