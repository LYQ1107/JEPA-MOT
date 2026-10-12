"""GeneralTrack adapter with queries frozen before any current-row assignment."""
from collections import deque

import numpy as np
import torch

from yolox.tracker.byte_tracker_bdd_jepa import BYTETracker
from .history import pack_history


class CausalHistoryTracker(BYTETracker):
    def __init__(self,args,learner=None,frame_rate=30):
        super().__init__(args,frame_rate)
        self.learner=learner
        self._queries={};self._packed={};self._observations={};self._query_frames={}
        self._native_evidence={};self._accepted={}
        self.causal_stats=dict(frames=0,empty_detection_frames=0,history_queries=0,
                               current_history_violations=0)

    def _track_query(self,track):
        # New births have no prior query. Never fall back to a feature just
        # committed in this frame, even when legacy birth/duplicate code asks.
        track_id=int(track.track_id)
        if track_id in self._queries and self._query_frames[track_id]>=self.frame_id:
            self.causal_stats['current_history_violations']+=1
            raise ValueError('A current observation entered the frozen query')
        return self._queries.get(track_id)

    def _commit_feature(self,track,det,allow=True):
        super()._commit_feature(track,det,allow)
        feature=getattr(det,'jepa_feature',None)
        if not allow or feature is None:
            return
        track_id=int(track.track_id)
        rows=self._observations.setdefault(track_id,deque(maxlen=self.jepa_history))
        if rows and rows[-1]['frame']==self.frame_id:
            return
        rows.append(dict(feature=np.asarray(feature,np.float32).copy(),
            box=np.asarray(det.tlbr,np.float32).copy(),frame=int(self.frame_id),
            birth=int(track.start_frame),confidence=float(det.score)))
        evidence=self._native_evidence.get((track_id,id(det)))
        if evidence is not None and track_id in self._packed:
            self._accepted[track_id]=dict(packed=self._packed[track_id],
                positive=np.asarray(feature,np.float32).copy(),evidence=evidence,
                frame=int(self.frame_id),score=float(det.score),
                candidates=self._current_candidates.copy())

    def _fuse_jepa(self,base_cost,tracks,detections,branch):
        # Reliability is measured with the original matcher, independently of
        # the learned query. Both row and column ambiguity are retained.
        cost=np.asarray(base_cost)
        for i,track in enumerate(tracks):
            for j,det in enumerate(detections):
                row=np.sort(cost[i]);column=np.sort(cost[:,j])
                row_margin=float(row[1]-row[0]) if len(row)>1 else 1.
                col_margin=float(column[1]-column[0]) if len(column)>1 else 1.
                self._native_evidence[(int(track.track_id),id(det))]=dict(
                    cost=float(cost[i,j]),row_top1=int(np.argmin(cost[i]))==j,
                    column_top1=int(np.argmin(cost[:,j]))==i,
                    row_margin=row_margin,column_margin=col_margin,branch=branch)
        return super()._fuse_jepa(base_cost,tracks,detections,branch)

    def update(self,output_results,img_info,img_size,imgs,jepa_maps=None):
        current=self.frame_id+1
        self._queries={};self._packed={};self._native_evidence={};self._accepted={};self._query_frames={}
        width,height=float(img_info[1]),float(img_info[0])
        existing=list(self.tracked_stracks)+list(self.lost_stracks)
        for track in existing:
            track_id=int(track.track_id);rows=self._observations.get(track_id)
            if not rows:continue
            self._query_frames[track_id]=int(rows[-1]['frame'])
            self._packed[track_id]=pack_history(rows,current,self.jepa_history,
                                               self.feature_dim,width,height)
        if self._packed:
            ids=list(self._packed)
            packed=[np.stack([self._packed[i][k] for i in ids]) for k in range(3)]
            if self.learner is None:
                queries=packed[0][:,-1]
            else:
                queries=self.learner.predict(*packed)
            self._queries={i:q.copy() for i,q in zip(ids,queries)}
            self.causal_stats['history_queries']+=len(ids)
        values=output_results.detach().cpu().numpy() if torch.is_tensor(output_results) else output_results
        self._current_candidates=np.asarray(values[:,6:6+self.feature_dim],np.float32)
        result=super().update(output_results,img_info,img_size,imgs,jepa_maps)
        self.causal_stats['frames']+=1
        self.causal_stats['empty_detection_frames']+=int(len(values)==0)
        if self.learner is not None:
            with torch.enable_grad():
                self.learner.after_assignment(self._accepted,current)
        alive={int(t.track_id) for t in self.tracked_stracks+self.lost_stracks}
        self._observations={i:h for i,h in self._observations.items() if i in alive}
        return result
