"""GeneralTrack adapter with queries frozen before any current-row assignment."""
from collections import deque

import numpy as np
import torch

from yolox.tracker.byte_tracker_bdd_jepa import BYTETracker
from yolox.tracker import matching_bdd as matching
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

    def _jepa_duplicate(self,det,tracks):
        if not getattr(self.args,'geometry_duplicates',False):
            return super()._jepa_duplicate(det,tracks)
        # The historical geometry-null baseline removes duplicate births
        # after native assignment. Apply that same box-only rule to every
        # learned objective when this common foundation is requested.
        current={int(t.track_id):t for t in list(self.tracked_stracks)+list(tracks)
            if int(getattr(t,'frame_id',-1))==self.frame_id and
            matching.sig(t.cls)==matching.sig(det.cls)}
        for track in current.values():
            self.jepa_stats['duplicate_candidates']+=1
            if float(1-matching.iou_distance([track],[det])[0,0])>=.2:
                self.jepa_stats['duplicate_suppressed']+=1
                return True
        return False

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
                query=self._queries[track_id].copy(),
                positive=np.asarray(feature,np.float32).copy(),evidence=evidence,
                frame=int(self.frame_id),score=float(det.score),
                candidates=self._current_candidates.copy())
            self._accepted[track_id]['positive_index']=int(np.argmin(
                np.linalg.norm(self._current_boxes-np.asarray(det.tlbr)[None],axis=1)))

    def _jepa_similarity(self,tracks,detections):
        similarity=super()._jepa_similarity(tracks,detections)
        groups=np.asarray([[matching.sig(t.cls)==matching.sig(d.cls) for d in detections]
                           for t in tracks],dtype=bool).reshape(similarity.shape)
        similarity=np.where(groups,similarity,0.)
        calibration=getattr(self.args,'source_calibration',None)
        if calibration is not None:
            valid=np.zeros_like(similarity,dtype=bool)
            for i,track in enumerate(tracks):
                for j,det in enumerate(detections):
                    valid[i,j]=(groups[i,j] and self._track_query(track) is not None and
                                getattr(det,'jepa_feature',None) is not None)
            logits=np.clip(calibration['slope']*(similarity-calibration['intercept']),-60,60)
            similarity=np.where(valid,1/(1+np.exp(-logits)),0.)
        constant=getattr(self.args,'semantic_constant',None)
        if constant is not None:
            # Isolate score-scale effects with the same missing-feature and
            # category eligibility as the real affinity. No identity signal.
            valid=np.asarray([[groups[i,j] and self._track_query(track) is not None and
                getattr(det,'jepa_feature',None) is not None for j,det in enumerate(detections)]
                for i,track in enumerate(tracks)],dtype=bool).reshape(similarity.shape)
            similarity=np.where(valid,float(constant),0.).astype(np.float32)
        return similarity

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
            selected_rows=list(rows)
            if getattr(self.args,'history_control','full')=='last_only':selected_rows=selected_rows[-1:]
            self._packed[track_id]=pack_history(selected_rows,current,self.jepa_history,
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
        self._current_boxes=np.asarray(values[:,:4],np.float32)
        result=super().update(output_results,img_info,img_size,imgs,jepa_maps)
        self.causal_stats['frames']+=1
        self.causal_stats['empty_detection_frames']+=int(len(values)==0)
        if self.learner is not None:
            with torch.enable_grad():
                self.learner.after_assignment(self._accepted,current)
        alive={int(t.track_id) for t in self.tracked_stracks+self.lost_stracks}
        self._observations={i:h for i,h in self._observations.items() if i in alive}
        return result
