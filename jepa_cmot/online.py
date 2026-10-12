"""A shared delayed pseudo-pair learner for matched continual objectives."""
from dataclasses import asdict,dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import default_collate
from torch.nn import functional as F

from .models import HistoryPredictor,PredictorConfig,objective


@dataclass(frozen=True)
class OnlineConfig:
    method: str = 'predictive_ranking'
    enabled: bool = False
    replay: int = 128
    batch: int = 8
    interval: int = 8
    learning_rate: float = 1e-4
    reliability: str = 'confidence'
    retention: str = 'none'
    seed: int = 17
    need_threshold: float = .05
    retention_weight: float = 1.


class ContinualLearner:
    def __init__(self,checkpoint,config,device='cuda'):
        payload=torch.load(checkpoint,map_location='cpu')
        self.model=HistoryPredictor(PredictorConfig(**payload['config'])).to(device)
        self.model.load_state_dict(payload['state_dict']);self.model.eval()
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=config.learning_rate,weight_decay=1e-4)
        self.config=config;self.device=device;self.rng=random.Random(config.seed)
        self.source_feature_metadata=payload.get('source_feature_metadata')
        self.pending={};self.memory=[];self.seen=0;self.frames=0;self.updates=0
        self.recent=[]
        self.stats=dict(proposed=0,delayed_confirmed=0,reliability_rejected=0,need_rejected=0,
                        video_resets=0,updates=0,losses=[])

    def predict(self,features,metadata,mask):
        self.model.eval()
        with torch.no_grad():
            query=self.model(*[torch.as_tensor(a,device=self.device) for a in [features,metadata,mask]])
        return query.cpu().numpy()

    @staticmethod
    def reliable(sample):
        evidence=sample['evidence']
        return (sample['score']>=.75 and evidence['cost']<=.2 and evidence['row_top1'] and
                evidence['column_top1'] and min(evidence['row_margin'],evidence['column_margin'])>=.1
                and sample['packed'][2].sum()>=3)

    def reset_video(self):
        # No pseudo-pair confirmation may cross independent-video boundaries.
        # Parameters, optimizer, replay and update budget remain unchanged.
        self.pending={};self.stats['video_resets']+=1

    def _example(self,sample):
        features,geometry,mask=sample['packed'];positive=sample['positive']
        positive_index=sample['positive_index']
        others=[c for index,c in enumerate(sample['candidates'])
                if index!=positive_index and np.linalg.norm(c)>1e-8]
        if not others:return None
        others.sort(key=lambda c:-float(features[-1]@c))
        candidates=np.zeros((16,len(positive)),np.float32);valid=np.zeros(16,np.bool_)
        rows=[positive]+others[:15];candidates[:len(rows)]=rows;valid[:len(rows)]=True
        # Preserve the prediction made before this pair was assigned. A later
        # confirming frame must not replace it with an already-updated query.
        query=sample['query']
        need=1-float(query@positive)
        if self.config.reliability=='prediction_need' and need<self.config.need_threshold:
            self.stats['need_rejected']+=1;return None
        with torch.no_grad():
            scores=torch.as_tensor(candidates@query/.07).masked_fill(~torch.from_numpy(valid),-1e4)
            teacher=scores.softmax(-1).numpy()
        return tuple(torch.from_numpy(np.asarray(a).copy()) for a in
                     (features,geometry,mask,candidates,valid,teacher,query))

    def _remember(self,example):
        self.seen+=1
        if len(self.memory)<self.config.replay:self.memory.append(example)
        elif self.config.replay:
            index=self.rng.randrange(self.seen)
            if index<len(self.memory):self.memory[index]=example

    def after_assignment(self,accepted,frame):
        self.frames+=1
        if not self.config.enabled:return
        # A frame-t pseudo pair is eligible only after independent frame-(t+1)
        # evidence confirms that same track. No update affects frame-t scoring.
        ready=[]
        for track_id,sample in self.pending.items():
            confirmation=accepted.get(track_id)
            if confirmation is None or frame!=sample['frame']+1 or not self.reliable(confirmation):
                continue
            example=self._example(sample)
            if example is not None:ready.append(example);self.stats['delayed_confirmed']+=1
        self.pending={}
        for track_id,sample in accepted.items():
            self.stats['proposed']+=1
            if self.reliable(sample):self.pending[track_id]=sample
            else:self.stats['reliability_rejected']+=1
        self.recent.extend(ready)
        self.recent=self.recent[-4*self.config.batch:]
        if self.frames%self.config.interval or len(self.recent)<self.config.batch:
            for example in ready:self._remember(example)
            return
        replay_count=self.config.batch//2 if self.memory and self.config.replay else 0
        fresh_count=self.config.batch-replay_count
        selected=self.recent[:fresh_count]
        self.recent=self.recent[fresh_count:]
        if replay_count:
            selected+=self.rng.choices(self.memory,k=replay_count)
        for example in ready:self._remember(example)
        self.model.train()
        batch=default_collate(selected)
        features,geometry,mask,candidates,valid,teacher,teacher_query=[a.to(self.device) for a in batch]
        query=self.model(features,geometry,mask)
        loss,parts=objective(query,candidates,torch.zeros(len(query),device=self.device,dtype=torch.long),
                             self.config.method,candidate_mask=valid)
        retention=torch.zeros((),device=self.device)
        if self.config.retention=='relations':
            logits=torch.einsum('bd,bnd->bn',query,F.normalize(candidates.detach(),dim=-1))/.07
            logits=logits.masked_fill(~valid,-1e4)
            retention=F.kl_div(F.log_softmax(logits,dim=-1),teacher,reduction='batchmean')
            loss=loss+self.config.retention_weight*retention
        elif self.config.retention=='features':
            retention=(1-F.cosine_similarity(query,teacher_query,dim=-1)).mean()
            loss=loss+self.config.retention_weight*retention
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('Online update is nonfinite')
        self.optimizer.step();self.model.eval();self.updates+=1
        self.stats['updates']=self.updates
        self.stats['losses'].append(dict(update=self.updates,frame=self.frames,loss=float(loss.detach()),
            prediction=float(parts['prediction']),ranking=float(parts['ranking']),retention=float(retention.detach())))

    def save(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(state_dict=self.model.state_dict(),config=asdict(self.model.config),
            online_config=asdict(self.config),optimizer=self.optimizer.state_dict(),memory=self.memory,
            sampler_state=self.rng.getstate(),seen=self.seen,frames=self.frames,updates=self.updates,
            stats=self.stats,pending=self.pending,recent=self.recent,
            source_feature_metadata=self.source_feature_metadata),path)

    def resume(self,path):
        payload=torch.load(path,map_location='cpu')
        if payload['online_config']!=asdict(self.config):raise ValueError('Online resume configuration changed')
        self.model.load_state_dict(payload['state_dict']);self.optimizer.load_state_dict(payload['optimizer'])
        self.memory=payload['memory'];self.rng.setstate(payload['sampler_state']);self.pending=payload['pending']
        self.recent=payload['recent']
        for key in ['seen','frames','updates','stats']:setattr(self,key,payload[key])
