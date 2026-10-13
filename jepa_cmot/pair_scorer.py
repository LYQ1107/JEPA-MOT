"""Ordinary scalar pair classifier; freeze history state before candidates."""
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .causal_tracker import CausalHistoryTracker
from yolox.tracker import matching_bdd as matching


@dataclass(frozen=True)
class PairConfig:
    feature_dim: int = 1024
    history: int = 4
    metadata_dim: int = 10
    hidden_dim: int = 122
    score_hidden: int = 128


class PairAssociationModel(nn.Module):
    def __init__(self, config=PairConfig()):
        super().__init__();self.config=config
        width=config.history*(config.feature_dim+config.metadata_dim+1)
        self.context=nn.Sequential(nn.Linear(width,config.hidden_dim),nn.LayerNorm(config.hidden_dim),nn.GELU())
        self.candidate=nn.Sequential(nn.Linear(config.feature_dim,config.hidden_dim),nn.LayerNorm(config.hidden_dim),nn.GELU())
        self.scorer=nn.Sequential(nn.Linear(2*config.hidden_dim,config.score_hidden),nn.GELU(),nn.Linear(config.score_hidden,1))

    def forward(self, features, metadata, mask):
        expanded=mask.unsqueeze(-1)
        packed=torch.cat([features*expanded,metadata*expanded,expanded],dim=-1)
        return self.context(packed.flatten(1))

    def score(self, history_state, candidates):
        candidate=self.candidate(candidates)
        state=history_state[:,None].expand(-1,candidates.shape[1],-1)
        return self.scorer(torch.cat([state,candidate],dim=-1)).squeeze(-1)


class FrozenPairReader:
    def __init__(self, checkpoint, device='cuda'):
        payload=torch.load(checkpoint,map_location='cpu')
        self.model=PairAssociationModel(PairConfig(**payload['config'])).to(device).eval()
        self.model.load_state_dict(payload['state_dict']);self.device=device
        self.source_feature_metadata=payload['source_feature_metadata']
        self.updates=0;self.frames=0;self.stats=dict(updates=0,video_resets=0)

    def predict(self, features, metadata, mask):
        with torch.no_grad():
            return self.model(*[torch.as_tensor(a,device=self.device) for a in [features,metadata,mask]]).cpu().numpy()

    def score(self, state, candidates):
        with torch.no_grad():
            return self.model.score(torch.as_tensor(state,device=self.device),
                                    torch.as_tensor(candidates,device=self.device)).cpu().numpy()

    def after_assignment(self, accepted, frame):self.frames+=1
    def reset_video(self):self.stats['video_resets']+=1

    def save(self, path, **kwargs):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(state_dict=self.model.state_dict(),config=asdict(self.model.config),
            source_feature_metadata=self.source_feature_metadata,updates=0,frames=self.frames,
            stats=self.stats,scope='frozen scalar pair reader, no online learning'),path)

    def resume(self, path):
        payload=torch.load(path,map_location='cpu');self.model.load_state_dict(payload['state_dict'])
        self.frames=payload['frames'];self.stats=payload['stats']


class PairHistoryTracker(CausalHistoryTracker):
    def _jepa_similarity(self, tracks, detections):
        result=np.zeros((len(tracks),len(detections)),np.float32)
        active=[i for i,t in enumerate(tracks) if self._track_query(t) is not None]
        available=[j for j,d in enumerate(detections) if getattr(d,'jepa_feature',None) is not None]
        if not active or not available:return result
        states=np.stack([self._track_query(tracks[i]) for i in active])
        descriptors=np.stack([detections[j].jepa_feature for j in available])
        logits=self.learner.score(states,np.broadcast_to(descriptors,(len(states),*descriptors.shape)).copy())
        calibration=self.args.source_calibration
        probability=1/(1+np.exp(-np.clip(calibration['slope']*(logits-calibration['intercept']),-60,60)))
        for row,i in enumerate(active):
            for column,j in enumerate(available):
                if matching.sig(tracks[i].cls)==matching.sig(detections[j].cls):result[i,j]=probability[row,column]
        return result
