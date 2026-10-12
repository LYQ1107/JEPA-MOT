"""Candidate-independent history models for matched-capacity objective controls."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PredictorConfig:
    feature_dim: int = 1664
    history: int = 4
    metadata_dim: int = 10
    hidden_dim: int = 128
    residual_scale: float = 0.25


class HistoryPredictor(nn.Module):
    """Predict a current identity latent using only strictly prior observations.

    The same architecture and initialization are used for the discriminative
    listwise classifier, predictive-only model and prediction+ranking model.
    Current candidates are scored separately, after this forward call.
    """
    def __init__(self, config: PredictorConfig):
        super().__init__()
        self.config = config
        input_dim = config.history * (config.feature_dim + config.metadata_dim + 1)
        self.context = nn.Sequential(nn.Linear(input_dim, config.hidden_dim),
                                     nn.LayerNorm(config.hidden_dim), nn.GELU())
        self.delta = nn.Linear(config.hidden_dim, config.feature_dim)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, history_features, history_metadata, history_mask):
        mask = history_mask.unsqueeze(-1)
        features = history_features * mask
        metadata = history_metadata * mask
        packed = torch.cat((features, metadata, mask), dim=-1).flatten(1)
        hidden = self.context(packed)
        positions = torch.arange(self.config.history, device=history_mask.device)
        last_index = (positions.unsqueeze(0) * history_mask).long().max(dim=1).values
        last = features[torch.arange(len(features), device=features.device), last_index]
        query = last + self.config.residual_scale * torch.tanh(self.delta(hidden))
        return F.normalize(query, dim=-1)


def association_scores(query, candidates, temperature=0.07):
    """Candidate input is permitted only after the history query exists."""
    return torch.einsum('bd,bnd->bn', query, F.normalize(candidates.detach(), dim=-1)) / temperature


def objective(query, candidates, positive_index, method, *, candidate_mask=None, temperature=0.07):
    target = candidates[torch.arange(len(query), device=query.device), positive_index].detach()
    prediction = (1 - F.cosine_similarity(query, target, dim=-1)).mean()
    logits = association_scores(query, candidates, temperature)
    if candidate_mask is not None:
        logits = logits.masked_fill(~candidate_mask, -1e4)
    ranking = F.cross_entropy(logits, positive_index)
    if method == 'discriminative':
        loss = ranking
    elif method == 'predictive':
        loss = prediction
    elif method == 'predictive_ranking':
        loss = prediction + ranking
    else:
        raise ValueError(method)
    return loss, {'prediction': prediction.detach(), 'ranking': ranking.detach()}
