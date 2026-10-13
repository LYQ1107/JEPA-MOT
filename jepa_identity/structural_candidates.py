"""Training-ready structural H1 candidates with an explicit causal boundary.

The two candidates in this module deliberately separate the history query from
candidate scoring:

* :class:`IdentityResidualPredictor` predicts a change around a stable EMA
  identity anchor built from past observations.
* :class:`CandidateRelativePredictor` predicts a relation query from the past
  and compares it with the current candidate set only in ``score``.

Neither history encoder accepts candidate features, candidate identities, or
ground-truth labels.  Labels are accepted only by the offline training loss.
This keeps the same interface usable for a common-state counterfactual audit.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _check_history_inputs(history_features, history_metadata, history_mask):
    """Validate the shared history tensor contract before a forward pass."""
    if history_features.ndim != 3:
        raise ValueError("history_features must have shape [batch, history, feature]")
    if history_metadata.ndim != 3:
        raise ValueError("history_metadata must have shape [batch, history, metadata]")
    if history_mask.ndim != 2:
        raise ValueError("history_mask must have shape [batch, history]")
    if history_features.shape[:2] != history_metadata.shape[:2] != history_mask.shape:
        raise ValueError("history feature, metadata, and mask shapes disagree")
    if history_features.shape[0] == 0 or history_features.shape[1] == 0:
        raise ValueError("history batch and length must be non-empty")


def validate_causal_history(history_frames, current_frames, history_mask) -> None:
    """Raise when an active history row is not strictly before its target.

    This helper is intentionally independent from model weights so it can be
    called by a source-pair builder and by an offline audit.  Padded rows are
    ignored; every active row must satisfy ``history_frame < current_frame``.
    """
    history_frames = torch.as_tensor(history_frames)
    current_frames = torch.as_tensor(current_frames)
    history_mask = torch.as_tensor(history_mask).bool()
    if history_frames.ndim != 2 or current_frames.ndim != 1:
        raise ValueError("history_frames must be [batch, history] and current_frames [batch]")
    if history_frames.shape != history_mask.shape or history_frames.shape[0] != current_frames.shape[0]:
        raise ValueError("causal frame and mask shapes disagree")
    if bool(((history_frames >= current_frames[:, None]) & history_mask).any()):
        raise ValueError("current or future observation entered causal history")


def stable_candidate_order(candidate_ids: Sequence, primary_scores: Sequence = None) -> Tuple[int, ...]:
    """Return a deterministic permutation, preserving input order on ties.

    At evaluation time ``primary_scores`` should be omitted and detector order
    is preserved.  A source-only reference score may be supplied for training
    hard-negative selection; ties still use the original index and therefore
    do not depend on a hash or an unordered container.
    """
    count = len(candidate_ids)
    if primary_scores is None:
        return tuple(range(count))
    if len(primary_scores) != count:
        raise ValueError("candidate ids and primary scores must have equal length")
    return tuple(sorted(range(count), key=lambda index: (-float(primary_scores[index]), index)))


def _masked_ema(features, mask, decay: float):
    """Compute a left-to-right EMA using only active history rows."""
    if not 0.0 <= decay < 1.0:
        raise ValueError("anchor_decay must be in [0, 1)")
    mask = mask.float()
    batch, length, _ = features.shape
    positions = torch.arange(length, device=features.device).view(1, length)
    last = (positions * mask.long()).max(dim=1).values
    distance = (last[:, None] - positions).clamp_min(0)
    weights = torch.where(mask.bool(), torch.pow(features.new_tensor(decay), distance.float()),
                          torch.zeros_like(distance, dtype=features.dtype))
    # The denominator is non-zero for every valid training example.  Keeping
    # the clamp also makes an empty padded row produce a finite zero anchor in
    # a smoke test instead of a NaN.
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return (features * weights.unsqueeze(-1)).sum(dim=1)


def _pack_history(history_features, history_metadata, history_mask):
    _check_history_inputs(history_features, history_metadata, history_mask)
    mask = history_mask.float().unsqueeze(-1)
    features = F.normalize(history_features.float(), dim=-1) * mask
    metadata = history_metadata.float() * mask
    return torch.cat((features, metadata, mask), dim=-1).flatten(1), features, mask


@dataclass(frozen=True)
class IdentityResidualConfig:
    feature_dim: int = 1024
    history: int = 4
    metadata_dim: int = 10
    hidden_dim: int = 128
    residual_scale: float = 0.25
    anchor_decay: float = 0.7


class IdentityResidualPredictor(nn.Module):
    """Predict identity-preserving feature change around a past-only anchor."""

    def __init__(self, config: IdentityResidualConfig):
        super().__init__()
        self.config = config
        input_dim = config.history * (config.feature_dim + config.metadata_dim + 1)
        self.context = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.delta = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.feature_dim),
        )
        # Starting at zero makes the first query exactly the stable anchor and
        # separates residual learning from a hidden identity re-initialization.
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def history_state(self, history_features, history_metadata, history_mask):
        """Return ``(query, anchor)``; no candidate or target is read here."""
        packed, features, mask = _pack_history(history_features, history_metadata, history_mask)
        hidden = self.context(packed)
        anchor = _masked_ema(features, mask.squeeze(-1), self.config.anchor_decay)
        residual = self.config.residual_scale * torch.tanh(self.delta(hidden))
        query = F.normalize(anchor + residual, dim=-1)
        return query, F.normalize(anchor, dim=-1)

    def forward(self, history_features, history_metadata, history_mask):
        """Build a candidate-independent query from strictly past history."""
        return self.history_state(history_features, history_metadata, history_mask)[0]


@dataclass(frozen=True)
class CandidateRelativeConfig:
    feature_dim: int = 1024
    history: int = 4
    metadata_dim: int = 10
    hidden_dim: int = 128
    relation_dim: int = 128
    temperature: float = 0.07


class CandidateRelativePredictor(nn.Module):
    """Predict a track-to-candidate relation query from past history only."""

    def __init__(self, config: CandidateRelativeConfig):
        super().__init__()
        self.config = config
        input_dim = config.history * (config.feature_dim + config.metadata_dim + 1)
        self.context = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.relation_query = nn.Linear(config.hidden_dim, config.relation_dim)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(config.feature_dim, config.relation_dim),
            nn.LayerNorm(config.relation_dim),
            nn.GELU(),
        )

    def encode_history(self, history_features, history_metadata, history_mask):
        """Return a relation query without reading the current candidate set."""
        packed, _, _ = _pack_history(history_features, history_metadata, history_mask)
        return self.relation_query(self.context(packed))

    def forward(self, history_features, history_metadata, history_mask):
        return self.encode_history(history_features, history_metadata, history_mask)

    def encode_candidates(self, candidates):
        if candidates.ndim != 3 or candidates.shape[-1] != self.config.feature_dim:
            raise ValueError("candidates must have shape [batch, candidates, feature]")
        return self.candidate_encoder(F.normalize(candidates.float(), dim=-1))

    def score(self, relation_query, candidates, candidate_mask=None):
        """Score candidates after the past-only query has been constructed."""
        if relation_query.ndim != 2:
            raise ValueError("relation_query must have shape [batch, relation]")
        encoded = F.normalize(self.encode_candidates(candidates), dim=-1)
        query = F.normalize(relation_query.float(), dim=-1)
        if query.shape[-1] != encoded.shape[-1]:
            raise ValueError("relation query and candidate relation dimensions disagree")
        scores = torch.einsum("bd,bnd->bn", query, encoded) / self.config.temperature
        if candidate_mask is not None:
            if candidate_mask.shape != scores.shape:
                raise ValueError("candidate mask shape disagrees with score shape")
            scores = scores.masked_fill(~candidate_mask.bool(), -1e4)
        return scores


def _positive_rows(values, positive_index):
    if positive_index.ndim != 1 or positive_index.shape[0] != values.shape[0]:
        raise ValueError("positive_index must have one entry per batch row")
    rows = torch.arange(values.shape[0], device=values.device)
    return values[rows, positive_index]


def identity_residual_objective(query, anchor, candidates, positive_index, method,
                                *, candidate_mask=None, temperature=0.07) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Train the residual query with prediction, CE, or both.

    ``positive_index`` is an offline source label.  It is not used to form the
    history query and must never be supplied by an online target-domain gate.
    """
    if query.shape != anchor.shape or query.ndim != 2:
        raise ValueError("query and anchor must have shape [batch, feature]")
    candidates = F.normalize(candidates.float(), dim=-1)
    target = _positive_rows(candidates, positive_index).detach()
    prediction = (1.0 - F.cosine_similarity(query, target, dim=-1)).mean()
    logits = torch.einsum("bd,bnd->bn", query, candidates) / temperature
    if candidate_mask is not None:
        logits = logits.masked_fill(~candidate_mask.bool(), -1e4)
    ranking = F.cross_entropy(logits, positive_index)
    anchor_error = (1.0 - F.cosine_similarity(anchor, target, dim=-1)).mean()
    if method == "predictive":
        loss = prediction
    elif method == "discriminative":
        loss = ranking
    elif method == "predictive_ranking":
        loss = prediction + ranking
    else:
        raise ValueError("unknown identity residual method: %s" % method)
    return loss, {
        "prediction": prediction.detach(),
        "ranking": ranking.detach(),
        "anchor_error": anchor_error.detach(),
    }


def candidate_relative_objective(model: CandidateRelativePredictor, relation_query,
                                 candidates, positive_index, method, *, candidate_mask=None,
                                 temperature=None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Train a relation query against the same-frame hard-negative set."""
    temperature = model.config.temperature if temperature is None else temperature
    encoded = F.normalize(model.encode_candidates(candidates), dim=-1)
    query = F.normalize(relation_query, dim=-1)
    logits = torch.einsum("bd,bnd->bn", query, encoded) / temperature
    if candidate_mask is not None:
        logits = logits.masked_fill(~candidate_mask.bool(), -1e4)
    ranking = F.cross_entropy(logits, positive_index)
    target = _positive_rows(encoded, positive_index).detach()
    prediction = (1.0 - F.cosine_similarity(query, target, dim=-1)).mean()
    teacher_logits = torch.einsum("bd,bnd->bn", target, encoded) / temperature
    if candidate_mask is not None:
        teacher_logits = teacher_logits.masked_fill(~candidate_mask.bool(), -1e4)
    relation = F.kl_div(
        F.log_softmax(logits, dim=-1),
        F.softmax(teacher_logits.detach(), dim=-1),
        reduction="batchmean",
    )
    if method == "predictive":
        loss = relation
    elif method == "discriminative":
        loss = ranking
    elif method == "predictive_ranking":
        loss = relation + ranking
    else:
        raise ValueError("unknown candidate-relative method: %s" % method)
    return loss, {
        "prediction": prediction.detach(),
        "ranking": ranking.detach(),
        "relation": relation.detach(),
    }

