"""History-guided RoI Prototype Memory (RoI association embedding regularizer).

Goal: stabilize the RoI association embedding across continuous target-domain
adaptation to reduce IDSw / improve AssA / IDF1 / HOTA. This is NOT a detection
or recall booster.

Design (phase-1, positive nearest-prototype alignment only):
  previous target-domain reliable RoI embeddings
    -> stage-end class-wise prototype memory (top-score or diversity selection)
    -> subsequent target-stage RoI embedding regularization
    -> stabilize association embedding -> reduce IDSw.

Hard constraints honored here:
  * never stores raw images / paths -- only detached, L2-normalized CPU tensors.
  * history-only: the current stage's candidates never supervise the current
    stage (collect != use).
  * positive-only: no same-class negatives, no class contrastive.
  * loss is None when history memory is empty or no valid RoI in the batch.

This module is independent from the causal bank / old roi_prototype_memory and
is only used by ProtoQuasiDenseTrackHead.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def roi_proto_loss(feats, prototypes, topk=3):
    """Positive nearest-prototype alignment loss.

    Args:
        feats (Tensor): [N, D] current-batch RoI association embeddings.
        prototypes (Tensor): [M, D] history prototypes (detached).
        topk (int): average the top-k most similar prototypes per RoI.
    Returns:
        (Tensor scalar, Tensor[N] per-roi mean top-k cosine sim)
    """
    loss, per_roi, _ = roi_proto_loss_with_gate(
        feats, prototypes, topk=topk, sim_gate_thr=None)
    return loss, per_roi


def roi_proto_loss_with_gate(feats, prototypes, topk=3, sim_gate_thr=None):
    """Positive nearest-prototype loss, optionally similarity-gated."""
    feats = F.normalize(feats, dim=1)
    prototypes = F.normalize(prototypes, dim=1).detach()
    sim = torch.matmul(feats, prototypes.t())
    k = min(topk, sim.size(1))
    topk_sim = sim.topk(k=k, dim=1).values
    per_roi = topk_sim.mean(dim=1)
    if sim_gate_thr is None:
        gate_mask = torch.ones_like(per_roi, dtype=torch.bool)
    else:
        gate_mask = per_roi.detach() >= float(sim_gate_thr)
    if gate_mask.sum() == 0:
        return None, per_roi, gate_mask
    loss = (1.0 - per_roi[gate_mask]).mean()
    return loss, per_roi, gate_mask


class RoIPrototypeMemory(nn.Module):
    """History-guided class-wise RoI prototype memory.

    Args:
        class_ids (list[int]): numeric class ids to track (e.g. [1] for car).
        score_thr (float): keep RoIs/candidates with score >= thr. If scores
            are None (not wired), filtering is by class only (logged).
        max_candidates_per_domain (int): per domain/class candidate cap.
        k_per_domain_class (int): #prototypes kept per domain/class at finalize.
        topk (int): top-k prototypes averaged in the loss.
        max_rois_per_batch (int): cap on RoIs used in a single loss call.
        diverse_select (bool): use diversity selection at finalize.
        diverse_preselect_top (int): pre-select pool by score before diversity.
        diverse_alpha (float): selection_score = a*norm_score + (1-a)*div_dist.
        feature_source (str): 'student' or 'teacher' (informational here).
    """

    def __init__(self,
                 class_ids=(1,),
                 score_thr=0.70,
                 max_candidates_per_domain=1000,
                 k_per_domain_class=100,
                 topk=3,
                 max_rois_per_batch=128,
                 diverse_select=False,
                 diverse_preselect_top=500,
                 diverse_alpha=0.5,
                 feature_source='student',
                 min_candidates=20,
                 use_sim_gate=False,
                 sim_gate_thr=0.50):
        super().__init__()
        self.class_ids = [int(c) for c in class_ids]
        self.score_thr = float(score_thr)
        self.max_candidates_per_domain = int(max_candidates_per_domain)
        self.k_per_domain_class = int(k_per_domain_class)
        self.topk = int(topk)
        self.max_rois_per_batch = int(max_rois_per_batch)
        self.diverse_select = bool(diverse_select)
        self.diverse_preselect_top = int(diverse_preselect_top)
        self.diverse_alpha = float(diverse_alpha)
        self.feature_source = str(feature_source)
        self.min_candidates = int(min_candidates)
        self.use_sim_gate = bool(use_sim_gate)
        self.sim_gate_thr = float(sim_gate_thr)

        # history prototypes: {domain: {class_id: Tensor[K, D] on CPU}}
        self._history = {}
        # current stage candidate buffer: {class_id: {'f': [tensors], 's': [tensors]}}
        self._cand = {}
        self._cand_scores = {}
        self.current_domain = None
        # last loss stats for logging
        self.last_stats = {}

    def start_stage(self, domain_name):
        """Clear current-stage candidate buffer; keep history intact."""
        self.current_domain = domain_name
        self._cand = {c: [] for c in self.class_ids}
        self._cand_scores = {c: [] for c in self.class_ids}

    @torch.no_grad()
    def collect(self, domain_name, feats, labels, scores=None):
        """Buffer detached, L2-normalized candidate RoI embeddings (CPU).

        Args:
            feats (Tensor): [N, D] RoI association embeddings.
            labels (Tensor): [N] pseudo class ids.
            scores (Tensor or None): [N] pseudo detection confidence.
        """
        if feats is None or feats.numel() == 0 or labels is None:
            return
        f = F.normalize(feats.detach().float(), dim=1).cpu()
        lab = labels.detach().cpu()
        sc = scores.detach().cpu() if scores is not None else None
        for c in self.class_ids:
            m = (lab == c)
            if sc is not None:
                m = m & (sc >= self.score_thr)
            if m.sum() == 0:
                continue
            self._cand[c].append(f[m])
            if sc is not None:
                self._cand_scores[c].append(sc[m])
            else:
                # no scores wired: use unit score so top-score == arbitrary order
                self._cand_scores[c].append(torch.ones(int(m.sum())))
            # cap candidate buffer per domain/class (keep highest score)
            self._trim_candidates(c)

    def _trim_candidates(self, c):
        total = sum(t.shape[0] for t in self._cand[c])
        if total <= self.max_candidates_per_domain:
            return
        f = torch.cat(self._cand[c], dim=0)
        s = torch.cat(self._cand_scores[c], dim=0)
        keep = s.topk(k=self.max_candidates_per_domain).indices
        self._cand[c] = [f[keep]]
        self._cand_scores[c] = [s[keep]]

    @torch.no_grad()
    def finalize_stage(self, domain_name):
        """Turn current-stage candidates into history prototypes."""
        protos = {}
        for c in self.class_ids:
            if not self._cand.get(c):
                continue
            f = torch.cat(self._cand[c], dim=0)
            s = torch.cat(self._cand_scores[c], dim=0)
            if f.shape[0] < self.min_candidates:
                import logging
                logging.getLogger('mmtrack').warning(
                    '[ROI_PROTO] domain=%s class=%d only %d candidates '
                    '(<%d); skip memory for this domain/class.',
                    domain_name, c, f.shape[0], self.min_candidates)
                continue
            if self.diverse_select:
                sel = self._select_diverse(f, s, self.k_per_domain_class)
            else:
                k = min(self.k_per_domain_class, f.shape[0])
                sel = f[s.topk(k=k).indices]
            protos[c] = sel
        if protos:
            self._history[domain_name] = protos
        # clear candidate buffer after finalize
        self._cand = {c: [] for c in self.class_ids}
        self._cand_scores = {c: [] for c in self.class_ids}

    @torch.no_grad()
    def _select_diverse(self, f, s, k):
        """Diversity selection: balance high score and low mutual similarity."""
        n = f.shape[0]
        pre = min(self.diverse_preselect_top, n)
        pool_idx = s.topk(k=pre).indices
        pool_f = F.normalize(f[pool_idx], dim=1)
        pool_s = s[pool_idx]
        norm_s = (pool_s - pool_s.min()) / (pool_s.max() - pool_s.min() + 1e-6)
        k = min(k, pre)
        selected = [int(torch.argmax(norm_s).item())]
        # max-similarity to already-selected set, init with first prototype
        max_sim = (pool_f @ pool_f[selected[0]].unsqueeze(1)).squeeze(1)
        for _ in range(1, k):
            div_dist = 1.0 - max_sim
            sel_score = self.diverse_alpha * norm_s + \
                (1 - self.diverse_alpha) * div_dist
            sel_score[selected] = -1e9
            nxt = int(torch.argmax(sel_score).item())
            selected.append(nxt)
            sim_nxt = (pool_f @ pool_f[nxt].unsqueeze(1)).squeeze(1)
            max_sim = torch.maximum(max_sim, sim_nxt)
        return f[pool_idx][selected]

    def get_history_prototypes(self, class_id, exclude_current=True):
        """Return [M, D] history prototypes for class_id (history domains only)."""
        mats = []
        for dom, protos in self._history.items():
            if exclude_current and dom == self.current_domain:
                continue
            if class_id in protos:
                mats.append(protos[class_id])
        if not mats:
            return None
        return torch.cat(mats, dim=0)

    def compute_loss(self, feats, labels, scores=None):
        """History-guided prototype loss for the current batch.

        Returns None when history memory is empty or no valid RoI in batch.
        """
        if feats is None or feats.numel() == 0 or labels is None:
            return None
        lab = labels.detach()
        device = feats.device

        total_loss = None
        n_classes = 0
        sel_rois_total = 0
        proto_total = 0
        sim_accum = 0.0
        gate_selected_total = 0
        gate_total = 0
        for c in self.class_ids:
            protos = self.get_history_prototypes(c, exclude_current=True)
            if protos is None or protos.shape[0] == 0:
                continue
            m = (lab == c)
            if scores is not None:
                m = m & (scores.detach() >= self.score_thr)
            if m.sum() == 0:
                continue
            cf = feats[m]
            if cf.shape[0] > self.max_rois_per_batch:
                idx = torch.randperm(cf.shape[0], device=cf.device)[
                    :self.max_rois_per_batch]
                cf = cf[idx]
            protos = protos.to(device, feats.dtype)
            loss_c, per_roi, gate_mask = roi_proto_loss_with_gate(
                cf, protos, topk=self.topk,
                sim_gate_thr=self.sim_gate_thr if self.use_sim_gate else None)
            if loss_c is None:
                gate_total += int(cf.shape[0])
                proto_total += protos.shape[0]
                self.last_stats = dict(
                    selected_rois=sel_rois_total,
                    prototypes=proto_total,
                    mean_topk_sim=(float(per_roi.mean().item())
                                   if per_roi.numel() > 0 else 0.0),
                    loss=0.0,
                    gate_enabled=self.use_sim_gate,
                    sim_gate_thr=self.sim_gate_thr,
                    gate_selected_rois=0,
                    gate_total_rois=int(cf.shape[0]),
                    gate_ratio=0.0,
                )
                continue
            gate_count = int(gate_mask.sum().item())
            gate_selected_total += gate_count
            gate_total += int(cf.shape[0])
            total_loss = loss_c if total_loss is None else total_loss + loss_c
            n_classes += 1
            sel_rois_total += gate_count if self.use_sim_gate else cf.shape[0]
            proto_total += protos.shape[0]
            sim_accum += float(per_roi[gate_mask].mean().item()) * max(gate_count, 1)

        if total_loss is None or n_classes == 0:
            return None
        loss = total_loss / n_classes
        self.last_stats = dict(
            selected_rois=sel_rois_total,
            prototypes=proto_total,
            mean_topk_sim=(sim_accum / max(sel_rois_total, 1)),
            loss=float(loss.item()),
            gate_enabled=self.use_sim_gate,
            sim_gate_thr=self.sim_gate_thr,
            gate_selected_rois=gate_selected_total,
            gate_total_rois=gate_total,
            gate_ratio=(gate_selected_total / max(gate_total, 1)),
        )
        return loss

    @torch.no_grad()
    def summary(self):
        """Return memory statistics for logging."""
        hist = {}
        for dom, protos in self._history.items():
            hist[dom] = {c: int(v.shape[0]) for c, v in protos.items()}
        cand = {}
        for c in self.class_ids:
            cand[c] = int(sum(t.shape[0] for t in self._cand.get(c, [])))
        # mean feature norm of history prototypes (should be ~1.0 after L2)
        norms = []
        for protos in self._history.values():
            for v in protos.values():
                norms.append(float(v.norm(dim=1).mean().item()))
        mean_norm = sum(norms) / len(norms) if norms else 0.0
        return dict(
            history_domains=list(self._history.keys()),
            prototypes_per_domain_class=hist,
            current_candidates=cand,
            mean_feature_norm=mean_norm,
            last_loss_stats=self.last_stats,
        )

    def state_dict_memory(self):
        """Serializable history (CPU tensors). Candidate buffer not saved."""
        return {dom: {c: v.clone() for c, v in protos.items()}
                for dom, protos in self._history.items()}

    def load_state_dict_memory(self, state):
        if not state:
            return
        self._history = {dom: {int(c): v for c, v in protos.items()}
                         for dom, protos in state.items()}
