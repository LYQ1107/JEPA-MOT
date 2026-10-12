"""RoI prototype memory (phase-1 interface only, default OFF).

Lightweight, batch-level positive RoI prototype memory. This is intentionally
minimal for the first phase: it is NOT enabled by default and does NOT do
tracklet-level reasoning. It only stores detached, normalized tensors plus
class id / domain id / score -- never images or paths.

Contract:
  * `memory_contrastive_loss` returns a 0 scalar when memory is empty.
  * Loss is computed against the HISTORY memory, then the memory is updated,
    so a batch never supervises itself.
  * Class ids are numeric; class weighting is configured by numeric id.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RoIPrototypeMemory(nn.Module):
    """Batch-level positive RoI prototype memory.

    Args:
        feat_dim (int): Embedding dimensionality D.
        max_prototypes_per_class (int): FIFO capacity per class.
        class_weights (dict or None): {numeric_class_id: weight}. If None, all
            classes are weighted 1.0.
        momentum (float): reserved for future EMA prototypes (unused phase-1).
        normalize (bool): L2-normalize stored/query features.
    """

    def __init__(
        self,
        feat_dim,
        max_prototypes_per_class=100,
        class_weights=None,
        momentum=0.9,
        normalize=True,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.max_per_class = int(max_prototypes_per_class)
        self.momentum = float(momentum)
        self.normalize = bool(normalize)
        self.class_weights = dict(class_weights) if class_weights else None

        # python-side storage of detached tensors; kept off the autograd graph.
        # {class_id: tensor[K, D]} on the module's device.
        self._bank = {}

    def _device(self):
        # follow a registered buffer-less module: infer from any stored tensor
        for v in self._bank.values():
            return v.device
        return None

    @torch.no_grad()
    def update(self, feats, labels, domain=None, scores=None):
        """Append detached normalized positive RoI feats to per-class FIFO."""
        if feats is None or feats.numel() == 0:
            return
        f = feats.detach()
        if self.normalize:
            f = F.normalize(f, dim=-1)
        labels = labels.detach()
        for cls in torch.unique(labels):
            cls_id = int(cls.item())
            sel = f[labels == cls]
            if sel.numel() == 0:
                continue
            if cls_id in self._bank:
                buf = torch.cat([self._bank[cls_id], sel], dim=0)
            else:
                buf = sel
            if buf.shape[0] > self.max_per_class:
                buf = buf[-self.max_per_class:]
            self._bank[cls_id] = buf

    def memory_contrastive_loss(self, feats, labels, domain=None,
                                temperature=0.07):
        """Supervised contrastive loss against HISTORY prototypes.

        Returns 0 (connected to feats' graph) when memory is empty so that
        the autograd graph stays valid and DDP does not complain.
        """
        zero = feats.sum() * 0.0 if (feats is not None and feats.numel() > 0) \
            else torch.tensor(0.0)
        if feats is None or feats.numel() == 0 or len(self._bank) == 0:
            return zero

        q = feats
        if self.normalize:
            q = F.normalize(q, dim=-1)

        # build prototype matrix from history
        proto_list, proto_labels = [], []
        for cls_id, buf in self._bank.items():
            proto_list.append(buf.to(q.device, q.dtype))
            proto_labels.append(
                torch.full((buf.shape[0],), cls_id,
                           device=q.device, dtype=torch.long))
        if not proto_list:
            return zero
        protos = torch.cat(proto_list, dim=0)            # [M, D]
        proto_labels = torch.cat(proto_labels, dim=0)    # [M]

        logits = (q @ protos.t()) / temperature          # [N, M]
        pos_mask = (labels.view(-1, 1) == proto_labels.view(1, -1)).float()

        has_pos = pos_mask.sum(dim=1) > 0
        if has_pos.sum() == 0:
            return zero

        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        pos_log_prob = (log_prob * pos_mask).sum(dim=1) / \
            pos_mask.sum(dim=1).clamp(min=1.0)
        loss_per = -pos_log_prob[has_pos]

        if self.class_weights is not None:
            w = torch.ones_like(loss_per)
            valid_labels = labels[has_pos]
            for i, lab in enumerate(valid_labels):
                w[i] = self.class_weights.get(int(lab.item()), 1.0)
            loss = (loss_per * w).sum() / w.sum().clamp(min=1e-6)
        else:
            loss = loss_per.mean()
        return loss

    @torch.no_grad()
    def size_by_class(self):
        """Return {class_id: num_prototypes} for logging."""
        return {k: int(v.shape[0]) for k, v in self._bank.items()}
