"""Retain column competition among track forecasts for a shared current detection."""
import torch
import numpy as np
from torch.nn import functional as F


def column_retention(query,candidates,valid,teacher_query,candidate_ids,query_ids,temperature=.07):
    normalized=F.normalize(candidates.detach(),dim=-1)
    scores=torch.einsum('bd,bnd->bn',query,normalized)/temperature
    old=torch.einsum('bd,bnd->bn',teacher_query.detach(),normalized)/temperature
    mask=valid.detach().cpu().numpy();ids=candidate_ids.detach().cpu().numpy()
    row_ids=query_ids.detach().cpu().tolist();groups={}
    # Identity keys, rather than descriptor equality, remove repeated replay
    # copies while retaining genuinely competing tracks with aliased latents.
    for row,slot in np.argwhere(mask):
        groups.setdefault(int(ids[row,slot]),{}).setdefault(row_ids[row],(int(row),int(slot)))
    groups=[list(rows.values()) for rows in groups.values() if len(rows)>1]
    if not groups:return query.sum()*0,0
    width=max(map(len,groups));rows=np.zeros((len(groups),width),np.int64)
    slots=np.zeros_like(rows);active=np.zeros_like(rows,dtype=bool)
    for i,group in enumerate(groups):
        rows[i,:len(group)],slots[i,:len(group)]=zip(*group);active[i,:len(group)]=True
    rows=torch.as_tensor(rows,device=query.device);slots=torch.as_tensor(slots,device=query.device)
    active=torch.as_tensor(active,device=query.device)
    student=scores[rows,slots].masked_fill(~active,-1e4)
    target=F.softmax(old[rows,slots].masked_fill(~active,-1e4),dim=1)
    loss=F.kl_div(F.log_softmax(student,dim=1),target,reduction='batchmean')
    return loss,len(groups)


def distance_retention(query,teacher_query,query_ids):
    """Generic RKD-D control over forecast embeddings, independent of candidates."""
    mask=query_ids[:,None]!=query_ids[None,:]
    if not mask.any():return query.sum()*0
    current=torch.cdist(query,query,p=2)[mask]
    old=torch.cdist(teacher_query.detach(),teacher_query.detach(),p=2)[mask]
    current=current/current.mean().clamp_min(1e-6)
    old=old/old.mean().clamp_min(1e-6)
    return F.smooth_l1_loss(current,old)
