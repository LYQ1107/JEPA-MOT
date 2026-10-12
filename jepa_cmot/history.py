"""Shared causal packing for supervised source and unlabeled online histories."""
import numpy as np


def pack_history(observations, current_frame, history, feature_dim, width, height):
    features = np.zeros((history, feature_dim), np.float32)
    metadata = np.zeros((history, 10), np.float32)
    mask = np.zeros(history, np.float32)
    rows = list(observations)[-history:]
    scale = np.asarray([width,height,width,height],np.float32)
    for index, row in enumerate(rows):
        slot = history-len(rows)+index
        gap = int(current_frame)-int(row['frame'])
        if gap <= 0:
            raise ValueError('Prediction history must end strictly before the current frame')
        feature = np.asarray(row['feature'],np.float32)
        features[slot] = feature/max(float(np.linalg.norm(feature)),1e-8)
        x1,y1,x2,y2 = np.asarray(row['box'],np.float32)/scale
        age = int(row['frame'])-int(row['birth'])+1
        metadata[slot] = [(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1,
            float(row['confidence']),min(gap,128)/128,min(age,1024)/1024,
            min(max(0,gap-1),128)/128,0,0]
        if index:
            previous = rows[index-1]
            center=(np.asarray(row['box'][:2])+np.asarray(row['box'][2:]))/2
            old=(np.asarray(previous['box'][:2])+np.asarray(previous['box'][2:]))/2
            metadata[slot,8:] = (center-old)/scale[:2]/max(1,row['frame']-previous['frame'])
        mask[slot] = 1
    return features,metadata,mask
