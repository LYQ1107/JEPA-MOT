"""Persistent source-only detector worker in the existing MMDetection environment."""
import argparse
import base64
from contextlib import redirect_stdout
import json
import sys
import time

import numpy as np
import cv2
import torch
from mmdet.apis import init_detector, inference_detector


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--return-pixels',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4)
    with redirect_stdout(sys.stderr):
        model=init_detector(args.config,args.checkpoint,device='cuda')
    torch.cuda.reset_peak_memory_stats()
    print(json.dumps(dict(status='READY',torch=torch.__version__)),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('stop'):break
        torch.cuda.synchronize();start=time.perf_counter()
        raw=cv2.imread(request['file_name']) if args.return_pixels else None
        with redirect_stdout(sys.stderr):
            results=inference_detector(model,raw if args.return_pixels else request['file_name'])
        rows=[]
        for category,detections in enumerate(results,1):
            if args.dataset=='ua_detrac' and category not in [2,3,4]:continue
            for detection in detections:
                if float(detection[4])>=.1:
                    rows.append(np.append(detection.astype(float),category-1).tolist())
        torch.cuda.synchronize()
        reply=dict(rows=rows,seconds=time.perf_counter()-start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if args.return_pixels:
            rgb=cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)
            reply['rgb_shape']=list(rgb.shape)
            reply['rgb_uint8_base64']=base64.b64encode(rgb.tobytes()).decode('ascii')
        print(json.dumps(reply),flush=True)


if __name__=='__main__':main()
