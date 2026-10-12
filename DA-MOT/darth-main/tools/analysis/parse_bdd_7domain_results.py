#!/usr/bin/env python3
import csv, json, re
from pathlib import Path
ROOT=Path('/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main')
BASE=ROOT/'work_dirs/bdd_7domain'
SUM=BASE/'summaries'; SUM.mkdir(parents=True, exist_ok=True)

def parse_log(p):
    txt=p.read_text(errors='ignore')
    out={k:'' for k in ['MOTA','HOTA','IDF1','DetA','AssA','mAP','mAP50','car_MOTA','car_HOTA','car_IDF1','ped_MOTA','ped_HOTA','ped_IDF1']}
    m=re.search(r'OVERALL\s+([\-\d\.]+)\s+([\-\d\.]+)\s+[\-\d\.nan]+\s+', txt)
    if m: out['IDF1'],out['MOTA']=m.group(1),m.group(2)
    m=re.search(r'OVERALL\s+([\-\d\.]+)\s+([\-\d\.]+)\s+([\-\d\.]+)\s+([\-\d\.]+)\s+([\-\d\.]+)\s+([\-\d\.]+)\s+([\-\d\.]+)\s+([\-\d\.]+)\s+\d+\s+\d+', txt)
    if m: out['AssA'],out['DetA'],out['HOTA']=m.group(3),m.group(7),m.group(8)
    m=re.search(r'OVERALL\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)', txt)
    if m: out['mAP'],out['mAP50']=m.group(1),m.group(2)
    m=re.search(r'car\s+([\-\d\.]+)\s+([\-\d\.]+)\s+', txt)
    if m: out['car_IDF1'],out['car_MOTA']=m.group(1),m.group(2)
    m=re.search(r'pedestrian\s+([\-\d\.]+)\s+([\-\d\.]+)\s+', txt)
    if m: out['ped_IDF1'],out['ped_MOTA']=m.group(1),m.group(2)
    return out

def infer(path):
    s=str(path)
    mode='source_only' if '/source_only_eval/' in s else ('tta' if '/tta_eval/' in s else 'unknown')
    source=target=source_epoch=tta_epoch=''
    if mode=='source_only':
        m=re.search(r'source_only_eval/([^_/]+(?:_[^_/]+)*)_epoch(\d+)_to_([^/]+)', s)
        if m: source,source_epoch,target=m.group(1),m.group(2),m.group(3)
    if mode=='tta':
        m=re.search(r'tta_eval/(.+)_to_(.+)_epoch(\d+)', s)
        if m: source,target,tta_epoch=m.group(1),m.group(2),m.group(3)
    return mode,source,target,source_epoch,tta_epoch

def main():
    rows=[]
    for p in sorted(BASE.rglob('*.log')):
        if 'source_only_eval' not in str(p) and 'tta_eval' not in str(p): continue
        met=parse_log(p); mode,source,target,se,te=infer(p)
        if not met.get('MOTA') and not met.get('HOTA') and not met.get('IDF1'): continue
        row=dict(source=source,target=target,mode=mode,source_epoch=se,tta_epoch=te,log_path=str(p),checkpoint_path='')
        row.update(met); rows.append(row)
        (p.parent/'parsed_metrics.json').write_text(json.dumps(row,indent=2))
    cols=['source','target','mode','source_epoch','tta_epoch','checkpoint_path','MOTA','HOTA','IDF1','DetA','AssA','mAP','mAP50','car_MOTA','car_HOTA','car_IDF1','ped_MOTA','ped_HOTA','ped_IDF1','log_path']
    with (SUM/'all_results_long.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,cols); w.writeheader(); w.writerows([{c:r.get(c,'') for c in cols} for r in rows])
    (SUM/'all_results_summary.md').write_text(f'# BDD 7-domain results\n\nParsed rows: {len(rows)}\n\nSee `all_results_long.csv`.\n')
    print(f'Parsed {len(rows)} result logs into {SUM}')
if __name__=='__main__': main()
