#!/usr/bin/env python3
"""Collect BDD seq3 episode results into long CSV, matrices and markdown."""
import csv, re, sys
from pathlib import Path

EVAL_DOMAINS=['dawn_dusk','night','partly_cloudy']
STAGE_ORDER=['stage00_source_only','stage01_after_dawn_dusk','stage02_after_night','stage03_after_partly_cloudy']
ADAPTED_SO_FAR={
    'stage00_source_only':'none',
    'stage01_after_dawn_dusk':'dawn_dusk',
    'stage02_after_night':'dawn_dusk,night',
    'stage03_after_partly_cloudy':'dawn_dusk,night,partly_cloudy',
}
ADAPTED_DOMAIN={
    'stage00_source_only':'none',
    'stage01_after_dawn_dusk':'dawn_dusk',
    'stage02_after_night':'night',
    'stage03_after_partly_cloudy':'partly_cloudy',
}

def parse_eval_log(log_path):
    if not Path(log_path).exists(): return None
    text=Path(log_path).read_text(errors='ignore')
    m=re.findall(r'OVERALL\s+([\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',text,re.DOTALL)
    if not m: return None
    LocA,DetA,MOTA,HOTA,IDF1,AssA=[float(x) for x in m[-1]]
    map_m=re.findall(r'bbox/OVERALL/mAP:?\s*([\d.]+)',text)
    map50_m=re.findall(r'bbox/OVERALL/(?:mAP_50|AP50):?\s*([\d.]+)',text)
    mAP=float(map_m[-1])*100 if map_m else ''
    mAP50=float(map50_m[-1])*100 if map50_m else ''
    return dict(HOTA=round(HOTA,2),MOTA=round(MOTA,2),IDF1=round(IDF1,2),AssA=round(AssA,2),DetA=round(DetA,2),LocA=round(LocA,2),mAP=round(mAP,2) if mAP!='' else '',mAP50=round(mAP50,2) if mAP50!='' else '')

def collect(root_work_dir):
    root=Path(root_work_dir); rows=[]
    for stage in STAGE_ORDER:
        for domain in EVAL_DOMAINS:
            d=root/'eval'/stage/domain
            log=d/'eval.log'
            met=parse_eval_log(log)
            if not met: continue
            rows.append(dict(stage=stage,adapted_domain=ADAPTED_DOMAIN[stage],adapted_domains_so_far=ADAPTED_SO_FAR[stage],checkpoint=str(root/'stage_snapshots'/(stage+'.pth')),eval_domain=domain,result_file=str(d/'scalabel_results.json') if (d/'scalabel_results.json').exists() else '',log_file=str(log),**met))
    return rows

def save_long(rows,out):
    fields=['stage','adapted_domain','adapted_domains_so_far','checkpoint','eval_domain','HOTA','MOTA','IDF1','AssA','DetA','LocA','mAP','mAP50','result_file','log_file']
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fields); w.writeheader(); [w.writerow({k:r.get(k,'') for k in fields}) for r in rows]
    print('✓ Saved:',out)

def save_matrices(rows,outdir):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    by={(r['stage'],r['eval_domain']):r for r in rows}
    for metric in ['HOTA','MOTA','IDF1','AssA','DetA']:
        out=outdir/f'seq3_matrix_{metric.lower()}.csv'
        with out.open('w',newline='') as f:
            w=csv.writer(f); w.writerow(['stage']+EVAL_DOMAINS)
            for stage in STAGE_ORDER:
                w.writerow([stage]+[by.get((stage,d),{}).get(metric,'') for d in EVAL_DOMAINS])
        print('✓ Saved:',out)

def save_md(rows,out):
    by={(r['stage'],r['eval_domain']):r for r in rows}
    lines=['# BDD Continuous Sequential 3-Target DARTH Episode Results','','## Setup','- Source: clear','- Target order: dawn_dusk → night → partly_cloudy','- Normal flow: one Python training episode, source checkpoint loaded once, in-memory model across stages','','## MOTA Matrix','','| Stage | Adapted domains so far | dawn_dusk | night | partly_cloudy |','|---|---|---:|---:|---:|']
    for stage in STAGE_ORDER:
        vals=[by.get((stage,d),{}).get('MOTA','') for d in EVAL_DOMAINS]
        lines.append(f"| {stage} | {ADAPTED_SO_FAR[stage]} | {vals[0]} | {vals[1]} | {vals[2]} |")
    lines += ['','## Detailed Metrics','','| Stage | Eval domain | HOTA | MOTA | IDF1 | AssA | DetA |','|---|---|---:|---:|---:|---:|---:|']
    for stage in STAGE_ORDER:
        for d in EVAL_DOMAINS:
            r=by.get((stage,d),{})
            lines.append(f"| {stage} | {d} | {r.get('HOTA','')} | {r.get('MOTA','')} | {r.get('IDF1','')} | {r.get('AssA','')} | {r.get('DetA','')} |")
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text('\n'.join(lines)+'\n')
    print('✓ Saved:',out)

def main():
    if len(sys.argv)<2:
        print('Usage: python tools/collect_bdd_seq3_results.py <root_work_dir>'); sys.exit(1)
    root=sys.argv[1]; rows=collect(root)
    if not rows:
        print('ERROR: No results found'); sys.exit(1)
    print(f'Found {len(rows)} entries')
    out=Path(root)/'summaries'; save_long(rows,out/'seq3_results_long.csv'); save_matrices(rows,out); save_md(rows,out/'seq3_results_summary.md')
    print('Done. View:',out/'seq3_results_summary.md')
if __name__=='__main__': main()
