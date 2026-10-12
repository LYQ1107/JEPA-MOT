#!/usr/bin/env python3
import csv, os, os.path as osp, re
ROOT='work_dirs/bdd_6domain/roiweak_continual_forgetting'
METHOD='spaP1_causalOld_roiV1Weak_6domain_5e'
WORK=osp.join(ROOT,METHOD)
LONG=osp.join(ROOT,'roiweak_6domain_forgetting_long.csv')
METR=osp.join(ROOT,'roiweak_6domain_forgetting_metrics.csv')
SUM=osp.join(ROOT,'roiweak_6domain_summary.csv')
MD=osp.join(ROOT,'roiweak_6domain_forgetting_compare.md')
MAT=osp.join(ROOT,'matrices')
DOMS=['overcast','partly_cloudy','rainy','snowy','dawn_dusk','night']
CM_HDR='IDF1 MOTA MOTP FP FN IDSw Rcll Prcn MT PT ML FM'; CM_COLS=['IDF1','MOTA','MOTP','FP','FN','IDSw','Rcll','Prcn','MT','PT','ML','FM']
HOTA_HDR='AssRe AssPr AssA LocA DetRe DetPr DetA HOTA Dets IDs'; HOTA_COLS=['AssRe','AssPr','AssA','LocA','DetRe','DetPr','DetA','HOTA','Dets','IDs']
VALS=['MOTA','IDF1','HOTA','DetA','AssA','LocA','FP','FN','IDSw','Dets','IDs','Rcll','Prcn','DetPr','DetRe','FM','MOTP','MT','PT','ML']
MATS=['MOTA','IDF1','HOTA','DetA','AssA','FP','FN','IDSw','Dets','Prcn','Rcll']
FMET=['HOTA','IDF1','AssA','IDSw','MOTA','DetA','FN','FP']

def rt(p):
    try: return open(p).read().strip()
    except OSError: return ''
def slog(p):
    if not osp.isfile(p): return {}
    lines=open(p,errors='ignore').read().splitlines(); res={}
    for hdr,cols in ((CM_HDR,CM_COLS),(HOTA_HDR,HOTA_COLS)):
        for i,l in enumerate(lines):
            if re.sub(r'\s+',' ',l.strip())==hdr:
                for j in range(i+1,min(i+30,len(lines))):
                    if lines[j].strip().upper().startswith('OVERALL'):
                        for c,v in zip(cols,lines[j].split()[1:]):
                            try: res[c]=float(v)
                            except Exception: pass
                        break
                break
    return res
def stage(i,d): return f'stage{i:02d}_after_{d}'
def fnum(v):
    try: return float(v)
    except Exception: return None
def mean(xs):
    xs=[x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else None
def fmt(v): return '' if v is None else f'{v:.4f}'
def write_csv(p,fields,rows):
    os.makedirs(osp.dirname(p),exist_ok=True)
    with open(p,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows([{k:r.get(k,'') for k in fields} for r in rows])
    print(f'[WRITE] {p} ({len(rows)} rows)')

def collect():
    rows=[]
    for si,adom in enumerate(DOMS,1):
        st=stage(si,adom)
        for di,edom in enumerate(DOMS,1):
            od=osp.join(WORK,'forgetting_eval',st,'epoch_5',edom); log=osp.join(od,'eval.log'); m=slog(log)
            r=dict(group=rt(osp.join(od,'group.txt')) or 'roiweak_6domain_forgetting',method=METHOD,target_order=' '.join(DOMS),stage=st,stage_idx=si,adapted_domain=adom,eval_domain=edom,eval_domain_idx=di,eval_epoch=5,thr_override=rt(osp.join(od,'cfg_options.txt')),checkpoint=rt(osp.join(od,'checkpoint_path.txt')),eval_log=log if osp.isfile(log) else 'NOT_FOUND')
            for k in VALS: r[k]=m.get(k,'')
            rows.append(r)
    return rows
def lk(rows,metric):
    d={}
    for r in rows:
        v=fnum(r.get(metric))
        if v is not None: d[(r['eval_domain'],int(r['stage_idx']))]=v
    return d

def matrices(rows):
    os.makedirs(MAT,exist_ok=True); cols=[stage(i+1,d) for i,d in enumerate(DOMS)]
    for metric in MATS:
        L=lk(rows,metric); p=osp.join(MAT,f'{METHOD}_{metric}_matrix.csv')
        with open(p,'w',newline='') as f:
            w=csv.writer(f); w.writerow(['eval_domain']+cols)
            for d in DOMS: w.writerow([d]+[L.get((d,i),'') for i in range(1,7)])
        print(f'[WRITE] {p}')

def metrics(rows):
    out=[]
    for metric in FMET:
        L=lk(rows,metric); A=lambda d,s:L.get((d,s))
        diag=[A(DOMS[i],i+1) for i in range(6)]; final=[A(d,6) for d in DOMS]; old=DOMS[:5]
        bwt=[]; fd=[]; fb=[]
        for i,d in enumerate(old):
            cur=A(d,i+1); fin=A(d,6); hist=[A(d,j) for j in range(i+1,7) if A(d,j) is not None]
            if cur is not None and fin is not None:
                bwt.append(fin-cur); fd.append(max(0,cur-fin))
            if hist and fin is not None: fb.append(max(0,max(hist)-fin))
        future=[mean([A(d,s) for d in DOMS[s:]]) for s in range(1,6)]
        seen=[mean([A(d,s) for d in DOMS[:s]]) for s in range(1,7)]
        out.append(dict(method=METHOD,metric=metric,diag_avg=mean(diag),final_avg_all_domains=mean(final),final_avg_seen_domains=mean(final),BWT_diag=mean(bwt),AvgForgetting_diag=mean(fd),AvgForgetting_best=mean(fb),old_domain_final_avg=mean([A(d,6) for d in old]),future_domain_avg=mean(future),stage_seen_avg_1=seen[0],stage_seen_avg_2=seen[1],stage_seen_avg_3=seen[2],stage_seen_avg_4=seen[3],stage_seen_avg_5=seen[4],stage_seen_avg_6=seen[5]))
    fields=['method','metric','diag_avg','final_avg_all_domains','final_avg_seen_domains','BWT_diag','AvgForgetting_diag','AvgForgetting_best','old_domain_final_avg','future_domain_avg','stage_seen_avg_1','stage_seen_avg_2','stage_seen_avg_3','stage_seen_avg_4','stage_seen_avg_5','stage_seen_avg_6']
    rows2=[{k:(r[k] if k in ['method','metric'] else fmt(r[k])) for k in fields} for r in out]
    write_csv(METR,fields,rows2); return out

def summary(mr):
    B={r['metric']:r for r in mr}; g=lambda m,k:B.get(m,{}).get(k)
    row=dict(method=METHOD,target_order=' '.join(DOMS),HOTA_final_avg=fmt(g('HOTA','final_avg_all_domains')),HOTA_BWT=fmt(g('HOTA','BWT_diag')),HOTA_AvgForgetBest=fmt(g('HOTA','AvgForgetting_best')),HOTA_old_final=fmt(g('HOTA','old_domain_final_avg')),IDF1_final_avg=fmt(g('IDF1','final_avg_all_domains')),IDF1_BWT=fmt(g('IDF1','BWT_diag')),IDF1_AvgForgetBest=fmt(g('IDF1','AvgForgetting_best')),IDF1_old_final=fmt(g('IDF1','old_domain_final_avg')),AssA_final_avg=fmt(g('AssA','final_avg_all_domains')),AssA_BWT=fmt(g('AssA','BWT_diag')),AssA_AvgForgetBest=fmt(g('AssA','AvgForgetting_best')),AssA_old_final=fmt(g('AssA','old_domain_final_avg')),MOTA_final_avg=fmt(g('MOTA','final_avg_all_domains')),IDSw_final_avg=fmt(g('IDSw','final_avg_all_domains')))
    write_csv(SUM,list(row.keys()),[row]); return row

def md_matrix(rows,metric):
    L=lk(rows,metric); cols=[stage(i+1,d) for i,d in enumerate(DOMS)]
    lines=[f'## {metric} matrix','', '| eval domain \\ stage | '+' | '.join(cols)+' |','|---|'+'---|'*6]
    for d in DOMS: lines.append('| '+d+' | '+' | '.join('—' if L.get((d,i)) is None else f'{L[(d,i)]:.2f}' for i in range(1,7))+' |')
    return lines

def write_md(rows,sr):
    lines=['# RoI-V1Weak 6-Domain Continual Forgetting','',f"Target order: `{' -> '.join(DOMS)}`",'', '## Summary','', '| HOTA final | HOTA BWT | HOTA ForgetBest | IDF1 final | AssA final | MOTA final | IDSw final |','|---:|---:|---:|---:|---:|---:|---:|',f"| {sr['HOTA_final_avg']} | {sr['HOTA_BWT']} | {sr['HOTA_AvgForgetBest']} | {sr['IDF1_final_avg']} | {sr['AssA_final_avg']} | {sr['MOTA_final_avg']} | {sr['IDSw_final_avg']} |",'']
    for m in ['HOTA','IDF1','MOTA']: lines += md_matrix(rows,m)+['']
    lines += ['## Output files','',f'- Long table: `{LONG}`',f'- Forgetting metrics: `{METR}`',f'- Summary: `{SUM}`',f'- Matrices: `{MAT}/`']
    with open(MD,'w') as f: f.write('\n'.join(lines))
    print(f'[WRITE] {MD}')

def main():
    rows=collect(); fields=['group','method','target_order','stage','stage_idx','adapted_domain','eval_domain','eval_domain_idx','eval_epoch','thr_override']+VALS+['checkpoint','eval_log']
    write_csv(LONG,fields,rows); matrices(rows); mr=metrics(rows); sr=summary(mr); write_md(rows,sr)
if __name__=='__main__': main()
