#!/usr/bin/env python3
"""Collect night-related baseline results and merge with old table."""
import csv, re, sys, os
from pathlib import Path
import pandas as pd

def parse_eval_log(log_path):
    """Parse HOTA/MOTA/IDF1 from eval log."""
    if not Path(log_path).exists():
        return None
    text = Path(log_path).read_text(errors='ignore')
    
    # OVERALL line: LocA DetA MOTA HOTA IDF1 AssA
    m = re.findall(r'OVERALL\s+([\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', text)
    if not m:
        return None
    LocA, DetA, MOTA, HOTA, IDF1, AssA = [float(x) for x in m[-1]]
    
    # Per-category metrics
    car_mota = car_idf1 = ped_mota = ped_idf1 = None
    car_m = re.findall(r'car\s+[\d.]+\s+[\d.]+\s+([-\d.]+)\s+[\d.]+\s+([\d.]+)\s+[\d.]+', text)
    if car_m:
        car_mota, car_idf1 = float(car_m[-1][0]), float(car_m[-1][1])
    ped_m = re.findall(r'pedestrian\s+[\d.]+\s+[\d.]+\s+([-\d.]+)\s+[\d.]+\s+([\d.]+)\s+[\d.]+', text)
    if ped_m:
        ped_mota, ped_idf1 = float(ped_m[-1][0]), float(ped_m[-1][1])
    
    return {
        'HOTA': round(HOTA, 1),
        'MOTA': round(MOTA, 1),
        'IDF1': round(IDF1, 1),
        'DetA': round(DetA, 1),
        'AssA': round(AssA, 1),
        'LocA': round(LocA, 1),
        'car_MOTA': round(car_mota, 1) if car_mota is not None else '',
        'car_IDF1': round(car_idf1, 1) if car_idf1 is not None else '',
        'ped_MOTA': round(ped_mota, 1) if ped_mota is not None else '',
        'ped_IDF1': round(ped_idf1, 1) if ped_idf1 is not None else '',
    }

def collect_source_only(root):
    """Collect source-only eval results."""
    rows = []
    root = Path(root)
    eval_dir = root / 'source_only_eval'
    if not eval_dir.exists():
        return rows
    
    for src_dir in eval_dir.iterdir():
        if not src_dir.is_dir():
            continue
        # source_<domain>_epoch20
        m = re.match(r'source_(\w+)_epoch(\d+)', src_dir.name)
        if not m:
            continue
        source, epoch = m.groups()
        
        for tgt_dir in src_dir.iterdir():
            if not tgt_dir.is_dir():
                continue
            target = tgt_dir.name
            log = tgt_dir / 'eval.log'
            met = parse_eval_log(log)
            if not met:
                continue
            
            # Determine mode: source_self_eval if source==target, else source_only
            mode = 'source_self_eval' if source == target else 'source_only'
            
            rows.append({
                'mode': mode,
                'source': source,
                'target': target,
                'epoch': int(epoch),
                **met,
                'checkpoint': str(root / 'source' / f'{source}_20e' / f'epoch_{epoch}.pth'),
                'eval_dir': str(tgt_dir),
                'log_file': str(log),
            })
    return rows

def collect_tta(root):
    """Collect TTA eval results."""
    rows = []
    root = Path(root)
    eval_dir = root / 'tta_eval'
    if not eval_dir.exists():
        return rows
    
    for pair_dir in eval_dir.iterdir():
        if not pair_dir.is_dir():
            continue
        # <source>_to_<target>
        m = re.match(r'(\w+)_to_(\w+)', pair_dir.name)
        if not m:
            continue
        source, target = m.groups()
        
        for epoch_dir in pair_dir.iterdir():
            if not epoch_dir.is_dir():
                continue
            # epoch06, epoch12, epoch20
            m2 = re.match(r'epoch(\d+)', epoch_dir.name)
            if not m2:
                continue
            epoch = int(m2.group(1))
            
            for tgt_dir in epoch_dir.iterdir():
                if not tgt_dir.is_dir():
                    continue
                # Should match target
                if tgt_dir.name != target:
                    continue
                log = tgt_dir / 'eval.log'
                met = parse_eval_log(log)
                if not met:
                    continue
                
                rows.append({
                    'mode': 'tta_eval',
                    'source': source,
                    'target': target,
                    'epoch': epoch,
                    **met,
                    'checkpoint': str(root / 'tta' / f'{source}_to_{target}_20e' / f'epoch_{epoch}.pth'),
                    'eval_dir': str(tgt_dir),
                    'log_file': str(log),
                })
    return rows

def save_long_csv(rows, out):
    """Save collected results to CSV."""
    fields = ['mode', 'source', 'target', 'epoch', 'MOTA', 'HOTA', 'IDF1', 'DetA', 'AssA', 'LocA',
              'car_MOTA', 'car_IDF1', 'ped_MOTA', 'ped_IDF1', 'checkpoint', 'eval_dir', 'log_file']
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})
    print(f'✓ Saved: {out}')

def merge_with_old(new_rows, old_csv, out):
    """Merge new night-related results with old table."""
    old_df = pd.read_csv(old_csv)
    print(f'\nOld table: {len(old_df)} rows')
    
    # Remove old night-related rows
    old_non_night = old_df[~((old_df['source'] == 'night') | (old_df['target'] == 'night'))]
    removed = len(old_df) - len(old_non_night)
    print(f'Removed night-related: {removed} rows')
    
    # Convert new rows to DataFrame
    new_df = pd.DataFrame(new_rows)
    print(f'New night-related: {len(new_df)} rows')
    
    # Merge
    merged = pd.concat([old_non_night, new_df], ignore_index=True)
    merged = merged.sort_values(['mode', 'source', 'target', 'epoch']).reset_index(drop=True)
    
    # Keep only columns from old table
    old_cols = ['mode', 'source', 'target', 'epoch', 'MOTA', 'HOTA', 'IDF1', 'DetA', 'AssA', 'LocA',
                'car_MOTA', 'car_IDF1', 'ped_MOTA', 'ped_IDF1']
    merged = merged[old_cols]
    
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f'✓ Merged table: {len(merged)} rows')
    print(f'✓ Saved: {out}')

def save_summary_md(rows, out):
    """Save markdown summary."""
    lines = ['# Night-related Baseline Results (Rerun)', '', '## Summary', '']
    
    # Count by mode
    by_mode = {}
    for r in rows:
        mode = r['mode']
        by_mode[mode] = by_mode.get(mode, 0) + 1
    
    lines.append(f"Total entries: {len(rows)}")
    for mode, count in sorted(by_mode.items()):
        lines.append(f"- {mode}: {count}")
    
    lines += ['', '## Source-only Results', '', '| Source | Target | HOTA | MOTA | IDF1 |', '|---|---|---:|---:|---:|']
    for r in sorted(rows, key=lambda x: (x['mode'], x['source'], x['target'])):
        if r['mode'] in ['source_only', 'source_self_eval']:
            lines.append(f"| {r['source']} | {r['target']} | {r['HOTA']} | {r['MOTA']} | {r['IDF1']} |")
    
    lines += ['', '## TTA Results (epoch 6, 12, 20)', '', '| Source | Target | Epoch | HOTA | MOTA | IDF1 |', '|---|---|---:|---:|---:|---:|']
    for r in sorted(rows, key=lambda x: (x['source'], x['target'], x['epoch'])):
        if r['mode'] == 'tta_eval':
            lines.append(f"| {r['source']} | {r['target']} | {r['epoch']} | {r['HOTA']} | {r['MOTA']} | {r['IDF1']} |")
    
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n')
    print(f'✓ Saved: {out}')

def main():
    if len(sys.argv) < 3:
        print('Usage: python tools/collect_bdd_night_related_baselines.py <root_work_dir> <old_csv>')
        sys.exit(1)
    
    root = sys.argv[1]
    old_csv = sys.argv[2]
    
    print('Collecting results...')
    srconly = collect_source_only(root)
    tta = collect_tta(root)
    rows = srconly + tta
    
    if not rows:
        print('ERROR: No results found')
        sys.exit(1)
    
    print(f'Found {len(rows)} entries ({len(srconly)} source-only, {len(tta)} TTA)')
    
    out_dir = Path(root) / 'summaries'
    save_long_csv(rows, out_dir / 'night_related_results_long.csv')
    save_summary_md(rows, out_dir / 'night_related_summary.md')
    
    if Path(old_csv).exists():
        merge_with_old(rows, old_csv, out_dir / 'complete_metrics_all_epochs_night_rerun_merged.csv')
    
    print('\nDone.')

if __name__ == '__main__':
    main()
