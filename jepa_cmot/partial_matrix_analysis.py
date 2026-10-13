"""Verified completed-stage evidence; never fill missing continual stages."""
import argparse
import json
from pathlib import Path

import torch

from .development_analysis import compare_domains
from .train_source import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--stages', type=int, nargs='+', required=True)
    parser.add_argument('--comparisons', nargs='*', default=[])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.sequence/'protocol.json').read_text())
    matrix = json.loads((args.sequence/'continual_matrix_partial.json').read_text())
    baseline = json.loads((args.baseline/'complete.json').read_text())
    snapshots = json.loads((args.sequence/'snapshots.json').read_text())
    domains = protocol['order']
    baseline_models = baseline['protocol']['checkpoint_sha256']
    source_parity = {}
    for control in protocol['controls']:
        fingerprint = sha256(control['checkpoint'])
        reference = next(name for name, value in baseline_models.items() if value == fingerprint)
        if matrix[control['name']]['0'] != baseline['matrix'][reference]:
            raise ValueError('Initial matrix differs from actual frozen source evaluation')
        source_parity[control['name']] = dict(reference=reference, checkpoint_sha256=fingerprint)
    selected = {}
    states = {}
    deltas = {}
    paired = {}
    for stage in args.stages:
        if not (args.sequence/'evaluation'/('stage'+str(stage))/'complete.json').exists():
            raise ValueError('Requested stage has no complete native evaluation')
        selected[str(stage)] = {}
        states[str(stage)] = {}
        deltas[str(stage)] = {}
        for name, values in matrix.items():
            current = values[str(stage)]
            if set(current) != set(domains):
                raise ValueError('Requested stage is missing registered domains')
            selected[str(stage)][name] = current
            deltas[str(stage)][name] = {metric: dict(
                by_domain={d: current[d][metric]-values['0'][d][metric] for d in domains},
                equal_domain_average=sum(current[d][metric]-values['0'][d][metric] for d in domains)/len(domains))
                for metric in ['HOTA', 'AssA', 'IDF1']}
            payload = torch.load(snapshots[stage][name], map_location='cpu')
            optimizer_steps = {int(value['step']) for value in payload['optimizer']['state'].values()}
            if optimizer_steps != {payload['updates']}:
                raise ValueError('Actual optimizer steps differ from the reported learner count')
            states[str(stage)][name] = dict(frames=payload['frames'], updates=payload['updates'],
                confirmed=payload['stats']['delayed_confirmed'], seen=payload['seen'], memory=len(payload['memory']),
                checkpoint_sha256=sha256(snapshots[stage][name]))
        if protocol.get('actual_update_counts_matched_each_frame'):
            if len({(v['frames'], v['updates'], v['confirmed'], v['seen']) for v in states[str(stage)].values()}) != 1:
                raise ValueError('Fixed-teacher source/update budgets differ')
        for comparison in args.comparisons:
            candidate, reference = comparison.split(':')
            def reports(name):
                return {d: json.loads((args.sequence/'evaluation'/('stage'+str(stage))/name/(d+'_metrics.json')).read_text())
                        for d in domains}
            paired.setdefault(str(stage), {})[comparison] = compare_domains(reports(reference), reports(candidate))
    result = dict(status='VERIFIED_COMPLETED_STAGES_NOT_FINAL_CONTINUAL_RESULT', dataset=protocol['dataset'],
        stages=args.stages, order=domains, actual_source_stage0_parity=source_parity, actual_boundary_states=states,
        stage_metrics=selected, delta_from_actual_stage0=deltas, paired_comparisons=paired,
        target_GT_used_for_training=protocol['target_GT_in_training'], sealed_used=protocol['sealed_test_used'],
        limitation='no final forgetting/BWT/FWT before complete matrix; fixed-seed development comparisons, no selection-adjusted or cross-seed claim')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print('COMPLETED_STAGE_ANALYSIS_VERIFIED', args.sequence, args.stages, flush=True)


if __name__ == '__main__':
    main()
