"""Complete pure-prediction versus CE/joint continual comparisons."""
import argparse
import json
from pathlib import Path
import sys

import torch

from .continual_analysis import summarize
from .development_analysis import compare_domains
from .train_source import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pure', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads((args.frozen / 'complete.json').read_text())
    matrices, metrics, protocols, budgets, reports = {}, {}, {}, {}, {}
    order = None
    for root in [args.prior, args.pure]:
        complete = json.loads((root / 'complete.json').read_text())
        protocol = complete['protocol']
        assert not protocol['target_GT_in_training'] and not protocol['sealed_test_used']
        assert protocol['dataset'] == baseline['protocol']['dataset']
        if order is None:
            order = protocol['order']
        assert protocol['order'] == order
        protocols[str(root)] = protocol
        snapshots = json.loads((root / 'snapshots.json').read_text())
        for control in protocol['controls']:
            name = control['name']
            assert name not in matrices
            matrix = complete['matrix'][name]
            fingerprint = sha256(control['checkpoint'])
            reference = next(n for n, s in baseline['protocol']['checkpoint_sha256'].items()
                             if s == fingerprint)
            assert matrix['0'] == baseline['matrix'][reference], 'Actual source stage0 differs'
            matrices[name] = matrix
            metrics[name] = summarize(matrix, order)
            reports[name] = {d: json.loads((root / 'evaluation' / f'stage{len(order)-1}' /
                                           name / (d + '_metrics.json')).read_text()) for d in order}
            budgets[name] = {}
            for stage in range(1, len(order)):
                checkpoint = snapshots[stage][name]
                state = torch.load(checkpoint, map_location='cpu')
                steps = sorted({int(v['step']) for v in state['optimizer']['state'].values()})
                assert steps == [state['updates']]
                budgets[name][str(stage)] = dict(checkpoint=checkpoint,
                    checkpoint_sha256=sha256(checkpoint), frames=state['frames'],
                    updates=state['updates'], optimizer_steps=steps,
                    delayed_confirmed=state['stats']['delayed_confirmed'], seen=state['seen'],
                    memory=len(state['memory']))
    assert len({v[str(len(order)-1)]['frames'] for v in budgets.values()}) == 1
    p0, pr = 'predictive_replay0_none', 'predictive_replay128_none'
    pairs = [(p0, 'discriminative_replay0_none'), (pr, 'discriminative_replay128_none'),
             (p0, 'predictive_ranking_replay0_none'), (pr, 'predictive_ranking_replay128_none'),
             (pr, p0)]
    comparisons = {a + ':' + b: compare_domains(reports[b], reports[a]) for a, b in pairs}
    for reference in ['predictive', 'geometry_null']:
        frozen = {d: json.loads((args.frozen / 'metrics' / reference / (d + '.json')).read_text())
                  for d in order}
        for candidate in [p0, pr]:
            comparisons[candidate + ':actual_frozen_' + reference] = compare_domains(frozen, reports[candidate])
    result = dict(status='COMPLETE_PURE_CE_JOINT_NATIVE_CONTINUAL_COMPARISON',
                  dataset=baseline['protocol']['dataset'], seed=17, order=order,
                  domain_matrices=matrices, metrics=metrics, actual_boundary_budgets=budgets,
                  actual_protocols=protocols, paired_final_comparisons=comparisons,
                  actual_frozen_reference=baseline['matrix'],
                  limitation='Matched nominal source/tracker/update settings; each self-teacher '
                             'trajectory supplies different pseudo pairs and actual updates, '
                             'reported explicitly. Frozen source initialization differences '
                             'must be separated from adaptation benefit. Fixed-seed development '
                             'uncertainty, not selection-adjusted, multi-seed or held-out evidence.',
                  invocation=[sys.executable, '-m', 'jepa_cmot.objective_continual_analysis'] + sys.argv[1:],
                  analysis_code_sha256=sha256(__file__), target_GT_in_training=False, sealed_used=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print('PURE_OBJECTIVE_CONTINUAL_COMPARISONS_COMPLETE',
          {n: m['HOTA']['final_equal_domain_average'] for n, m in metrics.items()}, flush=True)


if __name__ == '__main__':
    main()
