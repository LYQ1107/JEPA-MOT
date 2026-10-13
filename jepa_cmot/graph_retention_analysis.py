"""Complete fixed-teacher retention comparisons, including generic RKD-D."""
import argparse
import json
from pathlib import Path
import sys

from .continual_analysis import summarize
from .development_analysis import compare_domains
from .train_source import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--rkd', type=Path, required=True)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--budget-audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    matrices, protocols, reports, summaries = {}, {}, {}, {}
    order = None
    dataset = None
    for root in [args.graph, args.rkd]:
        complete = json.loads((root / 'complete.json').read_text())
        protocol = complete['protocol']
        assert not protocol['target_GT_in_training'] and not protocol['sealed_test_used']
        assert protocol['actual_update_counts_matched_each_frame']
        if order is None:
            order, dataset = protocol['order'], protocol['dataset']
        assert protocol['order'] == order and protocol['dataset'] == dataset
        protocols[str(root)] = protocol
        for name, matrix in complete['matrix'].items():
            assert name not in matrices
            summaries[name] = summarize(matrix, order)  # Requires all actual stages/domains.
            matrices[name] = matrix
            reports[name] = {d: json.loads((root / 'evaluation' / f'stage{len(order)-1}' /
                                           name / (d + '_metrics.json')).read_text()) for d in order}
    names = ['predictive_ranking_' + r + '_graph' for r in
             ['none', 'features', 'relations', 'competition', 'distance']]
    assert set(matrices) == set(names)
    initial = matrices[names[0]]['0']
    assert all(matrices[n]['0'] == initial for n in names), 'Actual source stage0 differs'
    # The preceding audit compares actual ordered buffers and optimizer steps.
    budget = json.loads(args.budget_audit.read_text())
    assert budget['scope']['dataset'] == dataset and budget['scope']['order'] == order
    assert {s['stage'] for s in budget['stages']} == {1, 2, 3}
    assert set(budget['protocols']) == {str(args.graph), str(args.rkd)}
    for root, fingerprint in budget['protocols'].items():
        assert sha256(Path(root) / 'protocol.json') == fingerprint
    for stage in budget['stages']:
        assert {c['control'] for c in stage['controls']} == set(names)
        for control in stage['controls']:
            assert sha256(control['checkpoint']) == control['checkpoint_sha256']
    geometry = {d: json.loads((args.frozen / 'metrics/geometry_null' / (d + '.json')).read_text())
                for d in order}
    frozen = json.loads((args.frozen / 'complete.json').read_text())
    assert frozen['protocol']['dataset'] == dataset and frozen['checkpoint_hashes_unchanged']
    none, feature, row, column, distance = names
    comparisons = {}
    for candidate, reference in [(column, distance), (distance, row), (distance, none),
                                 (column, row), (column, none), (column, feature)]:
        comparisons[candidate + ':' + reference] = compare_domains(reports[reference], reports[candidate])
    for candidate in names:
        comparisons[candidate + ':actual_frozen_geometry'] = compare_domains(geometry, reports[candidate])
    result = dict(status='COMPLETE_FIXED_TEACHER_RETENTION_AND_RKD_MATRICES', dataset=dataset,
                  seed=17, order=order, domain_matrices=matrices, metrics=summaries,
                  actual_protocols=protocols, actual_boundary_budget=budget,
                  invocation=[sys.executable, '-m', 'jepa_cmot.graph_retention_analysis'] + sys.argv[1:],
                  analysis_code_sha256=sha256(__file__),
                  paired_final_comparisons=comparisons,
                  frozen_geometry={d: r['metrics'] for d, r in geometry.items()},
                  native_frames_by_domain={d: sum(v['native_frames'] for v in r['videos'].values())
                                           for d, r in geometry.items()},
                  uncertainty_scope='Fixed-seed development, paired videos within each domain; '
                                    'no cross-seed or method-selection adjustment.',
                  limitation='Boundary sample parity does not prove every training batch equal; '
                             'generic RKD and column retention have different compute costs. '
                             'Full matrices and final paired scores do not establish novelty.',
                  target_GT_in_training=False, sealed_used=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print('COMPLETE_GRAPH_RKD_COMPARISONS',
          {n: (s['HOTA']['final_equal_domain_average'], s['HOTA']['average_forgetting'])
           for n, s in summaries.items()}, flush=True)


if __name__ == '__main__':
    main()
