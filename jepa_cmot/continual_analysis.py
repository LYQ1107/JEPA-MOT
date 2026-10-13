"""Summarize complete native domain matrices with explicit transfer definitions."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def _magnitude_summary(rows):
    if not rows:
        return dict(record_count=0, mean_l2=None, median_l2=None, p95_l2=None, max_l2=None,
                    mean_relative=None, median_relative=None, p95_relative=None, max_relative=None,
                    mean_parameter_l2=None)
    l2 = np.asarray([float(row['l2']) for row in rows], dtype=float)
    relative = np.asarray([float(row['relative']) for row in rows], dtype=float)
    parameter = np.asarray([float(row['parameter_l2']) for row in rows], dtype=float)
    return dict(record_count=int(len(rows)), mean_l2=float(l2.mean()), median_l2=float(np.median(l2)),
                p95_l2=float(np.percentile(l2, 95)), max_l2=float(l2.max()),
                mean_relative=float(relative.mean()), median_relative=float(np.median(relative)),
                p95_relative=float(np.percentile(relative, 95)), max_relative=float(relative.max()),
                mean_parameter_l2=float(parameter.mean()))


def update_magnitude_summary(root):
    """Collect actual learner deltas from completed adaptation video reports.

    Older experiments do not contain the new field.  They are reported as
    unavailable rather than reconstructed from gradient norms or update
    counts.  Direct per-video reports are used; their parent control summaries
    and video-completion caches are intentionally ignored.
    """
    adaptation = root / 'adaptation'
    rows = []
    report_count = 0
    missing_reports = 0
    malformed_records = 0
    report_groups = {}
    expected_controls = []
    root_report = root / 'complete.json'
    if root_report.is_file():
        try:
            payload = json.loads(root_report.read_text())
            controls = payload.get('protocol', {}).get('controls', [])
            if isinstance(controls, list):
                expected_controls = [str(item['name']) for item in controls
                                     if isinstance(item, dict) and 'name' in item]
        except (OSError, json.JSONDecodeError):
            pass
    if adaptation.is_dir():
        for path in sorted(adaptation.glob('**/complete.json')):
            try:
                report = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            stats = report.get('learner_stats')
            if not isinstance(stats, dict):
                continue
            report_count += 1
            records = stats.get('update_magnitude')
            if not isinstance(records, list):
                if int(stats.get('updates', 0)) > 0:
                    missing_reports += 1
                continue
            parts = path.relative_to(root).parts
            stage = next((part for part in parts if part.startswith('stage')), 'unknown_stage')
            domain = stage.split('_', 1)[1] if '_' in stage else stage
            # adaptation/stage_domain/video/control/video/complete.json
            video = parts[-2] if len(parts) >= 2 else 'unknown_video'
            control = parts[-3] if len(parts) >= 3 else 'unknown_control'
            group = (stage, video)
            report_groups.setdefault(group, {})[control] = 0
            for record in records:
                required = {'l2', 'relative', 'parameter_l2', 'update', 'frame'}
                if not isinstance(record, dict) or not required.issubset(record):
                    malformed_records += 1
                    continue
                try:
                    values = dict(l2=float(record['l2']), relative=float(record['relative']),
                                  parameter_l2=float(record['parameter_l2']),
                                  update=int(record['update']), frame=int(record['frame']))
                except (TypeError, ValueError, OverflowError):
                    malformed_records += 1
                    continue
                if not all(math.isfinite(values[key]) for key in ('l2', 'relative', 'parameter_l2')):
                    malformed_records += 1
                    continue
                rows.append(dict(stage=stage, domain=domain, video=video, control=control,
                                 **values))
                report_groups[group][control] += 1

    def by(key):
        groups = {}
        for row in rows:
            groups.setdefault(row[key], []).append(row)
        return {name: _magnitude_summary(values) for name, values in sorted(groups.items())}

    mismatches = []
    controls = expected_controls or sorted({control for values in report_groups.values() for control in values})
    for (stage, video), values in sorted(report_groups.items()):
        counts = {control: values.get(control, 0) for control in controls}
        missing = [control for control in controls if control not in values]
        if missing or len(set(counts.values())) != 1:
            mismatches.append(dict(stage=stage, video=video, counts=counts, missing_controls=missing))
    status = 'COMPLETE_UPDATE_MAGNITUDE_AUDIT' if rows else 'UPDATE_MAGNITUDE_UNAVAILABLE'
    return dict(status=status, source='adaptation/**/complete.json', report_count=report_count,
                expected_controls=controls, missing_update_magnitude_reports=missing_reports,
                malformed_update_magnitude_records=malformed_records, record_count=len(rows),
                matched_update_counts=(None if not report_groups else not mismatches),
                update_count_mismatches=mismatches[:20], overall=_magnitude_summary(rows),
                by_control=by('control'), by_stage=by('stage'), by_domain=by('domain'))


def summarize(matrix,order):
    count=len(order)
    if set(matrix)!={str(i) for i in range(count)}:
        raise ValueError('A final continual summary requires every stage')
    for stage in matrix.values():
        if set(stage)!=set(order):raise ValueError('A continual stage is missing a domain')
    values={metric:np.asarray([[matrix[str(stage)][domain][metric] for domain in order]
            for stage in range(count)],float) for metric in ['HOTA','AssA','IDF1']}
    report={}
    for metric,table in values.items():
        forgetting={domain:max(0.,float(table[j:count-1,j].max()-table[-1,j]))
                    for j,domain in enumerate(order[:-1])}
        signed_forgetting={domain:float(table[j:count-1,j].max()-table[-1,j])
                           for j,domain in enumerate(order[:-1])}
        backward={domain:float(table[-1,j]-table[j,j]) for j,domain in enumerate(order[:-1])}
        forward={domain:float(table[j-1,j]-table[0,j]) for j,domain in enumerate(order[1:],1)}
        report[metric]=dict(final_by_domain=dict(zip(order,table[-1].tolist())),
            final_equal_domain_average=float(table[-1].mean()),
            average_forgetting=float(np.mean(list(forgetting.values()))),forgetting_by_domain=forgetting,
            signed_average_forgetting=float(np.mean(list(signed_forgetting.values()))),
            signed_forgetting_by_domain=signed_forgetting,
            backward_transfer_by_domain=backward,backward_transfer=float(np.mean(list(backward.values()))),
            forward_transfer_by_domain=forward,forward_transfer=float(np.mean(list(forward.values()))))
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--complete',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();experiment=json.loads(args.complete.read_text())
    order=experiment['protocol']['order'];matrices=dict(experiment['matrix'])
    frozen={}
    for name,matrix in experiment['matrix'].items():
        initial=matrix['0']
        # This reference is copied from an actual complete stage0 evaluation.
        # Frozen parameters/calibration and per-video physical resets make it
        # independent of adaptation stage; no scores are inferred for new inputs.
        frozen[name]={str(i):initial for i in range(len(order))}
    result=dict(dataset=experiment['protocol']['dataset'],order=order,
        metrics={name:summarize(matrix,order) for name,matrix in matrices.items()},
        frozen_source_checkpoint_references={name:summarize(matrix,order) for name,matrix in frozen.items()},
        update_magnitude=update_magnitude_summary(args.complete.parent),
        frozen_reference_scope='actual stage0 weights/inputs; repeat the same immutable evaluation at each stage',
        definitions=dict(final_average='equal weight per registered domain; no reweighting by video count',
            forgetting='max(0, best learned-domain score before final stage minus final score)',
            signed_forgetting='best learned-domain score before final minus final, may be negative',
            backward_transfer='final minus score immediately after that domain was learned; exclude final new domain',
            forward_transfer='score before learning each target domain minus its actual stage0 score'))
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2))
    print('CONTINUAL_ANALYSIS_COMPLETE',args.out,flush=True)


if __name__=='__main__':main()
