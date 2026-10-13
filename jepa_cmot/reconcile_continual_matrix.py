"""Reconcile complete continual-control metric files into a verified matrix.

This is used only when the control runner has written every native stage and
metric file but its final matrix write was interrupted.  It never invents a
score: each cell must come from an existing ``*_metrics.json`` file under a
stage with a native ``complete.json`` marker.
"""
import argparse
import json
from pathlib import Path


def _metric(path):
    payload = json.loads(path.read_text())
    if not isinstance(payload.get("metrics"), dict):
        raise ValueError(f"metric payload has no metrics object: {path}")
    return payload["metrics"]


def _reconcile_matrix(sequence, protocol, partial, stages):
    controls = [item["name"] for item in protocol["controls"]]
    order = protocol["order"]
    matrix = {name: dict(values) for name, values in partial.items()}
    if set(matrix) != set(controls):
        raise ValueError("partial matrix controls differ from protocol")
    for stage in stages:
        marker = sequence / "evaluation" / f"stage{stage}" / "complete.json"
        if not marker.is_file():
            raise ValueError(f"stage has no native complete marker: {marker}")
        for name in controls:
            stage_dir = sequence / "evaluation" / f"stage{stage}" / name
            values = {}
            for domain in order:
                metric_path = stage_dir / f"{domain}_metrics.json"
                if not metric_path.is_file():
                    raise ValueError(f"missing metric file: {metric_path}")
                values[domain] = _metric(metric_path)
            matrix[name][str(stage)] = values
    for name in controls:
        for stage in stages:
            if set(matrix[name].get(str(stage), {})) != set(protocol["order"]):
                raise ValueError(f"stage {stage} is incomplete for {name}")
    return matrix


def _forgetting(matrix, order):
    last_stage = len(order) - 1
    result = {}
    for name, stages in matrix.items():
        values = {}
        for domain in order[:-1]:
            start = 0 if domain == order[0] else order.index(domain) + 1
            values[domain] = max(
                stages[str(stage)][domain]["HOTA"]
                for stage in range(start, last_stage + 1)
            ) - stages[str(last_stage)][domain]["HOTA"]
        result[name] = dict(by_domain=values, average=sum(values.values()) / len(values))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--stages", type=int, nargs="+", required=True)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    sequence = args.sequence
    protocol = json.loads((sequence / "protocol.json").read_text())
    partial_path = sequence / "continual_matrix_partial.json"
    partial = json.loads(partial_path.read_text()) if partial_path.exists() else {}
    stages = sorted(set(args.stages))
    expected = list(range(len(protocol["order"])))
    if any(stage < 0 or stage >= len(protocol["order"]) for stage in stages):
        raise ValueError("requested stage is outside protocol order")
    matrix = _reconcile_matrix(sequence, protocol, partial, stages)
    partial_path.write_text(json.dumps(matrix, indent=2) + "\n")
    if args.finalize:
        if stages != expected:
            raise ValueError("--finalize requires every stage in protocol order")
        complete = dict(
            protocol=protocol,
            matrix=matrix,
            forgetting=_forgetting(matrix, protocol["order"]),
            reference="historical-best-after-domain-learned",
            reconciled_from_complete_metric_files=True,
        )
        (sequence / "complete.json").write_text(json.dumps(complete, indent=2) + "\n")
    print("CONTINUAL_MATRIX_RECONCILED", sequence, stages, "finalized=" + str(args.finalize), flush=True)


if __name__ == "__main__":
    main()
