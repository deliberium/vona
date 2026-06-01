#!/usr/bin/env python3
"""Select vona-wake thresholds from calibration split and verify evaluation split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-report", type=Path, required=True)
    parser.add_argument("--min-precision", type=float, default=0.98)
    parser.add_argument("--min-recall", type=float, default=0.98)
    parser.add_argument("--max-repeated-positive-wake-events", type=int, default=0)
    parser.add_argument("--max-false-wakes-per-hour", type=float, default=0.05)
    parser.add_argument("--max-detection-latency-ms", type=int, default=1500)
    parser.add_argument(
        "--min-calibration-passing-points",
        type=int,
        default=2,
        help="Minimum threshold-sweep points that must pass calibration gates",
    )
    parser.add_argument(
        "--min-evaluation-passing-points",
        type=int,
        default=1,
        help="Minimum threshold-sweep points that must pass evaluation gates",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--enforce", action="store_true", help="Exit non-zero unless selected threshold passes evaluation")
    args = parser.parse_args()

    report = json.loads(args.real_report.read_text(encoding="utf-8"))
    points = report.get("threshold_sweep", [])
    calibration_passing_points = passing_points(points, "calibration", args)
    evaluation_passing_points = passing_points(points, "evaluation", args)
    selected = select_point(calibration_passing_points)
    failures = []
    if selected is None:
        failures.append("no threshold point passed calibration gates")
    else:
        evaluation = group_by_id(selected.get("splits", []), "evaluation")
        if evaluation is None:
            failures.append("selected threshold is missing evaluation split metrics")
        else:
            failures.extend(metric_failures("evaluation", evaluation, args))
    if len(calibration_passing_points) < args.min_calibration_passing_points:
        failures.append(
            f"calibration passing threshold points {len(calibration_passing_points)} below required "
            f"{args.min_calibration_passing_points}"
        )
    if len(evaluation_passing_points) < args.min_evaluation_passing_points:
        failures.append(
            f"evaluation passing threshold points {len(evaluation_passing_points)} below required "
            f"{args.min_evaluation_passing_points}"
        )

    result = {
        "accepted": selected is not None and not failures,
        "real_report": str(args.real_report),
        "real_report_sha256": sha256(args.real_report),
        "corpus": report.get("corpus"),
        "threshold_sweep_points": len(points),
        "calibration_passing_points": len(calibration_passing_points),
        "evaluation_passing_points": len(evaluation_passing_points),
        "selected": selected_summary(selected) if selected else None,
        "failures": failures,
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_text_result(result)
    return 1 if args.enforce and not result["accepted"] else 0


def passing_points(points: list[dict[str, object]], split: str, args) -> list[dict[str, object]]:
    passing = []
    for point in points:
        metrics = group_by_id(point.get("splits", []), split)
        if metrics is None:
            continue
        if metric_failures(split, metrics, args):
            continue
        passing.append(point)
    return passing


def select_point(passing: list[dict[str, object]]):
    if not passing:
        return None
    return sorted(
        passing,
        key=lambda point: (
            group_by_id(point.get("splits", []), "calibration").get("recall", 0.0),
            group_by_id(point.get("splits", []), "calibration").get("precision", 0.0),
            -float(group_by_id(point.get("splits", []), "calibration").get("false_wakes_per_hour", 0.0)),
            point.get("accept_threshold", 0.0),
        ),
        reverse=True,
    )[0]


def metric_failures(prefix: str, group: dict[str, object], args) -> list[str]:
    failures: list[str] = []
    if float(group.get("precision", 0.0)) < args.min_precision:
        failures.append(f"{prefix} precision {group.get('precision')} below required {args.min_precision}")
    if float(group.get("recall", 0.0)) < args.min_recall:
        failures.append(f"{prefix} recall {group.get('recall')} below required {args.min_recall}")
    repeated = group.get("repeated_positive_wake_events")
    if repeated is None:
        failures.append(f"{prefix} repeated positive wake events is missing")
    elif int(repeated) > args.max_repeated_positive_wake_events:
        failures.append(
            f"{prefix} repeated positive wake events {repeated} exceeded allowed "
            f"{args.max_repeated_positive_wake_events}"
        )
    if float(group.get("false_wakes_per_hour", 0.0)) > args.max_false_wakes_per_hour:
        failures.append(
            f"{prefix} false wakes/hour {group.get('false_wakes_per_hour')} "
            f"exceeded allowed {args.max_false_wakes_per_hour}"
        )
    latency = group.get("max_detection_latency_ms")
    if latency is None:
        failures.append(f"{prefix} max detection latency ms is missing")
    elif float(latency) > args.max_detection_latency_ms:
        failures.append(
            f"{prefix} max detection latency ms {latency} exceeded allowed "
            f"{args.max_detection_latency_ms}"
        )
    return failures


def group_by_id(groups, group_id: str):
    for group in groups or []:
        if group.get("id") == group_id:
            return group
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_summary(point):
    if point is None:
        return None
    return {
        "candidate_threshold": point.get("candidate_threshold"),
        "accept_threshold": point.get("accept_threshold"),
        "calibration": group_by_id(point.get("splits", []), "calibration"),
        "evaluation": group_by_id(point.get("splits", []), "evaluation"),
    }


def print_text_result(result: dict[str, object]) -> None:
    print(f"status={'accepted' if result['accepted'] else 'rejected'}")
    selected = result.get("selected")
    if selected:
        print(f"threshold_sweep_points={result.get('threshold_sweep_points')}")
        print(f"calibration_passing_points={result.get('calibration_passing_points')}")
        print(f"evaluation_passing_points={result.get('evaluation_passing_points')}")
        print(f"candidate_threshold={selected['candidate_threshold']}")
        print(f"accept_threshold={selected['accept_threshold']}")
        for split in ["calibration", "evaluation"]:
            metrics = selected.get(split) or {}
            print(f"{split}_precision={metrics.get('precision')}")
            print(f"{split}_recall={metrics.get('recall')}")
            print(f"{split}_repeated_positive_wake_events={metrics.get('repeated_positive_wake_events')}")
            print(f"{split}_false_wakes_per_hour={metrics.get('false_wakes_per_hour')}")
            print(f"{split}_max_detection_latency_ms={metrics.get('max_detection_latency_ms')}")
    failures = result["failures"]
    if failures:
        print("failures:")
        for failure in failures:
            print(f"- {failure}")


if __name__ == "__main__":
    raise SystemExit(main())
