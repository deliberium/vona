#!/usr/bin/env python3
"""Plan corpus sizes for statistically meaningful vona-wake real evaluations."""

from __future__ import annotations

import argparse
import json
import math


Z_95_TWO_SIDED = 1.959963984540054
Z_95_ONE_SIDED = 1.6448536269514722


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observed-precision",
        type=float,
        default=0.98,
        help="Expected observed precision for planning (default: 0.98)",
    )
    parser.add_argument(
        "--precision-lower-bound",
        type=float,
        default=0.95,
        help="Required 95%% Wilson lower bound for precision (default: 0.95)",
    )
    parser.add_argument(
        "--observed-recall",
        type=float,
        default=0.98,
        help="Expected observed recall for planning (default: 0.98)",
    )
    parser.add_argument(
        "--recall-lower-bound",
        type=float,
        default=0.95,
        help="Required 95%% Wilson lower bound for recall (default: 0.95)",
    )
    parser.add_argument(
        "--false-wake-events",
        type=int,
        default=0,
        help="Allowed observed false wake events for exposure planning (default: 0)",
    )
    parser.add_argument(
        "--false-wakes-per-hour-upper-bound",
        type=float,
        default=0.05,
        help="Required 95%% upper confidence bound for false wakes/hour (default: 0.05)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a shell-friendly summary",
    )
    args = parser.parse_args()

    precision_trials = required_wilson_trials(
        args.observed_precision, args.precision_lower_bound
    )
    recall_trials = required_wilson_trials(args.observed_recall, args.recall_lower_bound)
    negative_audio_hours = required_negative_audio_hours(
        args.false_wake_events, args.false_wakes_per_hour_upper_bound
    )

    result = {
        "confidence_level": 0.95,
        "observed_precision": args.observed_precision,
        "precision_lower_bound": args.precision_lower_bound,
        "minimum_precision_trials": precision_trials,
        "observed_recall": args.observed_recall,
        "recall_lower_bound": args.recall_lower_bound,
        "minimum_positive_cases": recall_trials,
        "false_wake_events": args.false_wake_events,
        "false_wakes_per_hour_upper_bound": args.false_wakes_per_hour_upper_bound,
        "minimum_negative_audio_hours": negative_audio_hours,
        "minimum_negative_audio_seconds": math.ceil(negative_audio_hours * 3600),
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"confidence_level={result['confidence_level']:.2f}")
        print(f"minimum_precision_trials={precision_trials}")
        print(f"minimum_positive_cases={recall_trials}")
        print(f"minimum_negative_audio_hours={negative_audio_hours:.3f}")
        print(
            f"minimum_negative_audio_seconds={result['minimum_negative_audio_seconds']}"
        )
    return 0


def required_wilson_trials(observed_rate: float, required_lower_bound: float) -> int:
    if not 0.0 < observed_rate <= 1.0:
        raise SystemExit("observed rates must be in the range (0, 1]")
    if not 0.0 < required_lower_bound < observed_rate:
        raise SystemExit("required lower bounds must be greater than 0 and below observed rates")

    trials = 1
    while True:
        successes = math.floor(observed_rate * trials)
        if successes == 0:
            trials += 1
            continue
        lower_bound = wilson_lower_bound(successes, trials)
        if lower_bound >= required_lower_bound:
            return trials
        trials += 1


def wilson_lower_bound(successes: int, trials: int) -> float:
    z = Z_95_TWO_SIDED
    n = float(trials)
    p = successes / n
    z2 = z * z
    center = p + z2 / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    denominator = 1.0 + z2 / n
    return max(0.0, (center - margin) / denominator)


def required_negative_audio_hours(
    false_wake_events: int, required_upper_bound: float
) -> float:
    if false_wake_events < 0:
        raise SystemExit("false wake events must be zero or greater")
    if required_upper_bound <= 0.0:
        raise SystemExit("false wakes/hour upper bound must be greater than 0")
    return poisson_count_upper_95(false_wake_events) / required_upper_bound


def poisson_count_upper_95(events: int) -> float:
    if events == 0:
        return -math.log(0.05)
    degrees_of_freedom = 2.0 * (events + 1.0)
    chi_square_upper = degrees_of_freedom * (
        1.0
        - 2.0 / (9.0 * degrees_of_freedom)
        + Z_95_ONE_SIDED * math.sqrt(2.0 / (9.0 * degrees_of_freedom))
    ) ** 3
    return 0.5 * chi_square_upper


if __name__ == "__main__":
    raise SystemExit(main())
