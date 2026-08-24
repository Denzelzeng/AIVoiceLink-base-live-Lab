from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("events", type=Path)
    args = parser.parse_args()
    metrics: dict[str, list[float]] = {}
    with args.events.open("r", encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("kind") != "turn.metrics":
                continue
            for name, value in event.get("data", {}).items():
                if isinstance(value, (int, float)):
                    metrics.setdefault(name, []).append(float(value))
    result = {
        name: {
            "count": len(values),
            "p50_ms": statistics.median(values),
            "p95_ms": percentile(values, 0.95),
            "p99_ms": percentile(values, 0.99),
            "min_ms": min(values),
            "max_ms": max(values),
        }
        for name, values in metrics.items()
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

