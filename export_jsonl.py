"""Export LongPIBench examples to JSONL for prompt-injection detection.

Each output line is a JSON object:
    {"text": ..., "label": 0 | 1, "suite": ..., "item_id": ..., "attack": ..., "goal": ...}

`text` is the data prompt the benchmark feeds to a model/detector (the same string
`longpibench detect` scores). `label` is 0 for clean data (attack "no") and 1 for
data containing an injected instruction.

Usage:
    pip install git+https://github.com/liu00222/LongPIBench.git
    longpibench download-data            # or pass --data-dir to an existing snapshot
    python export_jsonl.py --suite email --output data/email.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from longpibench import benchmark
from longpibench.attacks import ATTACKS, GOALS
from longpibench.benchmark import SUITES, prepare_example
from longpibench.data import ITEM_IDS

INJECTED_ATTACKS = tuple(attack for attack in ATTACKS if attack != "no")


def iter_records(suites, attacks, goals, include_clean, with_system_prompt):
    for suite in suites:
        conditions = [("no", "0")] if include_clean else []
        conditions += [(attack, goal) for attack in attacks for goal in goals]
        for attack, goal in conditions:
            for item_id in ITEM_IDS:
                example = prepare_example(suite, item_id, attack, goal)
                text = example.data_prompt
                if with_system_prompt:
                    text = f"{example.system_prompt}\n\n{text}"
                yield {
                    "text": text,
                    "label": int(attack != "no"),
                    "suite": suite,
                    "item_id": item_id,
                    "attack": attack,
                    "goal": goal,
                }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--suite", choices=(*SUITES, "all"), default="email")
    parser.add_argument(
        "--attacks", nargs="+", choices=INJECTED_ATTACKS, default=list(INJECTED_ATTACKS)
    )
    parser.add_argument("--goals", nargs="+", choices=GOALS, default=list(GOALS))
    parser.add_argument("--no-clean", action="store_true", help="omit label-0 clean examples")
    parser.add_argument(
        "--with-system-prompt",
        action="store_true",
        help="prepend the task's system prompt to text",
    )
    parser.add_argument("--data-dir", type=Path, help="dataset snapshot root (default: package)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.data_dir:
        benchmark.DATA_ROOT = args.data_dir.resolve()
    suites = SUITES if args.suite == "all" else (args.suite,)

    counts = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in iter_records(
            suites, args.attacks, sorted(set(args.goals)), not args.no_clean, args.with_system_prompt
        ):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[record["label"]] += 1
    print(f"Wrote {sum(counts.values())} lines to {args.output} (label 0: {counts[0]}, label 1: {counts[1]})")


if __name__ == "__main__":
    main()
