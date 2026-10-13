"""Create a fixed random Target cohort from label-free acquisition evidence."""

import argparse
import math
import numpy as np
from wapa.utils.io import read_rows, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    users = sorted({r["original_persona_id"] for r in read_rows(args.evidence)})
    if not users or not math.isfinite(args.fraction) or not 0 < args.fraction <= 1:
        raise ValueError("nonempty users and fraction in (0,1] required")
    selected = sorted(
        np.random.default_rng(args.seed)
        .choice(users, max(1, int(len(users) * args.fraction)), replace=False)
        .tolist()
    )
    write_json(
        args.output,
        {"target_group": selected, "seed": args.seed, "fraction": args.fraction},
    )
    print("Copy target_group into acquisition welfare and evaluation configurations.")


if __name__ == "__main__":
    main()
