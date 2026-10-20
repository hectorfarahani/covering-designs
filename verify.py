#!/usr/bin/env python3
"""Stand-alone verifier for covering designs (Python 3 standard library only).
Usage: python verify.py V K T FILE
FILE: one block per line, K distinct integers in 1..V separated by spaces.
Checks that every T-element subset of {1..V} lies in at least one block."""
import sys, itertools
from math import comb
def main():
    v, k, t, path = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    blocks = [list(map(int, line.split())) for line in open(path) if line.strip()]
    for i, b in enumerate(blocks, 1):
        if len(b) != k or len(set(b)) != k or not all(1 <= x <= v for x in b):
            sys.exit(f"block {i} is not a set of {k} distinct numbers in 1..{v}: {b}")
    covered = set()
    for b in blocks:
        covered.update(itertools.combinations(sorted(b), t))
    total = comb(v, t); missing = total - len(covered)
    print(f"C({v},{k},{t}): {len(blocks)} blocks, {len(set(map(tuple, map(sorted, blocks))))} distinct; "
          f"{len(covered)} of {total} {t}-subsets covered; missing {missing}")
    sys.exit(0 if missing == 0 else 1)
if __name__ == "__main__":
    main()
