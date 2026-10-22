# covering-designs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23489072.svg)](https://doi.org/10.5281/zenodo.23489072)

New covering designs found with GPU-parallel weighted local search, together with a stand-alone verifier and the search code.

**Author:** Hossein Farahani ([hossein.pro](https://hossein.pro), research@hossein.pro, ORCID [0009-0005-0297-8082](https://orcid.org/0009-0005-0297-8082))

A (v,k,t) covering design is a set of k-element subsets (blocks) of {1,...,v} such that every t-element subset
lies in at least one block. C(v,k,t) is the smallest possible number of blocks.

## What is new

- **Fifteen improved upper bounds:** C(29,17,4) ≤ 22 (previously 23), C(32,18,4) ≤ 25 (previously 26),
  C(30,12,3) ≤ 29 (previously 30), C(28,9,3) ≤ 55 (previously 56), and, new in v1.2.0,
  C(34,11,3) ≤ 50 (previously 51), C(35,11,3) ≤ 52 (previously 56), C(38,17,3) ≤ 21 (previously 22),
  C(47,21,3) ≤ 21 (previously 22), C(48,18,3) ≤ 34 (previously 35), C(50,21,3) ≤ 25 (previously 26),
  C(52,21,3) ≤ 28 (previously 29), C(52,23,3) ≤ 22 (previously 23), C(56,24,3) ≤ 23 (previously 24) and,
  new in v1.3.0, C(47,15,3) ≤ 51 (previously 52) and C(56,22,3) ≤ 30 (previously 31). The previous records had stood for 14 to 30 years (see Results).
- **An open, reproducible GPU implementation** of weighting local search for covering designs: one complete search per
  32-thread warp, thousands of searches in parallel on a single consumer GPU, every result recounted independently on the CPU.
  v1.2.0 adds `src/covsearch_gpu_t64.py`: designs on up to 64 points, with a focused move (see Method).
  v1.3.0 adds `src/covsearch.py`: the same search for up to 127 points, and faster (see Method).
- **A simple stand-alone verifier.** Anyone can check the designs in about a second with standard Python.

The search techniques themselves are established (see Method). The contribution is their GPU implementation, its application
to covering designs, and the new bounds it produced.

## Results

| Parameters | New size | Previous best | Previous method | Lower bound |
|---|---|---|---|---|
| C(29,17,4) | **22** | 23 | Dynamic construction (LJCR, 2009) | 16 |
| C(32,18,4) | **25** | 26 | Multiple of (16,9,4) covering (LJCR, 2009) | 21 |
| C(30,12,3) | **29** | 30 | JCD article (LJCR, 1996) | 25 |
| C(28,9,3) | **55** | 56 | JCD article (LJCR, 1996) | 50 |
| C(34,11,3) | **50** | 51 | Private tools, J. de Heer and S. Muir (LJCR, 2009) | 44 |
| C(35,11,3) | **52** | 56 | Simulated annealing, M. Dambmann (LJCR, 2008) | 48 |
| C(38,17,3) | **21** | 22 | Private tools, J. de Heer and S. Muir (LJCR, 2009) | 18 |
| C(47,21,3) | **21** | 22 | Private tools, J. de Heer and S. Muir (LJCR, 2010) | 18 |
| C(48,18,3) | **34** | 35 | Multiple of (24,9,3) covering (LJCR, 2009) | 30 |
| C(50,21,3) | **25** | 26 | Private tools, J. de Heer and S. Muir (LJCR, 2012) | 22 |
| C(52,21,3) | **28** | 29 | Dynamic programming construction (LJCR, 2009) | 25 |
| C(52,23,3) | **22** | 23 | Private tools, J. de Heer and S. Muir (LJCR, 2011) | 19 |
| C(56,24,3) | **23** | 24 | Multiple of (28,12,3) covering (LJCR, 2009) | 19 |
| C(47,15,3) | **51** | 52 | Private tools, J. de Heer and S. Muir (LJCR, 2009) | 44 |
| C(56,22,3) | **30** | 31 | Multiple of (28,11,3) covering (LJCR, 2009) | 26 |

Previous best values as listed in the La Jolla Covering Repository archive (Zenodo record 19472830, v1.2, April 2026),
and checked against the live tables at coveringrepository.com on 2026-10-20, before upload (23, 26, 30 and 56 blocks).
The first four designs have been listed in the live Covering Repository tables since 2026-10-20.
C(30,12,3) and C(28,9,3) were added in v1.1.0.
C(34,11,3), C(35,11,3), C(38,17,3), C(47,21,3), C(48,18,3), C(50,21,3), C(52,21,3), C(52,23,3) and C(56,24,3) were added
in v1.2.0; their previous values are from the same archive and were checked against the live tables at coveringrepository.com
on 2026-10-22, before upload (51, 56, 22, 22, 35, 26, 29, 23 and 24 blocks).
C(47,15,3) and C(56,22,3) were added in v1.3.0; their previous values are from the same archive and were checked against the
live tables at coveringrepository.com on 2026-10-22, before upload (52 and 31 blocks).

## Verify (Python 3 standard library only, about one second)

    python verify.py 29 17 4 records/C29_17_4_22blocks.txt
    python verify.py 32 18 4 records/C32_18_4_25blocks.txt
    python verify.py 30 12 3 records/C30_12_3_29blocks.txt
    python verify.py 28 9 3 records/C28_9_3_55blocks.txt
    python verify.py 34 11 3 records/C34_11_3_50blocks.txt
    python verify.py 35 11 3 records/C35_11_3_52blocks.txt
    python verify.py 38 17 3 records/C38_17_3_21blocks.txt
    python verify.py 47 21 3 records/C47_21_3_21blocks.txt
    python verify.py 48 18 3 records/C48_18_3_34blocks.txt
    python verify.py 50 21 3 records/C50_21_3_25blocks.txt
    python verify.py 52 21 3 records/C52_21_3_28blocks.txt
    python verify.py 52 23 3 records/C52_23_3_22blocks.txt
    python verify.py 56 24 3 records/C56_24_3_23blocks.txt
    python verify.py 47 15 3 records/C47_15_3_51blocks.txt
    python verify.py 56 22 3 records/C56_22_3_30blocks.txt

All fifteen report `missing 0`. Each line of a record file is one block.

## Method

A GPU-parallel weighted local search with tabu and restart kicks. It applies established techniques to covering designs:
local search for coverings (Nurmela and Östergård, 1993), row/clause weighting with tabu for set covering
(e.g. Gao, Yao, Weise and Li, 2015, RWLS), and focused WalkSAT-style moves.

- The block count is fixed at the target size. Searches start from random, point-balanced blocks.
- Each step chooses a random uncovered t-subset and evaluates every swap that would cover it
  (one point out of one block, one point of the t-subset in; in `covsearch_gpu_t64.py`, only in the blocks with maximal
  overlap with that t-subset). It applies the best swap, with random tie-breaking.
- Uncovered t-subsets carry weights, which increase whenever no improving swap exists. A point just removed from a block
  is tabu for 40 steps. A search that has not improved for 60 s is perturbed.
- Many independent searches run at once on an NVIDIA GPU, one search per 32-thread warp. The threads of a warp share the
  scoring of candidate swaps.
- Every reported design is recounted independently on the CPU.

C(29,17,4) at 22 and C(32,18,4) at 25 were found with `src/covsearch_gpu.py` on a single consumer NVIDIA GPU with 256
parallel searches (random seed 7), in 54 s and 71 s, after the search had first reached the previous record size.
C(30,12,3) at 29 was found with `src/covsearch_gpu_t.py`, 512 parallel searches (seed 7), in 6 s. C(28,9,3) at 55 was
found with the same code, 1024 parallel searches (seed 43) and `dmin=16`, in 28 s. (`dmin` is a minimum number of blocks per
point, used only to restrict the search.)

The v1.2.0 designs were found with `src/covsearch_gpu_t64.py` in a screen of table entries on 33 to 63 points: one 60 s
attempt per entry at one block below the listed value, 2048 parallel searches from fresh random starts.
This version differs from `src/covsearch_gpu_t.py` in one move rule (a focused move): for the chosen uncovered t-subset it
only considers swaps in the blocks that already share the most points with that t-subset, instead of in every block.

| Design | Seed | `dmin` | Time to find (incl. setup) |
|---|---|---|---|
| C(34,11,3) at 50 | 1 | 15 | about 20 s |
| C(35,11,3) at 52 | 4 | 16 | about 20 s (after fresh runs at 55, 54 and 53 with seeds 1, 2 and 3) |
| C(38,17,3) at 21 | 1 | 8 | about 14 s |
| C(47,21,3) at 21 | 1 | 7 | about 11 s |
| C(48,18,3) at 34 | 1 | 11 | about 46 s |
| C(50,21,3) at 25 | 1 | 9 | about 23 s |
| C(52,21,3) at 28 | 1 | 10 | about 27 s |
| C(52,23,3) at 22 | 1 | 8 | about 40 s |
| C(56,24,3) at 23 | 1 | 8 | about 52 s |

C(47,15,3) at 51 (`dmin=15`, about 59 s) and C(56,22,3) at 30 (`dmin=10`, about 22 s) were found with `src/covsearch.py`,
2048 parallel searches, seed 1.
`src/covsearch.py` uses the same move rule as `src/covsearch_gpu_t64.py` and is faster.

Structure: the 52-block C(35,11,3) design has an automorphism group of order 6,912 that is transitive on 32 of the points.
The other 3 points lie in exactly the same 20 blocks, and the remaining 8 points of those 20 blocks cover every pair of the
32 points. The 50-block C(34,11,3) design has a similar structure, with 2 such points, 18 blocks and 9 remaining points per block.

## Code

| Path | Content |
|---|---|
| `verify.py` | stand-alone verifier |
| `records/` | the new designs |
| `src/covsearch_gpu.py` | GPU search for C(v,k,4), v <= 32 (found C(29,17,4) and C(32,18,4)) |
| `src/covsearch_gpu_t.py` | the same search for any 2 <= t <= 5, v <= 32 (found C(30,12,3) and C(28,9,3)) |
| `src/covsearch_gpu_t64.py` | search for 2 <= t <= 7, v <= 64 with the focused move (found the v1.2.0 designs) |
| `src/covsearch.py` | the same search for 2 <= t <= 7, v <= 127, faster (found the v1.3.0 designs) |
| `tests/test_calibration.py` | checks against proven values C(10,6,4)=20, C(8,4,3)=14, C(7,3,2)=7, for `covsearch_gpu_t.py`, `covsearch_gpu_t64.py` and `covsearch.py` |
| `environment.yml` | conda environment (numpy, CuPy for CUDA 12) |

Minimal use:

    import sys; sys.path.insert(0, "src")
    import covsearch_gpu_t as CT
    P = CT.Problem(29, 17, 4)
    res = CT.run(P, R=256, nb=22, dmin=0, seconds=180, seed=7)
    print(min(f["missing"] for f in res["final"]))

On Windows with a GPU that drives the display, keep each kernel launch short (the default `launch_steps=200`).
The driver resets the GPU if a single launch runs longer than about 2 s.

## AI assistance

This work was carried out with AI assistance. The search code, verification scripts, analysis and documentation were
developed together with Claude Science (Anthropic), an AI research assistant, under the author's direction. Every design was
verified independently with `verify.py`, which anyone can rerun.

## Data sources and credit

Previous bounds: La Jolla Covering Repository, D. M. Gordon, archived on Zenodo (record 19472830, v1.2), CC BY 4.0, and the
live Covering Repository tables (coveringrepository.com). No previous designs were used as input. Each search starts from random blocks and is then improved by the directed
local search described under Method.

## Licence

Designs and documentation: CC BY 4.0. Code: MIT. See `LICENSE`.

## Citation

Farahani, H. (2026). covering-designs: new covering designs by GPU-parallel weighted local search (v1.3.0).
Zenodo. https://doi.org/10.5281/zenodo.23489072

Version 1.0.0 (C(29,17,4) and C(32,18,4) only): https://doi.org/10.5281/zenodo.23489073

The concept DOI [10.5281/zenodo.23489072](https://doi.org/10.5281/zenodo.23489072) always resolves to the latest version.
See also `CITATION.cff`.
