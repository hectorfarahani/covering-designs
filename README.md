# covering-designs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23489072.svg)](https://doi.org/10.5281/zenodo.23489072)

New covering designs found with GPU-parallel weighted local search, together with a stand-alone verifier and the search code.

**Author:** Hossein Farahani ([hossein.pro](https://hossein.pro), research@hossein.pro, ORCID [0009-0005-0297-8082](https://orcid.org/0009-0005-0297-8082))

A (v,k,t) covering design is a set of k-element subsets (blocks) of {1,...,v} such that every t-element subset
lies in at least one block. C(v,k,t) is the smallest possible number of blocks.

## Results

| Parameters | New size | Previous best | Previous method | Lower bound |
|---|---|---|---|---|
| C(29,17,4) | **22** | 23 | Dynamic construction (LJCR, 2009) | 16 |
| C(32,18,4) | **25** | 26 | Multiple of (16,9,4) covering (LJCR, 2009) | 21 |

Previous best values as listed in the La Jolla Covering Repository archive (Zenodo record 19472830, v1.2, April 2026),
and checked against the live tables at coveringrepository.com on 2026-10-20 (23 and 26 blocks).

## Verify (Python 3 standard library only, about one second)

    python verify.py 29 17 4 records/C29_17_4_22blocks.txt
    python verify.py 32 18 4 records/C32_18_4_25blocks.txt

Both report `missing 0`. Each line of a record file is one block.

## Method

A GPU-parallel weighted local search with tabu and restart kicks. It applies established techniques to covering designs:
local search for coverings (Nurmela and Östergård, 1993), row/clause weighting with tabu for set covering
(e.g. Gao, Yao, Weise and Li, 2015, RWLS), and focused WalkSAT-style moves.

- The block count is fixed at the target size. Searches start from random, point-balanced blocks.
- Each step chooses a random uncovered t-subset and evaluates every swap that would cover it
  (one point out of one block, one point of the t-subset in). It applies the best swap, with random tie-breaking.
- Uncovered t-subsets carry weights, which increase whenever no improving swap exists. A point just removed from a block
  is tabu for 40 steps. A search that has not improved for 60 s is perturbed.
- Many independent searches run at once on an NVIDIA GPU, one search per 32-thread warp. The threads of a warp share the
  scoring of candidate swaps.
- Every reported design is recounted independently on the CPU.

The designs above were found on one NVIDIA RTX 4080 with 256 parallel searches (random seed 7), in 54 s (C(29,17,4))
and 71 s (C(32,18,4)), after the search had first reached the previous record size.

## Code

| Path | Content |
|---|---|
| `verify.py` | stand-alone verifier |
| `records/` | the new designs |
| `src/covsearch_gpu.py` | GPU search for C(v,k,4), v <= 32 (the version that found the records) |
| `src/covsearch_gpu_t.py` | the same search for any 2 <= t <= 5, v <= 32 |
| `tests/test_calibration.py` | checks against proven values C(10,6,4)=20, C(8,4,3)=14, C(7,3,2)=7 |
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
live Covering Repository tables (coveringrepository.com). No previous designs were used as input. The new designs were
built from random starting blocks.

## Licence

Designs and documentation: CC BY 4.0. Code: MIT. See `LICENSE`.

## Citation

Farahani, H. (2026). covering-designs: new covering designs C(29,17,4) ≤ 22 and C(32,18,4) ≤ 25
by GPU-parallel weighted local search (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.23489073

The concept DOI [10.5281/zenodo.23489072](https://doi.org/10.5281/zenodo.23489072) always resolves to the latest version.
See also `CITATION.cff`.
