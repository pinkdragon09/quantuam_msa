#!/usr/bin/env python3
"""Toy experiments for quantum/hybrid sequence-alignment proposals.

The code intentionally uses only the Python standard library.  It validates the
alignment objective and the hybrid structure on small DNA/protein examples:

1. QSABA: quantum-shift/anchor assisted banded Needleman-Wunsch.
   The quantum primitive is simulated by classical k-mer anchors; on hardware,
   this candidate-generation step would be replaced by QFT/Grover/QiBAM-style
   search.

2. QEPS: quantum edit-path search surrogate.
   The QUBO/Ising objective is represented by legal edit-operation strings and
   optimized with simulated annealing as a stand-in for QAOA/annealing/tensor
   train ground-state search.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import random
from typing import Callable, Iterable


ScoreFn = Callable[[str, str], int]


@dataclass(frozen=True)
class AlignmentResult:
    score: int
    aligned_a: str
    aligned_b: str
    cells_evaluated: int
    method: str


@dataclass(frozen=True)
class AnchorPlan:
    diagonals: tuple[int, ...]
    anchor_counts: dict[int, int]
    k: int


def simple_score(match_score: int = 2, mismatch_score: int = -1) -> ScoreFn:
    def score(a: str, b: str) -> int:
        return match_score if a == b else mismatch_score

    return score


def global_align(
    seq_a: str,
    seq_b: str,
    score_fn: ScoreFn,
    gap_score: int = -2,
    band_diagonals: Iterable[int] | None = None,
    band_width: int = 0,
    method: str = "Needleman-Wunsch",
) -> AlignmentResult:
    """Needleman-Wunsch global alignment, optionally restricted to diagonal bands.

    The diagonal convention is i - j for DP cell (i, j).  The endpoint diagonal
    len(seq_a) - len(seq_b) is forced into the band by the caller in QSABA.
    """

    m, n = len(seq_a), len(seq_b)
    neg_inf = -10**9
    diagonals = set(band_diagonals or [])

    def allowed(i: int, j: int) -> bool:
        if not diagonals:
            return True
        if (i, j) in {(0, 0), (m, n)}:
            return True
        d = i - j
        return any(abs(d - center) <= band_width for center in diagonals)

    dp = [[neg_inf] * (n + 1) for _ in range(m + 1)]
    bt: list[list[str | None]] = [[None] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0
    cells = 1

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                continue
            if not allowed(i, j):
                continue
            cells += 1

            best_score = neg_inf
            best_move: str | None = None

            if i > 0 and j > 0:
                candidate = dp[i - 1][j - 1] + score_fn(seq_a[i - 1], seq_b[j - 1])
                if candidate > best_score:
                    best_score = candidate
                    best_move = "D"

            if i > 0:
                candidate = dp[i - 1][j] + gap_score
                if candidate > best_score:
                    best_score = candidate
                    best_move = "H"

            if j > 0:
                candidate = dp[i][j - 1] + gap_score
                if candidate > best_score:
                    best_score = candidate
                    best_move = "V"

            dp[i][j] = best_score
            bt[i][j] = best_move

    if dp[m][n] <= neg_inf // 2:
        raise ValueError(f"{method} band excluded all feasible paths")

    aligned_a: list[str] = []
    aligned_b: list[str] = []
    i, j = m, n
    while i > 0 or j > 0:
        move = bt[i][j]
        if move == "D":
            aligned_a.append(seq_a[i - 1])
            aligned_b.append(seq_b[j - 1])
            i -= 1
            j -= 1
        elif move == "H":
            aligned_a.append(seq_a[i - 1])
            aligned_b.append("-")
            i -= 1
        elif move == "V":
            aligned_a.append("-")
            aligned_b.append(seq_b[j - 1])
            j -= 1
        else:
            raise ValueError(f"traceback failed at {(i, j)}")

    return AlignmentResult(
        score=dp[m][n],
        aligned_a="".join(reversed(aligned_a)),
        aligned_b="".join(reversed(aligned_b)),
        cells_evaluated=cells,
        method=method,
    )


def anchor_plan(seq_a: str, seq_b: str, k: int, top_k: int = 4) -> AnchorPlan:
    """Find candidate DP diagonals using exact shared k-mer anchors."""

    positions_b: dict[str, list[int]] = defaultdict(list)
    for j in range(max(0, len(seq_b) - k + 1)):
        positions_b[seq_b[j : j + k]].append(j)

    counts: Counter[int] = Counter()
    for i in range(max(0, len(seq_a) - k + 1)):
        kmer = seq_a[i : i + k]
        for j in positions_b.get(kmer, []):
            counts[i - j] += 1

    diagonals = [diag for diag, _ in counts.most_common(top_k)]
    for forced in (0, len(seq_a) - len(seq_b)):
        if forced not in diagonals:
            diagonals.append(forced)

    return AnchorPlan(
        diagonals=tuple(diagonals),
        anchor_counts=dict(sorted(counts.items())),
        k=k,
    )


def qsaba_align(
    seq_a: str,
    seq_b: str,
    score_fn: ScoreFn,
    gap_score: int,
    k: int,
    top_k: int = 4,
    band_width: int = 2,
) -> tuple[AlignmentResult, AnchorPlan]:
    """Quantum Shift/Anchor Banded Alignment, simulated with k-mer anchors."""

    plan = anchor_plan(seq_a, seq_b, k=k, top_k=top_k)
    result = global_align(
        seq_a,
        seq_b,
        score_fn=score_fn,
        gap_score=gap_score,
        band_diagonals=plan.diagonals,
        band_width=band_width,
        method="QSABA simulated hybrid",
    )
    return result, plan


def score_ops(
    seq_a: str,
    seq_b: str,
    ops: Iterable[str],
    score_fn: ScoreFn,
    gap_score: int,
) -> int:
    i = j = 0
    total = 0
    for op in ops:
        if op == "D":
            total += score_fn(seq_a[i], seq_b[j])
            i += 1
            j += 1
        elif op == "H":
            total += gap_score
            i += 1
        elif op == "V":
            total += gap_score
            j += 1
        else:
            raise ValueError(f"unknown edit op {op!r}")
    if i != len(seq_a) or j != len(seq_b):
        raise ValueError("edit path does not terminate at sequence endpoints")
    return total


def alignment_from_ops(
    seq_a: str,
    seq_b: str,
    ops: Iterable[str],
    score: int,
    method: str,
) -> AlignmentResult:
    i = j = 0
    aligned_a: list[str] = []
    aligned_b: list[str] = []
    op_count = 0

    for op in ops:
        op_count += 1
        if op == "D":
            aligned_a.append(seq_a[i])
            aligned_b.append(seq_b[j])
            i += 1
            j += 1
        elif op == "H":
            aligned_a.append(seq_a[i])
            aligned_b.append("-")
            i += 1
        elif op == "V":
            aligned_a.append("-")
            aligned_b.append(seq_b[j])
            j += 1
        else:
            raise ValueError(f"unknown edit op {op!r}")

    return AlignmentResult(
        score=score,
        aligned_a="".join(aligned_a),
        aligned_b="".join(aligned_b),
        cells_evaluated=op_count,
        method=method,
    )


def random_path(m: int, n: int, rng: random.Random) -> tuple[str, ...]:
    i = j = 0
    ops: list[str] = []
    while i < m or j < n:
        choices: list[str] = []
        if i < m and j < n:
            choices.extend(["D", "D", "H", "V"])
        elif i < m:
            choices.append("H")
        else:
            choices.append("V")

        op = rng.choice(choices)
        ops.append(op)
        if op == "D":
            i += 1
            j += 1
        elif op == "H":
            i += 1
        else:
            j += 1
    return tuple(ops)


def mutate_path(ops: tuple[str, ...], rng: random.Random) -> tuple[str, ...]:
    """Mutation preserving edit-path endpoint constraints."""

    if not ops:
        return ops

    work = list(ops)
    action = rng.choice(["swap", "split", "merge"])

    if action == "swap" and len(work) >= 2:
        pos = rng.randrange(len(work) - 1)
        work[pos], work[pos + 1] = work[pos + 1], work[pos]

    elif action == "split":
        diagonal_positions = [idx for idx, op in enumerate(work) if op == "D"]
        if diagonal_positions:
            pos = rng.choice(diagonal_positions)
            replacement = ["H", "V"] if rng.random() < 0.5 else ["V", "H"]
            work[pos : pos + 1] = replacement

    elif action == "merge" and len(work) >= 2:
        candidates = [
            idx
            for idx in range(len(work) - 1)
            if {work[idx], work[idx + 1]} == {"H", "V"}
        ]
        if candidates:
            pos = rng.choice(candidates)
            work[pos : pos + 2] = ["D"]

    return tuple(work)


def qeps_annealed_align(
    seq_a: str,
    seq_b: str,
    score_fn: ScoreFn,
    gap_score: int,
    restarts: int = 96,
    steps: int = 2500,
    seed: int = 17,
) -> AlignmentResult:
    """Simulated annealing surrogate for edit-path Hamiltonian optimization."""

    rng = random.Random(seed)
    best_ops: tuple[str, ...] | None = None
    best_score = -10**9

    for _ in range(restarts):
        current = random_path(len(seq_a), len(seq_b), rng)
        current_score = score_ops(seq_a, seq_b, current, score_fn, gap_score)

        for step in range(steps):
            candidate = mutate_path(current, rng)
            candidate_score = score_ops(seq_a, seq_b, candidate, score_fn, gap_score)
            temperature = 3.0 * (0.02 / 3.0) ** (step / max(1, steps - 1))
            delta = candidate_score - current_score
            if delta >= 0 or rng.random() < math.exp(delta / temperature):
                current = candidate
                current_score = candidate_score

        if current_score > best_score:
            best_ops = current
            best_score = current_score

    if best_ops is None:
        raise RuntimeError("annealing did not produce a path")

    return alignment_from_ops(
        seq_a,
        seq_b,
        best_ops,
        score=best_score,
        method="QEPS annealing surrogate",
    )


def report_dataset(
    name: str,
    seq_a: str,
    seq_b: str,
    residue_type: str,
    k: int,
    band_width: int,
) -> None:
    score_fn = simple_score()
    gap_score = -2

    full = global_align(seq_a, seq_b, score_fn=score_fn, gap_score=gap_score)
    qsaba, plan = qsaba_align(
        seq_a,
        seq_b,
        score_fn=score_fn,
        gap_score=gap_score,
        k=k,
        band_width=band_width,
    )
    qeps = qeps_annealed_align(
        seq_a,
        seq_b,
        score_fn=score_fn,
        gap_score=gap_score,
        restarts=96,
        steps=2500,
        seed=17 + len(seq_a) * 11 + len(seq_b),
    )

    print(f"\n{name} ({residue_type})")
    print(f"  A: {seq_a}")
    print(f"  B: {seq_b}")
    print(f"  full NW score: {full.score:>3} cells={full.cells_evaluated}")
    print(
        "  QSABA score:   "
        f"{qsaba.score:>3} cells={qsaba.cells_evaluated} "
        f"diagonals={list(plan.diagonals)} anchors={plan.anchor_counts}"
    )
    print(
        "  QEPS score:    "
        f"{qeps.score:>3} sampled_path_len={qeps.cells_evaluated} "
        f"optimal={qeps.score == full.score}"
    )
    print("  representative alignment:")
    print(f"    {qsaba.aligned_a}")
    print(f"    {qsaba.aligned_b}")


def main() -> None:
    datasets = [
        (
            "dna_single_insertion",
            "ACGTACGTGACG",
            "ACGTTACGTGACG",
            "DNA",
            3,
            2,
        ),
        (
            "dna_substitutions",
            "GATTACA",
            "GCATGCA",
            "DNA",
            2,
            2,
        ),
        (
            "protein_substitution",
            "MEEPQSD",
            "MEESQSD",
            "protein",
            2,
            1,
        ),
        (
            "protein_deletion",
            "MKTAYIAK",
            "MKTYIAK",
            "protein",
            2,
            2,
        ),
    ]

    print("Scoring: match=+2 mismatch=-1 gap=-2")
    for dataset in datasets:
        report_dataset(*dataset)


if __name__ == "__main__":
    main()
