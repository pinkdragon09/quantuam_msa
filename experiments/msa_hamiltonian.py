#!/usr/bin/env python3
"""Multiple sequence alignment as a path Hamiltonian.

This implementation models a
multiple sequence alignment (MSA) as a monotone path in a k-dimensional edit
lattice.  The score is the sum of local column scores, and the Hamiltonian
energy is the negative score.

Implemented pieces:

1. Exact k-dimensional dynamic programming over the full lattice.
2. Bijection helpers between paths and alignment columns.
3. A path-state simulated annealer as a classical stand-in for quantum
   annealing/QAOA/tensor-train ground-state search.

The examples are intentionally small so exact DP can provide a rigorous
baseline for the annealer.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
import random
from typing import Iterable, Sequence

from substitution_matrices import MATRICES


Step = tuple[int, ...]
State = tuple[int, ...]
Column = tuple[str, ...]


@dataclass(frozen=True)
class Scoring:
    """Sum-of-pairs scoring with optional substitution matrix and affine gaps.

    Substitution of two residues uses ``match``/``mismatch`` by default, or a
    named matrix (e.g. ``"blosum62"``) when ``matrix`` is set.  Gaps are linear
    by default: each gap position in a pair costs ``gap``.  Setting
    ``gap_open`` to a nonzero (negative) value makes gaps *affine* -- a gap run of
    length ``L`` in a pairwise projection costs ``gap_open + L * gap``.  Thus the
    opening position incurs both ``gap_open`` and the per-position ``gap`` term;
    a length-one run costs ``gap_open + gap``.  Some tools instead use
    ``gap_open + (L - 1) * gap``, so named parameters require conversion before
    comparison.  Affine scoring no longer decomposes column by column and is
    therefore evaluated on whole alignments by :func:`score_alignment` (used by
    the annealer and brute-force search), while the per-column :meth:`column`
    method (used by the exact DP and the QUBO encoder) requires ``gap_open == 0``.
    """

    match: int = 2
    mismatch: int = -1
    gap: int = -2          # per-position cost, charged on opening and later positions
    gap_gap: int = 0
    gap_open: int = 0      # extra penalty at the start of a gap run (0 = linear)
    matrix: str = "identity"  # "identity" (match/mismatch) or e.g. "blosum62"

    @property
    def is_affine(self) -> bool:
        return self.gap_open != 0

    def sub(self, a: str, b: str) -> int:
        """Substitution score for two residues (neither is a gap)."""

        if self.matrix == "identity":
            return self.match if a == b else self.mismatch
        return MATRICES[self.matrix](a, b)

    def pair(self, a: str, b: str) -> int:
        """Linear per-column pair score (ignores ``gap_open``)."""

        if a == "-" and b == "-":
            return self.gap_gap
        if a == "-" or b == "-":
            return self.gap
        return self.sub(a, b)

    def column(self, column: Column) -> int:
        if self.is_affine:
            raise ValueError(
                "per-column scoring is invalid for affine gaps (gap_open != 0); "
                "use score_alignment() on the whole alignment instead"
            )
        total = 0
        for i in range(len(column)):
            for j in range(i + 1, len(column)):
                total += self.pair(column[i], column[j])
        return total


@dataclass(frozen=True)
class MSAResult:
    sequences: tuple[str, ...]
    score: int
    energy: int
    path: tuple[Step, ...]
    alignment: tuple[str, ...]
    states_evaluated: int
    method: str

    @property
    def columns(self) -> tuple[Column, ...]:
        return tuple(zip(*self.alignment))


def all_steps(k: int) -> tuple[Step, ...]:
    """All nonzero binary step vectors in {0,1}^k."""

    return tuple(step for step in product((0, 1), repeat=k) if any(step))


def path_endpoint(path: Iterable[Step], k: int) -> State:
    totals = [0] * k
    for step in path:
        for idx, value in enumerate(step):
            totals[idx] += value
    return tuple(totals)


def step_fits(step: Step, state: State) -> bool:
    return all(s <= x for s, x in zip(step, state))


def subtract_step(state: State, step: Step) -> State:
    return tuple(x - s for x, s in zip(state, step))


def add_step(state: State, step: Step) -> State:
    return tuple(x + s for x, s in zip(state, step))


def path_to_alignment(sequences: Sequence[str], path: Sequence[Step]) -> tuple[str, ...]:
    """Convert a valid monotone path into aligned MSA rows."""

    k = len(sequences)
    positions = [0] * k
    rows = [[] for _ in range(k)]

    for step in path:
        if len(step) != k or not any(step) or any(value not in (0, 1) for value in step):
            raise ValueError(f"invalid step {step!r}")

        for seq_idx, advance in enumerate(step):
            if advance:
                positions[seq_idx] += 1
                if positions[seq_idx] > len(sequences[seq_idx]):
                    raise ValueError("path advances beyond sequence length")
                rows[seq_idx].append(sequences[seq_idx][positions[seq_idx]])
            else:
                rows[seq_idx].append("-")

    lengths = tuple(len(seq) for seq in sequences)
    if tuple(positions) != lengths:
        raise ValueError(f"path endpoint {tuple(positions)} does not match {lengths}")

    return tuple("".join(row) for row in rows)


def alignment_to_path(alignment: Sequence[str]) -> tuple[Step, ...]:
    """Convert aligned MSA rows back into lattice step vectors."""

    if not alignment:
        raise ValueError("alignment must contain at least one row")

    width = len(alignment[0])
    if any(len(row) != width for row in alignment):
        raise ValueError("all alignment rows must have equal length")

    path: list[Step] = []
    for col_idx in range(width):
        step = tuple(0 if row[col_idx] == "-" else 1 for row in alignment)
        if not any(step):
            raise ValueError("all-gap columns are not valid MSA columns")
        path.append(step)
    return tuple(path)


def column_from_previous_state(
    sequences: Sequence[str], previous_state: State, step: Step
) -> Column:
    chars: list[str] = []
    for seq_idx, advance in enumerate(step):
        if advance:
            chars.append(sequences[seq_idx][previous_state[seq_idx]])
        else:
            chars.append("-")
    return tuple(chars)


def score_path(sequences: Sequence[str], path: Sequence[Step], scoring: Scoring) -> int:
    state = tuple(0 for _ in sequences)
    total = 0
    endpoint = tuple(len(seq) for seq in sequences)

    for step in path:
        if len(step) != len(sequences) or not any(step):
            raise ValueError(f"invalid step {step!r}")
        next_state = add_step(state, step)
        if any(a > b for a, b in zip(next_state, endpoint)):
            raise ValueError("path advances beyond endpoint")
        total += scoring.column(column_from_previous_state(sequences, state, step))
        state = next_state

    if state != endpoint:
        raise ValueError(f"path endpoint {state} does not match {endpoint}")

    return total


def score_pair_rows(row_a: str, row_b: str, scoring: Scoring) -> int:
    """Sum-of-pairs score of two aligned rows with affine gaps.

    Uses this project's explicitly defined pairwise-projection convention:
    columns where both rows have a gap are dropped, and a maximal run of gaps in
    one row costs ``gap_open + L * gap`` for run length ``L``.  The opening
    position is charged both terms.  Another common convention uses
    ``gap_open + (L - 1) * gap``; its named parameters are not directly
    comparable.  With ``gap_open == 0`` our convention reduces exactly to the
    linear per-position gap cost.
    """

    total = 0
    in_gap_a = False  # currently extending a gap run against a residue of b
    in_gap_b = False
    for x, y in zip(row_a, row_b):
        if x == "-" and y == "-":
            continue  # both-gap columns are transparent in the projection
        if x != "-" and y != "-":
            total += scoring.sub(x, y)
            in_gap_a = in_gap_b = False
        elif x == "-":  # gap in row a, residue in row b
            total += scoring.gap if in_gap_a else scoring.gap_open + scoring.gap
            in_gap_a, in_gap_b = True, False
        else:  # residue in row a, gap in row b
            total += scoring.gap if in_gap_b else scoring.gap_open + scoring.gap
            in_gap_a, in_gap_b = False, True
    return total


def score_alignment(alignment: Sequence[str], scoring: Scoring) -> int:
    """Sum-of-pairs score of a full alignment (affine- and matrix-aware)."""

    total = 0
    k = len(alignment)
    for a in range(k):
        for b in range(a + 1, k):
            total += score_pair_rows(alignment[a], alignment[b], scoring)
    return total


def evaluate_path(sequences: Sequence[str], path: Sequence[Step], scoring: Scoring) -> int:
    """Score a path, routing affine scoring through the whole-alignment scorer."""

    if scoring.is_affine:
        return score_alignment(path_to_alignment(sequences, path), scoring)
    return score_path(sequences, path, scoring)


def all_alignments(sequences: Sequence[str]) -> Iterable[tuple[Step, ...]]:
    """Enumerate every valid monotone path (alignment) -- tiny instances only."""

    seqs = tuple(sequences)
    k = len(seqs)
    lengths = tuple(len(s) for s in seqs)
    steps = all_steps(k)

    def rec(state: State, path: list[Step]):
        if state == lengths:
            yield tuple(path)
            return
        for step in steps:
            nxt = add_step(state, step)
            if all(a <= b for a, b in zip(nxt, lengths)):
                path.append(step)
                yield from rec(nxt, path)
                path.pop()

    yield from rec(tuple(0 for _ in seqs), [])


def brute_force_msa(
    sequences: Sequence[str], scoring: Scoring, max_alignments: int = 2_000_000
) -> MSAResult:
    """Exact optimum by enumerating all alignments (validates affine scoring)."""

    seqs = tuple(sequences)
    best_score = -(10**12)
    best_path: tuple[Step, ...] | None = None
    count = 0
    for path in all_alignments(seqs):
        count += 1
        if count > max_alignments:
            raise ValueError(
                f"instance too large for brute force (> {max_alignments} alignments)"
            )
        score = evaluate_path(seqs, path, scoring)
        if score > best_score:
            best_score = score
            best_path = path

    if best_path is None:
        raise RuntimeError("no alignment enumerated")
    return MSAResult(
        sequences=seqs,
        score=best_score,
        energy=-best_score,
        path=best_path,
        alignment=path_to_alignment(seqs, best_path),
        states_evaluated=count,
        method="brute-force enumeration",
    )


def exact_msa_dp(sequences: Sequence[str], scoring: Scoring) -> MSAResult:
    """Exact MSA by dynamic programming over the full k-dimensional lattice."""

    seqs = tuple(sequences)
    k = len(seqs)
    dimensions = tuple(len(seq) + 1 for seq in seqs)
    steps = all_steps(k)
    neg_inf = -10**12

    dp: dict[State, int] = {tuple(0 for _ in seqs): 0}
    back: dict[State, Step] = {}
    states_evaluated = 0

    for state in product(*(range(dim) for dim in dimensions)):
        state = tuple(state)
        states_evaluated += 1
        if not any(state):
            continue

        best_score = neg_inf
        best_step: Step | None = None

        for step in steps:
            if not step_fits(step, state):
                continue
            previous = subtract_step(state, step)
            previous_score = dp.get(previous, neg_inf)
            if previous_score <= neg_inf // 2:
                continue
            column = column_from_previous_state(seqs, previous, step)
            candidate = previous_score + scoring.column(column)
            if candidate > best_score:
                best_score = candidate
                best_step = step

        if best_step is None:
            raise RuntimeError(f"no valid predecessor for state {state}")

        dp[state] = best_score
        back[state] = best_step

    endpoint = tuple(len(seq) for seq in seqs)
    state = endpoint
    reversed_path: list[Step] = []
    while any(state):
        step = back[state]
        reversed_path.append(step)
        state = subtract_step(state, step)

    path = tuple(reversed(reversed_path))
    alignment = path_to_alignment(seqs, path)
    score = dp[endpoint]

    return MSAResult(
        sequences=seqs,
        score=score,
        energy=-score,
        path=path,
        alignment=alignment,
        states_evaluated=states_evaluated,
        method="exact k-dimensional DP",
    )


def greedy_initial_path(sequences: Sequence[str]) -> tuple[Step, ...]:
    """A deterministic starting path: advance all unfinished sequences together."""

    lengths = tuple(len(seq) for seq in sequences)
    state = tuple(0 for _ in sequences)
    path: list[Step] = []

    while state != lengths:
        step = tuple(1 if state[idx] < lengths[idx] else 0 for idx in range(len(sequences)))
        path.append(step)
        state = add_step(state, step)

    return tuple(path)


def random_subpath(counts: State, rng: random.Random) -> tuple[Step, ...]:
    """Generate a random monotone path from 0 to counts."""

    remaining = list(counts)
    k = len(counts)
    path: list[Step] = []

    while any(remaining):
        available = [idx for idx, value in enumerate(remaining) if value > 0]
        rng.shuffle(available)
        take_count = rng.randint(1, len(available))
        selected = set(available[:take_count])
        step = tuple(1 if idx in selected else 0 for idx in range(k))
        path.append(step)
        for idx in selected:
            remaining[idx] -= 1

    return tuple(path)


def random_initial_path(sequences: Sequence[str], rng: random.Random) -> tuple[Step, ...]:
    return random_subpath(tuple(len(seq) for seq in sequences), rng)


def mutate_path(path: tuple[Step, ...], rng: random.Random) -> tuple[Step, ...]:
    """Local path mutation preserving the endpoint exactly."""

    if not path:
        return path

    work = list(path)
    actions = ["swap", "merge", "split", "resample_window"]
    action = rng.choice(actions)

    if action == "swap" and len(work) >= 2:
        idx = rng.randrange(len(work) - 1)
        work[idx], work[idx + 1] = work[idx + 1], work[idx]

    elif action == "merge" and len(work) >= 2:
        candidates = []
        for idx in range(len(work) - 1):
            merged = tuple(a + b for a, b in zip(work[idx], work[idx + 1]))
            if all(value in (0, 1) for value in merged):
                candidates.append((idx, merged))
        if candidates:
            idx, merged = rng.choice(candidates)
            work[idx : idx + 2] = [merged]

    elif action == "split":
        candidates = [idx for idx, step in enumerate(work) if sum(step) >= 2]
        if candidates:
            idx = rng.choice(candidates)
            active = [pos for pos, value in enumerate(work[idx]) if value]
            rng.shuffle(active)
            cut = rng.randrange(1, len(active))
            first = set(active[:cut])
            second = set(active[cut:])
            step_a = tuple(1 if pos in first else 0 for pos in range(len(work[idx])))
            step_b = tuple(1 if pos in second else 0 for pos in range(len(work[idx])))
            if rng.random() < 0.5:
                work[idx : idx + 1] = [step_a, step_b]
            else:
                work[idx : idx + 1] = [step_b, step_a]

    elif action == "resample_window" and len(work) >= 2:
        width = rng.randint(2, min(5, len(work)))
        start = rng.randrange(len(work) - width + 1)
        counts = tuple(sum(step[idx] for step in work[start : start + width]) for idx in range(len(work[0])))
        replacement = random_subpath(counts, rng)
        work[start : start + width] = replacement

    return tuple(work)


def anneal_msa(
    sequences: Sequence[str],
    scoring: Scoring,
    restarts: int = 200,
    steps_per_restart: int = 2500,
    seed: int = 20260630,
) -> MSAResult:
    """Simulated annealing over valid path states."""

    seqs = tuple(sequences)
    rng = random.Random(seed)
    best_path: tuple[Step, ...] | None = None
    best_score = -10**12
    evaluations = 0

    initial_paths = [greedy_initial_path(seqs)]
    initial_paths.extend(random_initial_path(seqs, rng) for _ in range(max(0, restarts - 1)))

    for current in initial_paths:
        current_score = evaluate_path(seqs, current, scoring)
        evaluations += 1

        for step_idx in range(steps_per_restart):
            candidate = mutate_path(current, rng)
            candidate_score = evaluate_path(seqs, candidate, scoring)
            evaluations += 1

            temperature = 4.0 * (0.01 / 4.0) ** (
                step_idx / max(1, steps_per_restart - 1)
            )
            delta = candidate_score - current_score
            if delta >= 0 or rng.random() < math.exp(delta / temperature):
                current = candidate
                current_score = candidate_score

        if current_score > best_score:
            best_path = current
            best_score = current_score

    if best_path is None:
        raise RuntimeError("annealer failed to produce a path")

    alignment = path_to_alignment(seqs, best_path)
    return MSAResult(
        sequences=seqs,
        score=best_score,
        energy=-best_score,
        path=best_path,
        alignment=alignment,
        states_evaluated=evaluations,
        method="path-state simulated annealing",
    )


def greedy_myopic_msa(sequences: Sequence[str], scoring: Scoring) -> MSAResult:
    """Greedy baseline: at each lattice vertex take the step with the best
    immediate column score (no lookahead).  A simple deterministic heuristic
    over the same path space, used to show the annealer beats naive greedy."""

    seqs = tuple(sequences)
    k = len(seqs)
    steps = all_steps(k)
    endpoint = tuple(len(s) for s in seqs)
    state = tuple(0 for _ in seqs)
    path: list[Step] = []

    while state != endpoint:
        best_step: Step | None = None
        best_key: tuple[int, int] | None = None
        for step in steps:
            if not step_fits(step, subtract_step(endpoint, state)):
                continue  # step would overshoot the endpoint
            column = column_from_previous_state(seqs, state, step)
            key = (scoring.column(column), sum(step))  # score, then prefer more advances
            if best_key is None or key > best_key:
                best_key = key
                best_step = step
        assert best_step is not None
        path.append(best_step)
        state = add_step(state, best_step)

    alignment = path_to_alignment(seqs, tuple(path))
    score = score_path(seqs, tuple(path), scoring)
    return MSAResult(
        sequences=seqs,
        score=score,
        energy=-score,
        path=tuple(path),
        alignment=alignment,
        states_evaluated=len(path),
        method="greedy myopic",
    )


def random_search_msa(
    sequences: Sequence[str],
    scoring: Scoring,
    evaluations: int,
    seed: int = 20260630,
) -> MSAResult:
    """Blind random-search baseline: sample ``evaluations`` random valid paths
    and keep the best.  Same evaluation budget as the annealer but with no
    Metropolis acceptance, so the SA-minus-random gap isolates the value of the
    energy-landscape structure."""

    seqs = tuple(sequences)
    rng = random.Random(seed)
    best_path = greedy_initial_path(seqs)
    best_score = score_path(seqs, best_path, scoring)
    for _ in range(max(0, evaluations - 1)):
        candidate = random_initial_path(seqs, rng)
        candidate_score = score_path(seqs, candidate, scoring)
        if candidate_score > best_score:
            best_score = candidate_score
            best_path = candidate

    alignment = path_to_alignment(seqs, best_path)
    return MSAResult(
        sequences=seqs,
        score=best_score,
        energy=-best_score,
        path=best_path,
        alignment=alignment,
        states_evaluated=evaluations,
        method="random search",
    )


def assert_bijection(result: MSAResult, scoring: Scoring = Scoring()) -> None:
    round_trip_path = alignment_to_path(result.alignment)
    if round_trip_path != result.path:
        raise AssertionError("alignment_to_path(path_to_alignment(path)) failed")
    score = evaluate_path(result.sequences, result.path, scoring)
    if score != result.score:
        raise AssertionError(f"score mismatch: {score} != {result.score}")


def format_alignment(alignment: Sequence[str]) -> str:
    return "\n".join(f"    S{idx + 1}: {row}" for idx, row in enumerate(alignment))


def run_dataset(
    name: str,
    sequences: Sequence[str],
    scoring: Scoring,
    restarts: int = 200,
    steps_per_restart: int = 2500,
) -> tuple[MSAResult, MSAResult]:
    exact = exact_msa_dp(sequences, scoring)
    annealed = anneal_msa(
        sequences,
        scoring,
        restarts=restarts,
        steps_per_restart=steps_per_restart,
        seed=20260630 + len(name),
    )

    assert_bijection(exact, scoring)
    assert_bijection(annealed, scoring)

    print(f"\n{name}")
    print(f"  sequences: {tuple(sequences)}")
    print(
        f"  exact:    score={exact.score:>3} energy={exact.energy:>4} "
        f"states={exact.states_evaluated} columns={len(exact.path)}"
    )
    print(
        f"  anneal:   score={annealed.score:>3} energy={annealed.energy:>4} "
        f"evals={annealed.states_evaluated} columns={len(annealed.path)} "
        f"optimal={annealed.score == exact.score}"
    )
    print("  exact alignment:")
    print(format_alignment(exact.alignment))
    print("  annealed alignment:")
    print(format_alignment(annealed.alignment))

    return exact, annealed


def main() -> None:
    scoring = Scoring(match=2, mismatch=-1, gap=-2)
    datasets = [
        ("dna_three_sequence", ("ACGTAC", "ACGTC", "ACTTAC")),
        ("protein_three_sequence", ("MEEPQSD", "MEESQSD", "MEPQSD")),
        ("synthetic_four_sequence", ("ACGTACGT", "ACGTTCGT", "ACGACGT", "ACGTACGTT")),
    ]

    print("MSA path Hamiltonian prototype")
    print("Scoring: match=+2 mismatch=-1 gap=-2 gap-gap=0")

    for name, sequences in datasets:
        run_dataset(name, sequences, scoring)


if __name__ == "__main__":
    main()
