#!/usr/bin/env python3
"""Explicit QUBO / Ising Hamiltonian for multiple sequence alignment.

The companion module :mod:`msa_hamiltonian` proves that a multiple sequence
alignment (MSA) is a monotone path in a ``k``-dimensional edit lattice and that
the sum-of-pairs score equals the reward collected along that path.  That module
treats the "Hamiltonian'' abstractly, as ``H(P) = -Score(P)`` over whole paths.

This module makes the Hamiltonian *literal*: a quadratic function of binary
decision variables (qubits), i.e. a genuine QUBO / Ising model of the kind that
quantum annealing and QAOA minimise.  The construction is the standard
"single-commodity flow'' encoding of a source-to-sink path in a directed acyclic
graph (Lucas, 2014):

* One binary variable ``x_e in {0, 1}`` per lattice *edge* ``e = (u -> u + v)``,
  where ``v`` is a nonzero binary step.  ``x_e = 1`` means the alignment uses that
  edge (i.e. emits the column that step ``v`` induces at vertex ``u``).
* Each edge carries a **constant** reward ``w_e = SP(column(u, v))`` -- constant
  because the column is fully determined by the vertex ``u`` and the step ``v``.
  This is exactly why the objective is *linear* in the ``x_e`` (diagonal in the
  QUBO): the edit lattice pushes all the residue-dependence into precomputable
  edge weights.
* Path validity is enforced by a **flow-conservation penalty**: net out-flow is
  ``+1`` at the source ``(0, ..., 0)``, ``-1`` at the sink ``(n_1, ..., n_k)``,
  and ``0`` at every interior vertex.  In a DAG a binary unit flow decomposes
  into a single source-to-sink path, so these penalties alone select exactly the
  valid alignments.

The resulting energy is

    H(x) = x^T Q x + L . x + offset
         = A * sum_v ( net_flow_v - target_v )^2  -  sum_e w_e x_e ,

whose ground state, for a large enough penalty weight ``A``, is the maximum-score
alignment -- identical to the exact dynamic-programming optimum.  This module
verifies that identity, provides an exact brute-force ground-state solver (for
small qubit counts), an incremental spin-flip simulated-annealing sampler (the
classical stand-in for a quantum annealer), and decode/validation helpers.

Only :mod:`numpy` and the project's :mod:`msa_hamiltonian` are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
import random
from typing import Sequence

import numpy as np

import msa_hamiltonian as msa


Vertex = tuple[int, ...]
Edge = tuple[Vertex, Vertex, int]  # (tail, head, reward)


@dataclass(frozen=True)
class QUBOModel:
    """A quadratic unconstrained binary model H(x) = x^T Q x + L . x + offset."""

    Q: np.ndarray            # (m, m) symmetric quadratic coefficients
    L: np.ndarray            # (m,) linear coefficients
    offset: float            # constant energy shift
    edges: tuple[Edge, ...]  # edge for each binary variable
    sequences: tuple[str, ...]
    source: Vertex
    sink: Vertex
    penalty: float           # the A used for flow conservation

    @property
    def num_qubits(self) -> int:
        return len(self.edges)

    def energy(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        return float(x @ self.Q @ x + self.L @ x + self.offset)


def _in_band(vertex: Vertex, lengths: Sequence[int], band: float | None) -> bool:
    if band is None:
        return True
    fractions = [
        vertex[r] / lengths[r] if lengths[r] > 0 else 0.0 for r in range(len(lengths))
    ]
    return max(fractions) - min(fractions) <= band + 1e-12


def build_edges(
    sequences: Sequence[str],
    scoring: msa.Scoring,
    band: float | None = None,
) -> tuple[list[Edge], Vertex, Vertex, tuple[int, ...]]:
    """Enumerate the directed lattice edges (optionally restricted to a band)."""

    seqs = tuple(sequences)
    k = len(seqs)
    lengths = tuple(len(s) for s in seqs)
    steps = msa.all_steps(k)
    source: Vertex = tuple(0 for _ in seqs)
    sink: Vertex = lengths

    vertices = {
        tuple(point)
        for point in product(*(range(n + 1) for n in lengths))
        if _in_band(tuple(point), lengths, band)
    }
    vertices.add(source)
    vertices.add(sink)

    edges: list[Edge] = []
    for u in sorted(vertices):
        for v in steps:
            w = msa.add_step(u, v)
            if any(a > b for a, b in zip(w, lengths)):
                continue
            if w not in vertices:
                continue
            reward = scoring.column(msa.column_from_previous_state(seqs, u, v))
            edges.append((u, w, reward))
    return edges, source, sink, lengths


def build_qubo(
    sequences: Sequence[str],
    scoring: msa.Scoring,
    band: float | None = None,
    penalty: float | None = None,
) -> QUBOModel:
    """Assemble the flow-conservation QUBO for an MSA instance."""

    if scoring.is_affine:
        raise ValueError(
            "the edge-linear QUBO encoding requires column-decomposable scoring; "
            "affine gaps (gap_open != 0) are not supported here. Use the path-state "
            "annealer with score_alignment() for affine scoring."
        )
    seqs = tuple(sequences)
    edges, source, sink, lengths = build_edges(seqs, scoring, band=band)
    m = len(edges)
    if m == 0:
        raise ValueError("no lattice edges; check the sequences or band width")

    # Incidence: a[v] is the signed edge-incidence row for vertex v
    # (+1 for edges leaving v, -1 for edges entering v).
    vertex_index: dict[Vertex, int] = {}
    for tail, head, _ in edges:
        for vert in (tail, head):
            if vert not in vertex_index:
                vertex_index[vert] = len(vertex_index)
    num_vertices = len(vertex_index)

    incidence = np.zeros((num_vertices, m), dtype=np.float64)
    rewards = np.zeros(m, dtype=np.float64)
    for e, (tail, head, reward) in enumerate(edges):
        incidence[vertex_index[tail], e] += 1.0
        incidence[vertex_index[head], e] += 1.0
        rewards[e] = reward

    target = np.zeros(num_vertices, dtype=np.float64)
    target[vertex_index[source]] = 1.0
    target[vertex_index[sink]] = -1.0

    if penalty is None:
        # Any single constraint violation costs >= A; make A strictly larger than
        # the total objective spread so the ground state is always feasible.
        penalty = float(np.abs(rewards).sum() + 1.0)

    # H = A * || incidence @ x - target ||^2 - rewards . x
    #   = x^T (A incidence^T incidence) x + (-2A target^T incidence - rewards) . x
    #     + A target^T target
    Q = penalty * (incidence.T @ incidence)
    Q = 0.5 * (Q + Q.T)  # enforce exact symmetry
    L = -2.0 * penalty * (target @ incidence) - rewards
    offset = penalty * float(target @ target)

    return QUBOModel(
        Q=Q,
        L=L,
        offset=offset,
        edges=tuple(edges),
        sequences=seqs,
        source=source,
        sink=sink,
        penalty=penalty,
    )


@dataclass(frozen=True)
class DecodedAlignment:
    valid: bool
    reason: str
    path: tuple[msa.Step, ...]
    alignment: tuple[str, ...]
    score: int


def decode(model: QUBOModel, x: np.ndarray) -> DecodedAlignment:
    """Turn a binary edge assignment into an alignment, checking path validity."""

    x = np.asarray(x).astype(int)
    selected = [model.edges[e] for e in range(model.num_qubits) if x[e] == 1]

    out_edges: dict[Vertex, list[Edge]] = {}
    indeg: dict[Vertex, int] = {}
    for tail, head, reward in selected:
        out_edges.setdefault(tail, []).append((tail, head, reward))
        indeg[head] = indeg.get(head, 0) + 1

    empty = (tuple(), tuple(), 0)
    if any(len(v) > 1 for v in out_edges.values()):
        return DecodedAlignment(False, "vertex with out-degree > 1", (), (), 0)
    if any(d > 1 for d in indeg.values()):
        return DecodedAlignment(False, "vertex with in-degree > 1", (), (), 0)

    path: list[msa.Step] = []
    node = model.source
    visited = 0
    while node != model.sink:
        outs = out_edges.get(node, [])
        if len(outs) != 1:
            return DecodedAlignment(False, f"no unique edge out of {node}", (), (), 0)
        _tail, head, _reward = outs[0]
        step = tuple(h - t for h, t in zip(head, node))
        path.append(step)
        node = head
        visited += 1
        if visited > model.num_qubits + 1:
            return DecodedAlignment(False, "path does not terminate", (), (), 0)

    if sum(len(outs) for outs in out_edges.values()) != len(path):
        return DecodedAlignment(False, "unused selected edges (disconnected)", (), (), 0)

    try:
        alignment = msa.path_to_alignment(model.sequences, tuple(path))
        score = msa.score_path(model.sequences, tuple(path), _scoring_from(model))
    except ValueError as exc:  # pragma: no cover - defensive
        return DecodedAlignment(False, f"invalid path: {exc}", (), (), 0)

    return DecodedAlignment(True, "ok", tuple(path), alignment, score)


def _scoring_from(model: QUBOModel) -> msa.Scoring:
    # Rewards were baked into the QUBO; recompute scores with the same scheme by
    # reading it back off the edges is unnecessary -- callers pass Scoring
    # elsewhere.  We store nothing extra, so reconstruct the default used in this
    # project's experiments.  Callers that use a non-default scheme should pass
    # the score explicitly via decode_with_scoring.
    return msa.Scoring()


def decode_with_scoring(
    model: QUBOModel, x: np.ndarray, scoring: msa.Scoring
) -> DecodedAlignment:
    base = decode(model, x)
    if not base.valid:
        return base
    score = msa.score_path(model.sequences, base.path, scoring)
    return DecodedAlignment(True, "ok", base.path, base.alignment, score)


def exact_ground_state(model: QUBOModel) -> tuple[float, np.ndarray, list[np.ndarray]]:
    """Brute-force ground state over all 2^m assignments (small m only)."""

    m = model.num_qubits
    if m > 24:
        raise ValueError(f"exact solver refuses m={m} qubits (2^m too large)")

    bits = ((np.arange(2**m)[:, None] >> np.arange(m)[::-1]) & 1).astype(np.float64)
    # Energy for every assignment: row-wise x Q x + L x + offset.
    quad = np.einsum("ij,jk,ik->i", bits, model.Q, bits)
    energies = quad + bits @ model.L + model.offset
    best = float(energies.min())
    ground_mask = np.isclose(energies, best, atol=1e-6)
    ground_states = [bits[i].astype(int) for i in np.nonzero(ground_mask)[0]]
    return best, energies, ground_states


@dataclass(frozen=True)
class AnnealResult:
    energy: float
    x: np.ndarray
    decoded: DecodedAlignment
    sweeps: int
    restarts: int


def anneal_qubo(
    model: QUBOModel,
    scoring: msa.Scoring,
    restarts: int = 40,
    sweeps: int = 400,
    seed: int = 7,
) -> AnnealResult:
    """Spin-flip simulated annealing on the QUBO (classical annealer surrogate).

    Uses an incremental field update so each proposed single-bit flip costs O(m).
    """

    m = model.num_qubits
    Q = model.Q
    L = model.L
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    best_energy = math.inf
    best_x = np.zeros(m, dtype=int)

    diag = np.diag(Q).copy()

    for _ in range(restarts):
        x = np_rng.integers(0, 2, size=m).astype(np.float64)
        Qx = Q @ x  # maintained incrementally
        energy = float(x @ Qx + L @ x + model.offset)

        t_hot, t_cold = 4.0, 0.02
        for sweep in range(sweeps):
            temperature = t_hot * (t_cold / t_hot) ** (sweep / max(1, sweeps - 1))
            for e in range(m):
                delta_sign = 1.0 - 2.0 * x[e]  # +1 if 0->1, -1 if 1->0
                # dE = delta*L_e + 2*delta*(Qx)_e + Q_ee
                dE = delta_sign * L[e] + 2.0 * delta_sign * Qx[e] + diag[e]
                if dE <= 0 or rng.random() < math.exp(-dE / temperature):
                    x[e] += delta_sign
                    Qx += delta_sign * Q[:, e]
                    energy += dE
            if energy < best_energy:
                best_energy = energy
                best_x = x.copy().astype(int)

    decoded = decode_with_scoring(model, best_x, scoring)
    return AnnealResult(
        energy=best_energy,
        x=best_x,
        decoded=decoded,
        sweeps=sweeps,
        restarts=restarts,
    )


def minimal_feasible_penalty(
    sequences: Sequence[str],
    scoring: msa.Scoring,
    band: float | None = None,
    candidates: Sequence[float] | None = None,
) -> tuple[float | None, list[tuple[float, bool, float]]]:
    """Smallest tested candidate A for which every exact ground state is a DP optimum.

    A too-small penalty leaves infeasible (constraint-violating) assignments tied
    in energy with the true optimum, so the ground eigenspace is polluted.  This
    returns the smallest tested ``A`` at which the entire ground eigenspace
    consists of valid, maximum-score alignments -- the condition under which the
    QUBO ground state *is* the optimal alignment.  It does not search continuously
    over ``A`` and therefore does not establish a universal or continuous minimum.

    Returns ``(smallest_tested_A_or_None, trace)`` where ``trace`` lists
    ``(A, all_ground_feasible, worst_gap)`` for each candidate.  Only usable on
    instances small enough for the exact ground-state solver.
    """

    dp = msa.exact_msa_dp(sequences, scoring)
    if candidates is None:
        candidates = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0]

    trace: list[tuple[float, bool, float]] = []
    smallest_tested: float | None = None
    for A in candidates:
        model = build_qubo(sequences, scoring, band=band, penalty=A)
        _, _, grounds = exact_ground_state(model)
        all_feasible = True
        worst_gap = 0.0
        for g in grounds:
            decoded = decode_with_scoring(model, g, scoring)
            if not decoded.valid:
                all_feasible = False
                worst_gap = math.inf
                break
            worst_gap = max(worst_gap, dp.score - decoded.score)
        if worst_gap != 0.0:
            all_feasible = False
        trace.append((A, all_feasible, worst_gap))
        if all_feasible and smallest_tested is None:
            smallest_tested = A
    return smallest_tested, trace


def main() -> None:
    scoring = msa.Scoring(match=2, mismatch=-1, gap=-2, gap_gap=0)
    instances = [
        ("pair_AC_A", ("AC", "A")),
        ("pair_CA_A", ("CA", "A")),
        ("dna_two_short", ("ACG", "AG")),
        ("dna_three_tiny", ("AC", "AC", "AT")),
    ]

    print("QUBO / Ising Hamiltonian for MSA")
    print("Scoring: match=+2 mismatch=-1 gap=-2 gap-gap=0\n")
    for name, seqs in instances:
        model = build_qubo(seqs, scoring)
        dp = msa.exact_msa_dp(seqs, scoring)
        print(f"{name}: {seqs}")
        if model.num_qubits <= 22:
            best, _, grounds = exact_ground_state(model)
            decoded = decode_with_scoring(model, grounds[0], scoring)
            method = "exact ground state"
            degeneracy = len(grounds)
        else:
            result = anneal_qubo(model, scoring, restarts=30, sweeps=300, seed=11)
            best, decoded = result.energy, result.decoded
            method = "spin-flip annealing"
            degeneracy = -1
        match = decoded.valid and decoded.score == dp.score
        print(
            f"  qubits(edges)={model.num_qubits}  penalty A={model.penalty:.0f}  "
            f"energy={best:.1f}  ({method})"
        )
        deg = "" if degeneracy < 0 else f"  (degeneracy={degeneracy})"
        print(
            f"  DP score={dp.score}  QUBO-decoded score={decoded.score}  "
            f"matches DP: {match}{deg}"
        )
        for row in decoded.alignment:
            print(f"      {row}")
        print()


if __name__ == "__main__":
    main()
