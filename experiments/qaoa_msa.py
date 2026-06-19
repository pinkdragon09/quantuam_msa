#!/usr/bin/env python3
"""A from-scratch QAOA statevector simulator for the MSA path Hamiltonian.

This module runs the Quantum Approximate Optimization Algorithm (QAOA) on the
QUBO / Ising Hamiltonian built by :mod:`qubo_msa`.  It is a genuine gate-level
simulation -- the full ``2^m`` statevector is evolved under the cost and mixer
unitaries -- implemented in NumPy so the project has no dependency on a quantum
SDK and stays fully reproducible.

QAOA prepares the state

    |gamma, beta> = U_B(beta_p) U_C(gamma_p) ... U_B(beta_1) U_C(gamma_1) |+>^{m}

with the cost unitary ``U_C(gamma) = exp(-i gamma H_C)`` (diagonal, since our
cost Hamiltonian ``H_C`` is diagonal in the computational basis) and the
transverse-field mixer ``U_B(beta) = prod_q exp(-i beta X_q)``.  We classically
optimise the ``2p`` angles to minimise ``<H_C>`` and then report the probability
that a measurement returns an exact ground state -- i.e. the maximum-score
alignment.  As the QAOA depth ``p`` grows, this ground-state probability rises
toward one, demonstrating that the alignment Hamiltonian is solvable by a bona
fide variational quantum algorithm on small instances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

import msa_hamiltonian as msa
import qubo_msa


def cost_diagonal(model: qubo_msa.QUBOModel) -> np.ndarray:
    """Energies H_C(x) for all 2^m computational-basis states.

    Bit convention: state index ``s`` assigns qubit ``q`` the value
    ``(s >> q) & 1`` (qubit 0 is the least significant bit).
    """

    m = model.num_qubits
    if m > 22:
        raise ValueError(f"statevector QAOA refuses m={m} qubits")
    states = np.arange(2**m)
    bits = ((states[:, None] >> np.arange(m)[None, :]) & 1).astype(np.float64)
    quad = np.einsum("ij,jk,ik->i", bits, model.Q, bits)
    return quad + bits @ model.L + model.offset


def _apply_mixer(state: np.ndarray, beta: float, m: int) -> np.ndarray:
    """Apply U_B(beta) = prod_q exp(-i beta X_q) to the statevector."""

    cos_b = np.cos(beta)
    sin_b = -1j * np.sin(beta)
    tensor = state.reshape([2] * m)
    for q in range(m):
        axis = m - 1 - q  # qubit 0 is least significant -> last tensor axis
        moved = np.moveaxis(tensor, axis, 0)
        zero, one = moved[0].copy(), moved[1].copy()
        moved[0] = cos_b * zero + sin_b * one
        moved[1] = sin_b * zero + cos_b * one
        tensor = np.moveaxis(moved, 0, axis)
    return tensor.reshape(-1)


def qaoa_state(
    diag: np.ndarray, gammas: np.ndarray, betas: np.ndarray, m: int
) -> np.ndarray:
    """Prepare the QAOA statevector for given angle schedules."""

    state = np.full(2**m, 1.0 / np.sqrt(2**m), dtype=np.complex128)
    for gamma, beta in zip(gammas, betas):
        state = np.exp(-1j * gamma * diag) * state
        state = _apply_mixer(state, beta, m)
    return state


def expected_cost(params: np.ndarray, diag: np.ndarray, p: int, m: int) -> float:
    gammas, betas = params[:p], params[p:]
    state = qaoa_state(diag, gammas, betas, m)
    probs = np.abs(state) ** 2
    return float(np.sum(probs * diag))


@dataclass(frozen=True)
class QAOAResult:
    p: int
    expected_cost: float
    ground_energy: float
    ground_probability: float
    best_bitstring: np.ndarray
    decoded: qubo_msa.DecodedAlignment
    num_ground_states: int


def run_qaoa(
    model: qubo_msa.QUBOModel,
    scoring: msa.Scoring,
    p: int,
    restarts: int = 12,
    seed: int = 2026,
) -> QAOAResult:
    """Optimise a depth-``p`` QAOA and report ground-state recovery."""

    m = model.num_qubits
    diag = cost_diagonal(model)
    ground_energy = float(diag.min())
    ground_mask = np.isclose(diag, ground_energy, rtol=0.0, atol=1e-6)
    num_ground = int(ground_mask.sum())

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        gamma0 = rng.uniform(0, np.pi, size=p)
        beta0 = rng.uniform(0, np.pi, size=p)
        x0 = np.concatenate([gamma0, beta0])
        res = minimize(
            expected_cost,
            x0,
            args=(diag, p, m),
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-5},
        )
        if best is None or res.fun < best.fun:
            best = res

    gammas, betas = best.x[:p], best.x[p:]
    state = qaoa_state(diag, gammas, betas, m)
    probs = np.abs(state) ** 2
    ground_prob = float(probs[ground_mask].sum())
    best_index = int(np.argmax(probs))
    best_bits = ((best_index >> np.arange(m)) & 1).astype(int)
    decoded = qubo_msa.decode_with_scoring(model, best_bits, scoring)

    return QAOAResult(
        p=p,
        expected_cost=float(best.fun),
        ground_energy=ground_energy,
        ground_probability=ground_prob,
        best_bitstring=best_bits,
        decoded=decoded,
        num_ground_states=num_ground,
    )


def run_qaoa_sweep(
    model: qubo_msa.QUBOModel,
    scoring: msa.Scoring,
    depths=(1, 2, 3, 4, 5),
    restarts_p1: int = 48,
    restarts_extra: int = 10,
    seed: int = 100,
) -> list[QAOAResult]:
    """Sweep QAOA depth with layerwise warm-starting (INTERP-style).

    Depth ``p`` is initialised from the optimised depth ``p-1`` angles padded with
    a zero final layer -- which reproduces the shallower circuit exactly -- plus a
    few random restarts.  This guarantees the optimised cost expectation is
    monotone non-increasing in ``p`` and avoids the spurious dips that pure random
    restarts produce on this non-convex landscape.
    """

    depths = tuple(depths)
    if list(depths) != list(range(1, len(depths) + 1)):
        raise ValueError(
            "run_qaoa_sweep requires consecutive depths starting at 1 "
            f"(got {depths}); the INTERP warm-start pads the previous layer by one."
        )

    m = model.num_qubits
    diag = cost_diagonal(model)
    ground_energy = float(diag.min())
    ground_mask = np.isclose(diag, ground_energy, rtol=0.0, atol=1e-6)
    num_ground = int(ground_mask.sum())

    rng = np.random.default_rng(seed)
    results: list[QAOAResult] = []
    prev: np.ndarray | None = None

    for p in depths:
        starts: list[np.ndarray] = []
        if prev is not None:
            g_prev, b_prev = prev[: p - 1], prev[p - 1 :]
            starts.append(np.concatenate([g_prev, [0.0], b_prev, [0.0]]))
        n_random = restarts_p1 if p == 1 else restarts_extra
        for _ in range(n_random):
            starts.append(
                np.concatenate(
                    [rng.uniform(0, np.pi, size=p), rng.uniform(0, np.pi, size=p)]
                )
            )

        best = None
        for x0 in starts:
            res = minimize(
                expected_cost,
                x0,
                args=(diag, p, m),
                method="Nelder-Mead",
                options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-6},
            )
            if best is None or res.fun < best.fun:
                best = res
        prev = best.x.copy()

        state = qaoa_state(diag, best.x[:p], best.x[p:], m)
        probs = np.abs(state) ** 2
        best_index = int(np.argmax(probs))
        best_bits = ((best_index >> np.arange(m)) & 1).astype(int)
        decoded = qubo_msa.decode_with_scoring(model, best_bits, scoring)
        results.append(
            QAOAResult(
                p=p,
                expected_cost=float(best.fun),
                ground_energy=ground_energy,
                ground_probability=float(probs[ground_mask].sum()),
                best_bitstring=best_bits,
                decoded=decoded,
                num_ground_states=num_ground,
            )
        )
    return results


def pick_small_instance(
    scoring: msa.Scoring, max_qubits: int = 12
) -> tuple[str, tuple[str, ...]]:
    """Choose a tiny pairwise instance whose optimum contains a gap."""

    candidates = [
        ("pair_CA_A", ("CA", "A")),
        ("pair_AC_A", ("AC", "A")),
        ("pair_GA_GT", ("GA", "GT")),
        ("pair_ACG_AG", ("ACG", "AG")),
    ]
    for name, seqs in candidates:
        model = qubo_msa.build_qubo(seqs, scoring)
        if model.num_qubits > max_qubits:
            continue
        dp = msa.exact_msa_dp(seqs, scoring)
        if any("-" in row for row in dp.alignment):
            return name, seqs
    return candidates[0]


def main() -> None:
    scoring = msa.Scoring(match=2, mismatch=-1, gap=-2, gap_gap=0)
    name, seqs = pick_small_instance(scoring)
    # Use the smallest penalty whose ground state is still the DP optimum: a loose
    # (large) penalty over-weights the constraints and stiffens the QAOA landscape.
    penalty, _ = qubo_msa.minimal_feasible_penalty(seqs, scoring)
    model = qubo_msa.build_qubo(seqs, scoring, penalty=penalty)
    dp = msa.exact_msa_dp(seqs, scoring)

    print("QAOA statevector simulation of the MSA path Hamiltonian")
    print(f"Instance {name}: {seqs}")
    print(
        f"  qubits(edges)={model.num_qubits}  penalty A={model.penalty:.0f}  "
        f"DP optimum score={dp.score}"
    )
    print("  optimal alignment:")
    for row in dp.alignment:
        print(f"      {row}")
    print()
    print("  depth p | <H_C>   | ground energy | P(ground state)")
    print("  --------+---------+---------------+----------------")
    for p in (1, 2, 3, 4):
        result = run_qaoa(model, scoring, p=p, restarts=16, seed=2026 + p)
        print(
            f"     {p}    | {result.expected_cost:7.2f} | {result.ground_energy:13.2f} "
            f"| {result.ground_probability:6.3f}"
        )


if __name__ == "__main__":
    main()
