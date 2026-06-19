#!/usr/bin/env python3
"""Checks for the QUBO / Ising Hamiltonian and QAOA extension."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import msa_hamiltonian as msa  # noqa: E402
import qubo_msa  # noqa: E402
import qaoa_msa  # noqa: E402


SCORING = msa.Scoring(match=2, mismatch=-1, gap=-2, gap_gap=0)


class QUBOTests(unittest.TestCase):
    def test_ground_state_matches_dp(self) -> None:
        for seqs in [("CA", "A"), ("AC", "A"), ("ACG", "AG"), ("AT", "AAT")]:
            model = qubo_msa.build_qubo(seqs, SCORING)
            dp = msa.exact_msa_dp(seqs, SCORING)
            energy, _, grounds = qubo_msa.exact_ground_state(model)
            decoded = qubo_msa.decode_with_scoring(model, grounds[0], SCORING)
            self.assertTrue(decoded.valid, f"invalid decode for {seqs}")
            self.assertEqual(decoded.score, dp.score, f"score mismatch for {seqs}")
            # Ground energy equals negative optimal score for a feasible ground state.
            self.assertAlmostEqual(energy, -dp.score, places=6)

    def test_decoded_alignment_is_dp_alignment_score(self) -> None:
        seqs = ("ACG", "AG")
        model = qubo_msa.build_qubo(seqs, SCORING)
        _, _, grounds = qubo_msa.exact_ground_state(model)
        decoded = qubo_msa.decode_with_scoring(model, grounds[0], SCORING)
        # Round-trip: the decoded alignment re-encodes to the same path.
        path = msa.alignment_to_path(decoded.alignment)
        self.assertEqual(msa.score_path(seqs, path, SCORING), decoded.score)

    def test_flow_penalty_rejects_non_path(self) -> None:
        # The all-zero assignment violates source/sink flow and must cost >= penalty.
        model = qubo_msa.build_qubo(("CA", "A"), SCORING)
        zero = np.zeros(model.num_qubits)
        self.assertGreaterEqual(model.energy(zero), model.penalty - 1e-9)
        # A decoded valid path exists with energy = -score.
        _, _, grounds = qubo_msa.exact_ground_state(model)
        self.assertTrue(qubo_msa.decode(model, grounds[0]).valid)

    def test_minimal_feasible_penalty(self) -> None:
        minimal, trace = qubo_msa.minimal_feasible_penalty(("CA", "A"), SCORING)
        self.assertIsNotNone(minimal)
        # Below the smallest successful tested candidate, at least one tested
        # value leaves an infeasible assignment in the ground space.
        self.assertTrue(any(not feasible for _, feasible, _ in trace))


class QAOATests(unittest.TestCase):
    def test_qaoa_recovers_optimum(self) -> None:
        seqs = ("CA", "A")
        penalty, _ = qubo_msa.minimal_feasible_penalty(seqs, SCORING)
        model = qubo_msa.build_qubo(seqs, SCORING, penalty=penalty)
        result = qaoa_msa.run_qaoa(model, SCORING, p=2, restarts=12, seed=0)
        baseline = result.num_ground_states / (2 ** model.num_qubits)
        # Depth-2 QAOA concentrates far more probability on the optimal alignment
        # than uniform random sampling would.
        self.assertGreater(result.ground_probability, 20 * baseline)
        # The ground eigenspace is exactly the optimal alignment (unique here).
        self.assertEqual(result.num_ground_states, 1)
        # The exact ground state of the model always decodes to a valid alignment.
        _, _, grounds = qubo_msa.exact_ground_state(model)
        self.assertTrue(qubo_msa.decode(model, grounds[0]).valid)

    def test_mixer_preserves_norm(self) -> None:
        m = 4
        rng = np.random.default_rng(0)
        state = rng.normal(size=2**m) + 1j * rng.normal(size=2**m)
        state /= np.linalg.norm(state)
        mixed = qaoa_msa._apply_mixer(state, 0.37, m)
        self.assertAlmostEqual(np.linalg.norm(mixed), 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
