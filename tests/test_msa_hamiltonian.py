#!/usr/bin/env python3
"""Basic checks for the MSA path Hamiltonian prototype."""

from __future__ import annotations

import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import msa_hamiltonian as msa  # noqa: E402


class MSAHamiltonianTests(unittest.TestCase):
    def test_nonzero_binary_steps(self) -> None:
        self.assertEqual(len(msa.all_steps(1)), 1)
        self.assertEqual(len(msa.all_steps(2)), 3)
        self.assertEqual(len(msa.all_steps(3)), 7)
        self.assertEqual(len(msa.all_steps(4)), 15)

    def test_path_alignment_bijection(self) -> None:
        sequences = ("ACGTAC", "ACGTC", "ACTTAC")
        path = (
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            (1, 0, 1),
            (1, 1, 1),
        )

        alignment = msa.path_to_alignment(sequences, path)
        self.assertEqual(
            alignment,
            (
                "ACGTAC",
                "ACGT-C",
                "ACTTAC",
            ),
        )
        self.assertEqual(msa.alignment_to_path(alignment), path)

    def test_exact_dp_known_scores(self) -> None:
        scoring = msa.Scoring(match=2, mismatch=-1, gap=-2)

        dna = msa.exact_msa_dp(("ACGTAC", "ACGTC", "ACTTAC"), scoring)
        self.assertEqual(dna.score, 22)
        self.assertEqual(dna.energy, -22)

        protein = msa.exact_msa_dp(("MEEPQSD", "MEESQSD", "MEPQSD"), scoring)
        self.assertEqual(protein.score, 28)
        self.assertEqual(protein.energy, -28)

    def test_hamiltonian_energy_is_negative_score(self) -> None:
        scoring = msa.Scoring(match=2, mismatch=-1, gap=-2)
        result = msa.exact_msa_dp(("AA", "A"), scoring)

        self.assertEqual(result.energy, -result.score)
        self.assertEqual(msa.score_path(result.sequences, result.path, scoring), result.score)


if __name__ == "__main__":
    unittest.main()
