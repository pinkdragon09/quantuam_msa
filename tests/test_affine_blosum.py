#!/usr/bin/env python3
"""Checks for affine gap penalties and BLOSUM62 substitution scoring."""

from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import msa_hamiltonian as msa  # noqa: E402
import qubo_msa  # noqa: E402
import substitution_matrices as sm  # noqa: E402


class BlosumTests(unittest.TestCase):
    def test_symmetry_and_known_values(self) -> None:
        order = sm._ORDER
        for a in order:
            for b in order:
                self.assertEqual(sm.BLOSUM62[(a, b)], sm.BLOSUM62[(b, a)])
        self.assertEqual(sm.BLOSUM62[("A", "A")], 4)
        self.assertEqual(sm.BLOSUM62[("W", "W")], 11)
        self.assertEqual(sm.BLOSUM62[("D", "N")], 1)
        self.assertEqual(sm.BLOSUM62[("Y", "F")], 3)

    def test_fallback_and_scoring(self) -> None:
        self.assertEqual(sm.blosum62("U", "A"), sm.BLOSUM62[("X", "A")])
        prot = msa.Scoring(matrix="blosum62")
        self.assertEqual(prot.sub("M", "E"), sm.BLOSUM62[("M", "E")])


class AffineTests(unittest.TestCase):
    def test_linear_matches_percolumn(self) -> None:
        # With gap_open == 0, score_alignment must equal the per-column score_path.
        lin = msa.Scoring(match=2, mismatch=-1, gap=-2, gap_gap=0)
        for seqs in [("ACGTAC", "ACGTC", "ACTTAC"), ("MEEPQSD", "MEESQSD", "MEPQSD")]:
            dp = msa.exact_msa_dp(seqs, lin)
            self.assertEqual(msa.score_alignment(dp.alignment, lin), dp.score)

    def test_gap_run_charged_once(self) -> None:
        aff = msa.Scoring(match=2, mismatch=-1, gap=-1, gap_open=-5)
        # The opening position incurs both terms: a length-one run costs -6.
        self.assertEqual(msa.score_pair_rows("AA", "A-", aff), 2 - 5 - 1)
        # 'AAA' vs 'A--': match(+2) then a length-2 run (open -5, position -1 twice).
        self.assertEqual(msa.score_pair_rows("AAA", "A--", aff), 2 - 5 - 1 - 1)
        # both-gap columns are transparent: (A,A) (-,-) (A,A) -> two matches
        self.assertEqual(msa.score_pair_rows("A-A", "A-A", aff), 4)

    def test_affine_reduces_to_linear(self) -> None:
        lin = msa.Scoring(match=2, mismatch=-1, gap=-2, gap_open=0)
        rows_a, rows_b = "AC-GT", "A-CGT"
        # gap_open == 0 => each gap position costs `gap`, independent of runs
        manual = 0
        for x, y in zip(rows_a, rows_b):
            if x == "-" and y == "-":
                continue
            manual += lin.gap if (x == "-" or y == "-") else lin.sub(x, y)
        self.assertEqual(msa.score_pair_rows(rows_a, rows_b, lin), manual)

    def test_bruteforce_equals_anneal(self) -> None:
        seqs = ("ACGT", "ACT", "ACGT")
        aff = msa.Scoring(match=2, mismatch=-1, gap=-1, gap_open=-6)
        bf = msa.brute_force_msa(seqs, aff)
        sa = msa.anneal_msa(seqs, aff, restarts=40, steps_per_restart=600, seed=5)
        self.assertEqual(sa.score, bf.score)

    def test_blosum_affine_bruteforce_equals_anneal(self) -> None:
        seqs = ("MEEP", "MEP", "MESP")
        prot = msa.Scoring(matrix="blosum62", gap=-1, gap_open=-11)
        bf = msa.brute_force_msa(seqs, prot)
        sa = msa.anneal_msa(seqs, prot, restarts=40, steps_per_restart=600, seed=3)
        self.assertEqual(sa.score, bf.score)

    def test_qubo_rejects_affine(self) -> None:
        aff = msa.Scoring(match=2, mismatch=-1, gap=-1, gap_open=-6)
        with self.assertRaises(ValueError):
            qubo_msa.build_qubo(("AC", "A"), aff)


if __name__ == "__main__":
    unittest.main()
