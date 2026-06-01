#!/usr/bin/env python3
"""Checks for the enumerative combinatorics of the alignment space."""

from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import msa_hamiltonian as msa  # noqa: E402
import alignment_counting as ac  # noqa: E402


class EnumerationTests(unittest.TestCase):
    def test_recurrence_matches_brute_force(self) -> None:
        # The transfer recurrence must equal the direct path enumeration.
        for lengths in [(1, 1), (2, 2), (2, 3), (3, 3), (1, 1, 1), (2, 2, 1), (2, 1, 2)]:
            seqs = tuple("A" * n for n in lengths)
            brute = sum(1 for _ in msa.all_alignments(seqs))
            self.assertEqual(ac.count_alignments(lengths), brute, f"lengths={lengths}")

    def test_delannoy_closed_form(self) -> None:
        for m in range(6):
            for n in range(6):
                self.assertEqual(ac.count_alignments((m, n)), ac.delannoy(m, n))

    def test_central_delannoy_gf(self) -> None:
        # 1/sqrt(1-6t+t^2): n D_n = 3(2n-1) D_{n-1} - (n-1) D_{n-2}.
        self.assertTrue(ac.central_delannoy_recurrence_holds(10))

    def test_single_residue_is_ordered_bell(self) -> None:
        # A(1,...,1) = the k-th ordered Bell (Fubini) number.
        expected = {1: 1, 2: 3, 3: 13, 4: 75, 5: 541, 6: 4683}
        for k, value in expected.items():
            self.assertEqual(ac.count_alignments((1,) * k), value)
            self.assertEqual(ac.ordered_bell(k), value)

    def test_growth_brackets(self) -> None:
        # A(1^k) <= rho_k <= (2^k-1)^k, checked via consecutive-length ratios.
        for k in (2, 3, 4):
            s = [ac.count_alignments((n,) * k) for n in range(2, 6)]
            ratios = [s[i] / s[i - 1] for i in range(1, len(s))]
            lower = ac.count_alignments((1,) * k)
            upper = (2**k - 1) ** k
            for r in ratios:
                self.assertGreater(r, 1.0)
                self.assertLess(r, upper)
            # ratios increase toward the true growth rate, staying above the
            # single-residue lower bound once n is not too small.
            self.assertGreaterEqual(max(ratios), lower)

    def test_carrillo_lipman_bound(self) -> None:
        sc = msa.Scoring(match=2, mismatch=-1, gap=-2, gap_gap=0)
        for seqs in [("ACGTAC", "ACGTC", "ACTTAC"),
                     ("ACGTACGT", "ACGTTCGT", "ACGACGT", "ACGTACGTT")]:
            sp = msa.exact_msa_dp(seqs, sc).score
            bound = ac.carrillo_lipman_bound(seqs, sc)
            self.assertLessEqual(sp, bound)


if __name__ == "__main__":
    unittest.main()
