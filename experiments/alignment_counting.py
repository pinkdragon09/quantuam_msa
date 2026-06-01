#!/usr/bin/env python3
"""Enumerative combinatorics of the alignment space.

The set of multiple sequence alignments of $k$ sequences of lengths
$n_1,\\dots,n_k$ is (Theorem, main report) in bijection with the monotone lattice
paths from $\\mathbf 0$ to $(n_1,\\dots,n_k)$ whose steps are the nonzero binary
vectors $\\{0,1\\}^k\\setminus\\{\\mathbf 0\\}$.  This module counts them and
verifies the closed forms used in the report's theory section:

* the transfer recurrence and the ordinary generating function
  $1/(2-\\prod_i(1+x_i))$;
* the $k=2$ specialisation to the Delannoy numbers, with the central-Delannoy
  diagonal generating function $1/\\sqrt{1-6t+t^2}$;
* the identity $A(1,\\dots,1)=$ the $k$-th ordered Bell (Fubini) number;
* the Carrillo--Lipman upper bound $\\mathrm{SP}(M)\\le\\sum_{a<b}\\mathrm{OPT}_2(a,b)$.

Everything here is exact integer arithmetic and is cross-checked against the
brute-force path enumerator :func:`msa_hamiltonian.all_alignments`.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product
from math import comb, factorial

import msa_hamiltonian as msa


@lru_cache(maxsize=None)
def count_alignments(lengths: tuple[int, ...]) -> int:
    """Number of alignments A(n_1,...,n_k) via the transfer recurrence.

    A(0)=1 and A(n)=sum over nonzero binary v<=n of A(n-v); this is exactly the
    recurrence encoded by the generating function 1/(2 - prod_i (1+x_i)).
    """

    if all(c == 0 for c in lengths):
        return 1
    k = len(lengths)
    total = 0
    for v in product((0, 1), repeat=k):
        if any(v) and all(vi <= ni for vi, ni in zip(v, lengths)):
            total += count_alignments(tuple(ni - vi for ni, vi in zip(lengths, v)))
    return total


def delannoy(m: int, n: int) -> int:
    """Delannoy number D(m,n) = sum_d (m+n-d)! / ((m-d)!(n-d)!d!)."""

    return sum(
        factorial(m + n - d) // (factorial(m - d) * factorial(n - d) * factorial(d))
        for d in range(min(m, n) + 1)
    )


def ordered_bell(k: int) -> int:
    """k-th ordered Bell (Fubini) number: sum_j j! * S(k,j)."""

    return sum(
        sum((-1) ** (j - i) * comb(j, i) * i**k for i in range(j + 1))
        for j in range(k + 1)
    )


def central_delannoy_recurrence_holds(upto: int) -> bool:
    """Check n D_n = 3(2n-1) D_{n-1} - (n-1) D_{n-2}, i.e. GF 1/sqrt(1-6t+t^2)."""

    d = [count_alignments((n, n)) for n in range(upto + 1)]
    return all(n * d[n] == 3 * (2 * n - 1) * d[n - 1] - (n - 1) * d[n - 2]
               for n in range(2, upto + 1))


def carrillo_lipman_bound(sequences, scoring: msa.Scoring) -> int:
    """Upper bound sum_{a<b} OPT_2(S_a, S_b) on the SP score of any alignment."""

    seqs = tuple(sequences)
    total = 0
    for a in range(len(seqs)):
        for b in range(a + 1, len(seqs)):
            total += msa.exact_msa_dp((seqs[a], seqs[b]), scoring).score
    return total


def main() -> None:
    print("Enumeration of the alignment space A(n_1,...,n_k)\n")

    print("k=2: A(m,n) vs Delannoy vs brute-force path count")
    for m, n in [(1, 1), (2, 2), (2, 3), (3, 3), (4, 4), (5, 5)]:
        brute = sum(1 for _ in msa.all_alignments(("A" * m, "A" * n)))
        print(f"  ({m},{n}): A={count_alignments((m, n))}  "
              f"Delannoy={delannoy(m, n)}  brute={brute}")

    diag = [count_alignments((n, n)) for n in range(9)]
    print(f"\ncentral Delannoy A(n,n): {diag}")
    print(f"  diagonal GF 1/sqrt(1-6t+t^2) recurrence holds: "
          f"{central_delannoy_recurrence_holds(8)}")
    print(f"  exponential growth rate 3+2*sqrt(2) = {3 + 2 * 2**0.5:.4f} "
          f"(ratios approach this)")

    print("\nA(1,...,1) equals the ordered Bell (Fubini) numbers:")
    for k in range(1, 7):
        print(f"  k={k}: A(1^{k})={count_alignments((1,) * k)}  "
              f"ordered_bell({k})={ordered_bell(k)}")

    print("\ngrowth rate brackets  A(1^k) <= rho_k <= (2^k-1)^k:")
    for k in (2, 3, 4):
        s = [count_alignments((n,) * k) for n in range(6)]
        ratios = [round(s[n] / s[n - 1], 2) for n in range(2, 6)]
        print(f"  k={k}: lower A(1^k)={count_alignments((1,)*k)}  "
              f"upper (2^k-1)^k={(2**k - 1)**k}  observed ratios {ratios}")

    print("\nCarrillo-Lipman bound  SP(optimal MSA) <= sum OPT_2:")
    sc = msa.Scoring(match=2, mismatch=-1, gap=-2, gap_gap=0)
    for name, seqs in [("dna3", ("ACGTAC", "ACGTC", "ACTTAC")),
                       ("syn4", ("ACGTACGT", "ACGTTCGT", "ACGACGT", "ACGTACGTT"))]:
        sp = msa.exact_msa_dp(seqs, sc).score
        u = carrillo_lipman_bound(seqs, sc)
        print(f"  {name}: SP={sp}  bound={u}  holds={sp <= u}  slack={u - sp}")


if __name__ == "__main__":
    main()
