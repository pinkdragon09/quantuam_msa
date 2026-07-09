# QUBO / QAOA Extension Experiment Summary

Scoring: match=+2, mismatch=-1, gap=-2, gap-gap=0.

## 1. QUBO ground state vs exact DP

| Instance | Lengths | Qubits | Penalty A | DP score | QUBO score | Matches DP |
| --- | --- | ---: | ---: | ---: | ---: | :--: |
| `pair_CA_A` | 2/1 | 9 | 18 | 0 | 0 | yes |
| `pair_AC_A` | 2/1 | 9 | 18 | 0 | 0 | yes |
| `pair_AT_AAT` | 2/3 | 23 | 44 | 2 | 2 | yes |
| `dna_two_ACG_AG` | 3/2 | 23 | 43 | 2 | 2 | yes |
| `dna_two_GATT_GAT` | 4/3 | 43 | 79 | 4 | 4 | yes |
| `dna_three_tiny` | 2/2/2 | 98 | 373 | 6 | 6 | yes |

## 2. Finite penalty sweep on two instances

Candidate grid: A = {0.5, 1, 2, 3, 5, 8, 13, 21, 34, 55}. The smallest successful tested value is not a universal or continuous minimum.

| Instance | A | Feasible | Score gap | Smallest successful tested A | Conservative bound |
| --- | ---: | :--: | ---: | ---: | ---: |
| `pair_CA_A` | 0.5 | no | inf | 2.0 | 18 |
| `pair_CA_A` | 1.0 | no | inf | 2.0 | 18 |
| `pair_CA_A` | 2.0 | yes | 0 | 2.0 | 18 |
| `pair_CA_A` | 3.0 | yes | 0 | 2.0 | 18 |
| `pair_CA_A` | 5.0 | yes | 0 | 2.0 | 18 |
| `pair_CA_A` | 8.0 | yes | 0 | 2.0 | 18 |
| `pair_CA_A` | 13.0 | yes | 0 | 2.0 | 18 |
| `pair_CA_A` | 21.0 | yes | 0 | 2.0 | 18 |
| `pair_CA_A` | 34.0 | yes | 0 | 2.0 | 18 |
| `pair_CA_A` | 55.0 | yes | 0 | 2.0 | 18 |
| `dna_two_ACG_AG` | 0.5 | no | inf | 2.0 | 43 |
| `dna_two_ACG_AG` | 1.0 | no | inf | 2.0 | 43 |
| `dna_two_ACG_AG` | 2.0 | yes | 0 | 2.0 | 43 |
| `dna_two_ACG_AG` | 3.0 | yes | 0 | 2.0 | 43 |
| `dna_two_ACG_AG` | 5.0 | yes | 0 | 2.0 | 43 |
| `dna_two_ACG_AG` | 8.0 | yes | 0 | 2.0 | 43 |
| `dna_two_ACG_AG` | 13.0 | yes | 0 | 2.0 | 43 |
| `dna_two_ACG_AG` | 21.0 | yes | 0 | 2.0 | 43 |
| `dna_two_ACG_AG` | 34.0 | yes | 0 | 2.0 | 43 |
| `dna_two_ACG_AG` | 55.0 | yes | 0 | 2.0 | 43 |

## 3. QAOA depth study (9 qubits, `pair_CA_A`)

Uniform computational-basis baseline P(optimal) = 0.0020; this includes flow-infeasible bitstrings.

| Depth p | <H_C> | P(optimal) | Optimal decoded |
| ---: | ---: | ---: | :--: |
| 1 | 8.700 | 0.054 | yes |
| 2 | 5.448 | 0.134 | no |
| 3 | 3.978 | 0.204 | no |
| 4 | 3.111 | 0.269 | yes |
| 5 | 3.093 | 0.264 | yes |

## 3b. Banded three-sequence QAOA (`k3_banded_ACGT_ACT_ACT`, k=3)

Diagonal band cuts 361 qubits (full lattice) to 13; band still contains the optimum (score 14). Uniform computational-basis baseline P(optimal) = 0.00012; this includes flow-infeasible bitstrings.

| Depth p | <H_C> | P(optimal) |
| ---: | ---: | ---: |
| 1 | 6.589 | 0.009 |
| 2 | -2.769 | 0.040 |
| 3 | -8.451 | 0.083 |
| 4 | -10.483 | 0.123 |
| 5 | -10.634 | 0.132 |

## 4. Beyond our exact-DP budget (constructed known optima)

| k | Lattice states | DP status | DP score | Known optimum | SA score | SA optimal | SA time (s) |
| ---: | ---: | --- | ---: | ---: | ---: | :--: | ---: |
| 4 | 5,832 | feasible | 84 | 84 | 84 | yes | 2.77 |
| 5 | 52,488 | feasible | 144 | 144 | 144 | yes | 3.49 |
| 6 | 472,392 | feasible | 220 | 220 | 220 | yes | 4.28 |
| 7 | 4,251,528 | not_run | - | 312 | 312 | yes | 5.25 |
| 8 | 38,263,752 | not_run | - | 420 | 420 | yes | 6.34 |
