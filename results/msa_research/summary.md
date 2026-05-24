# MSA Hamiltonian Research Experiment Summary

Scoring: match=+2, mismatch=-1, gap=-2, gap-gap=0.

## Exact DP

| Dataset | k | Lengths | States | Score | Energy | Time (s) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `dna_three_sequence` | 3 | 6/5/6 | 294 | 22 | -22 | 0.0046 |
| `protein_three_sequence` | 3 | 7/7/6 | 448 | 28 | -28 | 0.0061 |
| `synthetic_four_sequence` | 4 | 8/8/7/9 | 6480 | 69 | -69 | 0.2254 |
| `dna_four_gattaca` | 4 | 7/7/6/8 | 4032 | 45 | -45 | 0.1359 |
| `dna_four_conserved_block` | 4 | 8/7/7/9 | 5760 | 60 | -60 | 0.1968 |

## Exact DP Scaling

| Length | k | Lengths | States | Step types | Time (s) |
| ---: | ---: | --- | ---: | ---: | ---: |
| 4 | 2 | 4/4 | 25 | 3 | 0.0001 |
| 4 | 3 | 4/4/4 | 125 | 7 | 0.0016 |
| 4 | 4 | 4/4/4/5 | 750 | 15 | 0.0225 |
| 4 | 5 | 4/4/4/5/4 | 3750 | 31 | 0.3195 |
| 6 | 2 | 6/6 | 49 | 3 | 0.0003 |
| 6 | 3 | 6/6/5 | 294 | 7 | 0.0038 |
| 6 | 4 | 6/6/5/7 | 2352 | 15 | 0.0752 |
| 6 | 5 | 6/6/5/7/6 | 16464 | 31 | 1.3568 |
| 8 | 2 | 8/8 | 81 | 3 | 0.0005 |
| 8 | 3 | 8/8/7 | 648 | 7 | 0.0090 |
| 8 | 4 | 8/8/7/9 | 6480 | 15 | 0.2226 |
| 8 | 5 | 8/8/7/9/8 | 58320 | 31 | 5.1162 |

## Annealing Summary

| Dataset | Budget | Trials | Successes | Success rate | Mean score gap | Mean time (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `dna_four_gattaca` | medium | 10 | 10 | 1.00 | 0.00 | 1.327 |
| `dna_four_gattaca` | small | 10 | 5 | 0.50 | 1.50 | 0.247 |
| `dna_three_sequence` | medium | 10 | 10 | 1.00 | 0.00 | 0.863 |
| `dna_three_sequence` | small | 10 | 10 | 1.00 | 0.00 | 0.149 |
| `protein_three_sequence` | medium | 10 | 10 | 1.00 | 0.00 | 0.959 |
| `protein_three_sequence` | small | 10 | 10 | 1.00 | 0.00 | 0.169 |
| `synthetic_four_sequence` | medium | 10 | 10 | 1.00 | 0.00 | 1.457 |
| `synthetic_four_sequence` | small | 10 | 10 | 1.00 | 0.00 | 0.258 |
