# MAFFT Comparison

MAFFT v7.526 (`--auto`). Each alignment scored under our sum-of-pairs
objective; our optimum is the exact DP ground state.

| Dataset | Objective | Our optimum | MAFFT | Gap | MAFFT optimal | Columns identical |
| --- | --- | ---: | ---: | ---: | :--: | :--: |
| `dna_three_sequence` | match/mismatch | 22 | 22 | 0 | yes | yes |
| `protein_three_sequence` | BLOSUM62 | 77 | 77 | 0 | yes | no |
| `synthetic_four_sequence` | match/mismatch | 69 | 18 | 51 | no | no |
| `dna_four_gattaca` | match/mismatch | 45 | 39 | 6 | no | no |
| `dna_four_conserved_block` | match/mismatch | 60 | 54 | 6 | no | no |
| `protein_HEAG` | BLOSUM62 | 168 | 168 | 0 | yes | yes |
