# Quantum-Inspired Multiple Sequence Alignment

This is the standalone code-and-results repository for a multiple sequence
alignment (MSA) project. It formulates an MSA as a monotone path through a
`k`-dimensional edit lattice, realizes the path objective as a Hamiltonian, and
studies exact dynamic programming (DP), simulated annealing, an explicit
flow-conservation quadratic unconstrained binary optimization (QUBO)
formulation and its equivalent Ising spin-model representation, and the Quantum
Approximate Optimization Algorithm (QAOA).

The repository is intentionally limited to runnable research code, tests, and
generated results. It excludes application manuscripts, literature files,
presentations, and videos.

## Terminology and abbreviations

- **MSA — multiple sequence alignment:** inserting gaps into three or more
  biological sequences so homologous or comparable symbols share columns.
- **DP — dynamic programming:** an exact classical method that solves the MSA
  problem by reusing optimal solutions for smaller edit-lattice states.
- **QUBO — quadratic unconstrained binary optimization:** an objective written
  as linear and quadratic terms in binary variables. Constraint penalties make
  invalid edge selections more expensive than valid alignment paths.
- **Ising model:** a mathematically equivalent spin representation of the QUBO.
  It is named after physicist Ernst Ising and is not an acronym; binary variables
  are replaced by spins with values `-1` or `+1`.
- **QAOA — Quantum Approximate Optimization Algorithm:** a variational quantum
  algorithm that alternates cost and mixing operations to concentrate
  probability on low-energy QUBO/Ising states.
- **BLOSUM62 — Blocks Substitution Matrix 62:** a protein substitution-scoring
  matrix used to reward or penalize aligned amino-acid pairs.
- **MAFFT — Multiple Alignment using Fast Fourier Transform:** an external MSA
  program used here as a comparison method.
- **DNA — deoxyribonucleic acid:** the sequence type used in the nucleotide
  examples.
- **CSV — comma-separated values:** the machine-readable result-table format.
- **PNG — Portable Network Graphics; PDF — Portable Document Format:** the
  raster-image and fixed-layout formats used for generated figures.
- **TeX/LaTeX:** document-typesetting systems used for generated report tables;
  these names are not acronyms in this context.

## Repository layout

```text
experiments/
  msa_hamiltonian.py                 edit lattice, exact DP, scoring, annealing
  qubo_msa.py                        flow-conservation QUBO/Ising encoding
  qaoa_msa.py                        NumPy statevector QAOA simulator
  substitution_matrices.py           embedded BLOSUM62 matrix
  alignment_counting.py              alignment-space enumeration
  run_msa_research_experiments.py    core exact/scale/annealing studies
  run_quantum_experiments.py         QUBO, penalty, QAOA, and scaling studies
  run_affine_blosum_experiments.py   affine-gap and BLOSUM62 studies
  mafft_comparison.py                optional comparison with external MAFFT
tests/                                24 unit tests
results/msa_research/
  *.csv                               comma-separated-value experiment outputs
  *.md                                human-readable Markdown summaries
  *.tex                               generated LaTeX table sources
  figures/                            generated plots in PNG and PDF formats
```

## Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The core path-Hamiltonian implementation uses only the Python standard library.
NumPy, SciPy, and Matplotlib are needed for the QUBO/QAOA studies and figures.

## Verify the code

```bash
python -m unittest discover -s tests -v
```

or:

```bash
make test
```

## Reproduce the committed results

Each driver writes only beneath `results/msa_research/`.

```bash
# Exact DP, scaling, and path-state simulated annealing
python experiments/run_msa_research_experiments.py

# Affine-gap and BLOSUM62 validation
python experiments/run_affine_blosum_experiments.py

# QUBO validation, exact penalty sweeps, QAOA, and figures
python experiments/run_quantum_experiments.py

# Optional external cross-check; requires MAFFT on PATH or MAFFT_BIN
python experiments/mafft_comparison.py
```

The quantum experiment driver includes exact enumeration of 23-bit QUBOs and
can take substantially longer than the unit tests. Runtime fields in regenerated
CSVs and summaries depend on the machine; scores and seeded stochastic outcomes
are the reproducibility targets.

The same commands are available as `make results-core`, `make results-affine`,
`make results-quantum`, and `make results-mafft`. `make results` runs the three
self-contained drivers and does not require MAFFT.

## Results snapshot

The committed outputs record these principal checks:

- The edit-lattice path score and Hamiltonian energy satisfy
  `energy = -score`, and exact DP finds the ground-state alignment.
- The explicit flow-conservation QUBO agrees with exact DP on tested instances;
  the largest validation instance contains 98 binary edge variables and is
  solved by the project's annealing path rather than exhaustive enumeration.
- On the 9-qubit demonstration, depth-4 QAOA assigns probability `0.269` to the
  optimum versus a `0.0020` uniform computational-basis baseline. That baseline
  includes infeasible edge assignments, so this is a proof of concept rather
  than evidence of quantum advantage.
- The constructed `k=8` annealing case reaches its certified score on a lattice
  with `38,263,752` states; this exceeds this implementation's configured
  full-lattice DP budget, not the theoretical reach of dynamic programming.
- Affine-gap and BLOSUM62 results match brute-force optima on the committed small
  validation cases.

See [results/msa_research/summary.md](results/msa_research/summary.md) and
[results/msa_research/quantum_summary.md](results/msa_research/quantum_summary.md)
for the complete tables and caveats.

## Scoring convention

The default linear scoring is match `+2`, mismatch `-1`, gap `-2`, and gap-gap
`0`. Affine scoring charges a gap run of length `L` as
`gap_open + L * gap`; the opening position incurs both terms. Some software uses
`gap_open + (L - 1) * gap`, so convert parameters before comparing scores.
