# Results

All experiment drivers write into `msa_research/` so the repository can be run
without the application manuscript tree from the parent project.

- CSV files are the machine-readable primary outputs.
- Markdown files summarize the exact-DP, annealing, QUBO, QAOA, and MAFFT runs.
- TeX files are generated table fragments for reuse in reports.
- `figures/` contains generated PNG and PDF plots.

Regeneration overwrites the corresponding tracked files. Runtime measurements
will vary across machines; scores and seeded stochastic outcomes should remain
stable for the same dependency versions.
