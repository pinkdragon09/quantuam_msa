PYTHON ?= python3

.PHONY: install test results results-core results-affine results-quantum results-mafft

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m unittest discover -s tests -v

results-core:
	$(PYTHON) experiments/run_msa_research_experiments.py

results-affine:
	$(PYTHON) experiments/run_affine_blosum_experiments.py

results-quantum:
	$(PYTHON) experiments/run_quantum_experiments.py

results-mafft:
	$(PYTHON) experiments/mafft_comparison.py

results: results-core results-affine results-quantum
