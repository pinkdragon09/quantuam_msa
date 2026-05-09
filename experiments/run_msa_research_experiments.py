#!/usr/bin/env python3
"""Run reproducible experiments for the MSA Hamiltonian study."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
import time
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import msa_hamiltonian as msa  # noqa: E402


RESULT_DIR = ROOT / "results" / "msa_research"
REPORT_TABLES = RESULT_DIR / "msa_research_tables.tex"


@dataclass(frozen=True)
class Dataset:
    name: str
    kind: str
    sequences: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class AnnealConfig:
    name: str
    restarts: int
    steps_per_restart: int
    seeds: tuple[int, ...]


SCORING = msa.Scoring(match=2, mismatch=-1, gap=-2, gap_gap=0)


def family(base: str, count: int) -> tuple[str, ...]:
    """Create deterministic synthetic sequence families for scaling tests."""

    alphabet = "ACGT"
    seqs = []
    for idx in range(count):
        chars = list(base)
        if idx % 4 == 1 and len(chars) >= 4:
            pos = len(chars) // 2
            chars[pos] = alphabet[(alphabet.index(chars[pos]) + 1) % len(alphabet)]
        elif idx % 4 == 2 and len(chars) >= 5:
            del chars[len(chars) // 2]
        elif idx % 4 == 3 and len(chars) >= 4:
            chars.insert(len(chars) // 2, "T")
        seqs.append("".join(chars))
    return tuple(seqs)


def datasets() -> list[Dataset]:
    return [
        Dataset(
            "dna_three_sequence",
            "DNA",
            ("ACGTAC", "ACGTC", "ACTTAC"),
            "three short DNA sequences with one deletion and one substitution pattern",
        ),
        Dataset(
            "protein_three_sequence",
            "protein",
            ("MEEPQSD", "MEESQSD", "MEPQSD"),
            "three short protein-like sequences with substitution/deletion ambiguity",
        ),
        Dataset(
            "synthetic_four_sequence",
            "DNA",
            ("ACGTACGT", "ACGTTCGT", "ACGACGT", "ACGTACGTT"),
            "four DNA sequences with substitution, deletion, and insertion",
        ),
        Dataset(
            "dna_four_gattaca",
            "DNA",
            ("GATTACA", "GCATGCA", "GATACA", "GATTACCA"),
            "four DNA sequences derived from a common short motif",
        ),
        Dataset(
            "dna_four_conserved_block",
            "DNA",
            ("TTACGTAA", "TACGTTA", "TTACGTA", "TTACGTTAA"),
            "four DNA sequences sharing a conserved ACGT block",
        ),
    ]


def run_exact_summary(rows: list[dict[str, str]]) -> None:
    for dataset in datasets():
        start = time.perf_counter()
        result = msa.exact_msa_dp(dataset.sequences, SCORING)
        elapsed = time.perf_counter() - start

        rows.append(
            {
                "dataset": dataset.name,
                "kind": dataset.kind,
                "k": str(len(dataset.sequences)),
                "lengths": "/".join(str(len(seq)) for seq in dataset.sequences),
                "states": str(result.states_evaluated),
                "score": str(result.score),
                "energy": str(result.energy),
                "columns": str(len(result.path)),
                "time_seconds": f"{elapsed:.4f}",
                "alignment": " | ".join(result.alignment),
                "description": dataset.description,
            }
        )


def run_scaling(rows: list[dict[str, str]]) -> None:
    for length in (4, 6, 8):
        base = ("ACGT" * ((length + 3) // 4))[:length]
        for k in (2, 3, 4, 5):
            seqs = family(base, k)
            start = time.perf_counter()
            result = msa.exact_msa_dp(seqs, SCORING)
            elapsed = time.perf_counter() - start

            rows.append(
                {
                    "length": str(length),
                    "k": str(k),
                    "lengths": "/".join(str(len(seq)) for seq in seqs),
                    "states": str(result.states_evaluated),
                    "steps": str(2**k - 1),
                    "score": str(result.score),
                    "time_seconds": f"{elapsed:.4f}",
                }
            )


def run_annealing_trials(rows: list[dict[str, str]]) -> None:
    selected = {
        "dna_three_sequence",
        "protein_three_sequence",
        "synthetic_four_sequence",
        "dna_four_gattaca",
    }
    configs = [
        AnnealConfig("small", 20, 300, tuple(range(10))),
        AnnealConfig("medium", 50, 700, tuple(range(10))),
    ]

    exact_by_name = {
        dataset.name: msa.exact_msa_dp(dataset.sequences, SCORING)
        for dataset in datasets()
        if dataset.name in selected
    }
    data_by_name = {dataset.name: dataset for dataset in datasets()}

    for dataset_name, exact in exact_by_name.items():
        dataset = data_by_name[dataset_name]
        for config in configs:
            for seed_offset in config.seeds:
                seed = 20260630 + seed_offset + len(dataset_name) * 101 + len(config.name)
                start = time.perf_counter()
                annealed = msa.anneal_msa(
                    dataset.sequences,
                    SCORING,
                    restarts=config.restarts,
                    steps_per_restart=config.steps_per_restart,
                    seed=seed,
                )
                elapsed = time.perf_counter() - start
                gap = exact.score - annealed.score
                rows.append(
                    {
                        "dataset": dataset_name,
                        "kind": dataset.kind,
                        "config": config.name,
                        "seed": str(seed),
                        "k": str(len(dataset.sequences)),
                        "exact_score": str(exact.score),
                        "anneal_score": str(annealed.score),
                        "score_gap": str(gap),
                        "optimal": "yes" if gap == 0 else "no",
                        "evaluations": str(annealed.states_evaluated),
                        "time_seconds": f"{elapsed:.4f}",
                        "columns": str(len(annealed.path)),
                    }
                )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def summarize_annealing(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["config"]), []).append(row)

    summary = []
    for (dataset, config), group in sorted(grouped.items()):
        success = sum(1 for row in group if row["optimal"] == "yes")
        gaps = [int(row["score_gap"]) for row in group]
        times = [float(row["time_seconds"]) for row in group]
        scores = [int(row["anneal_score"]) for row in group]
        summary.append(
            {
                "dataset": dataset,
                "config": config,
                "trials": str(len(group)),
                "successes": str(success),
                "success_rate": format_float(success / len(group), 2),
                "best_score": str(max(scores)),
                "mean_score_gap": format_float(mean(gaps), 2),
                "max_score_gap": str(max(gaps)),
                "mean_time_seconds": format_float(mean(times), 3),
            }
        )
    return summary


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def write_latex_tables(
    exact_rows: list[dict[str, str]],
    scaling_rows: list[dict[str, str]],
    anneal_summary: list[dict[str, str]],
) -> None:
    lines = [
        "% Auto-generated by experiments/run_msa_research_experiments.py",
        "",
        r"\newcommand{\ExactResultsTable}{%",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Instance & Type & $k$ & Lengths & States & Score & $H_{\mathrm{path}}$ \\",
        r"\midrule",
    ]
    for row in exact_rows:
        lines.append(
            f"{latex_escape(row['dataset'])} & {row['kind']} & {row['k']} & "
            f"{row['lengths']} & {row['states']} & {row['score']} & {row['energy']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Exact-DP implementation results on short constructed instances. The reported classical path energy equals the negative score by definition.}",
            r"\label{tab:exact-results}",
            r"\end{table}",
            r"}",
            "",
            r"\newcommand{\ScalingResultsTable}{%",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{rrrrrr}",
            r"\toprule",
            r"Length & $k$ & Lengths & Lattice states & Step types & Time (s) \\",
            r"\midrule",
        ]
    )
    for row in scaling_rows:
        lines.append(
            f"{row['length']} & {row['k']} & {row['lengths']} & {row['states']} & "
            f"{row['steps']} & {row['time_seconds']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Scaling of exact lattice DP in uncontrolled timing runs. Lattice states are $\prod_r(n_r+1)$ and step types are $2^k-1$.}",
            r"\label{tab:scaling-results}",
            r"\end{table}",
            r"}",
            "",
            r"\newcommand{\AnnealingResultsTable}{%",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{llrrrrr}",
            r"\toprule",
            r"Instance & Budget & Trials & Successes & Success rate & Mean gap & Mean time (s) \\",
            r"\midrule",
        ]
    )
    for row in anneal_summary:
        lines.append(
            f"{latex_escape(row['dataset'])} & {row['config']} & {row['trials']} & "
            f"{row['successes']} & {row['success_rate']} & {row['mean_score_gap']} & "
            f"{row['mean_time_seconds']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Classical path-state simulated annealing over ten deterministic-seed trials per cell. Score gap is exact score minus annealed score; timings are uncontrolled.}",
            r"\label{tab:annealing-results}",
            r"\end{table}",
            r"}",
            "",
        ]
    )
    REPORT_TABLES.write_text("\n".join(lines) + "\n")


def write_markdown_summary(
    exact_rows: list[dict[str, str]],
    scaling_rows: list[dict[str, str]],
    anneal_summary: list[dict[str, str]],
) -> None:
    lines = [
        "# MSA Hamiltonian Research Experiment Summary",
        "",
        "Scoring: match=+2, mismatch=-1, gap=-2, gap-gap=0.",
        "",
        "## Exact DP",
        "",
        "| Dataset | k | Lengths | States | Score | Energy | Time (s) |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in exact_rows:
        lines.append(
            f"| `{row['dataset']}` | {row['k']} | {row['lengths']} | {row['states']} | "
            f"{row['score']} | {row['energy']} | {row['time_seconds']} |"
        )

    lines.extend(
        [
            "",
            "## Exact DP Scaling",
            "",
            "| Length | k | Lengths | States | Step types | Time (s) |",
            "| ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in scaling_rows:
        lines.append(
            f"| {row['length']} | {row['k']} | {row['lengths']} | {row['states']} | "
            f"{row['steps']} | {row['time_seconds']} |"
        )

    lines.extend(
        [
            "",
            "## Annealing Summary",
            "",
            "| Dataset | Budget | Trials | Successes | Success rate | Mean score gap | Mean time (s) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in anneal_summary:
        lines.append(
            f"| `{row['dataset']}` | {row['config']} | {row['trials']} | {row['successes']} | "
            f"{row['success_rate']} | {row['mean_score_gap']} | {row['mean_time_seconds']} |"
        )

    (RESULT_DIR / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    exact_rows: list[dict[str, str]] = []
    scaling_rows: list[dict[str, str]] = []
    anneal_rows: list[dict[str, str]] = []

    print("Running exact DP examples...")
    run_exact_summary(exact_rows)
    write_csv(RESULT_DIR / "exact_summary.csv", exact_rows)

    print("Running exact DP scaling sweep...")
    run_scaling(scaling_rows)
    write_csv(RESULT_DIR / "scaling_summary.csv", scaling_rows)

    print("Running annealing trials...")
    run_annealing_trials(anneal_rows)
    write_csv(RESULT_DIR / "annealing_trials.csv", anneal_rows)
    anneal_summary = summarize_annealing(anneal_rows)
    write_csv(RESULT_DIR / "annealing_summary.csv", anneal_summary)

    write_latex_tables(exact_rows, scaling_rows, anneal_summary)
    write_markdown_summary(exact_rows, scaling_rows, anneal_summary)

    print(f"Wrote results to {RESULT_DIR}")
    print(f"Wrote LaTeX tables to {REPORT_TABLES}")


if __name__ == "__main__":
    main()
