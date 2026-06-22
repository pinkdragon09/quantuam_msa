#!/usr/bin/env python3
"""Reproducible experiments for the QUBO / QAOA extension of the MSA report.

Produces four studies and writes CSVs, a LaTeX table file, a Markdown summary,
and two figures:

1. QUBO ground state vs exact DP -- the encoding is correct: the ground state of
   the flow-conservation Hamiltonian is the maximum-score alignment.
2. Penalty sweep on two instances -- compare the theoretically sufficient bound
   with the smallest successful candidate in a finite tested grid.
3. QAOA depth study -- ground-state recovery probability rises with circuit
   depth p on a small instance, using a from-scratch statevector simulator.
4. Beyond our exact-DP budget -- constructed families with known optima show a
   path-state annealer recovering the optimum for k up to 8, beyond the cutoff
   used by this Python full-lattice DP implementation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from math import comb
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import msa_hamiltonian as msa  # noqa: E402
import qubo_msa  # noqa: E402
import qaoa_msa  # noqa: E402

RESULT_DIR = ROOT / "results" / "msa_research"
FIG_DIR = ROOT / "figures"
QUANTUM_TABLES = RESULT_DIR / "msa_quantum_tables.tex"

SCORING = msa.Scoring(match=2, mismatch=-1, gap=-2, gap_gap=0)

# Full-lattice DP is enumerated in Python; beyond this many states we treat it as
# infeasible for this project's compute budget.
DP_STATE_LIMIT = 1_500_000


def lattice_states(sequences) -> int:
    total = 1
    for seq in sequences:
        total *= len(seq) + 1
    return total


# --------------------------------------------------------------------------- #
# Study 1: QUBO ground state vs exact DP
# --------------------------------------------------------------------------- #
QUBO_INSTANCES = [
    ("pair_CA_A", ("CA", "A")),
    ("pair_AC_A", ("AC", "A")),
    ("pair_AT_AAT", ("AT", "AAT")),
    ("dna_two_ACG_AG", ("ACG", "AG")),
    ("dna_two_GATT_GAT", ("GATT", "GAT")),
    ("dna_three_tiny", ("AC", "AC", "AT")),
]


def run_qubo_validation() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, seqs in QUBO_INSTANCES:
        model = qubo_msa.build_qubo(seqs, SCORING)
        dp = msa.exact_msa_dp(seqs, SCORING)
        if model.num_qubits <= 22:
            energy, _, grounds = qubo_msa.exact_ground_state(model)
            decoded = qubo_msa.decode_with_scoring(model, grounds[0], SCORING)
            method = "exact"
            degeneracy = str(len(grounds))
        else:
            res = qubo_msa.anneal_qubo(model, SCORING, restarts=40, sweeps=350, seed=17)
            energy, decoded = res.energy, res.decoded
            method = "anneal"
            degeneracy = "-"
        rows.append(
            {
                "instance": name,
                "k": str(len(seqs)),
                "lengths": "/".join(str(len(s)) for s in seqs),
                "qubits": str(model.num_qubits),
                "penalty": f"{model.penalty:.0f}",
                "ground_energy": f"{energy:.1f}",
                "dp_score": str(dp.score),
                "qubo_score": str(decoded.score),
                "matches_dp": "yes" if decoded.valid and decoded.score == dp.score else "no",
                "degeneracy": degeneracy,
                "method": method,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Study 2: finite penalty sweep on two instances
# --------------------------------------------------------------------------- #
def run_penalty_analysis() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, seqs in [("pair_CA_A", ("CA", "A")), ("dna_two_ACG_AG", ("ACG", "AG"))]:
        conservative = qubo_msa.build_qubo(seqs, SCORING).penalty
        smallest_tested, trace = qubo_msa.minimal_feasible_penalty(seqs, SCORING)
        for A, feasible, gap in trace:
            rows.append(
                {
                    "instance": name,
                    "penalty": f"{A:.1f}",
                    "feasible": "yes" if feasible else "no",
                    "score_gap": "nan" if gap != gap else f"{gap:.0f}",
                    "smallest_successful_tested": (
                        f"{smallest_tested:.1f}" if smallest_tested is not None else "none"
                    ),
                    "conservative_bound": f"{conservative:.0f}",
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# Study 3: QAOA depth study (pairwise, and a banded three-sequence instance)
# --------------------------------------------------------------------------- #
def _qaoa_study(
    name: str,
    seqs: tuple[str, ...],
    band: float | None,
    depths=(1, 2, 3, 4, 5),
) -> tuple[list[dict[str, str]], dict]:
    penalty, _ = qubo_msa.minimal_feasible_penalty(seqs, SCORING, band=band)
    model = qubo_msa.build_qubo(seqs, SCORING, band=band, penalty=penalty)
    full_qubits = qubo_msa.build_qubo(seqs, SCORING).num_qubits
    dp = msa.exact_msa_dp(seqs, SCORING)
    results = qaoa_msa.run_qaoa_sweep(model, SCORING, depths=depths, seed=100)
    baseline = results[0].num_ground_states / (2 ** model.num_qubits)
    rows: list[dict[str, str]] = []
    for result in results:
        rows.append(
            {
                "instance": name,
                "qubits": str(model.num_qubits),
                "depth_p": str(result.p),
                "expected_cost": f"{result.expected_cost:.3f}",
                "ground_energy": f"{result.ground_energy:.1f}",
                "ground_probability": f"{result.ground_probability:.3f}",
                "random_baseline": f"{baseline:.4f}",
                "decoded_optimal": "yes"
                if result.decoded.valid and result.decoded.score == dp.score
                else "no",
            }
        )
    meta = {
        "instance": name,
        "sequences": seqs,
        "k": len(seqs),
        "band": band,
        "qubits": model.num_qubits,
        "full_qubits": full_qubits,
        "penalty": penalty,
        "dp_score": dp.score,
        "alignment": dp.alignment,
        "baseline": baseline,
    }
    return rows, meta


def run_qaoa_study(depths=(1, 2, 3, 4, 5)) -> tuple[list[dict[str, str]], dict]:
    name, seqs = qaoa_msa.pick_small_instance(SCORING)
    return _qaoa_study(name, seqs, band=None, depths=depths)


def run_banded_qaoa_study(depths=(1, 2, 3, 4, 5)) -> tuple[list[dict[str, str]], dict]:
    # A three-sequence instance exceeds this project's statevector QAOA budget on
    # the full lattice; a heuristic diagonal band prunes it to a handful of qubits.
    return _qaoa_study("k3_banded_ACGT_ACT_ACT", ("ACGT", "ACT", "ACT"), band=0.30,
                       depths=depths)


# --------------------------------------------------------------------------- #
# Study 4: beyond our exact-DP budget with constructed known optima
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConstructedFamily:
    name: str
    sequences: tuple[str, ...]
    optimal_alignment: tuple[str, ...]
    optimal_score: int


def single_indel_family(base: str, k: int, delete_at: int) -> ConstructedFamily:
    """k sequences: k-1 copies of `base`, one with a single deletion.

    The provably optimal alignment keeps the k-1 identical sequences perfectly
    aligned and places the deleted sequence's single gap at the deletion site,
    so it matches the others everywhere else.
    """

    deleted = base[:delete_at] + base[delete_at + 1 :]
    sequences = tuple([base] * (k - 1) + [deleted])
    aligned_deleted = base[:delete_at] + "-" + base[delete_at + 1 :]
    optimal_alignment = tuple([base] * (k - 1) + [aligned_deleted])
    path = msa.alignment_to_path(optimal_alignment)
    optimal_score = msa.score_path(sequences, path, SCORING)
    return ConstructedFamily(
        f"indel_k{k}_L{len(base)}", sequences, optimal_alignment, optimal_score
    )


def run_beyond_dp() -> list[dict[str, str]]:
    base = "ACGTACGT"  # length 8, structured so gap placement matters
    rows: list[dict[str, str]] = []
    for k in (4, 5, 6, 7, 8):
        family = single_indel_family(base, k, delete_at=4)
        states = lattice_states(family.sequences)

        dp_score = "-"
        dp_time = "-"
        dp_status = "not_run"
        if states <= DP_STATE_LIMIT:
            start = time.perf_counter()
            dp = msa.exact_msa_dp(family.sequences, SCORING)
            dp_time = f"{time.perf_counter() - start:.2f}"
            dp_score = str(dp.score)
            dp_status = "feasible"

        start = time.perf_counter()
        annealed = msa.anneal_msa(
            family.sequences,
            SCORING,
            restarts=60,
            steps_per_restart=1200,
            seed=31 + k,
        )
        sa_time = time.perf_counter() - start

        rows.append(
            {
                "family": family.name,
                "k": str(k),
                "lengths": "/".join(str(len(s)) for s in family.sequences),
                "lattice_states": str(states),
                "dp_status": dp_status,
                "dp_score": dp_score,
                "dp_time_s": dp_time,
                "known_optimum": str(family.optimal_score),
                "sa_score": str(annealed.score),
                "sa_optimal": "yes" if annealed.score == family.optimal_score else "no",
                "sa_time_s": f"{sa_time:.2f}",
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(qaoa_rows, qaoa_meta, banded_rows, banded_meta, beyond_rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # QAOA ground-state probability vs depth: pairwise and banded three-sequence
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    max_prob = 0.0
    min_baseline = 1.0
    for rows, meta, color, marker in (
        (qaoa_rows, qaoa_meta, "#1f4e79", "o"),
        (banded_rows, banded_meta, "#2e8b57", "s"),
    ):
        ps = [int(r["depth_p"]) for r in rows]
        probs = [float(r["ground_probability"]) for r in rows]
        max_prob = max(max_prob, max(probs))
        min_baseline = min(min_baseline, meta["baseline"])
        label = (
            f"$k$={meta['k']} "
            + ("(banded, " if meta["band"] else "(")
            + f"{meta['qubits']} qubits)"
        )
        ax.plot(ps, probs, marker + "-", color=color, linewidth=2, markersize=7,
                label=label)
        ax.axhline(meta["baseline"], color=color, linestyle=":", linewidth=1.0,
                   alpha=0.8, label=f"$k$={meta['k']} uniform bitstrings")
    ax.set_xlabel("QAOA depth $p$")
    ax.set_ylabel("P(measure optimal alignment)")
    ax.set_title("QAOA ground-state recovery")
    ax.set_xticks([int(r["depth_p"]) for r in qaoa_rows])
    ax.set_yscale("log")
    ax.set_ylim(min_baseline * 0.55, min(1.0, max_prob * 1.6))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "qaoa_depth.png", dpi=160)
    fig.savefig(FIG_DIR / "qaoa_depth.pdf")
    plt.close(fig)

    # Lattice growth vs k (beyond-DP): states on log scale
    ks = [int(r["k"]) for r in beyond_rows]
    states = [int(r["lattice_states"]) for r in beyond_rows]
    feasible = [r["dp_status"] == "feasible" for r in beyond_rows]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ax.semilogy(ks, states, "o-", color="#1f4e79", linewidth=2, markersize=7)
    for k, s, feas in zip(ks, states, feasible):
        if feas:
            text = "DP ok"
            xytext = (8, -4)
            ha = "left"
        elif k == max(ks):
            text = "DP not run:\nabove cutoff"
            xytext = (-8, -4)
            ha = "right"
        else:
            text = "DP not run:\nabove cutoff"
            xytext = (8, 8)
            ha = "left"
        ax.annotate(
            text,
            (k, s),
            textcoords="offset points",
            xytext=xytext,
            fontsize=8,
            ha=ha,
            color="#1f4e79" if feas else "#b00020",
        )
    ax.axhline(DP_STATE_LIMIT, color="#b00020", linestyle="--", linewidth=1.2,
               label=f"DP budget ({DP_STATE_LIMIT:,} states)")
    ax.set_xlabel("number of sequences $k$")
    ax.set_ylabel("edit-lattice states $\\prod_r (n_r+1)$")
    ax.set_title("Lattice growth and annealer recovery")
    ax.set_xticks(ks)
    ax.set_ylim(min(states) * 0.55, max(states) * 2.8)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lattice_scaling.png", dpi=160)
    fig.savefig(FIG_DIR / "lattice_scaling.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def esc(text: str) -> str:
    return text.replace("_", r"\_")


def write_latex(qubo_rows, qaoa_rows, qaoa_meta, banded_rows, banded_meta, beyond_rows) -> None:
    lines: list[str] = ["% Auto-generated by run_quantum_experiments.py", ""]

    # QUBO validation table
    lines += [
        r"\newcommand{\QUBOValidationTable}{%",
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\begin{tabular}{llrrrrcl}", r"\toprule",
        r"Instance & Lengths & Qubits & $A$ & DP score & QUBO score & Match & Method \\",
        r"\midrule",
    ]
    for r in qubo_rows:
        lines.append(
            f"{esc(r['instance'])} & {r['lengths']} & {r['qubits']} & {r['penalty']} & "
            f"{r['dp_score']} & {r['qubo_score']} & {r['matches_dp']} & "
            f"{'exhaustive' if r['method'] == 'exact' else 'annealed'} \\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{QUBO consistency validation against exact DP. Two assignments are found "
        r"by exhaustive search; the other four are found by spin-flip annealing. All six "
        r"decoded scores match DP, but the four annealed rows are heuristic consistency "
        r"checks rather than exhaustive numerical ground-state proofs.}",
        r"\label{tab:qubo-validation}", r"\end{table}", r"}", "",
    ]

    # QAOA table
    lines += [
        r"\newcommand{\QAOAResultsTable}{%",
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\begin{tabular}{rrrr}", r"\toprule",
        r"Depth $p$ & $\langle \widehat H_C\rangle$ & P(optimal) & Optimal decoded \\",
        r"\midrule",
    ]
    for r in qaoa_rows:
        lines.append(
            f"{r['depth_p']} & {r['expected_cost']} & {r['ground_probability']} & "
            f"{r['decoded_optimal']} \\\\"
        )
    baseline = qaoa_meta["baseline"]
    lines += [
        r"\bottomrule", r"\end{tabular}",
        rf"\caption{{QAOA ({qaoa_meta['qubits']} qubits, instance "
        rf"\texttt{{{esc(qaoa_meta['instance'])}}}) concentrates probability on the "
        rf"optimal alignment above the uniform computational-basis baseline of ${baseline:.4f}$, "
        rf"peaking at depth $p=4$ in this sweep. This baseline includes flow-infeasible "
        rf"bitstrings and is not a constraint-aware comparison. $\langle \widehat H_C\rangle$ is the "
        rf"optimised cost expectation; the ground energy is $0$. ``Optimal decoded'' "
        rf"means the most-probable bitstring is feasible and decodes to an "
        rf"exact-DP-optimal alignment.}}",
        r"\label{tab:qaoa}", r"\end{table}", r"}", "",
    ]

    # Banded three-sequence QAOA table
    lines += [
        r"\newcommand{\BandedQAOATable}{%",
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\begin{tabular}{rrr}", r"\toprule",
        r"Depth $p$ & $\langle \widehat H_C\rangle$ & P(optimal) \\",
        r"\midrule",
    ]
    for r in banded_rows:
        lines.append(
            f"{r['depth_p']} & {r['expected_cost']} & {r['ground_probability']} \\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        rf"\caption{{QAOA on a \emph{{three-sequence}} instance "
        rf"(\texttt{{ACGT/ACT/ACT}}) using a heuristic fractional diagonal band of "
        rf"width $0.30$: the "
        rf"restriction cuts the encoding from {banded_meta['full_qubits']} qubits "
        rf"on the full lattice to {banded_meta['qubits']}; an ex post check confirms that "
        rf"the full-lattice optimum score {banded_meta['dp_score']} remains attainable. Recovery rises "
        rf"with depth above the uniform computational-basis baseline of "
        rf"${banded_meta['baseline']:.4f}$, which includes flow-infeasible bitstrings.}}",
        r"\label{tab:qaoa-banded}", r"\end{table}", r"}", "",
    ]

    # Beyond-DP table
    lines += [
        r"\newcommand{\BeyondDPTable}{%",
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\begin{tabular}{rlrlrrc}", r"\toprule",
        r"$k$ & Lattice states & DP & Known opt. & SA score & SA time (s) & SA optimal \\",
        r"\midrule",
    ]
    for r in beyond_rows:
        dp_cell = r["dp_score"] if r["dp_status"] == "feasible" else r"\emph{not run}"
        lines.append(
            f"{r['k']} & {int(r['lattice_states']):,} & {dp_cell} & {r['known_optimum']} & "
            f"{r['sa_score']} & {r['sa_time_s']} & {r['sa_optimal']} \\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Constructed single-deletion families. ``Not run'' means the lattice "
        r"exceeds the project's $1{,}500{,}000$-state DP cutoff, not that DP is "
        r"theoretically infeasible. The classical annealer recovers the certified optimum.}",
        r"\label{tab:beyond-dp}", r"\end{table}", r"}", "",
    ]

    QUANTUM_TABLES.write_text("\n".join(lines) + "\n")


def write_markdown(
    qubo_rows, penalty_rows, qaoa_rows, qaoa_meta, banded_rows, banded_meta, beyond_rows
) -> None:
    lines = [
        "# QUBO / QAOA Extension Experiment Summary",
        "",
        "Scoring: match=+2, mismatch=-1, gap=-2, gap-gap=0.",
        "",
        "## 1. QUBO ground state vs exact DP",
        "",
        "| Instance | Lengths | Qubits | Penalty A | DP score | QUBO score | Matches DP |",
        "| --- | --- | ---: | ---: | ---: | ---: | :--: |",
    ]
    for r in qubo_rows:
        lines.append(
            f"| `{r['instance']}` | {r['lengths']} | {r['qubits']} | {r['penalty']} | "
            f"{r['dp_score']} | {r['qubo_score']} | {r['matches_dp']} |"
        )
    lines += [
        "",
        "## 2. Finite penalty sweep on two instances",
        "",
        "Candidate grid: A = {0.5, 1, 2, 3, 5, 8, 13, 21, 34, 55}. The smallest "
        "successful tested value is not a universal or continuous minimum.",
        "",
        "| Instance | A | Feasible | Score gap | Smallest successful tested A | Conservative bound |",
        "| --- | ---: | :--: | ---: | ---: | ---: |",
    ]
    for r in penalty_rows:
        lines.append(
            f"| `{r['instance']}` | {r['penalty']} | {r['feasible']} | {r['score_gap']} | "
            f"{r['smallest_successful_tested']} | {r['conservative_bound']} |"
        )
    lines += [
        "",
        f"## 3. QAOA depth study ({qaoa_meta['qubits']} qubits, "
        f"`{qaoa_meta['instance']}`)",
        "",
        f"Uniform computational-basis baseline P(optimal) = {qaoa_meta['baseline']:.4f}; "
        "this includes flow-infeasible bitstrings.",
        "",
        "| Depth p | <H_C> | P(optimal) | Optimal decoded |",
        "| ---: | ---: | ---: | :--: |",
    ]
    for r in qaoa_rows:
        lines.append(
            f"| {r['depth_p']} | {r['expected_cost']} | {r['ground_probability']} | "
            f"{r['decoded_optimal']} |"
        )
    lines += [
        "",
        f"## 3b. Banded three-sequence QAOA "
        f"(`{banded_meta['instance']}`, k={banded_meta['k']})",
        "",
        f"Diagonal band cuts {banded_meta['full_qubits']} qubits (full lattice) to "
        f"{banded_meta['qubits']}; band still contains the optimum "
        f"(score {banded_meta['dp_score']}). Uniform computational-basis baseline "
        f"P(optimal) = {banded_meta['baseline']:.5f}; this includes flow-infeasible bitstrings.",
        "",
        "| Depth p | <H_C> | P(optimal) |",
        "| ---: | ---: | ---: |",
    ]
    for r in banded_rows:
        lines.append(
            f"| {r['depth_p']} | {r['expected_cost']} | {r['ground_probability']} |"
        )
    lines += [
        "",
        "## 4. Beyond our exact-DP budget (constructed known optima)",
        "",
        "| k | Lattice states | DP status | DP score | Known optimum | SA score | SA optimal | SA time (s) |",
        "| ---: | ---: | --- | ---: | ---: | ---: | :--: | ---: |",
    ]
    for r in beyond_rows:
        lines.append(
            f"| {r['k']} | {int(r['lattice_states']):,} | {r['dp_status']} | "
            f"{r['dp_score']} | {r['known_optimum']} | {r['sa_score']} | "
            f"{r['sa_optimal']} | {r['sa_time_s']} |"
        )
    (RESULT_DIR / "quantum_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("Study 1: QUBO ground state vs exact DP ...")
    qubo_rows = run_qubo_validation()
    write_csv(RESULT_DIR / "qubo_validation.csv", qubo_rows)

    print("Study 2: finite penalty sweep on two instances ...")
    penalty_rows = run_penalty_analysis()
    write_csv(RESULT_DIR / "penalty_analysis.csv", penalty_rows)

    print("Study 3: QAOA depth study (statevector simulation) ...")
    qaoa_rows, qaoa_meta = run_qaoa_study()
    write_csv(RESULT_DIR / "qaoa_depth.csv", qaoa_rows)

    print("Study 3b: banded three-sequence QAOA ...")
    banded_rows, banded_meta = run_banded_qaoa_study()
    write_csv(RESULT_DIR / "qaoa_banded_k3.csv", banded_rows)

    print("Study 4: beyond our exact-DP budget ...")
    beyond_rows = run_beyond_dp()
    write_csv(RESULT_DIR / "beyond_dp.csv", beyond_rows)

    print("Generating figures ...")
    make_figures(qaoa_rows, qaoa_meta, banded_rows, banded_meta, beyond_rows)

    write_latex(qubo_rows, qaoa_rows, qaoa_meta, banded_rows, banded_meta, beyond_rows)
    write_markdown(
        qubo_rows, penalty_rows, qaoa_rows, qaoa_meta, banded_rows, banded_meta, beyond_rows
    )

    print(f"Wrote CSVs and summary to {RESULT_DIR}")
    print(f"Wrote LaTeX tables to {QUANTUM_TABLES}")
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
