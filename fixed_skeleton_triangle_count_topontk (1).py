#!/usr/bin/env python3
"""

Copyright (c) 2026 Sanjukta Krishnagopal
 
Fixed-skeleton simplex-count prediction with graph/TopoNTK-style kernels.

run fixed_skeleton_triangle_count_topontk.py --n 30 --runs 5 --samples_per_density 200 --q_values 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

Experiment
----------
We fix a 1-skeleton on n vertices with cycle edges (i,i+1) and second-neighbor
chords (i,i+2), using indices modulo n. The n triples (i,i+1,i+2) are the
candidate 2-simplices. For each density q, each candidate 2-simplex is filled
independently with probability q. The target is the number of filled triangles.

Baselines
---------
  graph_ntk : line-graph/shared-vertex edge propagation using the fixed 1-skeleton
  lower     : lower TopoNTK channel using L_down = B1^T B1
  upper     : upper TopoNTK channel using L_up = B2 B2^T
  full      : full TopoNTK using L_down + L_up

The 1-skeleton is fixed, so graph_ntk and lower cannot see which candidate
2-simplices are filled except through the training-set mean. Upper and full
kernels receive the sample-specific B2 and therefore encode filled 2-simplices.

Defaults match the main-text description:
  n = 30
  q in {0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0}
  60 samples per density
  5 independent repetitions
  70/30 train/test split
  lambda = 1e-4

Outputs
-------
  <out_dir>/raw_results.csv
  <out_dir>/summary.csv
  <out_dir>/sample_diagnostics.csv
  <out_dir>/triangle_count_rmse_vs_q.pdf/png
  <out_dir>/triangle_count_mae_vs_q.pdf/png
  <out_dir>/triangle_count_r2_vs_q.pdf/png
  <out_dir>/triangle_count_rel_rmse_vs_q.pdf/png
  <out_dir>/triangle_count_dataset_diagnostics_vs_q.pdf/png

Example
-------
  python fixed_skeleton_triangle_count_topontk.py
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Fixed simplicial complex utilities
# -----------------------------------------------------------------------------

def canonical_edge(i: int, j: int, n: int) -> Tuple[int, int]:
    i %= n
    j %= n
    if i == j:
        raise ValueError("Self edge requested.")
    return (i, j) if i < j else (j, i)


def fixed_cycle_chord_edges(n: int) -> List[Tuple[int, int]]:
    """Cycle edges (i,i+1) and second-neighbor chords (i,i+2), modulo n."""
    edges = set()
    for i in range(n):
        edges.add(canonical_edge(i, i + 1, n))
        edges.add(canonical_edge(i, i + 2, n))
    return sorted(edges)


def fixed_candidate_triangles(n: int) -> List[Tuple[int, int, int]]:
    """The n cyclic triples (i,i+1,i+2), stored as sorted vertex triples."""
    tris = set()
    for i in range(n):
        tri = tuple(sorted(((i % n), ((i + 1) % n), ((i + 2) % n))))
        tris.add(tri)
    return sorted(tris)


def sample_filled_triangles(
    candidates: Sequence[Tuple[int, int, int]],
    q: float,
    rng: np.random.Generator,
) -> List[Tuple[int, int, int]]:
    return [tri for tri in candidates if rng.random() < q]


def boundary_B1(n: int, edges: Sequence[Tuple[int, int]]) -> np.ndarray:
    """B1: C1 -> C0, shape n x m. Edge orientation low -> high."""
    B1 = np.zeros((n, len(edges)), dtype=float)
    for e, (i, j) in enumerate(edges):
        B1[i, e] = -1.0
        B1[j, e] = 1.0
    return B1


def boundary_B2(
    edges: Sequence[Tuple[int, int]],
    triangles: Sequence[Tuple[int, int, int]],
) -> np.ndarray:
    """B2: C2 -> C1, shape m x t.

    For sorted triangle (i,j,k), orientation is [j,k] - [i,k] + [i,j].
    Edges are oriented low -> high.
    """
    edge_index = {e: idx for idx, e in enumerate(edges)}
    B2 = np.zeros((len(edges), len(triangles)), dtype=float)
    for c, tri in enumerate(triangles):
        i, j, k = sorted(tri)
        for edge, sign in [((j, k), 1.0), ((i, k), -1.0), ((i, j), 1.0)]:
            if edge not in edge_index:
                raise ValueError(f"Triangle {tri} uses missing edge {edge}.")
            B2[edge_index[edge], c] = sign
    return B2


# -----------------------------------------------------------------------------
# Edge architecture kernels
# -----------------------------------------------------------------------------

def sym(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def normalize_symmetric(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if M.size == 0:
        return M.copy()
    w = np.linalg.eigvalsh(sym(M))
    scale = float(np.max(np.abs(w))) if w.size else 0.0
    if scale < eps:
        return M.copy()
    return M / scale


def relu_covariance_map(S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Infinite-width ReLU covariance and derivative-covariance maps."""
    diag = np.clip(np.diag(S), 1e-12, None)
    denom = np.sqrt(np.outer(diag, diag))
    C = np.clip(S / denom, -1.0, 1.0)
    theta = np.arccos(C)
    Phi = denom * (np.sin(theta) + (np.pi - theta) * C) / (2.0 * np.pi)
    Phi_dot = (np.pi - theta) / (2.0 * np.pi)
    return Phi, Phi_dot


def line_graph_adjacency_from_Ldown(Ldown: np.ndarray) -> np.ndarray:
    Aline = (np.abs(Ldown) > 1e-12).astype(float)
    np.fill_diagonal(Aline, 0.0)
    return sym(Aline)


def edge_arch_kernel(
    B1: np.ndarray,
    B2: np.ndarray,
    kernel: str,
    depth: int = 2,
    alpha: float = 1.0,
    beta: float = 1.0,
    add_identity: float = 0.5,
    use_linear_kernel: bool = False,
) -> np.ndarray:
    """Edge-level architecture kernel on the fixed edge set."""
    m = B1.shape[1]
    Ldown = B1.T @ B1
    Lup = B2 @ B2.T if B2.shape[1] else np.zeros((m, m))

    kernel = kernel.lower()
    if kernel in {"graph", "graph_ntk", "gntk"}:
        P0 = line_graph_adjacency_from_Ldown(Ldown)
        P = add_identity * np.eye(m) + alpha * normalize_symmetric(P0)
    elif kernel in {"lower", "lower_topontk", "tntk_lower"}:
        P = add_identity * np.eye(m) + alpha * normalize_symmetric(Ldown)
    elif kernel in {"upper", "upper_topontk", "tntk_upper"}:
        P = add_identity * np.eye(m) + beta * normalize_symmetric(Lup)
    elif kernel in {"full", "full_topontk", "tntk_full"}:
        P = add_identity * np.eye(m) + alpha * normalize_symmetric(Ldown) + beta * normalize_symmetric(Lup)
    else:
        raise ValueError(f"Unknown kernel '{kernel}'.")

    P = normalize_symmetric(sym(P))

    if use_linear_kernel:
        K = np.zeros((m, m), dtype=float)
        Pl = np.eye(m)
        for _ in range(depth + 1):
            K += Pl @ Pl.T
            Pl = P @ Pl
    else:
        Sigma = np.eye(m)
        Theta = Sigma.copy()
        for _ in range(depth):
            Phi, Phi_dot = relu_covariance_map(Sigma)
            Sigma_next = P @ Phi @ P.T
            Theta = P @ (Theta * Phi_dot) @ P.T + Sigma_next
            Sigma = sym(Sigma_next)
            Theta = sym(Theta)
        K = Theta

    K = sym(K)
    tr = float(np.trace(K))
    if tr > 1e-12:
        K *= m / tr
    return K + 1e-10 * np.eye(m)


# -----------------------------------------------------------------------------
# Dataset and kernel ridge regression
# -----------------------------------------------------------------------------

@dataclass
class Sample:
    x: np.ndarray
    y: float
    filled_triangles: List[Tuple[int, int, int]]


def draw_samples_for_q(
    n: int,
    q: float,
    n_samples: int,
    rng: np.random.Generator,
) -> List[Sample]:
    edges = fixed_cycle_chord_edges(n)
    candidates = fixed_candidate_triangles(n)
    x = np.ones(len(edges), dtype=float)
    x /= max(np.linalg.norm(x), 1e-12)

    samples: List[Sample] = []
    for _ in range(n_samples):
        filled = sample_filled_triangles(candidates, q, rng)
        samples.append(Sample(x=x.copy(), y=float(len(filled)), filled_triangles=filled))
    return samples


def pair_kernel(X: np.ndarray, K_list: Sequence[np.ndarray]) -> np.ndarray:
    """Symmetrized sample kernel G_ij = x_i^T ((K_i+K_j)/2) x_j."""
    N = X.shape[0]
    KX = [K_list[i] @ X[i] for i in range(N)]
    G = np.empty((N, N), dtype=float)
    for i in range(N):
        G[i, i] = float(X[i] @ KX[i])
        for j in range(i + 1, N):
            val = 0.5 * (float(X[i] @ KX[j]) + float(X[j] @ KX[i]))
            G[i, j] = val
            G[j, i] = val
    return sym(G)


def cross_kernel(
    Xte: np.ndarray,
    Kte: Sequence[np.ndarray],
    Xtr: np.ndarray,
    Ktr: Sequence[np.ndarray],
) -> np.ndarray:
    Nte, Ntr = Xte.shape[0], Xtr.shape[0]
    Kte_Xte = [Kte[i] @ Xte[i] for i in range(Nte)]
    Ktr_Xtr = [Ktr[j] @ Xtr[j] for j in range(Ntr)]
    G = np.empty((Nte, Ntr), dtype=float)
    for i in range(Nte):
        for j in range(Ntr):
            G[i, j] = 0.5 * (
                float(Xte[i] @ Ktr_Xtr[j]) + float(Xtr[j] @ Kte_Xte[i])
            )
    return G


def krr_predict(
    Xtr: np.ndarray,
    ytr_raw: np.ndarray,
    Ktr: Sequence[np.ndarray],
    Xte: np.ndarray,
    Kte: Sequence[np.ndarray],
    lam: float,
) -> np.ndarray:
    y_mean = float(np.mean(ytr_raw))
    ytr = ytr_raw - y_mean
    Gtr = pair_kernel(Xtr, Ktr)
    Gte = cross_kernel(Xte, Kte, Xtr, Ktr)
    diag_scale = max(float(np.mean(np.diag(Gtr))), 1e-12)
    coef = np.linalg.solve(Gtr + lam * diag_scale * np.eye(Gtr.shape[0]), ytr)
    return Gte @ coef + y_mean


def rmse(yhat: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((yhat - y) ** 2)))


def mae(yhat: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(yhat - y)))


def rel_rmse(yhat: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(yhat - y) / (np.linalg.norm(y) + 1e-12))


def r2_score(yhat: np.ndarray, y: np.ndarray) -> float:
    ss_res = float(np.sum((yhat - y) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


@dataclass
class Config:
    runs: int
    n: int
    q_values: List[float]
    samples_per_density: int
    train_frac: float
    lam: float
    depth: int
    alpha: float
    beta: float
    add_identity: float
    kernels: List[str]
    seed: int
    out_dir: str
    use_linear_kernel: bool


def build_K_list_for_kernel(
    cfg: Config,
    samples: Sequence[Sample],
    kernel: str,
    B1: np.ndarray,
) -> List[np.ndarray]:
    edges = fixed_cycle_chord_edges(cfg.n)
    K_list: List[np.ndarray] = []
    for s in samples:
        B2 = boundary_B2(edges, s.filled_triangles)
        K_list.append(edge_arch_kernel(
            B1, B2, kernel=kernel, depth=cfg.depth, alpha=cfg.alpha,
            beta=cfg.beta, add_identity=cfg.add_identity,
            use_linear_kernel=cfg.use_linear_kernel,
        ))
    return K_list


def run_one_q_rep(cfg: Config, q: float, run: int) -> Tuple[List[Dict[str, float]], pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed + 1009 * run + int(round(1000 * q)))
    samples = draw_samples_for_q(cfg.n, q, cfg.samples_per_density, rng)
    rng.shuffle(samples)

    n_train = int(round(cfg.train_frac * cfg.samples_per_density))
    n_train = min(max(n_train, 1), cfg.samples_per_density - 1)
    train = samples[:n_train]
    test = samples[n_train:]

    edges = fixed_cycle_chord_edges(cfg.n)
    candidates = fixed_candidate_triangles(cfg.n)
    B1 = boundary_B1(cfg.n, edges)

    Xtr = np.vstack([s.x for s in train])
    Xte = np.vstack([s.x for s in test])
    ytr = np.array([s.y for s in train], dtype=float)
    yte = np.array([s.y for s in test], dtype=float)

    rows: List[Dict[str, float]] = []
    for kernel in cfg.kernels:
        Ktr = build_K_list_for_kernel(cfg, train, kernel, B1)
        Kte = build_K_list_for_kernel(cfg, test, kernel, B1)
        yhat = krr_predict(Xtr, ytr, Ktr, Xte, Kte, cfg.lam)
        rows.append(dict(
            run=run,
            q=q,
            kernel=kernel,
            rmse=rmse(yhat, yte),
            mae=mae(yhat, yte),
            rel_rmse=rel_rmse(yhat, yte),
            r2=r2_score(yhat, yte),
            n_train=len(train),
            n_test=len(test),
            n_edges=len(edges),
            n_candidate_triangles=len(candidates),
            y_mean_train=float(np.mean(ytr)),
            y_std_train=float(np.std(ytr)),
            y_mean_test=float(np.mean(yte)),
            y_std_test=float(np.std(yte)),
            mean_filled_triangles=float(np.mean([s.y for s in samples])),
            std_filled_triangles=float(np.std([s.y for s in samples])),
        ))

    diag = pd.DataFrame({
        "run": run,
        "q": q,
        "sample": np.arange(len(samples)),
        "split": ["train"] * len(train) + ["test"] * len(test),
        "n_edges": len(edges),
        "n_candidate_triangles": len(candidates),
        "n_filled_triangles": [s.y for s in samples],
    })
    return rows, diag


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.groupby(["q", "kernel"]).agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rel_rmse_mean=("rel_rmse", "mean"),
        rel_rmse_std=("rel_rmse", "std"),
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        mean_filled_triangles=("mean_filled_triangles", "mean"),
        n_train=("n_train", "mean"),
        n_test=("n_test", "mean"),
        reps=("rmse", "count"),
    ).reset_index()
    for metric in ["rmse", "mae", "rel_rmse", "r2"]:
        out[f"{metric}_se"] = out[f"{metric}_std"].fillna(0.0) / np.sqrt(np.maximum(out["reps"], 1))
    return out


def run_experiment(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    os.makedirs(cfg.out_dir, exist_ok=True)
    rows: List[Dict[str, float]] = []
    diags: List[pd.DataFrame] = []
    total = cfg.runs * len(cfg.q_values)
    done = 0
    for q in cfg.q_values:
        for run in range(cfg.runs):
            r, d = run_one_q_rep(cfg, q, run)
            rows.extend(r)
            diags.append(d)
            done += 1
            print(f"finished {done}/{total}: q={q:.2f}, rep={run + 1}/{cfg.runs}")
    raw = pd.DataFrame(rows)
    diagnostics = pd.concat(diags, ignore_index=True)
    summary = summarize(raw)
    raw.to_csv(os.path.join(cfg.out_dir, "raw_results.csv"), index=False)
    summary.to_csv(os.path.join(cfg.out_dir, "summary.csv"), index=False)
    diagnostics.to_csv(os.path.join(cfg.out_dir, "sample_diagnostics.csv"), index=False)
    return raw, summary, diagnostics


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def savefig(path_base: str) -> None:
    plt.tight_layout()
    plt.savefig(path_base + ".pdf", bbox_inches="tight")
    plt.savefig(path_base + ".png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_metric_vs_q(summary: pd.DataFrame, metric: str, ylabel: str, title: str, out_base: str) -> None:
    labels = {
        "graph_ntk": "Graph = Lower",
        "lower": "Lower TopoNTK",
        "upper": "Upper TopoNTK",
        "full": "Full TopoNTK",
    }
    kernels = [k for k in ["graph_ntk", "lower", "upper", "full"] if k in set(summary["kernel"])]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for kernel in kernels:
        # In the fixed-1-skeleton experiment, graph_ntk and lower depend
        # only on the same fixed lower-order structure. Their sample Gram
        # matrices coincide, so the plot shows one shared baseline.
        if kernel == "lower" and "graph_ntk" in kernels:
            continue
        sub = summary[summary.kernel == kernel].sort_values("q")
        x = sub["q"].to_numpy()
        y = sub[f"{metric}_mean"].to_numpy()
        se = sub[f"{metric}_se"].to_numpy()
        ax.plot(x, y, marker="o", linewidth=2, label=labels.get(kernel, kernel))
        ax.fill_between(x, y - se, y + se, alpha=0.18)
    ax.set_xlabel(r"Triangle density $q$", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(out_base)


def plot_dataset_diagnostics(diagnostics: pd.DataFrame, out_base: str) -> None:
    sub = diagnostics.groupby(["q", "run"]).agg(
        mean_filled=("n_filled_triangles", "mean"),
        std_filled=("n_filled_triangles", "std"),
        n_edges=("n_edges", "mean"),
        n_candidate_triangles=("n_candidate_triangles", "mean"),
    ).reset_index()
    agg = sub.groupby("q").agg(
        mean_filled=("mean_filled", "mean"),
        mean_filled_se=("mean_filled", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0),
        n_edges=("n_edges", "mean"),
        n_candidate_triangles=("n_candidate_triangles", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(agg.q, agg.n_edges, marker="o", label="Edges")
    ax.plot(agg.q, agg.n_candidate_triangles, marker="s", label="Candidate 2-simplices")
    ax.plot(agg.q, agg.mean_filled, marker="^", label="Filled 2-simplices")
    ax.fill_between(agg.q, agg.mean_filled - agg.mean_filled_se, agg.mean_filled + agg.mean_filled_se, alpha=0.18)
    ax.set_xlabel(r"Triangle density $q$", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Fixed-skeleton dataset diagnostics", fontsize=14)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(out_base)


def make_plots(summary: pd.DataFrame, diagnostics: pd.DataFrame, out_dir: str) -> None:
    plot_metric_vs_q(summary, "rmse", "Test RMSE", "Triangle-count prediction", os.path.join(out_dir, "triangle_count_rmse_vs_q"))
    plot_metric_vs_q(summary, "mae", "Test MAE", "Triangle-count prediction", os.path.join(out_dir, "triangle_count_mae_vs_q"))
    plot_metric_vs_q(summary, "rel_rmse", "Relative test RMSE", "Triangle-count prediction", os.path.join(out_dir, "triangle_count_rel_rmse_vs_q"))
    plot_metric_vs_q(summary, "r2", r"Test $R^2$", "Triangle-count prediction", os.path.join(out_dir, "triangle_count_r2_vs_q"))
    plot_dataset_diagnostics(diagnostics, os.path.join(out_dir, "triangle_count_dataset_diagnostics_vs_q"))


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--n", type=int, default=30)
    p.add_argument(
        "--q_values", type=float, nargs="+",
        default=[0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0],
        help="Triangle fill densities to sweep.",
    )
    p.add_argument("--samples_per_density", type=int, default=60)
    p.add_argument("--train_frac", type=float, default=0.70)
    p.add_argument("--lam", type=float, default=1e-4)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--add_identity", type=float, default=0.5)
    p.add_argument(
        "--kernels", nargs="+", default=["graph_ntk", "lower", "upper", "full"],
        choices=["graph_ntk", "lower", "upper", "full"],
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="fixed_skeleton_triangle_count_out")
    p.add_argument("--use_linear_kernel", action="store_true")
    a = p.parse_args()
    return Config(**vars(a))


def main() -> None:
    cfg = parse_args()
    edges = fixed_cycle_chord_edges(cfg.n)
    candidates = fixed_candidate_triangles(cfg.n)
    print("Fixed skeleton:")
    print(f"  n vertices                 = {cfg.n}")
    print(f"  edges                      = {len(edges)}")
    print(f"  candidate 2-simplices      = {len(candidates)}")
    print(f"  samples per density        = {cfg.samples_per_density}")
    print(f"  train/test split           = {cfg.train_frac:.2f}/{1.0 - cfg.train_frac:.2f}")
    print(f"  ridge lambda               = {cfg.lam:g}")

    raw, summary, diagnostics = run_experiment(cfg)
    make_plots(summary, diagnostics, cfg.out_dir)

    print("\nExpected filled triangles per sample:")
    for q in cfg.q_values:
        print(f"  q={q:.2f}: {q * len(candidates):.3f}")

    print("\nSummary, mean ± SE over repetitions:")
    show = summary.copy()
    for metric in ["rmse", "mae", "rel_rmse", "r2"]:
        show[metric] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(show[f"{metric}_mean"], show[f"{metric}_se"])]
    print(show[["q", "kernel", "rmse", "mae", "rel_rmse", "r2", "mean_filled_triangles"]].to_string(index=False))
    print(f"\nSaved outputs in: {cfg.out_dir}")


if __name__ == "__main__":
    main()
