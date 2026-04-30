#!/usr/bin/env python3
"""
DBLP simplicial-closure prediction with TopoNTK-style kernels.

Copyright (c) 2026 Sanjukta Krishnagopal

run dblp_simplicial_closure.py --runs 5 --max-simplices
 50000 --n-pos 120 --n-neg 120 --ego-size 10

Downloads ScHoLP DBLP files directly from GitHub .txt.gz files:
  coauth-DBLP-nverts.txt.gz
  coauth-DBLP-simplices.txt.gz
  coauth-DBLP-times.txt.gz

Task: future 3-author collaboration closure. Given historical coauthorship,
predict whether a candidate author triple appears together in a future paper.
By default candidates are closed historical triads: all three pairwise edges
already exist historically, so the task tests higher-order closure.

Install:
  pip install numpy scipy pandas matplotlib scikit-learn requests tqdm

Quick:
  python dblp_simplicial_closure.py --quick

More serious:
  python dblp_simplicial_closure.py --max-simplices 300000 --n-pos 200 --n-neg 200 --runs 3 --ego-size 30
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

RAW_BASE = "https://raw.githubusercontent.com/arbenson/ScHoLP-Data/master/coauth-DBLP"
FILES = [
    "coauth-DBLP-nverts.txt",
    "coauth-DBLP-simplices.txt",
    "coauth-DBLP-times.txt",
]


# -----------------------------------------------------------------------------
# Download and parse ScHoLP DBLP
# -----------------------------------------------------------------------------

def download_file(url: str, path: Path, chunk_size: int = 2**20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        first = True
        with open(path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=path.name) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                if first:
                    first = False
                    head = chunk[:256].lstrip().lower()
                    if head.startswith(b"<!") or b"<html" in head:
                        raise RuntimeError(f"Downloaded HTML instead of data from {url}")
                f.write(chunk)
                pbar.update(len(chunk))


def gunzip_to_text(gz_path: Path, txt_path: Path) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(gz_path, "rb") as fin, open(txt_path, "wb") as fout:
        while True:
            buf = fin.read(2**20)
            if not buf:
                break
            fout.write(buf)


def ensure_dblp_files(data_dir: Path, force: bool = False) -> Path:
    dataset_dir = data_dir / "coauth-DBLP"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for fname in FILES:
        txt_path = dataset_dir / fname
        gz_path = dataset_dir / f"{fname}.gz"
        if force:
            if txt_path.exists():
                txt_path.unlink()
            if gz_path.exists():
                gz_path.unlink()

        if txt_path.exists() and txt_path.stat().st_size > 0:
            continue

        if not (gz_path.exists() and gz_path.stat().st_size > 0):
            url = f"{RAW_BASE}/{fname}.gz"
            print(f"Downloading {url}")
            download_file(url, gz_path)

        print(f"Decompressing {gz_path.name}")
        gunzip_to_text(gz_path, txt_path)
        if not (txt_path.exists() and txt_path.stat().st_size > 0):
            raise RuntimeError(f"Failed to create nonempty file: {txt_path}")

    return dataset_dir


def read_ints_linewise(path: Path, max_items=None) -> List[int]:
    vals = []
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            vals.append(int(s))
            if max_items is not None and len(vals) >= max_items:
                break
    return vals


@dataclass
class SimplexRecord:
    nodes: Tuple[int, ...]
    time: int


def read_simplices(dataset_dir: Path, max_simplices: int, max_simplex_size: int, min_simplex_size: int = 2) -> List[SimplexRecord]:
    nverts = read_ints_linewise(dataset_dir / "coauth-DBLP-nverts.txt", max_simplices)
    times = read_ints_linewise(dataset_dir / "coauth-DBLP-times.txt", len(nverts))
    if len(times) != len(nverts):
        raise RuntimeError(f"times length {len(times)} != nverts length {len(nverts)}")

    out = []
    with open(dataset_dir / "coauth-DBLP-simplices.txt", "rt", encoding="utf-8") as f:
        for i, k in enumerate(tqdm(nverts, desc="Parsing simplices")):
            nodes = []
            for _ in range(k):
                line = f.readline()
                if line == "":
                    raise RuntimeError("Unexpected EOF in simplices file")
                nodes.append(int(line.strip()))
            nodes = tuple(sorted(set(nodes)))
            if min_simplex_size <= len(nodes) <= max_simplex_size:
                out.append(SimplexRecord(nodes=nodes, time=int(times[i])))
    out.sort(key=lambda r: r.time)
    return out


# -----------------------------------------------------------------------------
# Historical structures and candidate triples
# -----------------------------------------------------------------------------

def all_pairs(nodes: Sequence[int]) -> Iterable[Tuple[int, int]]:
    for a, b in itertools.combinations(nodes, 2):
        if a != b:
            yield tuple(sorted((a, b)))


@dataclass
class HistoryFuture:
    history: List[SimplexRecord]
    future: List[SimplexRecord]
    edge_weight: Counter
    neighbors: Dict[int, Counter]
    hist_triangles: Set[Tuple[int, int, int]]
    future_triangles: Set[Tuple[int, int, int]]
    tri_by_node: Dict[int, List[Tuple[int, int, int]]]
    degree: Counter
    cutoff_time: int


def build_history_future(records: List[SimplexRecord], train_fraction_time: float, max_group_size_for_triples: int) -> HistoryFuture:
    split_idx = int(round(train_fraction_time * len(records)))
    split_idx = min(max(split_idx, 1), len(records) - 1)
    cutoff_time = records[split_idx].time
    history = [r for r in records if r.time <= cutoff_time]
    future = [r for r in records if r.time > cutoff_time]

    edge_weight = Counter()
    neighbors = defaultdict(Counter)
    degree = Counter()
    hist_triangles = set()
    future_triangles = set()

    for r in history:
        nodes = r.nodes
        for u, v in all_pairs(nodes):
            edge_weight[(u, v)] += 1
            neighbors[u][v] += 1
            neighbors[v][u] += 1
        for u in nodes:
            degree[u] += 1
        if len(nodes) <= max_group_size_for_triples:
            for tri in itertools.combinations(nodes, 3):
                hist_triangles.add(tuple(sorted(tri)))

    for r in future:
        nodes = r.nodes
        if len(nodes) <= max_group_size_for_triples:
            for tri in itertools.combinations(nodes, 3):
                future_triangles.add(tuple(sorted(tri)))

    tri_by_node = defaultdict(list)
    for tri in hist_triangles:
        for u in tri:
            tri_by_node[u].append(tri)

    return HistoryFuture(history, future, edge_weight, neighbors, hist_triangles, future_triangles, tri_by_node, degree, cutoff_time)


def is_closed_triad(tri: Tuple[int, int, int], edge_weight: Counter) -> bool:
    a, b, c = tri
    return (tuple(sorted((a, b))) in edge_weight and tuple(sorted((a, c))) in edge_weight and tuple(sorted((b, c))) in edge_weight)


def sample_positive_triples(hf: HistoryFuture, n_pos: int, rng: random.Random, require_closed: bool = True) -> List[Tuple[int, int, int]]:
    candidates = []
    for tri in hf.future_triangles:
        if tri in hf.hist_triangles:
            continue
        if require_closed and not is_closed_triad(tri, hf.edge_weight):
            continue
        candidates.append(tri)
    rng.shuffle(candidates)
    if len(candidates) < n_pos:
        print(f"Warning: requested {n_pos} positives but found {len(candidates)}")
    return candidates[:n_pos]


def sample_negative_triples(hf: HistoryFuture, n_neg: int, rng: random.Random, require_closed: bool = True, max_attempts: int = 300000) -> List[Tuple[int, int, int]]:
    edges = list(hf.edge_weight.keys())
    forbidden = hf.future_triangles | hf.hist_triangles
    negatives = set()
    attempts = 0
    while len(negatives) < n_neg and attempts < max_attempts:
        attempts += 1
        a, b = rng.choice(edges)
        if require_closed:
            common = set(hf.neighbors[a].keys()).intersection(hf.neighbors[b].keys())
            if not common:
                continue
            c = rng.choice(tuple(common))
        else:
            pool = set(hf.neighbors[a].keys()).union(hf.neighbors[b].keys())
            if not pool:
                continue
            c = rng.choice(tuple(pool))
        if c == a or c == b:
            continue
        tri = tuple(sorted((a, b, c)))
        if tri in forbidden:
            continue
        if require_closed and not is_closed_triad(tri, hf.edge_weight):
            continue
        negatives.add(tri)
    if len(negatives) < n_neg:
        print(f"Warning: requested {n_neg} negatives but found {len(negatives)}")
    return list(negatives)


# -----------------------------------------------------------------------------
# Local simplicial complexes
# -----------------------------------------------------------------------------

@dataclass
class LocalComplex:
    nodes_global: List[int]
    candidate_global: Tuple[int, int, int]
    edges_local: List[Tuple[int, int]]
    triangles_local: List[Tuple[int, int, int]]
    B1: np.ndarray
    B2: np.ndarray
    A_node: np.ndarray
    A_lower: np.ndarray
    A_upper: np.ndarray
    node_features: np.ndarray
    edge_features: np.ndarray
    meta: Dict


def row_normalized_with_self(A: np.ndarray) -> np.ndarray:
    M = A + np.eye(A.shape[0])
    d = M.sum(axis=1, keepdims=True)
    d[d == 0] = 1.0
    return M / d


def make_B1(n_nodes: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    B1 = np.zeros((n_nodes, len(edges)), dtype=float)
    for j, (u, v) in enumerate(edges):
        B1[u, j] = -1.0
        B1[v, j] = 1.0
    return B1


def make_B2(edges: List[Tuple[int, int]], triangles: List[Tuple[int, int, int]]) -> np.ndarray:
    edge_index = {e: i for i, e in enumerate(edges)}
    B2 = np.zeros((len(edges), len(triangles)), dtype=float)
    for col, (a, b, c) in enumerate(triangles):
        for e, sign in [((b, c), 1.0), ((a, c), -1.0), ((a, b), 1.0)]:
            e = tuple(sorted(e))
            if e in edge_index:
                B2[edge_index[e], col] = sign
    return B2


def local_triangles_from_index(ego_set: Set[int], hf: HistoryFuture, max_local_triangles: int, rng: random.Random) -> List[Tuple[int, int, int]]:
    cand = set()
    for u in ego_set:
        for tri in hf.tri_by_node.get(u, []):
            if tri[0] in ego_set and tri[1] in ego_set and tri[2] in ego_set:
                cand.add(tri)
    out = list(cand)
    if len(out) > max_local_triangles:
        rng.shuffle(out)
        out = out[:max_local_triangles]
    return out


def build_local_complex(tri: Tuple[int, int, int], hf: HistoryFuture, ego_size: int, max_local_triangles: int, rng: random.Random) -> LocalComplex:
    cand_set = set(tri)
    score = Counter()
    for u in tri:
        for v, w in hf.neighbors.get(u, {}).items():
            if v not in cand_set:
                score[v] += w
    ranked = [v for v, _ in score.most_common(max(ego_size - 3, 0))]
    ego_nodes = list(tri) + ranked
    seen = set()
    ego_nodes = [u for u in ego_nodes if not (u in seen or seen.add(u))]
    ego_nodes = ego_nodes[:ego_size]
    ego_set = set(ego_nodes)
    local_index = {u: i for i, u in enumerate(ego_nodes)}
    n = len(ego_nodes)

    edges_global = [e for e in hf.edge_weight.keys() if e[0] in ego_set and e[1] in ego_set]
    edges_local = sorted(tuple(sorted((local_index[u], local_index[v]))) for u, v in edges_global)
    edge_set_local = set(edges_local)

    tris_global = local_triangles_from_index(ego_set, hf, max_local_triangles, rng)
    triangles_local = []
    for a, b, c in tris_global:
        la, lb, lc = sorted((local_index[a], local_index[b], local_index[c]))
        if (la, lb) in edge_set_local and (la, lc) in edge_set_local and (lb, lc) in edge_set_local:
            triangles_local.append((la, lb, lc))
    triangles_local = sorted(set(triangles_local))

    B1 = make_B1(n, edges_local)
    B2 = make_B2(edges_local, triangles_local)

    A_node = np.zeros((n, n), dtype=float)
    for u, v in edges_local:
        A_node[u, v] = 1.0
        A_node[v, u] = 1.0
    A_node = row_normalized_with_self(A_node)

    m = len(edges_local)
    A_lower = np.zeros((m, m), dtype=float)
    edge_sets = [set(e) for e in edges_local]
    for i in range(m):
        for j in range(i + 1, m):
            if edge_sets[i].intersection(edge_sets[j]):
                A_lower[i, j] = A_lower[j, i] = 1.0
    A_lower = row_normalized_with_self(A_lower) if m > 0 else np.zeros((0, 0))

    A_upper = np.zeros((m, m), dtype=float)
    edge_index = {e: i for i, e in enumerate(edges_local)}
    for a, b, c in triangles_local:
        tri_edges = [tuple(sorted((a, b))), tuple(sorted((a, c))), tuple(sorted((b, c)))]
        idx = [edge_index[e] for e in tri_edges if e in edge_index]
        for i in idx:
            for j in idx:
                if i != j:
                    A_upper[i, j] = 1.0
    A_upper = row_normalized_with_self(A_upper) if m > 0 else np.zeros((0, 0))

    cand_local = {local_index[u] for u in tri if u in local_index}
    node_features = np.zeros((n, 5), dtype=float)
    max_deg = max([hf.degree[u] for u in ego_nodes] + [1])
    for u_global, i in local_index.items():
        co_with_candidate = sum(hf.edge_weight.get(tuple(sorted((u_global, c))), 0) for c in tri if c != u_global)
        node_features[i, 0] = 1.0 if i in cand_local else 0.0
        node_features[i, 1] = math.log1p(hf.degree[u_global]) / math.log1p(max_deg)
        node_features[i, 2] = math.log1p(co_with_candidate) / math.log1p(max_deg)
        node_features[i, 3] = A_node[i].sum() / max(n, 1)
        node_features[i, 4] = 1.0

    cand_pairs_local = {
        tuple(sorted((local_index[a], local_index[b])))
        for a, b in itertools.combinations(tri, 2)
        if a in local_index and b in local_index
    }
    edge_features = np.zeros((m, 7), dtype=float)
    max_w = max([hf.edge_weight[e] for e in edges_global] + [1])
    upper_diag = np.diag(B2 @ B2.T) if m > 0 else np.zeros(0)
    lower_deg = A_lower.sum(axis=1) if m > 0 else np.zeros(0)
    for j, (u, v) in enumerate(edges_local):
        ug, vg = ego_nodes[u], ego_nodes[v]
        w = hf.edge_weight.get(tuple(sorted((ug, vg))), 0)
        edge_features[j, 0] = 1.0 if (u, v) in cand_pairs_local else 0.0
        edge_features[j, 1] = (1.0 if u in cand_local else 0.0) + (1.0 if v in cand_local else 0.0)
        edge_features[j, 2] = math.log1p(w) / math.log1p(max_w)
        edge_features[j, 3] = upper_diag[j]
        edge_features[j, 4] = lower_deg[j] / max(m, 1)
        edge_features[j, 5] = abs(node_features[u, 1] - node_features[v, 1])
        edge_features[j, 6] = 1.0

    beta1 = 0
    if m > 0:
        L1 = B1.T @ B1 + B2 @ B2.T
        beta1 = int(np.sum(np.linalg.eigvalsh((L1 + L1.T) / 2) < 1e-8))

    return LocalComplex(
        nodes_global=ego_nodes,
        candidate_global=tri,
        edges_local=edges_local,
        triangles_local=triangles_local,
        B1=B1,
        B2=B2,
        A_node=A_node,
        A_lower=A_lower,
        A_upper=A_upper,
        node_features=node_features,
        edge_features=edge_features,
        meta=dict(
            candidate=tri,
            n_nodes=n,
            n_edges=m,
            n_triangles=len(triangles_local),
            beta1=beta1,
            candidate_pair_weight_sum=sum(hf.edge_weight.get(tuple(sorted(e)), 0) for e in itertools.combinations(tri, 2)),
        ),
    )


# -----------------------------------------------------------------------------
# NTK-style kernels
# -----------------------------------------------------------------------------

def activation_covariance(S_xx: np.ndarray, S_xy: np.ndarray, S_yy: np.ndarray):
    q11 = np.maximum(np.diag(S_xx), 1e-12)[:, None]
    q22 = np.maximum(np.diag(S_yy), 1e-12)[None, :]
    denom = np.sqrt(q11 * q22)
    c = np.clip(S_xy / denom, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = np.arccos(c)
    sigma = 2.0 * (denom / (2.0 * np.pi)) * (np.sin(theta) + (np.pi - theta) * np.cos(theta))
    sigma_dot = 2.0 * ((np.pi - theta) / (2.0 * np.pi))
    return sigma, sigma_dot


def ntk_pair_features(X: np.ndarray, Y: np.ndarray, Ax: np.ndarray, Ay: np.ndarray, depth: int = 2) -> float:
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return 0.0
    S_xx = X @ X.T
    S_xy = X @ Y.T
    S_yy = Y @ Y.T
    T_xy = S_xy.copy()
    for _ in range(depth):
        P_xx, _ = activation_covariance(S_xx, S_xx, S_xx)
        P_xy, Pdot_xy = activation_covariance(S_xx, S_xy, S_yy)
        P_yy, _ = activation_covariance(S_yy, S_yy, S_yy)
        S_xx = Ax @ P_xx @ Ax.T
        S_xy = Ax @ P_xy @ Ay.T
        S_yy = Ay @ P_yy @ Ay.T
        T_xy = Ax @ (T_xy * Pdot_xy) @ Ay.T + S_xy
    return float(T_xy.sum())


def pair_kernel(Cx: LocalComplex, Cy: LocalComplex, kind: str, depth: int = 2) -> float:
    if kind == "graph_ntk":
        return ntk_pair_features(Cx.node_features, Cy.node_features, Cx.A_node, Cy.A_node, depth)
    if kind == "lower":
        return ntk_pair_features(Cx.edge_features, Cy.edge_features, Cx.A_lower, Cy.A_lower, depth)
    if kind == "upper":
        return ntk_pair_features(Cx.edge_features, Cy.edge_features, Cx.A_upper, Cy.A_upper, depth)
    if kind == "full":
        Ax = 0.5 * (Cx.A_lower + Cx.A_upper) if Cx.A_lower.size else Cx.A_lower
        Ay = 0.5 * (Cy.A_lower + Cy.A_upper) if Cy.A_lower.size else Cy.A_lower
        return ntk_pair_features(Cx.edge_features, Cy.edge_features, Ax, Ay, depth)
    raise ValueError(f"Unknown kernel: {kind}")


def gram_matrix(complexes: List[LocalComplex], kind: str, depth: int = 2) -> np.ndarray:
    n = len(complexes)
    K = np.zeros((n, n), dtype=float)
    for i in tqdm(range(n), desc=f"Gram {kind}"):
        for j in range(i, n):
            val = pair_kernel(complexes[i], complexes[j], kind, depth)
            K[i, j] = K[j, i] = val
    K = 0.5 * (K + K.T)
    d = np.sqrt(np.maximum(np.diag(K), 1e-12))
    K = K / np.outer(d, d)
    return K + 1e-8 * np.eye(n)


# -----------------------------------------------------------------------------
# Learning/evaluation
# -----------------------------------------------------------------------------

def kernel_ridge_scores(K: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, lam: float):
    Ktr = K[np.ix_(train_idx, train_idx)]
    Kte = K[np.ix_(test_idx, train_idx)]
    ytr = y[train_idx].astype(float)
    alpha = np.linalg.solve(Ktr + lam * np.eye(len(train_idx)), ytr)
    scores = Kte @ alpha
    true = y[test_idx]
    auroc = float(roc_auc_score(true, scores)) if len(np.unique(true)) == 2 else float("nan")
    ap = float(average_precision_score(true, scores)) if len(np.unique(true)) == 2 else float("nan")

    train_scores = Ktr @ alpha
    thresholds = np.quantile(train_scores, np.linspace(0.05, 0.95, 41))
    best_thr, best_bal = 0.0, -1.0
    for thr in thresholds:
        pred_tr = (train_scores >= thr).astype(int)
        bal = balanced_accuracy_score(ytr, pred_tr)
        if bal > best_bal:
            best_bal, best_thr = bal, thr
    pred = (scores >= best_thr).astype(int)
    return scores, pred, dict(
        auroc=auroc,
        average_precision=ap,
        accuracy=float(accuracy_score(true, pred)),
        balanced_accuracy=float(balanced_accuracy_score(true, pred)),
        f1=float(f1_score(true, pred, zero_division=0)),
    )


def build_dataset(args, run_seed: int):
    data_dir = ensure_dblp_files(Path(args.data_dir), force=args.force_download)
    records = read_simplices(data_dir, args.max_simplices, args.max_simplex_size, 2)
    if len(records) < 100:
        raise RuntimeError(f"Too few records loaded: {len(records)}")
    hf = build_history_future(records, args.train_fraction_time, args.max_group_size_for_triples)
    rng = random.Random(run_seed)
    positives = sample_positive_triples(hf, args.n_pos, rng, require_closed=not args.allow_open_positive_triads)
    negatives = sample_negative_triples(hf, args.n_neg, rng, require_closed=not args.allow_open_negative_triads)
    triples = positives + negatives
    y = np.array([1] * len(positives) + [0] * len(negatives), dtype=int)
    if len(np.unique(y)) < 2:
        raise RuntimeError("Need both positive and negative examples.")
    complexes = [build_local_complex(tri, hf, args.ego_size, args.max_local_triangles, rng) for tri in tqdm(triples, desc="Building local complexes")]
    print(f"Loaded records={len(records)}, cutoff_time={hf.cutoff_time}")
    print(f"History simplices={len(hf.history)}, future simplices={len(hf.future)}")
    print(f"History edges={len(hf.edge_weight)}, history filled triples={len(hf.hist_triangles)}, future triples={len(hf.future_triangles)}")
    print(f"Dataset positives={len(positives)}, negatives={len(negatives)}")
    return complexes, y


def summarize_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    return raw_df.groupby("kernel").agg(
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
        ap_mean=("average_precision", "mean"), ap_std=("average_precision", "std"),
        bacc_mean=("balanced_accuracy", "mean"), bacc_std=("balanced_accuracy", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        n=("auroc", "count"),
    ).reset_index()


def plot_summary(summary: pd.DataFrame, outdir: Path) -> None:
    kernels = ["graph_ntk", "lower", "upper", "full"]
    labels = {"graph_ntk": "Graph NTK", "lower": "Lower", "upper": "Upper", "full": "Full"}
    metrics = [("auroc", "AUROC"), ("ap", "Average precision"), ("bacc", "Balanced accuracy"), ("f1", "F1")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, (metric, ylabel) in zip(axes, metrics):
        means, stds = [], []
        for k in kernels:
            row = summary[summary.kernel == k]
            if row.empty:
                means.append(np.nan); stds.append(0.0)
            else:
                means.append(float(row[f"{metric}_mean"].iloc[0]))
                stds.append(float(row[f"{metric}_std"].iloc[0]) if not pd.isna(row[f"{metric}_std"].iloc[0]) else 0.0)
        ax.bar(np.arange(len(kernels)), means, yerr=stds, capsize=3, edgecolor="black")
        ax.set_xticks(np.arange(len(kernels)))
        ax.set_xticklabels([labels[k] for k in kernels], rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("DBLP future 3-author collaboration closure")
    fig.tight_layout()
    fig.savefig(outdir / "closure_metrics_barplot.pdf")
    fig.savefig(outdir / "closure_metrics_barplot.png", dpi=300)
    plt.close(fig)


def run_experiment(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_rows, prediction_rows = [], []
    for run in range(args.runs):
        run_seed = args.seed + 1000 * run
        print(f"\n=== Run {run+1}/{args.runs}; seed={run_seed} ===")
        complexes, y = build_dataset(args, run_seed)
        meta_df = pd.DataFrame([c.meta for c in complexes])
        meta_df["label"] = y
        meta_df.to_csv(outdir / f"metadata_run{run}.csv", index=False)
        train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=args.test_size, stratify=y, random_state=run_seed)
        for kernel in args.kernels:
            K = gram_matrix(complexes, kernel, depth=args.depth)
            if args.save_grams and run == 0:
                np.save(outdir / f"gram_{kernel}_run0.npy", K)
            scores, pred, metrics = kernel_ridge_scores(K, y, train_idx, test_idx, args.lam)
            all_rows.append(dict(run=run, kernel=kernel, **metrics))
            print(f"{kernel:10s} AUROC={metrics['auroc']:.3f} AP={metrics['average_precision']:.3f} BAcc={metrics['balanced_accuracy']:.3f} F1={metrics['f1']:.3f}")
            for local_i, idx in enumerate(test_idx):
                prediction_rows.append({"run": run, "kernel": kernel, "sample_index": int(idx), "true": int(y[idx]), "score": float(scores[local_i]), "pred": int(pred[local_i])})
    raw_df = pd.DataFrame(all_rows)
    pred_df = pd.DataFrame(prediction_rows)
    summary = summarize_results(raw_df)
    raw_df.to_csv(outdir / "raw_results.csv", index=False)
    pred_df.to_csv(outdir / "predictions.csv", index=False)
    summary.to_csv(outdir / "summary.csv", index=False)
    plot_summary(summary, outdir)
    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to {outdir.resolve()}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="higher_order_data")
    p.add_argument("--outdir", type=str, default="dblp_simplicial_closure_outputs")
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--max-simplices", type=int, default=250000)
    p.add_argument("--max-simplex-size", type=int, default=10)
    p.add_argument("--max-group-size-for-triples", type=int, default=8)
    p.add_argument("--train-fraction-time", type=float, default=0.7)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--n-pos", type=int, default=120)
    p.add_argument("--n-neg", type=int, default=120)
    p.add_argument("--ego-size", type=int, default=28)
    p.add_argument("--max-local-triangles", type=int, default=200)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--lam", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--kernels", nargs="+", default=["graph_ntk", "lower", "upper", "full"], choices=["graph_ntk", "lower", "upper", "full"])
    p.add_argument("--save-grams", action="store_true")
    p.add_argument("--allow-open-positive-triads", action="store_true")
    p.add_argument("--allow-open-negative-triads", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.max_simplices = min(args.max_simplices, 50000)
        args.runs = 1
        args.n_pos = 30
        args.n_neg = 30
        args.ego_size = min(args.ego_size, 20)
        args.max_simplex_size = min(args.max_simplex_size, 8)
        args.max_group_size_for_triples = min(args.max_group_size_for_triples, 7)
        args.depth = min(args.depth, 1)
    return args


if __name__ == "__main__":
    run_experiment(parse_args())
