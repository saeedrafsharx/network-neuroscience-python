"""Build the cached per-participant metrics used by notebook 10's group comparison.

For each of the 50 participants in dev_fmri_timeseries.npz, this builds a
graph at a FIXED, density-matched threshold (proportional thresholding, the
fix from session 6 — every participant's graph has exactly the same density,
regardless of how strong their raw correlations happen to be), then computes:

  - mean participation coefficient and mean |within-module z-score|
    (session 4's hub cartography, averaged across all regions)
  - the degree-preserving-null-normalised rich-club coefficient (session 5)
    at a shared range of degree thresholds

The rich-club step alone runs 300 degree-preserving randomisations per
participant per degree threshold — clearly a "cache it" job under this
series' pacing rule, not something to run live in a recorded session.

Run once:  python scripts/build_capstone_group_metrics.py
Takes a few minutes.
"""

import os
import warnings

import numpy as np
import pandas as pd
import networkx as nx

warnings.filterwarnings("ignore")

OUT = "data"
os.makedirs(OUT, exist_ok=True)

TARGET_DENSITY = 0.15
N_NULL = 300
KS_COMMON = list(range(2, 9))


def graph_at_density(matrix, target_edges, node_labels):
    n = matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = matrix[iu]
    top_idx = np.argsort(vals)[-target_edges:]
    adj = np.zeros((n, n), dtype=int)
    rows, cols = iu[0][top_idx], iu[1][top_idx]
    adj[rows, cols] = 1
    adj[cols, rows] = 1
    g = nx.from_numpy_array(adj)
    return nx.relabel_nodes(g, dict(enumerate(node_labels)))


def participation_coefficient(G, communities):
    node_to_comm = {n: ci for ci, members in enumerate(communities) for n in members}
    P = {}
    for i in G.nodes():
        k_i = G.degree(i)
        if k_i == 0:
            P[i] = 0.0
            continue
        neighbour_comms = [node_to_comm[j] for j in G.neighbors(i)]
        total = sum((neighbour_comms.count(c) / k_i) ** 2 for c in set(neighbour_comms))
        P[i] = 1 - total
    return P


def within_module_degree_zscore(G, communities):
    node_to_comm = {n: ci for ci, members in enumerate(communities) for n in members}
    z = {}
    for members in communities:
        within_deg = {i: sum(1 for j in G.neighbors(i) if node_to_comm[j] == node_to_comm[i])
                      for i in members}
        vals = np.array(list(within_deg.values()), dtype=float)
        mean, std = vals.mean(), vals.std()
        for i in members:
            z[i] = 0.0 if std == 0 else (within_deg[i] - mean) / std
    return z


def degree_preserving_null(G, n_swaps_per_edge=10, seed=0):
    rng = np.random.default_rng(seed)
    G_null = G.copy()
    n_swaps = max(10, n_swaps_per_edge * G.number_of_edges())
    nx.double_edge_swap(G_null, nswap=n_swaps, max_tries=n_swaps * 20,
                         seed=int(rng.integers(1_000_000_000)))
    return G_null


print("loading dev_fmri_timeseries.npz ...")
d = np.load(f"{OUT}/dev_fmri_timeseries.npz", allow_pickle=True)
series = d["timeseries"]
group = d["group"]
age = d["age"]
participant_id = d["participant_id"]
labels = [str(x) for x in d["regions"]]

conn = np.array([np.corrcoef(subject.T) for subject in series])
n_regions = conn.shape[1]
max_edges = n_regions * (n_regions - 1) // 2
target_edges = round(TARGET_DENSITY * max_edges)
print(f"{len(conn)} participants, {n_regions} regions, "
      f"target density={TARGET_DENSITY} -> {target_edges} edges each")

rows = []
for i in range(len(conn)):
    g = graph_at_density(conn[i], target_edges, labels)
    comms = nx.community.louvain_communities(g, seed=0)
    P = participation_coefficient(g, comms)
    Z = within_module_degree_zscore(g, comms)

    rc_obs = nx.rich_club_coefficient(g, normalized=False)
    rng = np.random.default_rng(1000 + i)
    null_curves = np.full((N_NULL, len(KS_COMMON)), np.nan)
    for j in range(N_NULL):
        gn = degree_preserving_null(g, seed=int(rng.integers(1_000_000_000)))
        rc_n = nx.rich_club_coefficient(gn, normalized=False)
        for k_idx, k in enumerate(KS_COMMON):
            if k in rc_n:
                null_curves[j, k_idx] = rc_n[k]
    null_mean = np.nanmean(null_curves, axis=0)

    row = {
        "participant_id": str(participant_id[i]),
        "group": str(group[i]),
        "age": float(age[i]),
        "mean_participation": float(np.mean(list(P.values()))),
        "mean_abs_zscore": float(np.mean(np.abs(list(Z.values())))),
    }
    for k_idx, k in enumerate(KS_COMMON):
        obs_k = rc_obs.get(k, np.nan)
        row[f"rc_norm_k{k}"] = float(obs_k / null_mean[k_idx]) if null_mean[k_idx] > 0 else np.nan
    rows.append(row)
    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(conn)} participants done")

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/capstone_group_metrics.csv", index=False)
print(f"\nsaved {OUT}/capstone_group_metrics.csv  ({df.shape[0]} rows, {df.shape[1]} columns)")
print(df.groupby("group")[["mean_participation", "rc_norm_k8"]].mean())
