import sys
import math
import torch
from torch_geometric.data import ClusterData
from torch_geometric.utils import is_undirected, to_undirected, degree, coalesce
import numpy as np
from torch_geometric.utils import (
    to_networkx,
    from_networkx,
)
from torch_geometric.data import Data
from algorithms.borf.OllivierRicci import OllivierRicci
from algorithms.utils import print_max_min_cluster_sizes


device = f"cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
sys.setrecursionlimit(99999)


def _preprocess_data(data):
    # Get necessary data information
    N = data.x.shape[0]

    # Convert graph to Networkx
    G = to_networkx(data)

    return G, N

def borf3(
    data,
    loops=10,
    batch_add=4,
    batch_remove=2,
):
    # Preprocess data
    G, N = _preprocess_data(data)

    # Rewiring begins
    for _ in range(loops):
        # Compute ORC
        orc = OllivierRicci(G, alpha=0)
        orc.compute_ricci_curvature()
        _C = sorted(orc.G.edges, key=lambda x: orc.G[x[0]][x[1]]['ricciCurvature']['rc_curvature'])

        # Get top negative and positive curved edges
        most_pos_edges = _C[-batch_remove:]
        most_neg_edges = _C[:batch_add]

        # Add edges
        for (u, v) in most_neg_edges:
            pi = orc.G[u][v]['ricciCurvature']['rc_transport_cost']
            p, q = np.unravel_index(pi.values.argmax(), pi.values.shape)
            p, q = pi.index[p], pi.columns[q]
            
            if(p != q and not G.has_edge(p, q)):
                G.add_edge(p, q)

        # Remove edges
        for (u, v) in most_pos_edges:
            if(G.has_edge(u, v)):
                G.remove_edge(u, v)

    edge_index = from_networkx(G).edge_index

    return edge_index

def borf_clusters(data, cluster_size, borf_num_iterations, borf_batch_add, borf_batch_remove):
    if is_undirected(data.edge_index) is False:  # METIS sometimes crush if the graph is directed
        data.edge_index = to_undirected(data.edge_index)
    num_parts = math.ceil(len(data.x)/cluster_size)
    cluster_data = ClusterData(data, num_parts=num_parts)
    part = cluster_data.partition
    cluster_data = [cluster for cluster in cluster_data if cluster.x.shape[0] > 0]  # only non empty clusters
    min_size, max_size, cluster_sizes = print_max_min_cluster_sizes(cluster_data)
    if min_size < 0.5*cluster_size:
        raise ValueError('Minimum cluster size must be at least 1/2 of cluster_size')

    all_edge_index = []
    node_offset = 0
    for cluster in cluster_data:
        borf_edge_index = borf3(
            cluster,
            loops=borf_num_iterations,
            batch_add=borf_batch_add,
            batch_remove=borf_batch_remove,
        )
        edge_index = borf_edge_index.to(device=device)

        edge_index = edge_index + node_offset
        all_edge_index.append(edge_index)
        node_offset += cluster.x.shape[0]

    edge_index = torch.cat(all_edge_index, dim=1)

    # Reorder the data back to original order
    x = data.x
    y = data.y
    train_mask = None  # just placeholder
    val_mask = None  # just placeholder
    test_mask = None  # just placeholder

    # Reorder edge indices back to original order
    row_indices, col_indices = edge_index
    row_indices = part.node_perm[row_indices]  # Reorder row indices
    col_indices = part.node_perm[col_indices]  # Reorder column indices
    edge_index = torch.stack([row_indices, col_indices], dim=0)

    # we need to merge remaining edges with inter-cluster edges
    # find inter-cluster edges
    node_offset = 0
    intra_edge_index = []
    for cluster in cluster_data:
        cluster_edge_index = cluster.edge_index + node_offset
        intra_edge_index.append(cluster_edge_index)
        node_offset += cluster.x.shape[0]
    intra_edge_index = torch.cat(intra_edge_index, dim=1)
    row_indices, col_indices = intra_edge_index
    row_indices = part.node_perm[row_indices]  # Reorder row indices
    col_indices = part.node_perm[col_indices]  # Reorder column indices
    intra_edge_index = torch.stack([row_indices, col_indices], dim=0)

    edge_index_set = set(map(tuple, data.edge_index.T.tolist()))
    intra_edge_index_set = set(map(tuple, intra_edge_index.T.tolist()))
    inter_edges_set = edge_index_set - intra_edge_index_set
    inter_edges = torch.tensor(list(inter_edges_set)).T.to(device)

    edge_index = torch.cat([edge_index, inter_edges], dim=1)
    edge_index = coalesce(edge_index, num_nodes=len(data.x))

    row_indices, col_indices = edge_index
    deg = degree(row_indices, num_nodes=len(data.x))
    edge_weight = 1.0 / deg[row_indices]

    # Create the merged data object
    merged_data = Data(x=x, edge_index=edge_index, y=y,
                       train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                       edge_weight=edge_weight)

    return merged_data, edge_weight

def BORF_rewiring(data, cluster_size, borf_num_iterations, borf_batch_add, borf_batch_remove):
    if cluster_size is None:  # no clustering
        borf_edge_index = borf3(
            data,
            loops=borf_num_iterations,
            batch_add=borf_batch_add,
            batch_remove=borf_batch_remove,
        )
        edge_index = borf_edge_index.to(device=device)

        new_data = Data(x=data.x, edge_index=edge_index, y=data.y,
                        train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask,
                        edge_weight=None)
        new_edge_weight = None  # we don't use edge_weight

    else:
        new_data, new_edge_weight = borf_clusters(data, cluster_size, borf_num_iterations, borf_batch_add, borf_batch_remove)
        new_edge_weight = None  # we don't use edge_weight

    return new_data, new_edge_weight

