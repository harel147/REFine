import sys
import math
from numba import cuda
import numpy as np
import torch
from torch_geometric.utils import (
    to_networkx,
    from_networkx,
    to_dense_adj,
    remove_self_loops,
    to_undirected,
)
from torch_geometric.data import ClusterData
from torch_geometric.utils import is_undirected, to_undirected, degree, coalesce
from torch_geometric.data import Data

from algorithms.sdrf.gdl.src.gdl.curvature.utils import softmax

from algorithms.utils import print_max_min_cluster_sizes


device = f"cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
sys.setrecursionlimit(99999)


@cuda.jit(
    "void(float32[:,:], float32[:,:], float32[:], float32[:], int32, float32[:,:])"
)
def _balanced_forman_curvature(A, A2, d_in, d_out, N, C):
    i, j = cuda.grid(2)

    if (i < N) and (j < N):
        if A[i, j] == 0:
            C[i, j] = 0
            return

        if d_in[i] > d_out[j]:
            d_max = d_in[i]
            d_min = d_out[j]
        else:
            d_max = d_out[j]
            d_min = d_in[i]

        if d_max * d_min == 0:
            C[i, j] = 0
            return

        sharp_ij = 0
        lambda_ij = 0
        for k in range(N):
            TMP = A[k, j] * (A2[i, k] - A[i, k]) * A[i, j]
            if TMP > 0:
                sharp_ij += 1
                if TMP > lambda_ij:
                    lambda_ij = TMP

            TMP = A[i, k] * (A2[k, j] - A[k, j]) * A[i, j]
            if TMP > 0:
                sharp_ij += 1
                if TMP > lambda_ij:
                    lambda_ij = TMP

        C[i, j] = (
            (2 / d_max) + (2 / d_min) - 2 + (2 / d_max + 1 / d_min) * A2[i, j] * A[i, j]
        )
        if lambda_ij > 0:
            C[i, j] += sharp_ij / (d_max * lambda_ij)


def balanced_forman_curvature(A, C=None):
    N = A.shape[0]
    A2 = torch.matmul(A, A)
    d_in = A.sum(axis=0)
    d_out = A.sum(axis=1)
    if C is None:
        C = torch.zeros(N, N).cuda()

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(N / threadsperblock[0])
    blockspergrid_y = math.ceil(N / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    _balanced_forman_curvature[blockspergrid, threadsperblock](A, A2, d_in, d_out, N, C)
    return C


@cuda.jit(
    "void(float32[:,:], float32[:,:], float32, float32, int32, float32[:,:], int32, int32, int32[:], int32[:], int32, int32)"
)
def _balanced_forman_post_delta(
    A, A2, d_in_x, d_out_y, N, D, x, y, i_neighbors, j_neighbors, dim_i, dim_j
):
    I, J = cuda.grid(2)

    if (I < dim_i) and (J < dim_j):
        i = i_neighbors[I]
        j = j_neighbors[J]

        if (i == j) or (A[i, j] != 0):
            D[I, J] = -1000
            return

        # Difference in degree terms
        if j == x:
            d_in_x += 1
        elif i == y:
            d_out_y += 1

        if d_in_x * d_out_y == 0:
            D[I, J] = 0
            return

        if d_in_x > d_out_y:
            d_max = d_in_x
            d_min = d_out_y
        else:
            d_max = d_out_y
            d_min = d_in_x

        # Difference in triangles term
        A2_x_y = A2[x, y]
        if (x == i) and (A[j, y] != 0):
            A2_x_y += A[j, y]
        elif (y == j) and (A[x, i] != 0):
            A2_x_y += A[x, i]

        # Difference in four-cycles term
        sharp_ij = 0
        lambda_ij = 0
        for z in range(N):
            A_z_y = A[z, y] + 0
            A_x_z = A[x, z] + 0
            A2_z_y = A2[z, y] + 0
            A2_x_z = A2[x, z] + 0

            if (z == i) and (y == j):
                A_z_y += 1
            if (x == i) and (z == j):
                A_x_z += 1
            if (z == i) and (A[j, y] != 0):
                A2_z_y += A[j, y]
            if (x == i) and (A[j, z] != 0):
                A2_x_z += A[j, z]
            if (y == j) and (A[z, i] != 0):
                A2_z_y += A[z, i]
            if (z == j) and (A[x, i] != 0):
                A2_x_z += A[x, i]

            TMP = A_z_y * (A2_x_z - A_x_z) * A[x, y]
            if TMP > 0:
                sharp_ij += 1
                if TMP > lambda_ij:
                    lambda_ij = TMP

            TMP = A_x_z * (A2_z_y - A_z_y) * A[x, y]
            if TMP > 0:
                sharp_ij += 1
                if TMP > lambda_ij:
                    lambda_ij = TMP

        D[I, J] = (
            (2 / d_max) + (2 / d_min) - 2 + (2 / d_max + 1 / d_min) * A2_x_y * A[x, y]
        )
        if lambda_ij > 0:
            D[I, J] += sharp_ij / (d_max * lambda_ij)


def balanced_forman_post_delta(A, x, y, i_neighbors, j_neighbors, D=None):
    N = A.shape[0]
    A2 = torch.matmul(A, A)
    d_in = A[:, x].sum()
    d_out = A[y].sum()
    if D is None:
        D = torch.zeros(len(i_neighbors), len(j_neighbors)).cuda()

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(D.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(D.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    _balanced_forman_post_delta[blockspergrid, threadsperblock](
        A,
        A2,
        d_in,
        d_out,
        N,
        D,
        x,
        y,
        np.array(i_neighbors),
        np.array(j_neighbors),
        D.shape[0],
        D.shape[1],
    )
    return D


def sdrf(
    data,
    loops=10,
    remove_edges=True,
    removal_bound=0.5,
    tau=1,
    is_undirected=False,
):
    edge_index = data.edge_index
    if is_undirected:
        edge_index = to_undirected(edge_index)
    A = to_dense_adj(remove_self_loops(edge_index)[0])[0]
    N = A.shape[0]
    G = to_networkx(data)
    if is_undirected:
        G = G.to_undirected()
    A = A.cuda()
    C = torch.zeros(N, N).cuda()

    for x in range(loops):
        can_add = True
        balanced_forman_curvature(A, C=C)
        ix_min = C.argmin().item()
        x = ix_min // N
        y = ix_min % N
        
        if is_undirected:
            x_neighbors = list(G.neighbors(x)) + [x]
            y_neighbors = list(G.neighbors(y)) + [y]
        else:
            x_neighbors = list(G.successors(x)) + [x]
            y_neighbors = list(G.predecessors(y)) + [y]
        candidates = []
        for i in x_neighbors:
            for j in y_neighbors:
                if (i != j) and (not G.has_edge(i, j)):
                    candidates.append((i, j))

        if len(candidates):
            D = balanced_forman_post_delta(A, x, y, x_neighbors, y_neighbors)
            improvements = []
            for (i, j) in candidates:
                improvements.append(
                    (D - C[x, y])[x_neighbors.index(i), y_neighbors.index(j)].item()
                )

            k, l = candidates[
                np.random.choice(
                    range(len(candidates)), p=softmax(np.array(improvements), tau=tau)
                )
            ]
            G.add_edge(k, l)
            if is_undirected:
                A[k, l] = A[l, k] = 1
            else:
                A[k, l] = 1
        else:
            can_add = False
            if not remove_edges:
                break

        if remove_edges:
            ix_max = C.argmax().item()
            x = ix_max // N
            y = ix_max % N
            if C[x, y] > removal_bound:
                G.remove_edge(x, y)
                if is_undirected:
                    A[x, y] = A[y, x] = 0
                else:
                    A[x, y] = 0
            else:
                if can_add is False:
                    break

    return from_networkx(G)

def sdrf_clusters(data, cluster_size, sdrf_max_iter_ratio, sdrf_removal_bound, sdrf_tau):
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
        altered_data = sdrf(
            cluster,
            loops=int(sdrf_max_iter_ratio * len(cluster.x)),
            remove_edges=True,
            removal_bound=sdrf_removal_bound,
            tau=sdrf_tau,
            is_undirected=True,
        )
        edge_index = altered_data.edge_index.to(device=device)

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

def SDRF_rewiring(data, cluster_size, sdrf_max_iter_ratio, sdrf_tau, sdrf_removal_bound):
    if cluster_size is None:  # no clustering
        altered_data = sdrf(
            data,
            loops=int(sdrf_max_iter_ratio * len(data.x)),
            remove_edges=True,
            removal_bound=sdrf_removal_bound,
            tau=sdrf_tau,
            is_undirected=True,
        )
        edge_index = altered_data.edge_index.to(device=device)

        new_data = Data(x=data.x, edge_index=edge_index, y=data.y,
                        train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask,
                        edge_weight=None)
        new_edge_weight = None  # we don't use edge_weight

    else:
        new_data, new_edge_weight = sdrf_clusters(data, cluster_size, sdrf_max_iter_ratio, sdrf_removal_bound, sdrf_tau)
        new_edge_weight = None  # we don't use edge_weight

    return new_data, new_edge_weight