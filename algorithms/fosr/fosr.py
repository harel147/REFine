from numba import jit, int64
import sys
import math
import numpy as np
from math import inf
import torch
from torch_geometric.data import ClusterData
from torch_geometric.utils import is_undirected, to_undirected, degree, coalesce
from torch_geometric.data import Data

from algorithms.utils import print_max_min_cluster_sizes


device = f"cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
sys.setrecursionlimit(99999)


@jit(nopython=True)
def choose_edge_to_add(x, edge_index, degrees):
	# chooses edge (u, v) to add which minimizes y[u]*y[v]
	n = x.size
	m = edge_index.shape[1]
	y = x / ((degrees + 1) ** 0.5)
	products = np.outer(y, y)
	for i in range(m):
		u = edge_index[0, i]
		v = edge_index[1, i]
		products[u, v] = inf
	for i in range(n):
		products[i, i] = inf
	smallest_product = np.argmin(products)
	return (smallest_product % n, smallest_product // n)

@jit(nopython=True)
def compute_degrees(edge_index, num_nodes=None):
	# returns array of degrees of all nodes
	if num_nodes is None:
		num_nodes = np.max(edge_index) + 1
	degrees = np.zeros(num_nodes)
	m = edge_index.shape[1]
	for i in range(m):
		degrees[edge_index[0, i]] += 1
	return degrees

@jit(nopython=True)
def add_edge(edge_index, u, v):
	new_edge = np.array([[u, v],[v, u]])
	return np.concatenate((edge_index, new_edge), axis=1)

@jit(nopython=True)
def adj_matrix_multiply(edge_index, x):
	# given an edge_index, computes Ax, where A is the corresponding adjacency matrix
	n = x.size
	y = np.zeros(n)
	m = edge_index.shape[1]
	for i in range(m):
		u = edge_index[0, i]
		v = edge_index[1, i]
		y[u] += x[v]
	return y

@jit(nopython=True)
def compute_spectral_gap(edge_index, x):
	m = edge_index.shape[1]
	n = np.max(edge_index) + 1
	degrees = compute_degrees(edge_index, num_nodes=n)
	y = adj_matrix_multiply(edge_index, x / (degrees ** 0.5)) / (degrees ** 0.5)
	for i in range(n):
		if x[i] > 1e-9:
			return 1 - y[i]/x[i]
	return 0.

@jit(nopython=True)
def _edge_rewire(edge_index, x=None, num_iterations=50, initial_power_iters=50):
	m = edge_index.shape[1]
	n = np.max(edge_index) + 1
	if x is None:
		x = 2 * np.random.random(n) - 1
	degrees = compute_degrees(edge_index, num_nodes=n)
	for i in range(initial_power_iters):
		x = x - x.dot(degrees ** 0.5) * (degrees ** 0.5)/sum(degrees)
		y = x + adj_matrix_multiply(edge_index, x / (degrees ** 0.5)) / (degrees ** 0.5)
		x = y / np.linalg.norm(y)
	for I in range(num_iterations):
		i, j = choose_edge_to_add(x, edge_index, degrees=degrees)
		edge_index = add_edge(edge_index, i, j)
		degrees[i] += 1
		degrees[j] += 1
		x = x - x.dot(degrees ** 0.5) * (degrees ** 0.5)/sum(degrees)
		y = x + adj_matrix_multiply(edge_index, x / (degrees ** 0.5)) / (degrees ** 0.5)
		x = y / np.linalg.norm(y)
	return edge_index

def fosr(edge_index, x=None, num_iterations=50, initial_power_iters=5, device='cuda'):
	if edge_index.shape[1] == 0:
		return edge_index

	edge_index = edge_index.cpu().numpy()
	n = np.max(edge_index) + 1
	if x is None:
		x = 2 * np.random.random(n) - 1

	edge_index = _edge_rewire(edge_index, x=x, num_iterations=num_iterations, initial_power_iters=initial_power_iters)
	edge_index = torch.from_numpy(edge_index).to(device)

	return edge_index

def fosr_clusters(data, cluster_size, fosr_num_iterations):
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
        fosr_edge_index = fosr(
            cluster.edge_index,
            num_iterations=fosr_num_iterations,
        )
        edge_index = fosr_edge_index.to(device=device)

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

def FoSR_rewiring(data, cluster_size, fosr_num_iterations):
	if cluster_size is None:  # no clustering
		fosr_edge_index = fosr(
			data.edge_index,
			num_iterations=fosr_num_iterations
		)
		edge_index = fosr_edge_index.to(device=device)

		new_data = Data(x=data.x, edge_index=edge_index, y=data.y,
						train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask,
						edge_weight=None)
		new_edge_weight = None  # we don't use edge_weight

	else:
		new_data, new_edge_weight = fosr_clusters(data, cluster_size, fosr_num_iterations)
		new_edge_weight = None  # we don't use edge_weight

	return new_data, new_edge_weight
