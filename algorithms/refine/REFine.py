import sys
import math
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from torch_geometric.data import ClusterData
from torch_geometric.utils import is_undirected, to_undirected, degree, coalesce
from torch_geometric.data import Data

from algorithms.utils import print_max_min_cluster_sizes


device = f"cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
sys.setrecursionlimit(99999)

def symmetric_normalize_tensor(mx):
    rowsum = torch.sum(mx, 1)
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.0
    r_mat_inv = torch.diag(r_inv)
    mx_norm1 = torch.mm(torch.mm(r_mat_inv, mx), r_mat_inv)

    rowsum = torch.sum(mx_norm1, 1)
    r_inv = torch.pow(rowsum, -0.5).flatten()
    r_inv[torch.isinf(r_inv)] = 0.0
    r_mat_inv = torch.diag(r_inv)
    mx_norm2 = torch.mm(torch.mm(r_mat_inv, mx_norm1), r_mat_inv)

    return mx_norm2

def compute_kernel_data(data, metric='euclidean', eps=0.01):
    # affinity matrix W
    distances = squareform(pdist(data, metric=metric))
    W = np.exp(-((distances ** 2) / eps))

    # uncomment for checking all labels
    np.fill_diagonal(W, 0)

    # diagonal normalization matrix Q
    row_sums = np.sum(W, axis=1)
    row_sums = row_sums ** -1
    row_sums[np.isinf(row_sums)] = 0.0
    Q = np.diag(row_sums)

    # normalized kernel
    K_norm1 = Q @ W @ Q

    # second diagonal normalization matrix Q2
    row_sums = np.sum(K_norm1, axis=1)
    row_sums = row_sums ** -0.5
    row_sums[np.isinf(row_sums)] = 0.0
    Q2 = np.diag(row_sums)

    # second normalized kernel
    K_norm2 = Q2 @ K_norm1 @ Q2

    return K_norm2

def kernel_to_edge_index_and_edge_weight(kernel):
    # Step 1: Get non-zero indices (edges) and values (weights)
    edge_indices = kernel.nonzero(as_tuple=False).T  # Shape will be [2, num_edges]
    edge_weights = kernel[edge_indices[0], edge_indices[1]]  # Get weights for each edge

    # Now, edge_indices is your edge_index and edge_weights is your edge_weight
    edge_index = edge_indices
    edge_weight = edge_weights

    return edge_index, edge_weight

def reconstruct_full_graph_for_REFine(cluster_data, split_idx, PDPs, part, data, sample_rate, add_or_delete):
    all_edge_index = []
    node_offset = 0
    for pdp, cluster in zip(PDPs, cluster_data):
        row_indices, col_indices = torch.nonzero(pdp, as_tuple=True)
        edge_index = torch.stack([row_indices, col_indices], dim=0)

        if add_or_delete == 'add':
            num_add = int(sample_rate * edge_index.shape[1])
            indices = torch.randperm(edge_index.shape[1])[:num_add]
            edge_index = edge_index[:, indices]
            # it's not a bug that I don't use here the original edges, I add them later.

        if add_or_delete == 'delete':
            # new deletion implementation, using kernels instead of using edge sets.
            pdp_comp = 1 - pdp

            num_nodes = len(pdp_comp)
            orig_graph = torch.zeros((num_nodes, num_nodes), dtype=torch.float).to(device)
            row, col = cluster.edge_index
            orig_graph[row, col] = 1

            pdp_comp_for_comparison = pdp_comp.clone()
            pdp_comp_for_comparison[pdp_comp_for_comparison == 0] = 2
            pdp_comp_for_comparison[pdp_comp_for_comparison == 1] = 4

            # 2 - both no edge, 3 - edge only for orig, 4- edge only for pdp_comp, 5 - common edge
            compare_orig_pdp_comp = orig_graph + pdp_comp_for_comparison
            only_orig_edges = torch.zeros((num_nodes, num_nodes), dtype=torch.float).to(device)
            only_orig_edges[compare_orig_pdp_comp == 3] = 1
            only_common_edges = torch.zeros((num_nodes, num_nodes), dtype=torch.float).to(device)
            only_common_edges[compare_orig_pdp_comp == 5] = 1

            mask = torch.rand(only_common_edges.shape) > sample_rate
            mask = mask.float().to(device)
            only_common_edges_sampled = only_common_edges * mask

            only_orig_and_common_sampled = only_orig_edges + only_common_edges_sampled
            only_orig_and_common_sampled[only_orig_and_common_sampled > 0] = 1

            edge_index, _ = kernel_to_edge_index_and_edge_weight(only_orig_and_common_sampled)

        edge_index = edge_index + node_offset
        all_edge_index.append(edge_index)
        node_offset += cluster.x.shape[0]


    edge_index = torch.cat(all_edge_index, dim=1)

    # Reorder the data back to original order
    x = data.x
    y = data.y
    train_mask = data.train_mask[:, split_idx]
    val_mask = data.val_mask[:, split_idx]
    test_mask = data.test_mask[:, split_idx]

    # Reorder edge indices back to original order
    row_indices, col_indices = edge_index
    row_indices = part.node_perm[row_indices]  # Reorder row indices
    col_indices = part.node_perm[col_indices]  # Reorder column indices
    edge_index = torch.stack([row_indices, col_indices], dim=0)

    if add_or_delete == 'add':  # if we add edges we need to merge added edges with the original edges
        edge_index = torch.cat([edge_index, data.edge_index], dim=1)
        edge_index = coalesce(edge_index, num_nodes=len(data.x))

    if add_or_delete == 'delete':  # if we delete edges we need to merge remaining edges with inter-cluster edges
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

def REFine_clusters(data, cluster_size, split_idx, eps, sample_rate, add_or_delete, scheme='PDP', thresh_factor=1):
    if is_undirected(data.edge_index) is False:  # METIS sometimes crush if the graph is directed
        data.edge_index = to_undirected(data.edge_index)
    num_parts = math.ceil(len(data.x)/cluster_size)
    cluster_data = ClusterData(data, num_parts=num_parts)
    part = cluster_data.partition
    cluster_data = [cluster for cluster in cluster_data if cluster.x.shape[0] > 0]  # only non empty clusters
    min_size, max_size, cluster_sizes = print_max_min_cluster_sizes(cluster_data)
    if min_size < 0.5*cluster_size:
        raise ValueError('Minimum cluster size must be at least 1/2 of cluster_size')

    data_kernels = torch.zeros((num_parts, max_size, max_size)).to(device)
    labels_kernels = torch.zeros((num_parts, max_size, max_size)).to(device)
    for i, cluster in enumerate(cluster_data):
        data_kernel = compute_kernel_data(cluster.x.detach().cpu().numpy(), eps=eps)
        data_kernel = torch.from_numpy(data_kernel.astype(np.float32)).to(device)

        labels_kernel = (cluster.y.unsqueeze(0) == cluster.y.unsqueeze(1)).float()
        train_mask = cluster.train_mask[:, split_idx]
        test_mask = cluster.test_mask[:, split_idx]
        val_mask = cluster.val_mask[:, split_idx]
        mask = val_mask | test_mask
        # we don't use val and test labels
        labels_kernel[mask, :] = 0  # Zero out rows
        labels_kernel[:, mask] = 0  # Zero out columns
        labels_kernel[mask, mask] = 1

        labels_kernel = symmetric_normalize_tensor(labels_kernel)

        data_kernel_padded = torch.zeros(max_size, max_size).to(device)
        data_kernel_padded[:data_kernel.shape[0], :data_kernel.shape[0]] = data_kernel
        labels_kernel_padded = torch.zeros(max_size, max_size).to(device)
        labels_kernel_padded[:labels_kernel.shape[0], :labels_kernel.shape[0]] = labels_kernel
        data_kernels[i] = data_kernel_padded
        labels_kernels[i] = labels_kernel_padded

    PDs_with_pad = torch.matmul(labels_kernels, data_kernels)
    PDPs_with_pad = torch.matmul(PDs_with_pad, labels_kernels)
    if scheme == 'D':
        PDPs_with_pad = data_kernels

    PDPs = []
    for (i, cluster_size), cluster in zip(enumerate(cluster_sizes), cluster_data):

        pdp_without_padding = PDPs_with_pad[i, :cluster_size, :cluster_size]

        thresh = thresh_factor*torch.mean(pdp_without_padding, dim=1, keepdim=True)
        pdp_without_padding[pdp_without_padding < thresh] = 0
        pdp_without_padding = pdp_without_padding.fill_diagonal_(0)
        pdp_without_padding[pdp_without_padding > 0] = 1

        PDPs.append(pdp_without_padding)

    merged_data, new_edge_weight = reconstruct_full_graph_for_REFine(cluster_data, split_idx, PDPs, part, data, sample_rate, add_or_delete)

    return merged_data, new_edge_weight

def REFine_no_clusters(data, split_idx, eps, sample_rate, add_or_delete, scheme='PDP', thresh_factor=1):
    data_kernel = compute_kernel_data(data.x.detach().cpu().numpy(), eps=eps)
    data_kernel = torch.from_numpy(data_kernel.astype(np.float32)).to(device)
    labels_kernel = (data.y.unsqueeze(0) == data.y.unsqueeze(1)).float()  # create P using all labels, will be masked accordingly for each data split

    train_mask = data.train_mask[:, split_idx]
    test_mask = data.test_mask[:, split_idx]
    val_mask = data.val_mask[:, split_idx]
    mask = val_mask | test_mask
    iter_labels_kernel = labels_kernel.clone()
    iter_labels_kernel[mask, :] = 0  # Zero out rows
    iter_labels_kernel[:, mask] = 0  # Zero out columns
    iter_labels_kernel[mask, mask] = 1

    iter_labels_kernel = symmetric_normalize_tensor(iter_labels_kernel)

    if scheme == 'D':
        result_kernel = data_kernel
    if scheme == 'PDP':
        result_kernel = iter_labels_kernel @ data_kernel @ iter_labels_kernel  # for PDP (dont forget to zero self loops for D)

    # clip by row means
    thresh = thresh_factor * torch.mean(result_kernel, dim=1, keepdim=True)
    result_kernel[result_kernel < thresh] = 0
    result_kernel = result_kernel.fill_diagonal_(0)
    result_kernel[result_kernel > 0] = 1

    edge_index, edge_weight = kernel_to_edge_index_and_edge_weight(result_kernel)  # reference graph edge set
    edge_weight = None

    if add_or_delete == 'add':
        num_add = int(sample_rate * edge_index.shape[1])
        indices = torch.randperm(edge_index.shape[1])[:num_add]
        sampled_edge_index = edge_index[:, indices]

        edge_index = torch.cat([sampled_edge_index, data.edge_index], dim=1)
        edge_index = coalesce(edge_index, num_nodes=len(data.x))
    if add_or_delete == 'delete':
        # new deletion implementation, using kernels instead of using edge sets.
        result_kernel_comp = 1 - result_kernel

        num_nodes = len(result_kernel_comp)
        orig_graph = torch.zeros((num_nodes, num_nodes), dtype=torch.float).to(device)
        row, col = data.edge_index
        orig_graph[row, col] = 1

        pdp_comp_for_comparison = result_kernel_comp.clone()
        pdp_comp_for_comparison[pdp_comp_for_comparison == 0] = 2
        pdp_comp_for_comparison[pdp_comp_for_comparison == 1] = 4

        # 2 - both no edge, 3 - edge only for orig, 4- edge only for pdp_comp, 5 - common edge
        compare_orig_pdp_comp = orig_graph + pdp_comp_for_comparison
        only_orig_edges = torch.zeros((num_nodes, num_nodes), dtype=torch.float).to(device)
        only_orig_edges[compare_orig_pdp_comp == 3] = 1
        only_common_edges = torch.zeros((num_nodes, num_nodes), dtype=torch.float).to(device)
        only_common_edges[compare_orig_pdp_comp == 5] = 1

        mask = torch.rand(only_common_edges.shape) > sample_rate
        mask = mask.float().to(device)
        only_common_edges_sampled = only_common_edges * mask

        only_orig_and_common_sampled = only_orig_edges + only_common_edges_sampled
        only_orig_and_common_sampled[only_orig_and_common_sampled > 0] = 1

        edge_index, _ = kernel_to_edge_index_and_edge_weight(only_orig_and_common_sampled)

    new_data = Data(x=data.x, edge_index=edge_index, y=data.y,
                       train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                       edge_weight=edge_weight)

    return new_data, edge_weight

def REFine_rewiring(data, cluster_size, split_idx, data_eps, sample_rate, add_or_delete, scheme):
    if cluster_size is None:  # no clustering
        new_data, new_edge_weight = REFine_no_clusters(data, split_idx, data_eps,
                                                  sample_rate, add_or_delete, scheme=scheme)
        new_edge_weight = None  # we don't use edge_weight

    else:
        new_data, new_edge_weight = REFine_clusters(data, cluster_size, split_idx, data_eps,
                                                                            sample_rate, add_or_delete, scheme=scheme)
        new_edge_weight = None  # we don't use edge_weight

    return new_data, new_edge_weight

