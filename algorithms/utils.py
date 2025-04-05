def print_max_min_cluster_sizes(cluster_data):
    min_size = float('inf')
    max_size = 0
    cluster_sizes = []
    for i, cluster in enumerate(cluster_data):
        cluster_size = cluster.x.shape[0]
        min_size = min(min_size, cluster_size)
        max_size = max(max_size, cluster_size)
        cluster_sizes.append(cluster_size)

    print(f"Minimum cluster size: {min_size}")
    print(f"Maximum cluster size: {max_size}")
    return min_size, max_size, cluster_sizes