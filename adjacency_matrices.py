import numpy as np
def get_adjacency_matrices(dataset):
    """
    Convert a dataset of graphs to a list of adjacency matrices.
    Args:
    dataset (TUDataset): A dataset of graphs.
    Returns:
    list: A list of adjacency matrices.
    """
    adjacency_matrices = []
    for data in dataset:
        edge_index = data.edge_index.numpy()
        num_nodes = data.num_nodes
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for i, j in edge_index.T:
            adj_matrix[i, j] = 1
        adjacency_matrices.append(adj_matrix)
    return adjacency_matrices