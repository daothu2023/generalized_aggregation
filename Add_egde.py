import torch
def create_adjacency_list(edge_index):
    adj_list = {}
    for i in range(edge_index.size(1)):
        src, dst = edge_index[:, i]
        if src.item() not in adj_list:
            adj_list[src.item()] = []
        if dst.item() not in adj_list:
            adj_list[dst.item()] = []
        adj_list[src.item()].append(dst.item())
        adj_list[dst.item()].append(src.item())
    return adj_list
d=0
def add_edges(edge_index, num_nodes, similarity_matrix, top_k):
    new_edges = []
    similarity_matrix = torch.from_numpy(similarity_matrix)

    if similarity_matrix.size(0) < top_k:
        print("Size of similarity matrix is smaller than the number of nodes. Skipping edge addition.")
        return torch.empty((2, 0), dtype=torch.long)

    _, top_indices = torch.topk(similarity_matrix, top_k, dim=-1)
    adj_list = create_adjacency_list(edge_index)

    for i in range(num_nodes):
        if i in adj_list and i < len(top_indices):
            for j in top_indices[i]:
                if j != i and j not in adj_list[i]:
                    new_edges.append((i, j))


    if len(new_edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(new_edges, dtype=torch.long).t()