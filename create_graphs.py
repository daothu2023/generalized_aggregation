import networkx as nx
def create_graphs(arry):
    """
    Parameters:
    - file_path: path to adjacency matrices
    - start_ID: Starting ID for graph nodes

    Return: An undirected, unweighted graph
    """

    start_ID = 0
    graphs = []
    for AM in [arry]:
        N = AM.shape[0]
        g = nx.Graph()

        g.add_nodes_from(range(start_ID, N + start_ID), label="")
        list_edges = []
        for u in range(N - 1):
            for v in range(u + 1, N):
                w = AM[u, v]
                if w != 0.0:
                    list_edges.append((u + start_ID, v + start_ID))
        g.add_edges_from(list_edges, label="")
        graphs.append(g)

        start_ID += start_ID + N

    return graphs