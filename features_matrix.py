from load_graph_datasets import dispatch
from WLVectorizer import WLVectorizer
def feature_WLvect(dataset):
    G_list = dispatch(dataset)
    n = len(G_list.graphs)
    node_order = G_list.graphs[1].graph['node_order']
    print(node_order)
    y = G_list.target
    print(y)
    print("Computing WL node kernel..")
    WLvect = WLVectorizer(r=3)
    features = WLvect.transform(G_list.graphs)
    total_matrices = []
    for j in range(len(G_list.graphs)):
      matrix = features[0][j]
      for k in range(1,4):
        matrix += features[k][j]
      total_matrices.append(matrix)
    del(features)
    return total_matrices