from ioskgraph import load_target
from graph import instance_to_graph
from scipy.sparse import csr_matrix
from torch_geometric.datasets import TUDataset
from torch_geometric.utils.convert import to_networkx
from sklearn.utils import Bunch
def dispatch(dataset):
    if dataset == "GDD":
        print("Loading GDD dataset")
        g_it = load_graphs_GDD()


def dispatch(dataset):
    if dataset == "PTC_MR":
        print("Loading PTC_MR dataset")
        g_it = load_graphs_PTC_MR()
    elif dataset == "Mutag":
        print("Loading MUTAG dataset")
        g_it = load_graphs_MUTAG()
    elif dataset == "KKI":
        print("Loading KKI dataset")
        g_it = load_graphs_KKI()
    elif dataset == "DD":
        print("Loading DD dataset")
        g_it = load_graphs_DD()

    elif dataset == "proteins":
        print("Loading proteins dataset")
        g_it = load_graphs_proteins()
    elif dataset == "AIDS":
        print("Loading AIDS dataset")
        g_it = load_graphs_AIDS()
    elif dataset == "NCI109":
        print("Loading NCI109 dataset")
        g_it = load_graphs_NCI109()
    elif dataset == "CPDB":
        print("Loading CPDB dataset")
        g_it = load_graphs_CPDB()
    elif dataset == "NCI1":
        print("Loading NCI1 dataset")
        g_it = load_graphs_NCI1()
    elif dataset == "NCI123":
        print("Loading NCI123 dataset")
        g_it = load_graphs_NCI123()
    elif dataset == "NCI_AIDS":
        print("Loading NCI_AIDS dataset")
        g_it = load_graphs_NCI_AIDS()
    elif dataset == "Chemical2":
        print("Loading LEUK40OV41LEUK47OV50 dataset")
        g_it = load_graphs_LEUK40OV41LEUK47OV50()
    elif dataset == "Chemical1":
        print("Loading LEUK40LEUK47OV41OV50 dataset")
        g_it = load_graphs_LEUK40LEUK47OV41OV50()
    elif dataset == "Chemical3":
        print("Loading LEUK40LEUK47OV41OV50LEUK40OV41LEUK47OV50 dataset")
        g_it = load_graphs_LEUK40LEUK47OV41OV50LEUK40OV41LEUK47OV50()
    elif dataset == "Chemical_reduced":
        print("Loading LEUK40OV41LEUK47OV50 REDUCED dataset")
        g_it = load_graphs_LEUK40OV41LEUK47OV50_reduced()
    elif dataset == "Enzymes":
        print("Loading enzymes dataset")
        g_it = load_graphs_enzymes()
    elif dataset == "proteins":
        print("Loading proteins dataset")
        g_it = load_graphs_proteins()

    elif dataset == "Synthetic":
        print("Loading synthetic dataset")
        g_it = load_graphs_synthetic()
    elif dataset == "BZR":
        print("Loading BZR dataset")
        g_it = load_graphs_BZR()
    elif dataset == "COX2":
        print("Loading COX2 dataset")
        g_it = load_graphs_COX2()
    elif dataset == "DHFR":
        print("Loading DHFR dataset")
        g_it = load_graphs_DHFR()
    elif dataset == "PROTEINS_full":
        print("Loading PROTEINS_full dataset")
        g_it = load_graphs_PROTEINS_full()
    elif dataset == "LMdata":
        print("Loading LMdata dataset")
        g_it = load_graphs_LMdata()
    elif dataset == "CAS":
        print("Loading bursi(CAS) dataset")
        g_it = load_graphs_bursi()
    else:
        print("Unknown dataset name")
    return g_it
def convert_to_sparse_matrix(km):

        if len(km) == 0:
            raise Exception('ERROR: something went wrong, empty feature_dict. Perhaps wrong data format, i.e. do nodes have the "viewpoint" attribute?')
        row, col, data = [], [], []
        ne = len(km)
        for i in range(ne):
            for j in range(ne):
                if (km[i][j]!= 0):
                    row.append(i)
                    col.append(j)
                    data.append(km[i][j])
        print(len(row), len(col), len(data))

        X = csr_matrix((data, (row, col)), shape=(ne, ne))

        return X
def load_graphs_GDD():

    input_target_url = 'http://www.math.unipd.it/~nnavarin/datasets/GDD/GDD_labels.txt'
    input_data_url = 'http://www.math.unipd.it/~nnavarin/datasets/GDD/graphs.gspan'
    _target = load_target(input_target_url)
    g_it = instance_to_graph(input=input_data_url)
    gra = [i for i in g_it]
    print('Loaded GDD graph dataset for graph classification.')
    print(len(gra), 'graphs.')
    return Bunch(graphs=gra, target=_target, labels=True, veclabels=False)


"""def load_graphs_MUTAG():
    import urllib.request
    #from obabel import obabel_to_eden
    from eden.converter.molecule.obabel import obabel_to_eden
    input_target_url = 'http://www.math.unipd.it/~nnavarin/datasets/MUTAG/mutag_188_target.txt'
    input_data_url = 'http://www.math.unipd.it/~nnavarin/datasets/MUTAG/mutag_188_data.can'
    # input_data_string = str(urllib.request.urlopen(input_data_url).read())
    # print(input_data_string)
    # print(input_data_string)
    input_data_string = load_data(input_data_url)
    _target = load_target(input_target_url)
    g_it = obabel_to_eden(input_data_url, file_format='smi')

    gra=[i for i in g_it]
    print('Loaded MUTAG graph dataset for graph classification.')
    print(len(gra),'graphs.')
    return Bunch(graphs=gra, target=_target, labels=True, veclabels=False)"""
from urllib.request import urlopen
def load_data(url):
    with urlopen(url) as f:
        return f.read().decode('utf-8').strip()

def load_graphs_MUTAG():
    # from eden.converter.molecule.obabel import obabel_to_eden
    # input_target_url = 'http://www.math.unipd.it/~nnavarin/datasets/MUTAG/mutag_188_target.txt'
    # input_data_url = 'http://www.math.unipd.it/~nnavarin/datasets/MUTAG/mutag_188_data.can'
    #
    # _target = load_target(input_target_url)
    # input_data_string = load_data(input_data_url)
    # g_it = obabel_to_eden(input_data_string, file_format='smi')
    #
    # gra = [i for i in g_it]
    # print('Loaded MUTAG graph dataset for graph classification.')
    # print(len(gra), 'graphs.')
    # return Bunch(graphs=gra, target=_target, labels=True, veclabels=False)
    dataset = TUDataset(root='data/TUDataset', name='Mutag')
    graphs = []
    targets = []
    veclabels = []  # Để lưu trữ các nhãn của các nút trong mỗi đồ thị

    for graph in dataset:
        # Convert graph to NetworkX format
        g = to_networkx(graph)  # Chuyển đổi Data thành NetworkX

        # Tạo danh sách nhãn cho các nút trong đồ thị
        node_labels = []
        for node in range(graph.num_nodes):
            # Tạo nhãn cho mỗi nút, ví dụ: 'label_0', 'label_1', 'label_2', ...
            label = f'label_{node}'
            node_labels.append(label)

        # Gán danh sách nhãn cho các nút trong đồ thị
        for node, label in enumerate(node_labels):
            g.nodes[node]['label'] = label

        # Thêm danh sách nhãn vào danh sách veclabels
        veclabels.append(node_labels)

        # Sắp xếp các nút theo id của chúng
        node_order = sorted(g.nodes())

        # Thêm thuộc tính 'node_order' vào đồ thị
        g.graph['node_order'] = node_order

        graphs.append(g)

        # Lấy nhãn (label) của đồ thị
        target = graph.y.item()  # Giả sử mỗi đồ thị chỉ có một nhãn
        targets.append(target)

    return Bunch(graphs=graphs, target=targets, labels=True, veclabels=True)

def load_graphs_KKI():
    dataset = TUDataset(root='data/TUDataset', name='KKI')
    graphs = []
    targets = []
    veclabels = []  # Để lưu trữ các nhãn của các nút trong mỗi đồ thị

    for graph in dataset:
        # Convert graph to NetworkX format
        g = to_networkx(graph)  # Chuyển đổi Data thành NetworkX

        # Tạo danh sách nhãn cho các nút trong đồ thị
        node_labels = []
        for node in range(graph.num_nodes):
            # Tạo nhãn cho mỗi nút, ví dụ: 'label_0', 'label_1', 'label_2', ...
            label = f'label_{node}'
            node_labels.append(label)

        # Gán danh sách nhãn cho các nút trong đồ thị
        for node, label in enumerate(node_labels):
            g.nodes[node]['label'] = label

        # Thêm danh sách nhãn vào danh sách veclabels
        veclabels.append(node_labels)

        # Sắp xếp các nút theo id của chúng
        node_order = sorted(g.nodes())

        # Thêm thuộc tính 'node_order' vào đồ thị
        g.graph['node_order'] = node_order

        graphs.append(g)

        # Lấy nhãn (label) của đồ thị
        target = graph.y.item()  # Giả sử mỗi đồ thị chỉ có một nhãn
        targets.append(target)

    return Bunch(graphs=graphs, target=targets, labels=True, veclabels=True)

def load_graphs_proteins():
    # input_target_url = 'http://www.math.unipd.it/~nnavarin/datasets/PROTEINS.labels'
    # input_data_url = 'http://www.math.unipd.it/~nnavarin/datasets/PROTEINS.gspan'
    #
    # _target = load_target(input_target_url)
    # g_it = instance_to_graph(input=input_data_url)  # url
    #
    # print('Loaded PROTEINS graph dataset for graph classification from:')
    # print('Dobson, P. D., & Doig, A. J. (2003)')
    # print('Distinguishing enzyme structures from non-enzymes without alignments.')
    # print('Journal of Molecular Biology, 330, 771–783. doi:10.1016/S0022-2836(03)00628-4')
    # return Bunch(graphs=[i for i in g_it], target=_target, labels=True, veclabels=True)
    dataset = TUDataset(root='data/TUDataset', name='proteins')
    graphs = []
    targets = []
    veclabels = []  # Để lưu trữ các nhãn của các nút trong mỗi đồ thị

    for graph in dataset:
        # Convert graph to NetworkX format
        g = to_networkx(graph)  # Chuyển đổi Data thành NetworkX

        # Tạo danh sách nhãn cho các nút trong đồ thị
        node_labels = []
        for node in range(graph.num_nodes):
            # Tạo nhãn cho mỗi nút, ví dụ: 'label_0', 'label_1', 'label_2', ...
            label = f'label_{node}'
            node_labels.append(label)

        # Gán danh sách nhãn cho các nút trong đồ thị
        for node, label in enumerate(node_labels):
            g.nodes[node]['label'] = label

        # Thêm danh sách nhãn vào danh sách veclabels
        veclabels.append(node_labels)

        # Sắp xếp các nút theo id của chúng
        node_order = sorted(g.nodes())

        # Thêm thuộc tính 'node_order' vào đồ thị
        g.graph['node_order'] = node_order

        graphs.append(g)

        # Lấy nhãn (label) của đồ thị
        target = graph.y.item()  # Giả sử mỗi đồ thị chỉ có một nhãn
        targets.append(target)

    return Bunch(graphs=graphs, target=targets, labels=True, veclabels=True)
def load_graphs_PTC_MR():

    dataset = TUDataset(root='data/TUDataset', name='PTC_MR')
    graphs = []
    targets = []
    veclabels = []  # Để lưu trữ các nhãn của các nút trong mỗi đồ thị

    for graph in dataset:
        # Convert graph to NetworkX format
        g = to_networkx(graph)  # Chuyển đổi Data thành NetworkX

        # Tạo danh sách nhãn cho các nút trong đồ thị
        node_labels = []
        for node in range(graph.num_nodes):
            # Tạo nhãn cho mỗi nút, ví dụ: 'label_0', 'label_1', 'label_2', ...
            label = f'label_{node}'
            node_labels.append(label)

        # Gán danh sách nhãn cho các nút trong đồ thị
        for node, label in enumerate(node_labels):
            g.nodes[node]['label'] = label

        # Thêm danh sách nhãn vào danh sách veclabels
        veclabels.append(node_labels)

        # Sắp xếp các nút theo id của chúng
        node_order = sorted(g.nodes())

        # Thêm thuộc tính 'node_order' vào đồ thị
        g.graph['node_order'] = node_order

        graphs.append(g)

        # Lấy nhãn (label) của đồ thị
        target = graph.y.item()  # Giả sử mỗi đồ thị chỉ có một nhãn
        targets.append(target)

    return Bunch(graphs=graphs, target=targets, labels=True, veclabels=True)
def load_graphs_AIDS():
    from eden.converter.molecule.obabel import obabel_to_eden
    input_target_url = 'http://www.math.unipd.it/~nnavarin/datasets/AIDS/CAvsCM.y'
    input_data_url = 'http://www.math.unipd.it/~nnavarin/datasets/AIDS/CAvsCM.can'

    _target = load_target(input_target_url)
    input_data_string = load_data(input_data_url)
    print(input_data_string)
    g_it = obabel_to_eden(input_data_string, file_format='smi')

    gra = [i for i in g_it]
    print('Loaded MUTAG graph dataset for graph classification.')
    print(len(gra), 'graphs.')
    return Bunch(graphs=gra, target=_target, labels=True, veclabels=False)
def load_graphs_NCI109():
    """Load the NCI109 graph dataset for graph classification..

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_target_url='http://www.math.unipd.it/~nnavarin/datasets/NCI109/NCI109_labels.txt'
    input_data_url='http://www.math.unipd.it/~nnavarin/datasets/NCI109/NCI109_graphs.gspan'
    _target=load_target(input_target_url)
    g_it=instance_to_graph(input = input_data_url)

    print('Loaded NCI109 graph dataset for graph classification.')
    return Bunch(graphs=[i for i in g_it], target=_target, labels=True, veclabels=False)
def load_graphs_NCI1():
    """Load the NCI1 graph dataset for graph classification..

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_target_url='http://www.math.unipd.it/~nnavarin/datasets/NCI1/NCI1_labels.txt'
    input_data_url='http://www.math.unipd.it/~nnavarin/datasets/NCI1/NCI1_graphs.gspan'
    _target=load_target(input_target_url)
    label_dict={}
    g_it=instance_to_graph(input=input_data_url)
    print('Loaded NCI1 graph dataset for graph classification.')
    return Bunch(graphs=[i for i in g_it], target=_target, labels=True, veclabels=False)
def load_graphs_bursi():
    """Load the Bursi graph dataset for graph classification..

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_target_url = 'http://www.bioinf.uni-freiburg.de/~costa/bursi.target'
    input_data_url = 'http://www.bioinf.uni-freiburg.de/~costa/bursi.gspan'
    _target = load_target(input_target_url)
    g_it = instance_to_graph(input=input_data_url)

    print('Loaded Bursi graph dataset for graph classification.')
    return Bunch(graphs=[i for i in g_it], target=_target, labels=True, veclabels=False)


def load_graphs_enzymes():
    """Load the ENZYMES graph dataset for (multiclass) graph classification from:
    Schomburg, I., Chang, A., Ebeling, C., Gremse, M., Heldt, C., Huhn, G., & Schomburg, D. (2004).
    BRENDA, the enzyme database: updates and major new developments.
    Nucleic Acids Research, 32, D431–D433. doi:10.1093/nar/gkh081

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    # input_target_url = 'http://www.math.unipd.it/~nnavarin/datasets/ENZYMES.labels'
    # input_data_url = 'http://www.math.unipd.it/~nnavarin/datasets/ENZYMES.gspan'
    # # input_target_url='datasets/ENZYMES.labels'
    # # input_data_url='datasets/ENZYMES.gspan'
    #
    # _target = load_target(input_target_url)
    # g_it = instance_to_graph(input=input_data_url)  # url
    # # return Bunch(data=flat_data,
    # #            target=target.astype(np.int),
    # #           target_names=np.arange(10),
    # #            images=images,
    # #            DESCR=descr)
    # print('Loaded ENZYMES graph dataset for (multiclass) graph classification from:')
    # print('Schomburg, I., Chang, A., Ebeling, C., Gremse, M., Heldt, C., Huhn, G., & Schomburg, D. (2004).')
    # print('BRENDA, the enzyme database: updates and major new developments.')
    # print('Nucleic Acids Research, 32, D431–D433. doi:10.1093/nar/gkh081')
    # return Bunch(graphs=[i for i in g_it], target=_target, labels=True, veclabels=True)
    dataset = TUDataset(root='data/TUDataset', name='Enzymes')
    graphs = []
    targets = []
    veclabels = []  # Để lưu trữ các nhãn của các nút trong mỗi đồ thị

    for graph in dataset:
        # Convert graph to NetworkX format
        g = to_networkx(graph)  # Chuyển đổi Data thành NetworkX

        # Tạo danh sách nhãn cho các nút trong đồ thị
        node_labels = []
        for node in range(graph.num_nodes):
            # Tạo nhãn cho mỗi nút, ví dụ: 'label_0', 'label_1', 'label_2', ...
            label = f'label_{node}'
            node_labels.append(label)

        # Gán danh sách nhãn cho các nút trong đồ thị
        for node, label in enumerate(node_labels):
            g.nodes[node]['label'] = label

        # Thêm danh sách nhãn vào danh sách veclabels
        veclabels.append(node_labels)

        # Sắp xếp các nút theo id của chúng
        node_order = sorted(g.nodes())

        # Thêm thuộc tính 'node_order' vào đồ thị
        g.graph['node_order'] = node_order

        graphs.append(g)

        # Lấy nhãn (label) của đồ thị
        target = graph.y.item()  # Giả sử mỗi đồ thị chỉ có một nhãn
        targets.append(target)

    return Bunch(graphs=graphs, target=targets, labels=True, veclabels=True)



def load_graphs_synthetic():
    """Load the SYNTHETIC graph dataset for graph classification from:
        Feragen, A., Kasenburg, N., Petersen, J., de Bruijne, M., & Borgwardt, K. M. (2013)
        Scalable kernels for graphs with continuous attributes.
        In Neural Information Processing Systems (NIPS) 2013 (pp. 216–224).
        Retrieved from http://papers.nips.cc/paper/5155-scalable-kernels-for-graphs-with-continuous-attributes.pdf
    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    # input_target_url = 'http://www.math.unipd.it/~nnavarin/datasets/SYNTHETICnew.labels'
    # input_data_url = 'http://www.math.unipd.it/~nnavarin/datasets/SYNTHETICnew.gspan'
    # # input_target_url='datasets/ENZYMES.labels'
    # # input_data_url='datasets/ENZYMES.gspan'
    #
    # _target = load_target(input_target_url)
    # g_it = instance_to_graph(input=input_data_url)  # url
    # # return Bunch(data=flat_data,
    # #            target=target.astype(np.int),
    # #           target_names=np.arange(10),
    # #            images=images,
    # #            DESCR=descr)
    # g = [i for i in g_it]
    # for i in g:
    #     for n in i.nodes():
    #         i.nodes[n]['label'] = str(i.degree(n))
    #
    # print('Loaded SYNTHETIC graph dataset for graph classification from:')
    # print('Feragen, A., Kasenburg, N., Petersen, J., de Bruijne, M., & Borgwardt, K. M. (2013)')
    # print('Scalable kernels for graphs with continuous attributes.')
    # print('In Neural Information Processing Systems (NIPS) 2013 (pp. 216–224).')
    # return Bunch(graphs=g, target=_target, labels=True, veclabels=True)
    dataset = TUDataset(root='data/TUDataset', name='Mutag')
    graphs = []
    targets = []
    veclabels = []  # Để lưu trữ các nhãn của các nút trong mỗi đồ thị

    for graph in dataset:
        # Convert graph to NetworkX format
        g = to_networkx(graph)  # Chuyển đổi Data thành NetworkX

        # Tạo danh sách nhãn cho các nút trong đồ thị
        node_labels = []
        for node in range(graph.num_nodes):
            # Tạo nhãn cho mỗi nút, ví dụ: 'label_0', 'label_1', 'label_2', ...
            label = f'label_{node}'
            node_labels.append(label)

        # Gán danh sách nhãn cho các nút trong đồ thị
        for node, label in enumerate(node_labels):
            g.nodes[node]['label'] = label

        # Thêm danh sách nhãn vào danh sách veclabels
        veclabels.append(node_labels)

        # Sắp xếp các nút theo id của chúng
        node_order = sorted(g.nodes())

        # Thêm thuộc tính 'node_order' vào đồ thị
        g.graph['node_order'] = node_order

        graphs.append(g)

        # Lấy nhãn (label) của đồ thị
        target = graph.y.item()  # Giả sử mỗi đồ thị chỉ có một nhãn
        targets.append(target)

    return Bunch(graphs=graphs, target=targets, labels=True, veclabels=True)


def load_graphs_BZR():
    """Load the BZR graph dataset for graph classification from:
        Neumann, M., Garnett R., Bauckhage Ch., Kersting K.: Propagation Kernels: Efficient Graph
        Kernels from Propagated Information. Under review at MLJ.

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    # input_target_url = 'http://www.math.unipd.it/~nnavarin/datasets/BZR_graph_labels.txt'
    # input_data_url = 'http://www.math.unipd.it/~nnavarin/datasets/BZR.gspan'
    # # input_target_url='datasets/ENZYMES.labels'
    # # input_data_url='datasets/ENZYMES.gspan'
    #
    # _target = load_target(input_target_url)
    # g_it = instance_to_graph(input=input_data_url)  # url
    # # return Bunch(data=flat_data,
    # #            target=target.astype(np.int),
    # #           target_names=np.arange(10),
    # #            images=images,
    # #            DESCR=descr)
    # print('Loaded BZR graph dataset for  graph classification from:')
    # print('Neumann, M., Garnett R., Bauckhage Ch., Kersting K.: Propagation Kernels: Efficient Graph')
    # print('Kernels from Propagated Information. MLJ 2015.')
    # return Bunch(graphs=[i for i in g_it], target=_target, labels=True, veclabels=True)
    dataset = TUDataset(root='data/TUDataset', name='BZR')
    graphs = []
    targets = []
    veclabels = []  # Để lưu trữ các nhãn của các nút trong mỗi đồ thị

    for graph in dataset:
        # Convert graph to NetworkX format
        g = to_networkx(graph)  # Chuyển đổi Data thành NetworkX

        # Tạo danh sách nhãn cho các nút trong đồ thị
        node_labels = []
        for node in range(graph.num_nodes):
            # Tạo nhãn cho mỗi nút, ví dụ: 'label_0', 'label_1', 'label_2', ...
            label = f'label_{node}'
            node_labels.append(label)

        # Gán danh sách nhãn cho các nút trong đồ thị
        for node, label in enumerate(node_labels):
            g.nodes[node]['label'] = label

        # Thêm danh sách nhãn vào danh sách veclabels
        veclabels.append(node_labels)

        # Sắp xếp các nút theo id của chúng
        node_order = sorted(g.nodes())

        # Thêm thuộc tính 'node_order' vào đồ thị
        g.graph['node_order'] = node_order

        graphs.append(g)

        # Lấy nhãn (label) của đồ thị
        target = graph.y.item()  # Giả sử mỗi đồ thị chỉ có một nhãn
        targets.append(target)

    return Bunch(graphs=graphs, target=targets, labels=True, veclabels=True)

def load_graphs_COX2():
    """Load the COX2 graph dataset for graph classification from:
        Neumann, M., Garnett R., Bauckhage Ch., Kersting K.: Propagation Kernels: Efficient Graph
        Kernels from Propagated Information. Under review at MLJ.

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    # input_target_url = 'http://www.math.unipd.it/~nnavarin/datasets/COX2_graph_labels.txt'
    # input_data_url = 'http://www.math.unipd.it/~nnavarin/datasets/COX2.gspan'
    # # input_target_url='datasets/ENZYMES.labels'
    # # input_data_url='datasets/ENZYMES.gspan'
    #
    # _target = load_target(input_target_url)
    # g_it = instance_to_graph(input=input_data_url)  # url
    # # return Bunch(data=flat_data,
    # #            target=target.astype(np.int),
    # #           target_names=np.arange(10),
    # #            images=images,
    # #            DESCR=descr)
    # print('Loaded COX2 graph dataset for  graph classification from:')
    # print('Neumann, M., Garnett R., Bauckhage Ch., Kersting K.: Propagation Kernels: Efficient Graph')
    # print('Kernels from Propagated Information. MLJ 2015.')
    # return Bunch(graphs=[i for i in g_it], target=_target, labels=True, veclabels=True)



    dataset = TUDataset(root='data/TUDataset', name='COX2')
    graphs = []
    targets = []
    veclabels = []  # Để lưu trữ các nhãn của các nút trong mỗi đồ thị

    for graph in dataset:
        # Convert graph to NetworkX format
        g = to_networkx(graph)  # Chuyển đổi Data thành NetworkX

        # Tạo danh sách nhãn cho các nút trong đồ thị
        node_labels = []
        for node in range(graph.num_nodes):
            # Tạo nhãn cho mỗi nút, ví dụ: 'label_0', 'label_1', 'label_2', ...
            label = f'label_{node}'
            node_labels.append(label)

        # Gán danh sách nhãn cho các nút trong đồ thị
        for node, label in enumerate(node_labels):
            g.nodes[node]['label'] = label

        # Thêm danh sách nhãn vào danh sách veclabels
        veclabels.append(node_labels)

        # Sắp xếp các nút theo id của chúng
        node_order = sorted(g.nodes())

        # Thêm thuộc tính 'node_order' vào đồ thị
        g.graph['node_order'] = node_order

        graphs.append(g)

        # Lấy nhãn (label) của đồ thị
        target = graph.y.item()  # Giả sử mỗi đồ thị chỉ có một nhãn
        targets.append(target)

    return Bunch(graphs=graphs, target=targets, labels=True, veclabels=True)
def load_graphs_DHFR():
    """Load the DHFR graph dataset for graph classification from:
        Neumann, M., Garnett R., Bauckhage Ch., Kersting K.: Propagation Kernels: Efficient Graph
        Kernels from Propagated Information. Under review at MLJ.

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    # input_target_url = 'http://www.math.unipd.it/~nnavarin/datasets/DHFR_graph_labels.txt'
    # input_data_url = 'http://www.math.unipd.it/~nnavarin/datasets/DHFR.gspan'
    # # input_target_url='datasets/ENZYMES.labels'
    # # input_data_url='datasets/ENZYMES.gspan'
    #
    # _target = load_target(input_target_url)
    # g_it = instance_to_graph(input=input_data_url)  # url
    # # return Bunch(data=flat_data,
    # #            target=target.astype(np.int),
    # #           target_names=np.arange(10),
    # #            images=images,
    # #            DESCR=descr)
    # print('Loaded DHFR graph dataset for  graph classification from:')
    # print('Neumann, M., Garnett R., Bauckhage Ch., Kersting K.: Propagation Kernels: Efficient Graph')
    # print('Kernels from Propagated Information. MLJ 2015.')
    # return Bunch(graphs=[i for i in g_it], target=_target, labels=True, veclabels=True)
    dataset = TUDataset(root='data/TUDataset', name='DHFR')
    graphs = []
    targets = []
    veclabels = []  # Để lưu trữ các nhãn của các nút trong mỗi đồ thị

    for graph in dataset:
        # Convert graph to NetworkX format
        g = to_networkx(graph)  # Chuyển đổi Data thành NetworkX

        # Tạo danh sách nhãn cho các nút trong đồ thị
        node_labels = []
        for node in range(graph.num_nodes):
            # Tạo nhãn cho mỗi nút, ví dụ: 'label_0', 'label_1', 'label_2', ...
            label = f'label_{node}'
            node_labels.append(label)

        # Gán danh sách nhãn cho các nút trong đồ thị
        for node, label in enumerate(node_labels):
            g.nodes[node]['label'] = label

        # Thêm danh sách nhãn vào danh sách veclabels
        veclabels.append(node_labels)

        # Sắp xếp các nút theo id của chúng
        node_order = sorted(g.nodes())

        # Thêm thuộc tính 'node_order' vào đồ thị
        g.graph['node_order'] = node_order

        graphs.append(g)

        # Lấy nhãn (label) của đồ thị
        target = graph.y.item()  # Giả sử mỗi đồ thị chỉ có một nhãn
        targets.append(target)

    return Bunch(graphs=graphs, target=targets, labels=True, veclabels=True)

def load_graphs_DD():

    dataset = TUDataset(root='data/TUDataset', name='DD')
    graphs = []
    targets = []
    veclabels = []  # Để lưu trữ các nhãn của các nút trong mỗi đồ thị

    for graph in dataset:
        # Convert graph to NetworkX format
        g = to_networkx(graph)  # Chuyển đổi Data thành NetworkX

        # Tạo danh sách nhãn cho các nút trong đồ thị
        node_labels = []
        for node in range(graph.num_nodes):
            # Tạo nhãn cho mỗi nút, ví dụ: 'label_0', 'label_1', 'label_2', ...
            label = f'label_{node}'
            node_labels.append(label)

        # Gán danh sách nhãn cho các nút trong đồ thị
        for node, label in enumerate(node_labels):
            g.nodes[node]['label'] = label

        # Thêm danh sách nhãn vào danh sách veclabels
        veclabels.append(node_labels)

        # Sắp xếp các nút theo id của chúng
        node_order = sorted(g.nodes())

        # Thêm thuộc tính 'node_order' vào đồ thị
        g.graph['node_order'] = node_order

        graphs.append(g)

        # Lấy nhãn (label) của đồ thị
        target = graph.y.item()  # Giả sử mỗi đồ thị chỉ có một nhãn
        targets.append(target)

    return Bunch(graphs=graphs, target=targets, labels=True, veclabels=True)
def load_graphs_PROTEINS_full():
    """Load the PROTEINS_full graph dataset for graph classification from:
        Neumann, M., Garnett R., Bauckhage Ch., Kersting K.: Propagation Kernels: Efficient Graph
        Kernels from Propagated Information. Under review at MLJ.

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_target_url = 'http://www.math.unipd.it/~nnavarin/datasets/PROTEINS_full_graph_labels.txt'
    input_data_url = 'http://www.math.unipd.it/~nnavarin/datasets/PROTEINS_full.gspan'
    # input_target_url='datasets/ENZYMES.labels'
    # input_data_url='datasets/ENZYMES.gspan'

    _target = load_target(input_target_url)
    g_it = instance_to_graph(input=input_data_url)  # url
    # return Bunch(data=flat_data,
    #            target=target.astype(np.int),
    #           target_names=np.arange(10),
    #            images=images,
    #            DESCR=descr)
    print('Loaded PROTEINS_full graph dataset for  graph classification from:')
    print('Neumann, M., Garnett R., Bauckhage Ch., Kersting K.: Propagation Kernels: Efficient Graph')
    print('Kernels from Propagated Information. MLJ 2015.')
    return Bunch(graphs=[i for i in g_it], target=_target, labels=True, veclabels=True)
def load_graphs_NCI_AIDS():
    """Load the NCI antiHIV graph dataset for graph classification..

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_target_url='http://www.math.unipd.it/~nnavarin/datasets/NCI_AIDS/AIDO99SD_numeric.labels'
    input_data_url='http://www.math.unipd.it/~nnavarin/datasets/NCI_AIDS/AIDO99SD.gspan'
    _target=load_target(input_target_url)
    g_it=instance_to_graph(input = input_data_url)

    print('Loaded NCI antiHIV dataset graph dataset for graph classification.')
    return Bunch(graphs=[i for i in g_it], target=_target, labels=True, veclabels=False)
def load_graphs_LMdata():
    """Load the LMdata graph dataset for graph classification..

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes :
        'graphs', the graphs in the dataset in Networkx format,  'target', the classification labels for each
        sample.
    """
    input_target_url='http://www.math.unipd.it/~nnavarin/datasets/LMdata/labels.txt.standardized'
    input_data_url='http://www.math.unipd.it/~nnavarin/datasets//LMdata/graphs.gspan.standardized'
    _target=load_target(input_target_url)
    label_dict={}
    counter=[1]
    g_it=instance_to_graph(input_data_url,label_dict,counter)

    print('Loaded LMdata graph dataset for graph classification.')
    return Bunch(graphs=[i for i in g_it], label_dict=label_dict, target=_target, labels=True, veclabels=False)
def load_graphs_CPDB():
    from eden.converter.molecule.obabel import obabel_to_eden
    input_target_url = 'http://www.math.unipd.it/~nnavarin/datasets/CPDB/mutagen_labels.tab'
    input_data_url = 'http://www.math.unipd.it/~nnavarin/datasets/CPDB/mutagen_smile.can'

    _target = load_target(input_target_url)
    input_data_string = load_data(input_data_url)
    g_it = obabel_to_eden(input_data_string, file_format='smi')

    gra = [i for i in g_it]
    print('Loaded MUTAG graph dataset for graph classification.')
    print(len(gra), 'graphs.')
    return Bunch(graphs=gra, target=_target, labels=True, veclabels=False)