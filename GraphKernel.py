"""from torch_geometric.datasets import TUDataset
import networkx as nx
from grakel.kernels import WeisfeilerLehman
from sklearn.metrics.pairwise import cosine_similarity

# Load the MUTAG dataset
dataset = TUDataset(root='data/TUDataset', name='MUTAG')

# Convert PyTorch Geometric graphs to networkx graphs
graphs = []
for data in dataset:
    # Convert PyTorch Geometric data to a networkx graph
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    edge_index = data.edge_index.t().tolist()
    graph.add_edges_from(edge_index)
    graphs.append(graph)

print(f'Number of graphs: {len(graphs)}')

# Khởi tạo đối tượng Weisfeiler-Lehman kernel với n_iter là một số nguyên dương
wl_kernel = WeisfeilerLehman(n_iter=1)  # Đặt n_iter là một số nguyên dương tùy chọn

# Tính toán biểu diễn của các đỉnh sau khi cập nhật
X_wl = wl_kernel.fit_transform(graphs)

# Khởi tạo độ tương đồng giữa các đỉnh
similarity_matrices = []

# Lặp qua từng đồ thị trong danh sách graphs
for i, graph in enumerate(graphs):
    # Lấy ma trận biểu diễn của các đỉnh đã được cập nhật
    transformed_graph = wl_kernel.transform([graph])

    # Tính độ tương đồng giữa các đỉnh dựa trên biểu diễn sau cập nhật
    similarity_matrix = cosine_similarity(transformed_graph[0])

    # Thêm ma trận độ tương đồng vào danh sách
    similarity_matrices.append(similarity_matrix)

# Kết quả là danh sách các ma trận tương đồng giữa các đỉnh trong từng đồ thị
print(similarity_matrices)"""
import urllib.request
import networkx as nx
from openbabel import openbabel, pybel

def can_to_networkx(can_url):
    """
    Chuyển đổi dữ liệu từ tập tin CAN thành đồ thị mạng NetworkX.

    Parameters:
    can_url (str): Đường dẫn đến tập tin CAN.

    Returns:
    nx.Graph: Đồ thị mạng NetworkX.
    """
    # Đọc dữ liệu từ URL
    with urllib.request.urlopen(can_url) as f:
        can_data = f.read().decode('utf-8')

    # Tạo đồ thị mạng NetworkX
    G = nx.Graph()

    # Tạo bộ chuyển đổi SMILES thành đồ thị mạng NetworkX
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("smi", "mol")

    # Duyệt qua các dòng trong tập tin CAN
    for line in can_data.split('\n'):
        # Loại bỏ khoảng trắng và dấu cách thừa
        line = line.strip()
        if line:
            # Chuyển đổi chuỗi SMILES thành đồ thị mạng NetworkX
            mol = pybel.readstring("smi", line.strip())
            graph = obabel_to_networkx(mol)
            G.add_nodes_from(graph.nodes(data=True))
            G.add_edges_from(graph.edges())

    return G

def obabel_to_networkx(mol):
    """
    Chuyển đổi một đối tượng mol của Pybel thành đồ thị mạng NetworkX.

    Parameters:
    mol: Đối tượng mol của Pybel.

    Returns:
    nx.Graph: Đồ thị mạng NetworkX.
    """
    G = nx.Graph()

    for atom in mol:
        atom_id = atom.idx - 1
        atom_label = str(atom.type)
        G.add_node(atom_id, label=atom_label)

    for bond in openbabel.OBMolBondIter(mol.OBMol):
        bond_label = str(bond.GetBO())
        atom1_id = bond.GetBeginAtomIdx() - 1
        atom2_id = bond.GetEndAtomIdx() - 1
        G.add_edge(atom1_id, atom2_id, label=bond_label)

    return G

# URL của tập tin CAN của dữ liệu MUTAG
input_data_url = 'http://www.math.unipd.it/~nnavarin/datasets/MUTAG/mutag_188_data.can'

# Chuyển đổi tập tin CAN thành đồ thị mạng NetworkX
graph = can_to_networkx(input_data_url)

# In thông tin về đồ thị mạng
print("NetworkX Graph Info:")
print("Number of nodes:", graph.number_of_nodes())
print("Number of edges:", graph.number_of_edges())