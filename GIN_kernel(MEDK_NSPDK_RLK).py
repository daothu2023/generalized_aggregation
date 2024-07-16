
from kernel_NSPDK import NSPDK
from kernel_RLK import get_RLK
from kernel_MEDK import get_MEDK
from create_graphs import create_graphs
from adjacency_matrices import get_adjacency_matrices
import argparse
from sklearn.model_selection import KFold, StratifiedKFold
import torch
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import MLP, GINConv, global_add_pool, global_mean_pool
from torch_geometric.datasets import TUDataset
import torch.nn.functional as F
import copy
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from Add_egdes import add_edges
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# Load dataset
dataset = TUDataset(root='data/TUDataset', name='COX2')
# Get the adjacency matrices for dataset
list_adjacency_matrices = get_adjacency_matrices(dataset)
list_kernel = []
d = 4
r = 4
for AM in list_adjacency_matrices:
  # G = create_graphs(AM)
  # K = NSPDK(G, d, r)
  K = get_RLK(AM, alpha=4.)
  # K = get_MEDK(AM, beta=0.04)
  list_kernel.append(K)
torch.manual_seed(42)
dataset5 = dataset.shuffle()
# Lấy thứ tự mới của dataset5
new_order = np.random.permutation(len(dataset5)).tolist()
top_k = 0
dataset1 = dataset
new_dataset1 = []
for idx in new_order:
    similarity_matrix = list_kernel[idx]
    new_data = dataset[idx]
    new_edges = add_edges(dataset[idx].edge_index, dataset[idx].num_nodes, similarity_matrix, top_k)
    if new_edges.size(1) > 0:  # Kiểm tra nếu new_edges không rỗng
        new_data.edge_index = torch.cat([new_data.edge_index, new_edges], dim=-1)
    new_dataset1.append(new_data)
import random
for run in range(10):
    print(f'Run {run + 1}')
    new_dataset = Batch.from_data_list(new_dataset1)
    # new_dataset.num_classes = dataset5.num_classes
    # new_dataset.num_features = dataset5.num_features

    # Trộn ngẫu nhiên dữ liệu new_dataset
    indices = torch.randperm(len(new_dataset))
    new_dataset = new_dataset[indices]
    new_dataset = Batch.from_data_list(new_dataset)
    new_dataset.num_classes = dataset5.num_classes
    new_dataset.num_features = dataset5.num_features


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='COX2')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_channels', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--wandb', action='store_true', help='Track experiment')
    # args = parser.parse_args()
    # Parse known arguments and ignore unknown
    args, unknown = parser.parse_known_args()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS is currently slower than CPU due to missing int64 min/max ops
        device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    init_wandb(
        name=f'GIN-{args.dataset}',
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        device=device,
    )

    # train_loader = DataLoader(new_dataset[:int(len(new_dataset) * 0.9)], args.batch_size, shuffle=True)
    # test_loader = DataLoader(new_dataset[int(len(new_dataset) * 0.9):], args.batch_size)

    class GIN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
            super().__init__()

            self.convs = torch.nn.ModuleList()
            for _ in range(num_layers):
                mlp = MLP([in_channels, hidden_channels, hidden_channels])
                self.convs.append(GINConv(nn=mlp, train_eps=False))
                in_channels = hidden_channels

            self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                          norm=None, dropout=0.5)

        def forward(self, x, edge_index, batch):
            for conv in self.convs:
                x = conv(x, edge_index).relu()
            x = global_add_pool(x, batch)
            return self.mlp(x)


    model = GIN(
        in_channels=new_dataset.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=new_dataset.num_classes,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Chuyển mô hình sang GPU
    def train():
        model.train()

        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * data.num_graphs
        return total_loss / len(train_loader.dataset)


    @torch.no_grad()
    def validate(loader):
        model.eval()
        total_loss = 0

        total_correct = 0
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=-1)
            total_correct += int((pred == data.y).sum())
            loss = F.cross_entropy(out, data.y)
            total_loss += float(loss) * data.num_graphs
        return total_loss/len(loader.dataset), total_correct / len(loader.dataset)

    # batch_size = 32
    # hidden_channels=32
    # Số lần lặp (số fold)
    seed = 42
    np.random.seed(seed)

    num_splits = 10
    kf = StratifiedKFold(n_splits=num_splits, shuffle=True)

    # Initialize list to store test accuracies
    test_accuracies = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(new_dataset, new_dataset.y)):
        print(f'Fold {fold_idx + 1}/{num_splits}')
        # Convert train_idx and test_idx to list
        train_idx = train_idx.tolist()
        test_idx = test_idx.tolist()

        # Split dataset into train and test sets
        train_dataset = new_dataset[train_idx]
        test_dataset = new_dataset[test_idx]

        # Further split train dataset into train and validation sets
        num_train = len(train_dataset)
        split_idx = int(0.8 * num_train)
        train_data, val_data = train_dataset[:split_idx], train_dataset[split_idx:]

        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # Initialize model, optimizer, and loss function
        model = GIN(
            in_channels=new_dataset.num_features,
            hidden_channels=args.hidden_channels,
            out_channels=new_dataset.num_classes,
            num_layers=args.num_layers,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Initialize variables to store best model and its performance
        best_model = None
        best_validation_loss = float('inf')
        patience = 0
        min_validation_loss = float('inf')

        # Training loop
        for epoch in range(1, 100):
            train_loss = train()
            validation_loss, validation_acc = validate(test_loader)
            print(f'Epoch: {epoch}, Train Loss: {train_loss}, Validation Loss: {validation_loss}, Validation Acc: {validation_acc}')

            # Check if validation loss has improved
            if validation_loss < min_validation_loss:
                best_model = copy.deepcopy(model.state_dict())
                min_validation_loss = validation_loss
                patience = 0
            else:
                patience += 1

            # Early stopping if validation loss does not improve for 5 epochs
            if patience >= 30:
                print("Early stopping at epoch", epoch)
                break

        print(f'Finished training for fold {fold_idx + 1}/{num_splits}')

        # Load best model
        model.load_state_dict(best_model)

        # Test best model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                outputs = model(data.x, data.edge_index, data.batch)
                _, predicted = torch.max(outputs, 1)
                total += data.y.size(0)
                correct += (predicted == data.y).sum().item()

        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)
        print(f'Test Accuracy of the network on fold {fold_idx + 1}: {test_accuracy:.2f}%')

    # Calculate average test accuracy
    avg_test_accuracy = sum(test_accuracies) / num_splits
    print(f'Average Test Accuracy: {avg_test_accuracy:.2f}%')