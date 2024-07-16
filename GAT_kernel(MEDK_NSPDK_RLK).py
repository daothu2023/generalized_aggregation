
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
from torch_geometric.nn import GATConv, global_mean_pool
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
new_dataset = Batch.from_data_list(new_dataset1)
indices = torch.randperm(len(new_dataset))
new_dataset = new_dataset[indices]
new_dataset = Batch.from_data_list(new_dataset)
new_dataset.num_classes = dataset5.num_classes
new_dataset.num_features = dataset5.num_features
# Define the GAT model
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=8, concat=True)
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=1, concat=True)
        self.fc = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = global_mean_pool(x, batch)  # Global mean pooling
        x = self.fc(x)
        return x
# Set up training parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_channels = 128
num_epochs = 100
patience = 30
batch_size = 32
lr = 0.01
weight_decay = 1e-5
# Training function
def train(loader):
    model.train()
    total_loss = 0
    correct = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return total_loss / len(loader.dataset), correct / len(loader.dataset)
# Validation function
def validate(loader):
    model.eval()
    total_loss = 0
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# Testing function
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

# seed = 42
# np.random.seed(seed)
for run in range(5):
    print(f'Run {run + 1}')
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
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # Initialize model, optimizer, and loss function
        model = GAT(new_dataset.num_node_features, hidden_channels=hidden_channels, num_classes=dataset.num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        # Initialize variables to store best model and its performance
        best_model = None
        best_validation_loss = float('inf')
        patience = 0
        min_validation_loss = float('inf')

        # Training loop
        for epoch in range(1, 150):
            train_loss = train(train_loader)
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
            if patience >= 40:
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
