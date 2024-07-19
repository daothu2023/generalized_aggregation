import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
import numpy as np
import copy
from torch_geometric.data import Data, InMemoryDataset

dataset_name = 'PROTEINS'
# Function to perform the required matrix operations and convert back to edge_index
def add_edge_index1(data):
    adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
    B = torch.matmul(adj, adj)
    B = B.numpy()
    n = B.shape[0]
    mask = np.ones((n, n)) - np.identity(n)
    C = np.multiply(B, mask)
    # Convert C back to edge_index format
    edge_index1, _ = dense_to_sparse(torch.tensor(C))
    # Add the new edge_index to the data object
    data.edge_index1 = edge_index1
    return data

# Define a custom dataset class
class CustomDataset(InMemoryDataset):
    def __init__(self, root, data_list, transform=None, pre_transform=None):
        self.data_list = data_list
        super(CustomDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = self.collate(self.data_list)
    def get(self, idx):
        return self.data_list[idx]
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return []
    def process(self):
        pass



# custom_dataset = DataLoader(custom_dataset, batch_size=32, shuffle=True)

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels * 2, hidden_channels * 2)
        self.fc = nn.Linear(hidden_channels * 4, num_classes)

    def forward(self, x, edge_index1, edge_index2, batch):
        x1 = self.conv1(x, edge_index1)
        x1 = F.relu(x1)
        x2 = self.conv1(x, edge_index2)
        x2 = F.relu(x2)
        x = torch.hstack((x1, x2))
        x1 = self.conv2(x, edge_index1)
        x1 = F.relu(x1)
        x2 = self.conv2(x, edge_index2)
        x2 = F.relu(x2)
        x = torch.hstack((x1, x2))
        x = global_mean_pool(x, batch)  # Global mean pooling
        x = self.fc(x)
        return x

# Set up training parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_channels = 32
num_epochs = 100
patience = 30
batch_size = 32
lr = 0.001
weight_decay = 1e-5

# Training function
def train(loader, criterion):
    model.train()
    total_loss = 0
    correct = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_index1, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# Validation function
def validate(loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_index1, data.batch)
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
        out = model(data.x, data.edge_index, data.edge_index1, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

# Load the dataset
dataset = TUDataset(root='data/TUDataset', name= dataset_name).shuffle()
print(dataset[0].num_nodes)
# Apply the function to all graphs in the dataset
processed_data_list = [add_edge_index1(data) for data in dataset]
# Create the custom dataset
custom_dataset = CustomDataset(root='/tmp/Custom' + dataset_name, data_list=processed_data_list)
print('------------------------')
custom_dataset = custom_dataset.shuffle()
print(custom_dataset[0].num_nodes)
print(custom_dataset[1].num_nodes)
custom_dataset = custom_dataset.shuffle()
print(custom_dataset[0].num_nodes)
print(custom_dataset[1].num_nodes)
print('------------------------')
# Verify the result for the first graph
print(custom_dataset[0])
list_all_acc = []
for run in range(5):
    print(f'Run {run + 1}')
    custom_dataset = custom_dataset.shuffle()
    print(custom_dataset[0].num_nodes)
    print(custom_dataset[1].num_nodes)

    num_splits = 10
    kf = StratifiedKFold(n_splits=num_splits, shuffle=True)

    # Initialize list to store test accuracies
    test_accuracies = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(custom_dataset, custom_dataset.data.y)):
        print(f'Fold {fold_idx + 1}/{num_splits}')
        # Convert train_idx and test_idx to list
        train_idx = train_idx.tolist()
        test_idx = test_idx.tolist()

        # Split dataset into train and test sets
        train_dataset = custom_dataset[train_idx]
        test_dataset = custom_dataset[test_idx]

        # Further split train dataset into train and validation sets
        num_train = len(train_dataset)
        split_idx = int(0.8 * num_train)
        train_data, val_data = train_dataset[:split_idx], train_dataset[split_idx:]

        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model, optimizer, and loss function
        model = GCN(custom_dataset.num_node_features, hidden_channels,custom_dataset.num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        # Initialize variables to store best model and its performance
        best_model = None
        best_validation_loss = float('inf')
        patience = 0
        min_validation_loss = float('inf')

        # Training loop
        for epoch in range(1, num_epochs + 1):
            train_loss = train(train_loader, criterion)
            validation_loss, validation_acc = validate(val_loader, criterion)
            print(f'Epoch: {epoch}, Train Loss: {train_loss}, Validation Loss: {validation_loss}, Validation Acc: {validation_acc}')

            # Check if validation loss has improved
            if validation_loss < min_validation_loss:
                best_model = copy.deepcopy(model.state_dict())
                min_validation_loss = validation_loss
                patience = 0
            else:
                patience += 1

            # Early stopping if validation loss does not improve for 'patience' epochs
            if patience >= 30:
                print("Early stopping at epoch", epoch)
                break

        print(f'Finished training for fold {fold_idx + 1}/{num_splits}')

        # Load best model
        model.load_state_dict(best_model)

        # Test best model
        test_accuracy = test(test_loader)
        test_accuracies.append(test_accuracy)
        print(f'Test Accuracy of the network on fold {fold_idx + 1}: {test_accuracy:.2f}%')
        list_all_acc.append(test_accuracy)

    # Calculate average test accuracy
    avg_test_accuracy = sum(test_accuracies) / num_splits
    print(f'Average Test Accuracy: {avg_test_accuracy:.2f}%')
for acc in list_all_acc:
  print(acc)
