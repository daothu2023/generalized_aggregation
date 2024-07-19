# generalized_aggregation
Graph Neural Networks (GNNs) are powerful tools for modeling graph structures, achieving high performance in various fields. Most GNNs are based on Message-Passing Neural Networks (MPNNs), updating node representations iteratively but only using information from k-hop neighbors.

This project proposes a new aggregation method that combines information from both neighboring nodes and distant nodes with high similarity, aiming to improve the performance of existing MPNN models. Evaluation results on multiple datasets demonstrate the potential of this method.

**Dependencies**

Compatible with PyTorch 1.12.1 and Python 3.9.13.

**Dataset**

This project utilizes the PTC, MUTAG, NCI1, DHFR, COX2, Proteins, and DD datasets for evaluation. These datasets are directly fetched from the TUDataset, a rich collection of graph datasets for research in Graph Neural Networks (GNNs).

**Data Retrieval**

These datasets are automatically downloaded and preprocessed via TUDataset. You do not need to manually download them; the code will handle this automatically when you run the training commands.

**Training model**

To train the models, you can use the following scripts corresponding to different methods:

- **GAT_kfold.py**: Runs the standard Graph Attention Network (GAT) method with k-fold cross-validation.
- **GCN_kfold.py**: Runs the standard Graph Convolutional Network (GCN) method with k-fold cross-validation.
- **GIN_kfold.py**: Runs the standard Graph Isomorphism Network (GIN) method with k-fold cross-validation.
- **GCN_kfold_WL.py**: Runs the GCN method with Weisfeiler-Lehman (WL) kernel with k-fold cross-validation.
- **GIN_kfold_WL.py**: Runs the GIN method with Weisfeiler-Lehman (WL) kernel with k-fold cross-validation.
- **GAT_kfold_WL.py**: Runs the GAT method with Weisfeiler-Lehman (WL) kernel with k-fold cross-validation.
- **GCN_kernel(MEDK_NSPDK_RLK).py**: Runs the GCN method with MEDK, NSPDK, and RLK kernels.
- **GIN_kernel(MEDK_NSPDK_RLK).py**: Runs the GIN method with MEDK, NSPDK, and RLK kernels.
- **GAT_kernel(MEDK_NSPDK_RLK).py**: Runs the GAT method with MEDK, NSPDK, and RLK kernels.
- **PGC_GCN.py**: Runs the PGC_GCN method.
