# generalized_aggregation
Graph Neural Networks (GNNs) are powerful tools for modeling graph structures, achieving high performance in various fields. Most GNNs are based on Message-Passing Neural Networks (MPNNs), updating node representations iteratively but only using information from k-hop neighbors.

This project proposes a new aggregation method that combines information from both neighboring nodes and distant nodes with high similarity, aiming to improve the performance of existing MPNN models. Evaluation results on multiple datasets demonstrate the potential of this method.

Dependencies

Compatible with PyTorch 1.12.1 and Python 3.9.13.

Dataset:

This project utilizes the PTC, MUTAG, NCI1, DHFR, COX2, Proteins, and DD datasets for evaluation. These datasets are directly fetched from the TUDataset, a rich collection of graph datasets for research in Graph Neural Networks (GNNs).

Data Retrieval

These datasets are automatically downloaded and preprocessed via TUDataset. You do not need to manually download them; the code will handle this automatically when you run the training commands.


