import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, to_undirected
from torch_geometric.datasets import (Planetoid, Actor, HeterophilousGraphDataset, LINKXDataset,
                                      AttributedGraphDataset, EllipticBitcoinDataset)
from torch_geometric.transforms import NormalizeFeatures
from utils.utils import set_seed, home_folder
from datasets.BGP.bgp import BGP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
set_seed(seed)


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def fixed_geom_gcn_data_splits(dataset_name, num_nodes):
    fixed_splits = 10
    train_mask = torch.zeros((num_nodes, fixed_splits), dtype=torch.bool)
    val_mask = torch.zeros((num_nodes, fixed_splits), dtype=torch.bool)
    test_mask = torch.zeros((num_nodes, fixed_splits), dtype=torch.bool)

    for idx in range(fixed_splits):
        path = f"{home_folder()}/repos/label-diffusion-rewiring"
        splits_file_path = f"{path}/splits/fixed_geom-gcn_splits/{dataset_name}_split_0.6_0.2_{str(idx)}.npz"
        with np.load(splits_file_path) as splits_file:
            train_mask_split = splits_file["train_mask"]
            val_mask_split = splits_file["val_mask"]
            test_mask_split = splits_file["test_mask"]
        train_mask[:, idx] = torch.BoolTensor(train_mask_split)
        val_mask[:, idx] = torch.BoolTensor(val_mask_split)
        test_mask[:, idx] = torch.BoolTensor(test_mask_split)

    return train_mask, val_mask, test_mask

def basic_random_splits(num_nodes, splits=100):
    train_ratio = 0.6
    val_ratio = 0.2

    train_mask = torch.zeros((num_nodes, splits), dtype=torch.bool)
    val_mask = torch.zeros((num_nodes, splits), dtype=torch.bool)
    test_mask = torch.zeros((num_nodes, splits), dtype=torch.bool)

    for idx in range(splits):

        indices = torch.arange(num_nodes)
        indices = indices[torch.randperm(indices.size(0))]  # Shuffle indices
        train_size = int(num_nodes * train_ratio)
        train_index = indices[:train_size]
        val_size = int(num_nodes * val_ratio)
        val_index = indices[train_size:train_size + val_size]
        test_index = indices[train_size + val_size:]

        # Create masks
        train_mask[:, idx] = index_to_mask(train_index, size=num_nodes)
        val_mask[:, idx] = index_to_mask(val_index, size=num_nodes)
        test_mask[:, idx] = index_to_mask(test_index, size=num_nodes)

    return train_mask, val_mask, test_mask


def load_data(datasetname, num_train_per_class=20, num_val=500):
    path = f"{home_folder()}/repos/data/{datasetname}"
    # datasets with official splits
    if datasetname in ['Cora', 'Citeseer', 'Pubmed', 'Roman-empire', 'Actor', 'Wiki', 'BlogCatalog',
                       'EllipticBitcoinDataset', 'genius', 'BGP']:
        if datasetname in ['Cora', 'Citeseer', 'Pubmed']:
            dataset = Planetoid(root=path, name=datasetname)
            train_mask, val_mask, test_mask = basic_random_splits(dataset[0].y.size(0), splits=5)
            data = Data(x=dataset[0].x, edge_index=dataset[0].edge_index, y=dataset[0].y,
                        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        elif datasetname in ['Roman-empire']:
            dataset = HeterophilousGraphDataset(root=path, name=datasetname, transform=NormalizeFeatures())  # these datasets features are not categorical so maybe add here transform=NormalizeFeatures()
            data = dataset[0]
        elif datasetname in ['genius']:
            dataset = LINKXDataset(root=path, name=datasetname, transform=NormalizeFeatures())
            train_mask, val_mask, test_mask = basic_random_splits(dataset[0].y.size(0), splits=5)
            data = Data(x=dataset[0].x, edge_index=dataset[0].edge_index, y=dataset[0].y,
                        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        elif datasetname in ['Actor']:  # this one is called film in ACM, these splits are actually the geom-gcn splits
            dataset = Actor(root=path)
            data = dataset[0]
        elif datasetname in ['Wiki']:
            dataset = AttributedGraphDataset(root=path, name=datasetname, transform=NormalizeFeatures())
            train_mask, val_mask, test_mask = basic_random_splits(dataset[0].y.size(0), splits=5)
            data = Data(x=dataset[0].x, edge_index=dataset[0].edge_index, y=dataset[0].y,
                        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        elif datasetname in ['BlogCatalog']:
            dataset = AttributedGraphDataset(root=path, name=datasetname)
            train_mask, val_mask, test_mask = basic_random_splits(dataset[0].y.size(0), splits=5)
            data = Data(x=dataset[0].x, edge_index=dataset[0].edge_index, y=dataset[0].y,
                        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        elif datasetname in ['EllipticBitcoinDataset']:
            dataset = EllipticBitcoinDataset(root=path)
            train_mask, val_mask, test_mask = basic_random_splits(dataset[0].y.size(0), splits=5)
            data = Data(x=dataset[0].x, edge_index=dataset[0].edge_index, y=dataset[0].y,
                        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        elif datasetname in ['BGP']:
            # taken from https://github.com/susheels/gnns-and-local-assortativity/tree/main
            dataset = BGP(root=path)
            train_mask, val_mask, test_mask = basic_random_splits(dataset[0].y.size(0), splits=5)
            data = Data(x=dataset[0].x, edge_index=dataset[0].edge_index, y=dataset[0].y,
                        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

        if datasetname == 'EllipticBitcoinDataset':
            num_features = dataset[0].x.size(1)
            num_classes = 3  # I found a bug in torch-geometric, dataset.num_classes set to 2 even though it should be 3
        else:
            num_features = dataset.num_features
            num_classes = dataset.num_classes

        data = data.to(device)
        print(data)
        num_train_nodes = data.train_mask.sum().item()
        num_val_nodes = data.val_mask.sum().item()
        num_test_nodes = data.test_mask.sum().item()

    # I took all the heterophilous from here:
    # https://github.com/yandex-research/heterophilous-graphs/tree/main
    # datasets with the 48-30-20 splits (geom-gcn)
    elif datasetname in ['cornell', 'texas', 'wisconsin', 'chameleon', 'squirrel', 'chameleon_filtered', 'squirrel_filtered']:
        path = f"{home_folder()}/repos/label-diffusion-rewiring/heterophilous-graphs-data"
        file_name = f"{datasetname}.npz"
        filepath = os.path.join(path, file_name)
        data = np.load(filepath)
        print("Converting to PyG dataset...")
        x = torch.tensor(data['node_features'], dtype=torch.float)
        y = torch.tensor(data['node_labels'], dtype=torch.long)
        edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
        train_mask = torch.tensor(data['train_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
        val_mask = torch.tensor(data['val_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
        test_mask = torch.tensor(data['test_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
        num_classes = len(torch.unique(y))
        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        data.num_classes = num_classes

        print("Splitting datasets train/val/test...")
        if '_filtered' not in datasetname:
            train_mask, val_mask, test_mask = fixed_geom_gcn_data_splits(datasetname, data.y.size(0))

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        data = data.to(device)
        print(data)
        num_train_nodes = data.train_mask.sum().item()
        num_val_nodes = data.val_mask.sum().item()
        num_test_nodes = data.test_mask.sum().item()
        num_features = data.num_features
        num_classes = data.num_classes

    else:
        raise ValueError(f"Dataset {datasetname} not found")

    if is_undirected(data.edge_index) is False:
        data.edge_index = to_undirected(data.edge_index)

    return data, num_classes, num_features, num_train_nodes, num_test_nodes, num_val_nodes
