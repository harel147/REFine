import warnings
warnings.filterwarnings('ignore')
import argparse
import numpy as np
import torch
from tqdm import tqdm

from utils.model import create_model
from utils.dataloader import load_data

from algorithms.hrgr.HRGR import HRGR_rewiring



from utils.utils import set_seed

seed = 42
set_seed(seed)


parser = argparse.ArgumentParser(description='Run NodeClassification+Rewiring script')
parser.add_argument('--dataset', type=str, help='Dataset')
parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GATv2', 'APPNPNet', 'MixHop', 'H2GCN', 'GPRGNN', 'OrderedGNN'], help='Model to use')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout')
parser.add_argument('--hidden_dimension', type=int, default=32, help='Hidden Dimension size')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')

parser.add_argument('--scheme', type=str, default='DPD', choices=['D', 'PDP'], help='propagation scheme')
parser.add_argument('--data_eps', type=float, default=0.1, help='data kernel eps')
parser.add_argument('--sample_rate', type=float, default=0.1, help='sample rate')
parser.add_argument('--add_or_delete', type=str, default='add', choices=['add', 'delete'], help='add or delete')
parser.add_argument('--cluster_size', type=int, default=None, help='cluster size, when set to None cluster size is chosen by graph size')


def run_hrgr_node_classification(data, model_type, num_features, num_classes, hidden_dimension, dropout, lr, weight_decay,
                       device, data_eps, scheme, sample_rate, add_or_delete, cluster_size, init_seeds=1):

    num_splits = data.train_mask.shape[1]

    model = create_model(model_type, num_features, num_classes, hidden_dimension, dropout).to(device)
    model = model.to(device)
    print(model)

    test_accuracies = []
    val_accuracies = []
    train_accuracies = []

    if cluster_size is None:
        if len(data.x) < 1000:
            cluster_size = None  # no clustering
        elif len(data.x) < 25000:
            cluster_size = 500
        else:
            cluster_size = 100

    def train(model, optimizer, edge_index, edge_weight):
        model.train()
        optimizer.zero_grad()
        out = model(new_data.x, edge_index, edge_weight)
        loss = criterion(out[train_mask], new_data.y[train_mask])
        loss.backward()
        optimizer.step()
        pred = out.argmax(dim=1)
        train_correct = pred[train_mask] == new_data.y[train_mask]
        train_acc = int(train_correct.sum()) / int(train_mask.sum())
        return loss, train_acc

    def val_and_test(model, edge_index, edge_weight):
        model.eval()
        with torch.no_grad():
            out = model(new_data.x, edge_index, edge_weight)
            pred = out.argmax(dim=1)
            val_correct = pred[val_mask] == new_data.y[val_mask]
            val_acc = int(val_correct.sum()) / int(val_mask.sum())

            test_correct = pred[test_mask] == new_data.y[test_mask]
            test_acc = int(test_correct.sum()) / int(test_mask.sum())

        return val_acc, test_acc


    for split_idx in range(0, num_splits):
        for init_seed_idx in range(init_seeds):
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = torch.nn.CrossEntropyLoss()
            model.reset_parameters()

            new_data, new_edge_weight = HRGR_rewiring(data, cluster_size, split_idx, data_eps, sample_rate, add_or_delete, scheme=scheme)
            num_add_or_delete = abs(data.edge_index.shape[1] - new_data.edge_index.shape[1])
            print(f"num_add_or_delete: {num_add_or_delete}, original graph has {data.edge_index.shape[1]} edges")

            train_mask = new_data.train_mask
            test_mask = new_data.test_mask
            val_mask = new_data.val_mask
            # Data leakage check
            train_nodes = train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            test_nodes = test_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            val_nodes = val_mask.nonzero(as_tuple=True)[0].cpu().numpy()

            if len(np.intersect1d(train_nodes, test_nodes)) > 0 or len(np.intersect1d(train_nodes, val_nodes)) > 0:
                print(f"Warning: Data leakage detected in split {split_idx}. Skipping this split.")
                continue

            print(f"Training for index = {split_idx}")
            best_val_acc = 0
            train_acc_of_best_epoch = 0
            test_acc_of_best_epoch = 0
            for epoch in tqdm(range(1, 501)):
                loss, train_acc = train(model, optimizer, new_data.edge_index, new_edge_weight)
                val_acc, test_acc = val_and_test(model, new_data.edge_index, new_edge_weight)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc_of_best_epoch = test_acc
                    train_acc_of_best_epoch = train_acc

            test_accuracies.append(test_acc_of_best_epoch * 100)
            val_accuracies.append(best_val_acc * 100)
            train_accuracies.append(train_acc_of_best_epoch * 100)

            print(f"Split {split_idx} (init_seed_idx {init_seed_idx}): Test Accuracy: {test_acc_of_best_epoch:.4f}")


    print(f"Average Test Accuracy: {np.mean(test_accuracies):.2f} ± {np.std(test_accuracies) / np.sqrt(len(test_accuracies)):.2f}")
    print(f"Average Validation Accuracy: {np.mean(val_accuracies):.2f} ± {np.std(val_accuracies) / np.sqrt(len(val_accuracies)):.2f}")
    print(f"Average Training Accuracy: {np.mean(train_accuracies):.2f} ± {np.std(train_accuracies) / np.sqrt(len(train_accuracies)):.2f}")

    avg_test_acc = np.mean(test_accuracies)
    sample_size = len(test_accuracies)
    sem_test = np.std(test_accuracies)/(np.sqrt(sample_size))

    print(f'Final test accuracy after {(avg_test_acc):.4f}\u00B1{(sem_test):.4f}')



def main():
    args = parser.parse_args()

    print("Loading dataset...")
    data, num_classes, num_features, num_train_nodes, num_test_nodes, num_val_nodes = load_data(args.dataset)

    print()
    print(f"Number of training nodes: {num_train_nodes / 100}")
    print(f"Number of validation nodes: {num_val_nodes / 100}")
    print(f"Number of test nodes: {num_test_nodes / 100}")
    print()

    print("Start Training...")

    run_hrgr_node_classification(data, args.model, num_features, num_classes, args.hidden_dimension, args.dropout, args.lr,
                       args.weight_decay, args.device, args.data_eps, args.scheme, args.sample_rate, args.add_or_delete,
                       args.cluster_size)

if __name__ == '__main__':
    main()