import warnings
warnings.filterwarnings('ignore')
import argparse
import numpy as np
import torch
from tqdm import tqdm

from utils.model import create_model
from utils.dataloader import load_data

from utils.utils import set_seed

seed = 42
set_seed(seed)


parser = argparse.ArgumentParser(description='Run NodeClassification+Rewiring script')
parser.add_argument('--dataset', type=str, help='Dataset')
parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GATv2', 'APPNPNet', 'MixHop', 'H2GCN', 'GPRGNN'], help='Model to use')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout')
parser.add_argument('--hidden_dimension', type=int, default=32, help='Hidden Dimension size')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')



def run_none_node_classification(data, model_type, num_features, num_classes, hidden_dimension, dropout, lr, weight_decay,
                       device, init_seeds=1):

    num_splits = data.train_mask.shape[1]

    model = create_model(model_type, num_features, num_classes, hidden_dimension, dropout).to(device)
    model = model.to(device)
    print(model)

    edge_weight = None

    test_accuracies = []
    val_accuracies = []
    train_accuracies = []

    def train(model, optimizer, edge_index, edge_weight):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, edge_index, edge_weight)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        pred = out.argmax(dim=1)
        train_correct = pred[train_mask] == data.y[train_mask]
        train_acc = int(train_correct.sum()) / int(train_mask.sum())
        return loss, train_acc

    def val_and_test(model, edge_index, edge_weight):
        model.eval()
        with torch.no_grad():
            out = model(data.x, edge_index, edge_weight)
            pred = out.argmax(dim=1)
            val_correct = pred[val_mask] == data.y[val_mask]
            val_acc = int(val_correct.sum()) / int(val_mask.sum())

            test_correct = pred[test_mask] == data.y[test_mask]
            test_acc = int(test_correct.sum()) / int(test_mask.sum())

        return val_acc, test_acc


    for split_idx in range(0, num_splits):
        for init_seed_idx in range(init_seeds):
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = torch.nn.CrossEntropyLoss()
            model.reset_parameters()

            train_mask = data.train_mask[:, split_idx]
            val_mask = data.val_mask[:, split_idx]
            test_mask = data.test_mask[:, split_idx]
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
                loss, train_acc = train(model, optimizer, data.edge_index, edge_weight)
                val_acc, test_acc = val_and_test(model, data.edge_index, edge_weight)
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

    run_none_node_classification(data, args.model, num_features, num_classes, args.hidden_dimension, args.dropout, args.lr,
                       args.weight_decay, args.device)

if __name__ == '__main__':
    main()