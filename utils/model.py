import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
import scipy.sparse
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import MessagePassing, GCNConv, GATv2Conv, APPNP, MixHopConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.0):
        super().__init__()
        torch.manual_seed(12345)

        cached = False
        add_self_loops = True
        normalize = True
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=cached, normalize=normalize, add_self_loops=add_self_loops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=cached, normalize=normalize, add_self_loops=add_self_loops))

        self.dropout = dropout
        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        x = self.convs[0](x, edge_index, edge_weight)
        x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[1](x, edge_index, edge_weight)
        return x  # no need for softmax, CrossEntropyLoss already do softmax

class GATv2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.0, heads=2):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GATv2Conv(in_channels, hidden_channels, edge_dim=1, heads=heads)
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, edge_dim=1, heads=1)
        self.dropout = dropout
        self.activation = F.relu

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_attr=edge_weight)
        x = self.activation(x)  # ReLU activation
        x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout

        # Second layer
        x = self.conv2(x, edge_index, edge_attr=edge_weight)
        return x  # No softmax, CrossEntropyLoss will handle softmax

class APPNPNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(12345)
        add_self_loops = True
        normalize = True
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.prop1 = APPNP(K=10, alpha=0.1, add_self_loops=add_self_loops, normalize=normalize)
        self.activation = F.relu

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.prop1(x, edge_index, edge_weight=edge_weight)
        return x  # no need for softmax, CrossEntropyLoss already do softmax

class MixHop(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        torch.manual_seed(12345)

        # default powers are powers=[0, 1, 2], so we define new hidden_channels
        hidden_channels = int(hidden_channels / 3)

        self.convs = nn.ModuleList()
        self.convs.append(MixHopConv(in_channels, hidden_channels))
        self.convs.append(MixHopConv(3 * hidden_channels, hidden_channels))

        self.lin = nn.Linear(3 * hidden_channels, out_channels)

        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        x = self.convs[0](x, edge_index, edge_weight)
        x = self.activation(x)
        x = self.convs[1](x, edge_index, edge_weight)
        x = self.lin(x)
        return x  # no need for softmax, CrossEntropyLoss already do softmax


# taken from https://github.com/CUAI/Non-Homophily-Large-Scale/blob/master/models.py
class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
        x = self.lins[-1](x)
        return x

# taken from https://github.com/CUAI/Non-Homophily-Large-Scale/blob/master/models.py
class H2GCNConv(nn.Module):
    """ Neighborhood aggregation step """

    def __init__(self):
        super(H2GCNConv, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, x, adj_t, adj_t2):
        x1 = matmul(adj_t, x)
        x2 = matmul(adj_t2, x)
        return torch.cat([x1, x2], dim=1)

# taken from https://github.com/CUAI/Non-Homophily-Large-Scale/blob/master/models.py
class H2GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, num_mlp_layers=1):
        super(H2GCN, self).__init__()

        # we concat 2 embedding vectors in H2GCNConv, so we define new hidden_channels
        hidden_channels = int(hidden_channels / 2)

        self.feature_embed = MLP(in_channels, hidden_channels, hidden_channels, num_layers=num_mlp_layers)

        self.convs = nn.ModuleList()
        self.convs.append(H2GCNConv())

        for l in range(num_layers - 1):
            self.convs.append(H2GCNConv())

        self.activation = F.relu

        last_dim = hidden_channels * (2 ** (num_layers + 1) - 1)
        self.final_project = nn.Linear(last_dim, out_channels)
        self.already_init_adj = False

    def reset_parameters(self):
        self.feature_embed.reset_parameters()
        self.final_project.reset_parameters()
        self.already_init_adj = False  # this is important, to apply self.init_adj for every data split

    def init_adj(self, edge_index, num_nodes):
        """ cache normalized adjacency and normalized strict two-hop adjacency,
        neither has self loops
        """
        n = num_nodes

        if isinstance(edge_index, SparseTensor):
            dev = edge_index.device
            adj_t = edge_index
            adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
            adj_t[adj_t > 0] = 1
            adj_t[adj_t < 0] = 0
            adj_t = SparseTensor.from_scipy(adj_t).to(dev)
        elif isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=None, sparse_sizes=(n, n))

        adj_t.remove_diag(0)
        adj_t2 = matmul(adj_t, adj_t)
        adj_t2.remove_diag(0)
        adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
        adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())
        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0

        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)

        adj_t = gcn_norm(adj_t, None, n, add_self_loops=False)
        adj_t2 = gcn_norm(adj_t2, None, n, add_self_loops=False)

        self.adj_t = adj_t.to(edge_index.device)
        self.adj_t2 = adj_t2.to(edge_index.device)

        self.already_init_adj = True

    def forward(self, x, edge_index, edge_weight=None):  # edge_weight for compatibility
        if self.already_init_adj == False:
            self.init_adj(edge_index, len(x))

        adj_t = self.adj_t
        adj_t2 = self.adj_t2

        x = self.feature_embed(x)
        x = self.activation(x)
        xs = [x]
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, adj_t2)
            xs.append(x)
        x = self.convs[-1](x, adj_t, adj_t2)
        xs.append(x)

        x = torch.cat(xs, dim=1)
        x = self.final_project(x)
        return x

# taken from https://github.com/CUAI/Non-Homophily-Large-Scale/blob/master/models.py
class GPR_prop(MessagePassing):
    '''
    GPRGNN, from original repo https://github.com/jianhao2016/GPRGNN
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0 * np.ones(K + 1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha) ** np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = nn.Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        nn.init.zeros_(self.temp)
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
        self.temp.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(edge_index, torch.Tensor):
            edge_index, norm = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
            norm = None

        hidden = x * (self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

# taken from https://github.com/CUAI/Non-Homophily-Large-Scale/blob/master/models.py
class GPRGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, Init='Random', K=10, alpha=.1,
                 Gamma=None, num_layers=2):
        super(GPRGNN, self).__init__()

        self.mlp = MLP(in_channels, hidden_channels, out_channels, num_layers=num_layers)
        self.prop1 = GPR_prop(K, alpha, Init, Gamma)

        self.Init = Init

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):  # edge_weight for compatibility
        x = self.mlp(x)

        x = self.prop1(x, edge_index)
        return x