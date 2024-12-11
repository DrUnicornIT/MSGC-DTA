import torch
import torch.nn as nn

from torch_geometric.nn import GATConv, global_mean_pool as gep

class GATBlock(nn.Module):
    def __init__(self, gcn_layers_dim, hidden_dim=1024, output_dim=128, dropout_rate=0.2):
        super(GATBlock, self).__init__()

        self.conv_layers = nn.ModuleList([GATConv(gcn_layers_dim[i], gcn_layers_dim[i + 1]) for i in range(3)])

        self.mol_fcs = nn.ModuleList(
            [nn.Linear(gcn_layers_dim[3], hidden_dim), nn.Linear(hidden_dim, output_dim)])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_weight, batch):
        output = x

        output = self.conv_layers[0](output, edge_index)

        for conv in self.conv_layers[1:]:
            output = self.relu(conv(output, edge_index))

        output = gep(output, batch)
        for fc in self.mol_fcs:
            output = fc(self.relu(output))
            output = self.dropout(output)

        return output
    
class GATModel(nn.Module):
    def __init__(self, layers_dim):
        super(GATModel, self).__init__()

        self.num_layers = len(layers_dim) - 1
        self.graph_conv = GATBlock(layers_dim)

    def forward(self, graph_batchs):
        embeddings = list(
                map(lambda graph: self.graph_conv(graph.x, graph.edge_index, None, graph.batch), graph_batchs))

        return embeddings