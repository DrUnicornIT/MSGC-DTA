import torch
import torch.nn as nn

from torch_geometric.nn import GATConv, global_mean_pool as gep

class GATBlock_MIX(nn.Module):
    def __init__(self, gcn_layers_dim, hidden_dim=1024, output_dim=128, dropout_rate=0.2):
        super(GATBlock_MIX, self).__init__()

        self.conv_layers = nn.ModuleList([GATConv(gcn_layers_dim[i], gcn_layers_dim[i + 1]) for i in range(3)])
        self.conv_neighbor_layers = nn.ModuleList([GATConv(gcn_layers_dim[i], gcn_layers_dim[i + 1]) for i in range(3)])
        self.fusion_layers = nn.ModuleList([nn.Linear(gcn_layers_dim[i+1] * 2, gcn_layers_dim[i+1]) for i in range(3)])

        # self.mol_fcs = nn.ModuleList(
        #     [nn.Linear(gcn_layers_dim[3], hidden_dim), nn.Linear(hidden_dim, output_dim)])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_weight, batch, edge_index_neighbor, edge_weight_neighbor, batch_neighbor):
        output = x

        output_origin = self.conv_layers[0](output, edge_index)
        output_neighbor = self.conv_neighbor_layers[0](output, edge_index_neighbor)
        output = torch.cat((output_origin, output_neighbor), dim = -1)
        output = self.fusion_layers[0](output)
        
        for i in range(1, len(self.conv_layers)):
            output_origin = self.relu(self.conv_layers[i](output_origin, edge_index))
            output_neighbor = self.relu(self.conv_neighbor_layers[i](output_neighbor, edge_index_neighbor))
            output = torch.cat((output_origin, output_neighbor), dim = -1)
            output = self.fusion_layers[i](output)
                
        output = gep(output, batch)
        # for fc in self.mol_fcs:
        #     output = fc(self.relu(output))
        #     output = self.dropout(output)

        return output

class GATModel_MIX(nn.Module):
    def __init__(self, layers_dim):
        super(GATModel_MIX, self).__init__()

        self.num_layers = len(layers_dim) - 1
        self.graph_conv = GATBlock_MIX(layers_dim)

    def forward(self, graph_batchs, graph_neighbor_batchs):
        embeddings = list(
                map(lambda pair: self.graph_conv(pair[0].x, pair[0].edge_index, None, pair[0].batch, pair[1].edge_index, None, pair[1].batch),
                     zip(graph_batchs, graph_neighbor_batchs)))

        return embeddings
