import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv, global_mean_pool as gep


class GCNBlock_MIX(nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(GCNBlock_MIX, self).__init__()

        self.conv_layers = nn.ModuleList()
        
        for i in range(len(gcn_layers_dim) - 1):
            conv_layer = GCNConv(gcn_layers_dim[i], gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)

        self.conv_neighbor_layers = nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            conv_layer = GCNConv(gcn_layers_dim[i], gcn_layers_dim[i + 1])
            self.conv_neighbor_layers.append(conv_layer)
        
        self.fusion_layers = nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            fusion = nn.Linear(gcn_layers_dim[i+1] * 2, gcn_layers_dim[i+1])
            self.fusion_layers.append(fusion)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, edge_index, edge_weight, batch, edge_index_neighbor, edge_weight_neighbor, batch_neighbor):
        output = x
        embeddings = []
        for conv_layer_index in range(len(self.conv_layers)):
            output_origin = self.conv_layers[conv_layer_index](output, edge_index, edge_weight)
            output_neighbor = self.conv_neighbor_layers[conv_layer_index](output, edge_index_neighbor, edge_weight_neighbor)
            output = torch.cat((output_origin, output_neighbor), dim = -1)
            output = self.fusion_layers[conv_layer_index](output)
            
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)


            embeddings.append(gep(output, batch))

        return embeddings
    
class GCNModel_MIX(nn.Module):
    def __init__(self, layers_dim):
        super(GCNModel_MIX, self).__init__()

        self.num_layers = len(layers_dim) - 1
        self.graph_conv = GCNBlock_MIX(layers_dim, relu_layers_index=list(range(self.num_layers)))

    def forward(self, graph_batchs, graph_neighbor_batchs):
        embedding_batchs = list(
            map(lambda pair: self.graph_conv(pair[0].x, pair[0].edge_index, None, pair[0].batch, pair[1].edge_index, None, pair[1].batch),
                zip(graph_batchs, graph_neighbor_batchs))
        )

        embeddings = []
        for i in range(self.num_layers):
            embeddings.append(torch.cat(list(map(lambda embedding_batch: embedding_batch[i], embedding_batchs)), 0))

        return embeddings