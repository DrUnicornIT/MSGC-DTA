import torch
import torch.nn as nn
from torch.nn.functional import normalize

from torch_geometric.nn import DenseGCNConv, GCNConv, GATConv, global_mean_pool as gep
from torch_geometric.utils import dropout_adj

class Feature_Fusion(nn.Module):
    def __init__(self, channels=128 * 2, r=4):
        super(Feature_Fusion, self).__init__()
        inter_channels = int(channels // r)
        layers = [nn.Linear(channels, inter_channels), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(inter_channels, channels)]
        self.att1 = nn.Sequential(*layers)
        self.att2 = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fd, fp):
        w1 = self.sigmoid(self.att1(fd + fp))
        fout1 = fd * w1 + fp * (1 - w1)
        w2 = self.sigmoid(self.att2(fout1))
        fout2 = fd * w2 + fp * (1 - w2)
        return fout2

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
    
class GCNBlock(nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(GCNBlock, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            conv_layer = GCNConv(gcn_layers_dim[i], gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, edge_index, edge_weight, batch):
        output = x
        embeddings = []
        for conv_layer_index in range(len(self.conv_layers)):
            output = self.conv_layers[conv_layer_index](output, edge_index, edge_weight)

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

class GATModel(nn.Module):
    def __init__(self, layers_dim):
        super(GATModel, self).__init__()

        self.num_layers = len(layers_dim) - 1
        self.graph_conv = GATBlock(layers_dim)

    def forward(self, graph_batchs):
        embeddings = list(
                map(lambda graph: self.graph_conv(graph.x, graph.edge_index, None, graph.batch), graph_batchs))

        return embeddings

class GCNModel(nn.Module):
    def __init__(self, layers_dim):
        super(GCNModel, self).__init__()

        self.num_layers = len(layers_dim) - 1
        self.graph_conv = GCNBlock(layers_dim, relu_layers_index=list(range(self.num_layers)))

    def forward(self, graph_batchs):
        embedding_batchs = list(
                map(lambda graph: self.graph_conv(graph.x, graph.edge_index, None, graph.batch), graph_batchs))
        embeddings = []
        for i in range(self.num_layers):
            embeddings.append(torch.cat(list(map(lambda embedding_batch: embedding_batch[i], embedding_batchs)), 0))

        return embeddings


class DenseGCNBlock(nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(DenseGCNBlock, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            conv_layer = DenseGCNConv(gcn_layers_dim[i], gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, adj):
        output = x
        embeddings = []
        for conv_layer_index in range(len(self.conv_layers)):
            output = self.conv_layers[conv_layer_index](output, adj, add_loop=False)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(torch.squeeze(output, dim=0))

        return embeddings


# Regression
class DenseRegressionModel(nn.Module):
    def __init__(self, layers_dim, edge_dropout_rate=0.):
        super(DenseRegressionModel, self).__init__()
        self.gcn = GCNConv(layers_dim[0], layers_dim[1])
        
        self.fc1 = nn.Linear(layers_dim[1], 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, 256)
    
    def forward(self, graph):
        xs, adj, num_d, num_t = graph.x, graph.adj, graph.num_drug, graph.num_target
        
        x = self.gcn(xs)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        embeddings = self.fc_out(x)
        return embeddings


class DenseGCNModel(nn.Module):
    def __init__(self, layers_dim, edge_dropout_rate=0.):
        super(DenseGCNModel, self).__init__()

        self.edge_dropout_rate = edge_dropout_rate
        self.num_layers = len(layers_dim) - 1
        self.graph_conv = DenseGCNBlock(layers_dim, 0.1, relu_layers_index=list(range(self.num_layers)),
                                        dropout_layers_index=list(range(self.num_layers)))

    def forward(self, graph):
        xs, adj, num_d, num_t = graph.x, graph.adj, graph.num_drug, graph.num_target
        indexs = torch.where(adj != 0)
        edge_indexs = torch.cat((torch.unsqueeze(indexs[0], 0), torch.unsqueeze(indexs[1], 0)), 0)
        edge_indexs_dropout, edge_weights_dropout = dropout_adj(edge_index=edge_indexs, edge_attr=adj[indexs],
                                                                p=self.edge_dropout_rate, force_undirected=True,
                                                                num_nodes=num_d + num_t, training=self.training)
        adj_dropout = torch.zeros_like(adj)
        adj_dropout[edge_indexs_dropout[0], edge_indexs_dropout[1]] = edge_weights_dropout

        embeddings = self.graph_conv(xs, adj_dropout)

        return embeddings


class LinearBlock(nn.Module):
    def __init__(self, linear_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(LinearBlock, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(linear_layers_dim) - 1):
            layer = nn.Linear(linear_layers_dim[i], linear_layers_dim[i + 1])
            self.layers.append(layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x):
        output = x
        embeddings = []
        for layer_index in range(len(self.layers)):
            output = self.layers[layer_index](output)
            if layer_index in self.relu_layers_index:
                output = self.relu(output)
            if layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(output)

        return embeddings


class Contrast(nn.Module):
    def __init__(self, hidden_dim, output_dim, tau, lam):
        super(Contrast, self).__init__()

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim))
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)

        return sim_matrix

    def forward(self, za, zb, pos):
        za_proj = self.proj(za)
        zb_proj = self.proj(zb)
        matrix_a2b = self.sim(za_proj, zb_proj)
        matrix_b2a = matrix_a2b.t()

        matrix_a2b = matrix_a2b / (torch.sum(matrix_a2b, dim=1).view(-1, 1) + 1e-8)
        lori_a = -torch.log(matrix_a2b.mul(pos.to_dense()).sum(dim=-1)).mean()

        matrix_b2a = matrix_b2a / (torch.sum(matrix_b2a, dim=1).view(-1, 1) + 1e-8)
        lori_b = -torch.log(matrix_b2a.mul(pos.to_dense()).sum(dim=-1)).mean()

        return self.lam * lori_a + (1 - self.lam) * lori_b, torch.cat((za_proj, zb_proj), 1)

class EnsembleEmbedding(nn.Module):
    def __init__(self, embeddings, sizes = [100, 300], target_size = 128):
        super(EnsembleEmbedding, self).__init__()
        num_dims = len(embeddings)
        self.params = nn.ParameterList([nn.Parameter(torch.from_numpy(embedding).float(), requires_grad = True) for embedding in embeddings])
        # self.linears = nn.ModuleList([nn.Linear(sizes[i], mid_size) for i in range(num_dims)])
        self.linear = nn.Linear(sum(sizes), target_size)
        self.relu = nn.ReLU()
    def forward(self):
        # embeds = []
        # for i, param in enumerate(self.params):
        #     embed_prj = self.relu(self.linears[i](normalize(param)))
        #     embeds.append(embed_prj)
        # embeds = torch.cat(embeds, dim = -1)
        embeds = torch.cat([normalize(param.data) for param in self.params], dim=-1)
        embeds = self.linear(embeds)
        return embeds
        

class CSCoDTA(nn.Module):
    def __init__(self, tau, lam, ns_dims, d_ms_dims, t_ms_dims, d_embeddings, t_embeddings, embedding_dim=128, dropout_rate=0.2):
        super(CSCoDTA, self).__init__()

        self.output_dim = embedding_dim * 2

        self.affinity_graph_conv = DenseRegressionModel(ns_dims, dropout_rate)
        
        self.drug_graph_conv = GATModel_MIX(d_ms_dims)
        self.target_graph_conv = GATModel_MIX(t_ms_dims)

        self.drug_embeddings = EnsembleEmbedding(d_embeddings, sizes = (100, 300, 512), target_size = 128)
        self.target_embeddings = EnsembleEmbedding(t_embeddings, sizes = (100, 768, 1280), target_size = 128)
        
        self.drug_contrast = Contrast(ns_dims[-1], embedding_dim, tau, lam)
        self.target_contrast = Contrast(ns_dims[-1], embedding_dim, tau, lam)

    def forward(self, affinity_graph, drug_graph_batchs, drug_graph_neighbor_batchs, target_graph_batchs, target_graph_neighbor_batchs, drug_pos, target_pos):
        num_d = affinity_graph.num_drug

        affinity_graph_embedding = self.affinity_graph_conv(affinity_graph)[-1]
        
        #______________
        drug_graph_embedding_dynamic = self.drug_graph_conv(drug_graph_batchs, drug_graph_neighbor_batchs)[-1]
        target_graph_embedding_dynamic = self.target_graph_conv(target_graph_batchs, target_graph_neighbor_batchs)[-1]
        #_________________
        drug_graph_embedding_static = self.drug_embeddings()
        target_graph_embedding_static = self.target_embeddings()

        drug_graph_embedding = torch.cat([drug_graph_embedding_dynamic, drug_graph_embedding_static], dim=-1)
        target_graph_embedding = torch.cat([target_graph_embedding_dynamic, target_graph_embedding_static], dim=-1)

        dru_loss, drug_embedding = self.drug_contrast(affinity_graph_embedding[:num_d], drug_graph_embedding, drug_pos)
        tar_loss, target_embedding = self.target_contrast(affinity_graph_embedding[num_d:], target_graph_embedding,
                                                          target_pos)

        return dru_loss + tar_loss, drug_embedding, target_embedding
        # return drug_graph_embedding, target_graph_embedding


class PredictModule(nn.Module):
    def __init__(self, embedding_dim=128, output_dim=1):
        super(PredictModule, self).__init__()
        self.dtf = Feature_Fusion(channels= 2 * embedding_dim)

        # self.prediction_func, prediction_dim_func = (lambda x, y: torch.cat((x, y), -1), lambda dim: 4 * dim)
        # mlp_layers_dim = [prediction_dim_func(embedding_dim), 1024, 512, output_dim]

        self.prediction_func = lambda x, y: torch.cat((x, y), -1)
        mlp_layers_dim = [2 * embedding_dim, 1024, 512, output_dim]

        self.mlp = LinearBlock(mlp_layers_dim, 0.1, relu_layers_index=[0, 1], dropout_layers_index=[0, 1])

    def forward(self, data, drug_embedding, target_embedding):
        drug_id, target_id, y = data.drug_id, data.target_id, data.y

        drug_feature = drug_embedding[drug_id.int().cpu().numpy()]
        target_feature = target_embedding[target_id.int().cpu().numpy()]
        
        concat_feature = self.dtf(drug_feature, target_feature)

        # concat_feature = self.prediction_func(drug_feature, target_feature)
        
        mlp_embeddings = self.mlp(concat_feature)
        link_embeddings = mlp_embeddings[-2]
        out = mlp_embeddings[-1]

        return out, link_embeddings

