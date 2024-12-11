import torch
import torch.nn as nn

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

class PredictModule(nn.Module):
    def __init__(self, embedding_dim=128, output_dim=1):
        super(PredictModule, self).__init__()

        self.prediction_func, prediction_dim_func = (lambda x, y: torch.cat((x, y), -1), lambda dim: 4 * dim)
        mlp_layers_dim = [prediction_dim_func(embedding_dim), 1024, 512, output_dim]

        self.mlp = LinearBlock(mlp_layers_dim, 0.1, relu_layers_index=[0, 1], dropout_layers_index=[0, 1])

    def forward(self, data, drug_embedding, target_embedding):
        drug_id, target_id, y = data.drug_id, data.target_id, data.y

        drug_feature = drug_embedding[drug_id.int().cpu().numpy()]
        target_feature = target_embedding[target_id.int().cpu().numpy()]
        
        concat_feature = self.prediction_func(drug_feature, target_feature)
        
        mlp_embeddings = self.mlp(concat_feature)
        link_embeddings = mlp_embeddings[-2]
        out = mlp_embeddings[-1]

        return out, link_embeddings