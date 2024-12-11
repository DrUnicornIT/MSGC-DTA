import torch
import torch.nn as nn
from torch.nn.functional import normalize

# class PretrainedEmbedding(nn.Module):
#     def __init__(self, embeddings, sizes = [100, 300], target_size = 128):
#         super(PretrainedEmbedding, self).__init__()
#         self.params = nn.ParameterList([nn.Parameter(torch.from_numpy(embedding).float(), requires_grad = True) for embedding in embeddings])
        
#         layers_dim = [sum(sizes), 512, 256, target_size]
#         self.linears = nn.ModuleList([nn.Sequential(
#             nn.Linear(layers_dim[i], layers_dim[i+1]),
#             nn.ReLU(),
#             nn.Dropout(0.2)
#         )for i in range(3)])

#     def forward(self):
#         embeds = torch.cat([normalize(param.data) for param in self.params], dim=-1)
#         for linear in self.linears:
#             embeds = linear(embeds)
#         return embeds

class PretrainedEmbedding(nn.Module):
    def __init__(self, embeddings, sizes = [100, 300], target_size = 128):
        super(PretrainedEmbedding, self).__init__()
        num_dims = len(embeddings)
        self.params = nn.ParameterList([nn.Parameter(torch.from_numpy(embedding).float(), requires_grad = True) for embedding in embeddings])
        self.linear = nn.Linear(sum(sizes), target_size)
        self.relu = nn.ReLU()
    def forward(self):
        embeds = torch.cat([normalize(param.data) for param in self.params], dim=-1)
        embeds = self.linear(embeds)
        return embeds