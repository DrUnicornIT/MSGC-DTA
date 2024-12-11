import torch
import torch.nn as nn

from .network_model import DenseGCNModel
from .gat import GATModel
from .pretrained import PretrainedEmbedding
from .contrastive import Contrast

class MSGCDTA(nn.Module):
    def __init__(self, tau, lam, ns_dims, d_ms_dims, t_ms_dims, d_embeddings, t_embeddings, embedding_dim=128, dropout_rate=0.2):
        super(MSGCDTA, self).__init__()

        self.output_dim = embedding_dim * 2

        self.affinity_graph_conv = DenseGCNModel(ns_dims, dropout_rate)
        
        self.drug_graph_conv = GATModel(d_ms_dims)
        self.target_graph_conv = GATModel(t_ms_dims)

        self.drug_embeddings = PretrainedEmbedding(d_embeddings, sizes = (100, 300, 512), target_size = 128)
        self.target_embeddings = PretrainedEmbedding(t_embeddings, sizes = (100, 768, 1280), target_size = 128)
        
        self.drug_contrast = Contrast(ns_dims[-1], embedding_dim, tau, lam)
        self.target_contrast = Contrast(ns_dims[-1], embedding_dim, tau, lam)

    def forward(self, affinity_graph, drug_graph_batchs, target_graph_batchs, drug_pos, target_pos):
        num_d = affinity_graph.num_drug

        affinity_graph_embedding = self.affinity_graph_conv(affinity_graph)[-1]

        drug_graph_embedding_dynamic = self.drug_graph_conv(drug_graph_batchs)[-1]
        target_graph_embedding_dynamic = self.target_graph_conv(target_graph_batchs)[-1]

        drug_graph_embedding_static = self.drug_embeddings()
        target_graph_embedding_static = self.target_embeddings()

        drug_graph_embedding = torch.cat([drug_graph_embedding_dynamic, drug_graph_embedding_static], dim=-1)
        target_graph_embedding = torch.cat([target_graph_embedding_dynamic, target_graph_embedding_static], dim=-1)
        
        dru_loss, drug_embedding = self.drug_contrast(affinity_graph_embedding[:num_d], drug_graph_embedding, drug_pos)
        tar_loss, target_embedding = self.target_contrast(affinity_graph_embedding[num_d:], target_graph_embedding,
                                                          target_pos)

        return dru_loss + tar_loss, drug_embedding, target_embedding