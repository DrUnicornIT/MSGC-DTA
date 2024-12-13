import os
import argparse
import torch
import json
import warnings
from collections import OrderedDict
import numpy as np
from torch import nn
from itertools import chain
from datetime import datetime
import wandb
import random

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def test(model, predictor, device, loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph, drug_pos,
         target_pos):
    model.eval()
    predictor.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))  # drug graphs

    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))  # target graphs

    
    with torch.no_grad():
        for data in loader:
            _, drug_embedding, target_embedding = model(affinity_graph.to(device), drug_graph_batchs, target_graph_batchs, drug_pos, target_pos)
            output, _ = predictor(data.to(device), drug_embedding, target_embedding)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()




from tools.process import load_data, process_data
from tools.drug_molecule import get_drug_molecule_graph
from tools.target_molecule import get_target_molecule_graph
from tools.utils import GraphDataset, collate, model_evaluate
from models.MSGC_DTA import MSGCDTA
from models.prediction import PredictModule


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=2024, help='Random Seed')
    parser.add_argument('--gpus', type=str, default='0', help='Number of GPUs') # 0 -> CPU
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='/kaggle/input/msgc-dta/MSGC-DTA/data/')
    parser.add_argument('--dataset', type=str, default='davis')
    parser.add_argument('--epochs', type=int, default=6000)    # --kiba 3000
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--edge_dropout_rate', type=float, default=0.2)   # --kiba 0.
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--num_pos', type=int, default=3)    # --kiba 10
    parser.add_argument('--pos_threshold', type=float, default=8.0)

    args, _ = parser.parse_known_args()
    # wandb.login(key="b67abb17df1ee7142cd9e8950d8b6d9aca0585cd")
    # Setup Wandb project
    set_seed(args.seed)


    print("Data preparation in progress for the {} dataset...".format(args.dataset))

    #-------------Loading affinity----------------
    affinity_mat = load_data(args.data_path, args.dataset)


    #-------------Process build train data and test data----------------
    train_data, test_data, affinity_graph, drug_pos, target_pos = process_data(args.data_path, affinity_mat, args.dataset, args.num_pos, args.pos_threshold)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)


    #-------------Graph drug molecule----------------
    drug_graphs_dict, drug_graphs_neighbor_dict = get_drug_molecule_graph(
        json.load(open(f'{args.data_path}{args.dataset}/drugs.txt'), object_pairs_hook=OrderedDict))
        
    drug_graphs_Data = GraphDataset(graphs_dict=drug_graphs_dict, dttype="drug")
    drug_graphs_DataLoader = torch.utils.data.DataLoader(drug_graphs_Data, shuffle=False, collate_fn=collate,
                                                         batch_size=affinity_graph.num_drug)
    
    ## drug_graphs_neighbor_Data = GraphDataset(graphs_dict=drug_graphs_neighbor_dict, dttype="drug")
    ## drug_graphs_neighbor_DataLoader = torch.utils.data.DataLoader(drug_graphs_neighbor_Data, shuffle=False, collate_fn=collate, batch_size=affinity_graph.num_drug)
    
    
    #-------------Graph target molecule----------------
    target_graphs_dict, target_graphs_neighbor_dict = get_target_molecule_graph(args.data_path,
        json.load(open(f'{args.data_path}{args.dataset}/targets.txt'), object_pairs_hook=OrderedDict), args.dataset)
    target_graphs_Data = GraphDataset(graphs_dict=target_graphs_dict, dttype="target")
    target_graphs_DataLoader = torch.utils.data.DataLoader(target_graphs_Data, shuffle=False, collate_fn=collate,
                                                           batch_size=affinity_graph.num_target)
    
    ## target_graphs_neighbor_Data = GraphDataset(graphs_dict=target_graphs_neighbor_dict, dttype="target")
    ## target_graphs_neighbor_DataLoader = torch.utils.data.DataLoader(target_graphs_neighbor_Data, shuffle=False, collate_fn=collate, batch_size=affinity_graph.num_target)
    

    #-------------Pretrained Embedding----------------

    d_1d_embeds = np.load(args.data_path + f'results/unique_drug_Mol2Vec_EMB_{args.dataset.upper()}.npy')
    d_1d_covid = np.load(args.data_path + f'results/unique_drug_Mol2Vec_EMB_COVID.npy')

    d_2d_embeds = np.load(args.data_path + f'results/unique_drug_GIN_EMB_{args.dataset.upper()}.npy')
    d_2d_covid = np.load(args.data_path + f'results/unique_drug_GIN_EMB_COVID.npy')
    d_3d_embeds = np.load(args.data_path + f'results/unique_drug_E3nn_EMB_{args.dataset.upper()}.npy')
    d_3d_covid = np.load(args.data_path + f'results/unique_drug_E3nn_EMB_COVID.npy')

    d_embeddings = (np.vstack((d_1d_embeds,d_1d_covid)), np.vstack((d_2d_embeds,d_2d_covid)), np.vstack((d_3d_embeds,d_3d_covid)))

    t_1d_embeds = np.load(args.data_path + f'results/unique_protein_ProVec_EMB_{args.dataset.upper()}.npy') 
    t_1d_covid = np.load(args.data_path + f'results/unique_protein_ProVec_EMB_COVID.npy') 

    t_2d_embeds = np.load(args.data_path + f'results/unique_protein_BERT_EMB_{args.dataset.upper()}.npy')
    t_2d_covid = np.load(args.data_path + f'results/unique_protein_BERT_EMB_COVID.npy') 

    t_3d_embeds = np.load(args.data_path + f'results/unique_protein_ESM_EMB_{args.dataset.upper()}.npy')
    t_3d_covid = np.load(args.data_path + f'results/unique_protein_ESM_EMB_COVID.npy') 

    t_embeddings = (np.vstack((t_1d_embeds,t_1d_covid)), np.vstack((t_2d_embeds,t_2d_covid)), np.vstack((t_3d_embeds,t_3d_covid)))


    print(t_embeddings[0].shape)
    #-------------Training Model----------------
    print(affinity_graph)
    print("_"*10)
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    model = MSGCDTA(tau=args.tau,
                    lam=args.lam,
                    ns_dims=[affinity_graph.num_drug + affinity_graph.num_target + 2, 512, 256],
                    d_ms_dims=[78, 78, 78 * 2, 128],
                    t_ms_dims=[54, 54, 54 * 2, 128],
                    d_embeddings=d_embeddings,
                    t_embeddings=t_embeddings,
                    embedding_dim=128,
                    dropout_rate=args.edge_dropout_rate)
    predictor = PredictModule()
    drug_pos = drug_pos.to(device)
    target_pos = target_pos.to(device)
    model.to(device)
    predictor.to(device)

    # model.load_state_dict(torch.load("davis_sota_main.pth", map_location=torch.device('cpu')))
    predictor.load_state_dict(torch.load("checkpoint/davis_sota_predictor_2.pth", map_location=torch.device('cpu')))
    state_dict = torch.load("checkpoint/davis_sota_main_2.pth", map_location=torch.device('cpu'))

    pretrained_weights = state_dict['affinity_graph_conv.graph_conv.conv_layers.0.lin.weight']
    adjusted_weights = torch.zeros(512, 521)  
    adjusted_weights[:, :512] = pretrained_weights
    state_dict['affinity_graph_conv.graph_conv.conv_layers.0.lin.weight'] = adjusted_weights

    drug_embedding_0 = state_dict['drug_embeddings.params.0'] 
    adjusted_drug_embedding_0 = torch.zeros(76, 100)           
    adjusted_drug_embedding_0[:68, :] = drug_embedding_0       
    state_dict['drug_embeddings.params.0'] = adjusted_drug_embedding_0


    drug_embedding_1 = state_dict['drug_embeddings.params.1'] 
    adjusted_drug_embedding_1 = torch.zeros(76, 300)           
    adjusted_drug_embedding_1[:68, :] = drug_embedding_1     
    state_dict['drug_embeddings.params.1'] = adjusted_drug_embedding_1

    drug_embedding_2 = state_dict['drug_embeddings.params.2']
    adjusted_drug_embedding_2 = torch.zeros(76, 512)          
    adjusted_drug_embedding_2[:68, :] = drug_embedding_2     
    state_dict['drug_embeddings.params.2'] = adjusted_drug_embedding_2

    target_embedding_0 = state_dict['target_embeddings.params.0'] 
    adjusted_target_embedding_0 = torch.zeros(443, 100)           
    adjusted_target_embedding_0[:442, :] = target_embedding_0    
    state_dict['target_embeddings.params.0'] = adjusted_target_embedding_0

    target_embedding_1 = state_dict['target_embeddings.params.1']
    adjusted_target_embedding_1 = torch.zeros(443, 768)           
    adjusted_target_embedding_1[:442, :] = target_embedding_1    
    state_dict['target_embeddings.params.1'] = adjusted_target_embedding_1

    target_embedding_2 = state_dict['target_embeddings.params.2']
    adjusted_target_embedding_2 = torch.zeros(443, 1280)         
    adjusted_target_embedding_2[:442, :] = target_embedding_2     
    state_dict['target_embeddings.params.2'] = adjusted_target_embedding_2



    model.load_state_dict(state_dict)

    G, P = test(model, predictor, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader,
                    affinity_graph, drug_pos, target_pos)
                    
    print(P[-8:])

    print(len(G[:-8]))
    # r = model_evaluate(G[:-8], P[:-8], full = False)
    # print("result:", r)
    # print({"test_MSE": r[0], "test_RM2": r[1], "test_CI_DeepDTA": r[2], "test_CI_GraphDTA": r[3]})