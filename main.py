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

from data_process import load_data, process_data, get_drug_molecule_graph, get_target_molecule_graph

from utils import GraphDataset, collate, model_evaluate
from models import CSCoDTA, PredictModule

def train(model, predictor, device, train_loader, drug_graphs_DataLoader, drug_graphs_neighbor_DataLoader, target_graphs_DataLoader, lr, epoch,
          batch_size, affinity_graph, drug_pos, target_pos, optimizer, scheduler):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    # Calculate total number of parameters
    
    total_params = sum(p.numel() for p in model.parameters())
    # Print total number of parameters
    print(f"Total number of parameters: {total_params}")
    # Optionally, print number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    
    model.train()
    predictor.train()
    LOG_INTERVAL = 10
    loss_fn = nn.MSELoss()
    # optimizer = torch.optim.Adam(
    #     filter(lambda p: p.requires_grad, chain(model.parameters(), predictor.parameters())), lr=lr, weight_decay=0)
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))
    drug_graph_neighbor_batchs = list(map(lambda graph: graph.to(device), drug_graphs_neighbor_DataLoader))

    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))

    epoch_loss = 0
    num_batches = len(train_loader)
    
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        ssl_loss, drug_embedding, target_embedding = model(affinity_graph.to(device), drug_graph_batchs, drug_graph_neighbor_batchs,
                                                                  target_graph_batchs, drug_pos, target_pos)
        output, _ = predictor(data.to(device), drug_embedding, target_embedding)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device)) + ssl_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))

    mean_loss = epoch_loss / num_batches

    # Log the mean loss for the epoch to wandb
    wandb.log({"mean_loss": mean_loss})


def test(model, predictor, device, loader, drug_graphs_DataLoader, drug_graphs_neighbor_DataLoader, target_graphs_DataLoader, affinity_graph, drug_pos,
         target_pos):
    model.eval()
    predictor.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))  # drug graphs
    drug_graph_neighbor_batchs = list(map(lambda graph: graph.to(device), drug_graphs_neighbor_DataLoader))

    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))  # target graphs
    with torch.no_grad():
        for data in loader:
            _, drug_embedding, target_embedding = model(affinity_graph.to(device), drug_graph_batchs, drug_graph_neighbor_batchs, target_graph_batchs, drug_pos, target_pos)
            output, _ = predictor(data.to(device), drug_embedding, target_embedding)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

# Training data
def train_predict(): 
    print("Data preparation in progress for the {} dataset...".format(args.dataset))

    # Loading affinity
    affinity_mat = load_data(args.dataset)
    print(affinity_mat)
    
    # Process build train data and test data
    train_data, test_data, affinity_graph, drug_pos, target_pos = process_data(affinity_mat, args.dataset, args.num_pos, args.pos_threshold)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    #________________________________________________
    drug_graphs_dict, drug_graphs_neighbor_dict = get_drug_molecule_graph(
        json.load(open(f'/kaggle/input/msgc-dta/MSGC-DTA/data/{args.dataset}/drugs.txt'), object_pairs_hook=OrderedDict))
    
    drug_graphs_Data = GraphDataset(graphs_dict=drug_graphs_dict, dttype="drug")
    drug_graphs_DataLoader = torch.utils.data.DataLoader(drug_graphs_Data, shuffle=False, collate_fn=collate,
                                                         batch_size=affinity_graph.num_drug)
    
    drug_graphs_neighbor_Data = GraphDataset(graphs_dict=drug_graphs_neighbor_dict, dttype="drug")
    drug_graphs_neighbor_DataLoader = torch.utils.data.DataLoader(drug_graphs_neighbor_Data, shuffle=False, collate_fn=collate,
                                                         batch_size=affinity_graph.num_drug)
    #________________________________________________
    target_graphs_dict, target_graphs_neighbor_dict = get_target_molecule_graph(
        json.load(open(f'/kaggle/input/msgc-dta/MSGC-DTA/data/{args.dataset}/targets.txt'), object_pairs_hook=OrderedDict), args.dataset)
    target_graphs_Data = GraphDataset(graphs_dict=target_graphs_dict, dttype="target")
    target_graphs_DataLoader = torch.utils.data.DataLoader(target_graphs_Data, shuffle=False, collate_fn=collate,
                                                           batch_size=affinity_graph.num_target)
    
    target_graphs_neighbor_Data = GraphDataset(graphs_dict=target_graphs_neighbor_dict, dttype="target")
    target_graphs_neighbor_DataLoader = torch.utils.data.DataLoader(target_graphs_neighbor_Data, shuffle=False, collate_fn=collate,
                                                           batch_size=affinity_graph.num_target)
    #________________________________________________
    
    d_1d_embeds = np.load('/kaggle/input/msgc-dta/MSGC-DTA/data/results/unique_drug_Mol2Vec_EMB_DAVIS.npy')
    d_2d_embeds = np.load('/kaggle/input/msgc-dta/MSGC-DTA/data/results/unique_drug_GIN_EMB_DAVIS.npy')
    d_3d_embeds = np.load('/kaggle/input/msgc-dta/MSGC-DTA/data/results/unique_drug_E3nn_EMB_DAVIS.npy')
    
    t_1d_embeds = np.load('/kaggle/input/msgc-dta/MSGC-DTA/data/results/unique_protein_ProVec_EMB_DAVIS.npy') 
    t_2d_embeds = np.load('/kaggle/input/msgc-dta/MSGC-DTA/data/results/unique_protein_BERT_EMB_DAVIS.npy')
    t_3d_embeds = np.load('/kaggle/input/msgc-dta/MSGC-DTA/data/results/unique_protein_ESM_EMB_DAVIS.npy')
    
    print(d_1d_embeds.shape)
    print(d_2d_embeds.shape)
    print(d_3d_embeds.shape)
    print(t_1d_embeds.shape)
    print(t_2d_embeds.shape)
    print(t_3d_embeds.shape)
    
    d_embeddings = (d_1d_embeds, d_2d_embeds, d_3d_embeds)
    t_embeddings = (t_1d_embeds, t_2d_embeds, t_3d_embeds)

    print("Model preparation... ")
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    model = CSCoDTA(tau=args.tau,
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
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, chain(model.parameters(), predictor.parameters())), lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs, pct_start = 0.0)
    print("Start training...")
    for epoch in range(args.epochs):
        train(model, predictor, device, train_loader, drug_graphs_DataLoader, drug_graphs_neighbor_DataLoader, target_graphs_DataLoader, target_graphs_neighbor_DataLoader, args.lr, epoch+1,
              args.batch_size, affinity_graph, drug_pos, target_pos, optimizer, scheduler)
        G, P = test(model, predictor, device, test_loader, drug_graphs_DataLoader, drug_graphs_neighbor_DataLoader, target_graphs_DataLoader, target_graphs_neighbor_DataLoader,
                    affinity_graph, drug_pos, target_pos)
        if epoch % 1000 == 0:
            r = model_evaluate(G, P, full = True)
        else: 
            r = model_evaluate(G, P, full = False)
        print("result:", r)
        wandb.log({"test_MSE": r[0], "test_RM2": r[1], "test_CI_DeepDTA": r[2], "test_CI_GraphDTA": r[3]})

    print('\npredicting for test data')
    G, P = test(model, predictor, device, test_loader, drug_graphs_DataLoader, drug_graphs_neighbor_DataLoader, target_graphs_DataLoader, target_graphs_neighbor_DataLoader,
                affinity_graph, drug_pos, target_pos)
    result = model_evaluate(G, P, full = True)
    print("result:", result)
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=2024, help='Random Seed')
    parser.add_argument('--gpus', type=str, default='0', help='Number of GPUs') # 0 -> CPU
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--workspace', type=str, default='/kaggle/input/msgc-dta/MSGC-DTA/')
    parser.add_argument('--dataset', type=str, default='davis')
    parser.add_argument('--epochs', type=int, default=2)    # --kiba 3000
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--edge_dropout_rate', type=float, default=0.2)   # --kiba 0.
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--num_pos', type=int, default=3)    # --kiba 10
    parser.add_argument('--pos_threshold', type=float, default=8.0)

    args, _ = parser.parse_known_args()

    # Setup Wandb project
    wandb.init(
        project="MSGC-DTA",
        config = vars(args)
    )

    train_predict()  # Training data