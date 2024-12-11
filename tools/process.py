import json
import pickle
import numpy as np

from .utils import DTADataset

from .drugtarget_network import get_affinity_graph

def load_data(data_path, dataset):
    affinity = pickle.load(open(data_path + dataset + '/affinities', 'rb'), encoding='latin1')
    if dataset == 'davis':
        affinity = -np.log10(affinity / 1e9)

    return affinity

def process_data(data_path, affinity_mat, dataset, num_pos, pos_threshold):
    dataset_path = data_path + dataset + '/'

    train_file = json.load(open(dataset_path + 'S1_train_set.txt'))
    train_index = []
    for i in range(len(train_file)):
        train_index += train_file[i]
    test_index = json.load(open(dataset_path + 'S1_test_set.txt'))
    print(train_index[0:10])
    rows, cols = np.where(np.isnan(affinity_mat) == False)
    print(len(rows))
    print(len(cols))
    train_rows, train_cols = rows[train_index], cols[train_index]
    train_Y = affinity_mat[train_rows, train_cols]
    train_dataset = DTADataset(drug_ids=train_rows, target_ids=train_cols, y=train_Y)
    test_rows, test_cols = rows[test_index], cols[test_index]
    test_Y = affinity_mat[test_rows, test_cols]
    test_dataset = DTADataset(drug_ids=test_rows, target_ids=test_cols, y=test_Y)

    train_affinity_mat = np.zeros_like(affinity_mat)
    train_affinity_mat[train_rows, train_cols] = train_Y
    affinity_graph, drug_pos, target_pos = get_affinity_graph(data_path, dataset, train_affinity_mat, num_pos, pos_threshold)

    return train_dataset, test_dataset, affinity_graph, drug_pos, target_pos