
import sys
print(f"ENV {sys.executable}")

import argparse
from tdc.multi_pred import DTI
import torch
import numpy as np
from tqdm import tqdm
import re
import os

from unimol_tools import UniMolRepr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drug')
    parser.add_argument('--dataset', type=str,
                        help='Name of dataset.', default="Kiba")
    parser.add_argument('--model_name', type=str,
                        help='Name of model.', default="bert-base")
    args = parser.parse_args()
    dataset = DTI(name = args.dataset)
    data = dataset.get_data()
    drug_dict = dict(zip(data['Drug_ID'], data['Drug']))
    data = list(drug_dict.values())
    data_batch = data
    BATCH_SIZE = 32
    data_batch = [data[i:min(i+BATCH_SIZE, len(data))] for i in range(0, len(data), BATCH_SIZE)]
    
    clf = UniMolRepr(data_type='molecule', remove_hs=False)
    
    print(f"Model: f{clf}")
    embeddings = []
    for smiles in data_batch:
        print(smiles)
        unimol_repr = clf.get_repr(smiles, return_atomic_reprs=True)        
        embeddings.extend(list(unimol_repr['cls_repr']))
        
    print(np.array(embeddings).shape)
    np.save(f'drug/unique_drug_E3NN_EMB_{args.dataset}.npy', np.array(embeddings))
