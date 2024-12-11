import sys
print(f"ENV {sys.executable}")

import argparse
from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer, EsmModel
from tdc.multi_pred import DTI
import torch
import numpy as np
from tqdm import tqdm
import re
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drug')
    parser.add_argument('--dataset', type=str,
                        help='Name of dataset.', default="Kiba")
    parser.add_argument('--model_name', type=str,
                        help='Name of model.', default="bert-base")
    args = parser.parse_args()
    dataset = DTI(name = args.dataset)
    data = dataset.get_data()
    protein_dict = dict(zip(data['Target_ID'], data['Target']))
    data = list(protein_dict.values())
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", do_lower_case=False)
    model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model.to(device)

    embeddings = []
    for seq in data:
        encoded_proteins = tokenizer(seq,
                                      return_tensors='pt',
                                      max_length=1024,
                                      truncation=True,
                                      padding=True)
    
        encoded_proteins = encoded_proteins.to(device)
        with torch.no_grad():
            target_embeddings = model(**encoded_proteins).last_hidden_state[:, 0, :]
        for target_embedding in target_embeddings.cpu().detach().numpy():
            embeddings.append(target_embedding)

        
    embeddings = np.array(embeddings)
    
    print(f"Shape: {embeddings.shape}")    
    np.save(f'protein/unique_protein_ESM_EMB_{args.name_dataset.lower()}.npy', embeddings)
