import sys
print(f"ENV {sys.executable}")

import argparse
import torch
import tape
import numpy as np
import json
from tqdm import tqdm
from tdc.multi_pred import DTI

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

    model = tape.ProteinBertModel.from_pretrained('bert-base')
    tokenizer = tape.TAPETokenizer(vocab='iupac')

    embeddings = []

    for target in tqdm(data):
        sequence = target
    
        input_ids = np.array(tokenizer.encode(sequence))
        with torch.no_grad():
            outputs = model(torch.tensor([input_ids]))
            embedding = outputs[0][:, 0, :]
    
        embeddings.append(embedding)
        
    embeddings = torch.cat(embeddings, dim=0)
    
    print(f"Shape: {embeddings.shape}")    
    np.save(f'protein/unique_protein_BERT_EMB_{args.dataset.lower()}.npy', embeddings)
