import sys
print(f"ENV: {sys.executable}")

import argparse
import biovec
import numpy as np

from tdc.multi_pred import DTI

def extracting_embedding_probert(model, data):

    embed = {}
    sumEmbed = []
    for key, value in data.items():
        embedding = model.to_vecs(value)

        embed[key] = embedding        
        sumE = embedding[0]+embedding[1]+embedding[2]
        sumEmbed.append(sumE)
        
    sumEmbed = np.array(sumEmbed)
    print(f"Shape Embedding: {sumEmbed.shape}")
    return sumEmbed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Protein ')
    parser.add_argument('--dataset', type=str,
                        help='Name of dataset.', default="Kiba")
    parser.add_argument('--model_name', type=str,
                        help='Name of model.', default="swissprot-reviewed-protvec")
    args = parser.parse_args()
    
    print("Extracting Embeddings for " + args.dataset + " dataset using " + args.model_name + " model")
    dataset = DTI(name = args.dataset)
    data = dataset.get_data()
    protein_dict = dict(zip(data['Target_ID'], data['Target']))

    protein_dict

    pv = biovec.models.load_protvec('model/swissprot-reviewed-protvec.model')

    print(f"Model: {pv}")
    embedding = extracting_embedding_probert(pv, protein_dict)
    np.save(f"protein/unique_protein_ProVec_EMB_{args.dataset.lower()}.npy", embedding)