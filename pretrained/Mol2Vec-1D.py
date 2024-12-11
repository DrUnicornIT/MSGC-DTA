import sys
print(f"ENV: {sys.executable}")

import argparse
import numpy as np
import pandas as pd
import json
from tdc.multi_pred import DTI

from rdkit import Chem
# from scipy import *
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import Word2Vec

def validate_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return True
    except:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drug')
    parser.add_argument('--dataset', type=str,
                        help='Name of dataset.', default="Kiba")
    parser.add_argument('--model_name', type=str,
                        help='Name of model.', default="model_300dim.pkl")
    args = parser.parse_args()

    model = Word2Vec.load('model/model_300dim.pkl')
    
    print("Extracting Embeddings for " + args.dataset + " dataset using " + args.model_name + " model")

    if args.dataset == "KibaCovid":
        drug_dict = json.load(open('../data/kiba_covid/drugs.txt'))
    else:
        dataset = DTI(name = args.dataset)
        data = dataset.get_data()
        drug_dict = dict(zip(data['Drug_ID'], data['Drug']))
        # protein_dict = dict(zip(data['Target_ID'], data['Target']))"

        print(f"Model: {model}")
        
    df_drug = pd.DataFrame.from_dict({"Drug_ID": drug_dict.keys(), "Drug": drug_dict.values()})
    
    df_drug['valid_smiles'] = df_drug['Drug'].apply(validate_smiles)
    
    df_drug['mol'] = df_drug['Drug'].apply(lambda x: Chem.MolFromSmiles(x))
    
    df_drug['sentence'] = df_drug.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
    
    df_drug['mol2vec'] = [DfVec(x) for x in sentences2vec(df_drug['sentence'], model, unseen='UNK')]
    df_drug.to_csv(f"drug/dataframe_Mol2Vec_{args.dataset.lower()}.csv")
    embedding = np.array([x.vec for x in df_drug['mol2vec']])
    print(f"Shape: {embedding.shape}")
    np.save(f"drug/unique_drug_Mol2Vec_EMB_{args.dataset.lower()}.npy", embedding)