python ProVec-1D.py --dataset Kiba --model swissprot-reviewed-protvec.model
python Mol2Vec-1D.py --dataset Kiba --model model_300dim.pkl
python GIN-2D.py --dataset Kiba --model 'gin_supervised_contextpred' --out-dir drug
python ProBERT-2D.py --dataset Kiba --model bert-base
python ESM2-3D.py --dataset Kiba --model bert-base
python E3NN-3D.py --dataset Kiba --model bert-base