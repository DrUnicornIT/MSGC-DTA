# MSGC-DTA
DỰ ĐOÁN MỨC BÁM ĐÍCH CỦA THUỐC SỬ DỤNG HỌC TƯƠNG PHẢN ĐỒ THỊ ĐA QUY MÔ

## Requirements
- numpy == 1.17.4 
- kreas == 2.3.1 
- pytorch == 1.8.0 
- matplotlib==3.2.2 
- pandas==1.2.4
- PyG (torch-geometric) == 1.3.2
- rdkit==2009.Q1-1
- tqdm==4.51.0
- torch_geometric
- scipy==1.10.1
- scikit_learn==0.24.2 <br />

## :rainbow: Datasets

All publicly accessible datasets used can be accessed here:

| Dataset Name        | Link                                                |
|---------------------|-----------------------------------------------------|
| Davis, KIBA         | https://github.com/hkmztrk/DeepDTA/tree/master/data |


## Pre-trained model

All publicly accessible models used can be accessed here:
Support link [Notebook]('https://colab.research.google.com/drive/1zEOEflHtHPmrYjz-zmr108qolSofv0S4?usp=sharing')

| Model Name | Link                                        |
|------------|---------------------------------------------|
| E3NN-3D    | https://github.com/deepmodeling/Uni-Mol |
| ESM2-3D    | https://huggingface.co/facebook/esm2_t33_650M_UR50D     |
| GIN-2D    | https://github.com/wenx00/dgl-lifesci/blob/master/examples/molecule_embeddings/main.py     |
| ProBERT-2D    | https://github.com/songlab-cal/tape   |
| Mol2Vec-1D    | https://github.com/samoturk/mol2vec     |
| ProVec-1D    | https://github.com/facebookresearch/esm     |

##  Install tutorial

1. Clone the repository
    ``` shell
   git clone https://github.com/DrUnicornIT/MSGC-DTA.git
   cd repository
   ```
2. Install the required dependencies
    ``` shell
    pip install -r requirements
    ```

## Preprocess script


1. Prepare the data need for train.
    ``` shell
   python cm_generation.py
   ```
2. Prepare embedding pretrained model
    ``` shell
    ./pretrained/run_extracting_embedding.sh
    ```

## Training


1. Prepare the data need for train.
    ``` shell
   python cm_generation.py
   ```
2. Prepare embedding pretrained model
    ``` shell
    ./pretrained/run_extracting_embedding.sh
    ```


# Inference Script Guide

This document provides guidance on how to use the `inference.py` script, including the various arguments and their purposes.

## Training

```shell
python training.py <data_path_> <cuda> ... <epochs> <batch_size>
```

The `training.py` script supports the following arguments:

| Argument             | Type   | Default Value                       | Description                                   |
|----------------------|--------|-------------------------------------|-----------------------------------------------|
| `--seed`             | `int`  | `2024`                              | Random seed for reproducibility.             |
| `--gpus`             | `str`  | `'0'`                               | Number of GPUs to use (`0` for CPU).         |
| `--cuda`             | `int`  | `0`                                 | Index of the CUDA device to use.             |
| `--data_path`        | `str`  | `'/kaggle/input/msgc-dta/MSGC-DTA/data/'` | Path to the dataset.                       |
| `--dataset`          | `str`  | `'davis'`                           | Dataset name (`davis` or `kiba`).            |
| `--epochs`           | `int`  | `6000`                              | Number of training epochs (`3000` for kiba). |
| `--batch_size`       | `int`  | `512`                               | Batch size for training.                     |
| `--lr`               | `float`| `0.0002`                            | Learning rate for the optimizer.             |
| `--edge_dropout_rate`| `float`| `0.2`                               | Dropout rate for edges (`0.0` for kiba).     |
| `--tau`              | `float`| `0.8`                               | Scaling factor for contrastive loss.         |
| `--lam`              | `float`| `0.5`                               | Weight for regularization term.              |
| `--num_pos`          | `int`  | `3`                                 | Number of positive samples (`10` for kiba).  |
| `--pos_threshold`    | `float`| `8.0`                               | Threshold for positive sample selection.     |