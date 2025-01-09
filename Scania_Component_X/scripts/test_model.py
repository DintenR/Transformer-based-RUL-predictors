import torch
import pytorch_lightning as pl
from dataset import ScaniaDataset
from scripts.utils import MODEL_MAP, scania_score
from torch.nn import functional as F
import pandas as pd
from torchmetrics.functional import mean_squared_error
from torchmetrics.functional.classification import multiclass_f1_score
import numpy as np
from itertools import product
from sklearn.model_selection import ParameterSampler
from Scania_training import ScaniaModule
from Scania_training_ordinal_regression import ScaniaModuleOrdinalRegression
import matplotlib.pyplot as plt


def load_model_from_checkpoint(checkpoint_path: str, model_version):
    if model_version == 'ordinal_regression':
        model = ScaniaModuleOrdinalRegression.load_from_checkpoint(checkpoint_path)
    else:
        model = ScaniaModule.load_from_checkpoint(checkpoint_path)
    return model

def print_confusion_matrix(checkpoints):
    data_dir=r'..\..\..\..\Datasets\Scania\data'
    sequence_len=200
    batch_size=1
    piecewise_rul=30
    train_loader, valid_loader, test_loader= ScaniaDataset.get_dataloaders(
        data_dir=data_dir,
        test_pct=0,
        piece_wise_rul=piecewise_rul,
        window_size=sequence_len,
        batch_size=batch_size,
        validation_rate=0,
        stored_subsets=False,
        undersample=0.3,
        cluster_specifications=False,
        include_specifications=False,
        histogram_normalizer=False,
        forward_fill=False,
        pca=False) 

    dataloaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }
    for model_name, path in checkpoints.items():
        print('###############################')
        print(f'Results for model {model_name}')
        print('###############################')
        print('\n')
        model = load_model_from_checkpoint(path)
        model.eval()
        for dataset, dataloader in dataloaders.items():
            
            y_hat = []
            y_true = []

            for batch in dataloader:
                x, y = batch
                x = x.to('cuda')
                y_true.extend(y.view(-1,5).argmax(dim=1).numpy().tolist())
                y_hat.extend(model(x).argmax(dim=1).cpu().detach().numpy().tolist())
            
            # print confusion matrix
            print(f'Confusion matrix {dataset} dataset')
            print('-'*50)
            confusion_matrix = pd.crosstab(pd.Series(y_true), pd.Series(y_hat), rownames=['Real'], colnames=['Predicted'])
            print(confusion_matrix)
            print('\n')
            print('-'*50)
            print('Accuracy:', (np.array(y_true) == np.array(y_hat)).mean())
            print('F1 Score:', multiclass_f1_score(torch.tensor(y_true), torch.tensor(y_hat), num_classes=5, average='macro').item())
            print('Score:   ', scania_score(torch.tensor(y_true), torch.tensor(y_hat)).item())
            print('-'*50)
            print('\n\n')

if __name__ == '__main__':
    #seed everything
    pl.seed_everything(42)
    checkpoints = {
        'version_1': r'..\checkpoints\version_1\checkpoint-epoch=01-val_score=46621.0000.ckpt',
        'ordinal_regression': r'..\checkpoints\ordinal_regression\checkpoint-epoch=01-val_accuracy=0.2230.ckpt', # Best model regression
    }
    
    print_confusion_matrix(checkpoints)
    
    
