import torch
import pytorch_lightning as pl
from dataset import MetroPTDataset
from scripts.utils import MODEL_MAP
from torch.nn import functional as F
import pandas as pd
from torchmetrics.functional import mean_squared_error
import numpy as np
from itertools import product
from sklearn.model_selection import ParameterSampler
from lightning_training import Module
import matplotlib.pyplot as plt


def load_model_from_checkpoint(checkpoint_path: str):
    model = Module.load_from_checkpoint(checkpoint_path)
    return model

def plot_real_vs_prediction(model,file_path):
    model.eval()
    dataloaders = MetroPTDataset.get_experiment_dataloaders(data_dir='../../../../Datasets/metropt',
            piece_wise_rul=30,
            window_size=30,
            test_pct=0.2,
            validation_rate=0.2,
            batch_size=32,)

    for i,dataloader in enumerate(dataloaders):
        
        y_hat = []
        y_true = []

        for batch in dataloader:
            x, y = batch
            x = x.to('cuda')
            y_true.extend(y.numpy().tolist())
            y_hat.extend(model(x).cpu().detach().numpy().tolist())
        
        # y_hat = np.array(y_hat).flatten()
        # print(y_hat.shape)
        # y_hat = np.array(pd.Series(y_hat).rolling(window=30).mean().bfill().values)
        plt.title(f'Experiment {i}')
        plt.plot(y_hat)
        plt.plot(y_true)
        plt.savefig(f'{file_path}experiment_{i}.png')
        plt.figure().clear()
        plt.close('all')

if __name__ == '__main__':

    checkpoints = [r'checkpoints\model-version_1-10min-w60-last5hours-only-analog\checkpoint-epoch=71-val_rmse=0.6419.ckpt',
                   r'checkpoints\model-version_1-10min-w60-last5hours-only-analog\checkpoint-epoch=39-val_rmse=0.7199.ckpt',]


    for i in range(len(checkpoints)):
        model = load_model_from_checkpoint(checkpoints[i])

        plot_real_vs_prediction(model,f'./images/model-version_1-10min-w60-last3hours-only-analog-all-subsets/experiments/')
        