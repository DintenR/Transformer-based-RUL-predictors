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
    model = Module.load_from_checkpoint(checkpoint_path)#, model_version='version_3', lr=0.001)
    return model

if __name__ == '__main__':

    version_1_1min_w60_4days_checkpoint_paths = [
        r'checkpoints\model-version_1-combination-0-1min-w60-last4days-only-analog\checkpoint-fold-0.ckpt',
        r'checkpoints\model-version_1-combination-0-1min-w60-last4days-only-analog\checkpoint-fold-1.ckpt',
        r'checkpoints\model-version_1-combination-0-1min-w60-last4days-only-analog\checkpoint-fold-2.ckpt',
        r'checkpoints\model-version_1-combination-0-1min-w60-last4days-only-analog\checkpoint-fold-3.ckpt',
        r'checkpoints\model-version_1-combination-0-1min-w60-last4days-only-analog\checkpoint-fold-4.ckpt',
    ]


    for i in range(len(version_1_1min_w60_4days_checkpoint_paths)):
        model = load_model_from_checkpoint(version_1_1min_w60_4days_checkpoint_paths[i])
        train_loader, val_loader, test_loader = MetroPTDataset.get_leave_one_out_dataloaders_by_index(
                data_dir='../../../Datasets/metropt',
                piece_wise_rul=30,
                window_size=60,
                batch_size=32,
                index=i
            )
        model = model.to('cuda')
        model.eval()

        y_hat = []
        y_true = []

        model_version = 'version_1'

        for batch in train_loader:
            x, y = batch
            x = x.to('cuda')
            y_true.extend(y.numpy().tolist())
            y_hat.extend(model(x).cpu().detach().numpy().tolist())
        
        plt.plot(y_hat)
        plt.plot(y_true)
        plt.savefig(f'./images/{model_version}/{model_version}_1min_w60_train_fold{i}.png')
        plt.figure().clear()

        y_hat = []
        y_true = []

        if val_loader is not None:
            for batch in val_loader:
                x, y = batch
                x = x.to('cuda')
                y_true.extend(y.numpy().tolist())
                y_hat.extend(model(x).cpu().detach().numpy().tolist())
            
            plt.figure(figsize=(20,10))
            plt.plot(y_hat)
            plt.plot(y_true)
            plt.savefig(f'./images/{model_version}/{model_version}_1min_w60_val_fold{i}.png')
            plt.figure().clear()        

        y_hat = []
        y_true = []
        for batch in test_loader:
            x, y = batch
            x = x.to('cuda')
            y_true.extend(y.numpy().tolist())
            y_hat.extend(model(x).cpu().detach().numpy().tolist())
        
        plt.plot(y_hat)
        plt.plot(y_true)
        plt.savefig(f'./images/{model_version}/{model_version}_1min_w60_test_fold{i}.png')
        plt.figure().clear()
        plt.close('all')
    
