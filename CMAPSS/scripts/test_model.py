import torch
import pytorch_lightning as pl
from dataset import CMAPSSDataset
from torch.nn import functional as F
from scripts.utils import MODEL_MAP, score
import pandas as pd
from pytorch_lightning.metrics.functional import mean_squared_error
import numpy as np
from cmapss_script import Module


def load_model_from_checkpoint(checkpoint_path: str):
    model = Module.load_from_checkpoint(checkpoint_path)#, model_version='version_3', lr=0.001)
    return model

if __name__ == '__main__':

    sub_dataset = 'FD001'
    checkpoint_path = 'path_to_your_checkpoint.ckpt'
    
    print(f'Loading model for {sub_dataset} from {checkpoint_path}')
    model = load_model_from_checkpoint(checkpoint_path)
    model.eval()
    # Load the dataset
    train_loader, test_loader, valid_loader = CMAPSSDataset.get_data_loaders(
    # dataset_root=data_dir,
    sequence_len=60,
    sub_dataset=sub_dataset,
    norm_type='z-score',
    max_rul=125,
    cluster_operations=True,
    norm_by_operations=True,
    use_max_rul_on_test=True,
    validation_rate=0.2,
    return_id=False,
    use_only_final_on_test=True,
    loader_kwargs={'batch_size': 32}
    )

    # Test the model
    test_results = []
    for batch in test_loader:
        x, y = batch
        y_hat = model(x)
        test_results.append((y, y_hat))

    # Calculate RMSE
    rmse = np.sqrt(np.mean([(y - y_hat).pow(2).mean().item() for y, y_hat in test_results]))
    # Calculate score
    y_true = torch.cat([y for y, _ in test_results])
    y_pred = torch.cat([y_hat for _, y_hat in test_results])
    print(f'Test RMSE for {sub_dataset}: {rmse}')
    print(f'Score for {sub_dataset}: {score(y_pred.detach().numpy(), y_true.detach().numpy())}')