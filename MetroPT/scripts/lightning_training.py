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



class Module(pl.LightningModule):
    def __init__(self, lr, model_version, **kwargs):
        super(Module, self).__init__()
        self.save_hyperparameters()
        self.net = MODEL_MAP[model_version](**kwargs)
        self.lr = lr
        self.validation_step_losses = []
        self.validation_step_lengths = []
        self.test_step_outputs = []
        self.test_step_targets = []

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.net(x)
        loss = F.mse_loss(x, y) 
        self.log('train_rmse', torch.sqrt(loss), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.net(x)
        loss = F.mse_loss(x, y, reduction='sum')
        self.validation_step_losses.append(loss)
        self.validation_step_lengths.append(len(y))

    def test_step(self, batch, batch_idx, reduction='sum'):
        x, y = batch
        x = self.net(x)
        self.test_step_outputs.extend(x)
        self.test_step_targets.extend(y)

    def on_test_epoch_end(self):
        rmse = mean_squared_error(torch.tensor(self.test_step_outputs), torch.tensor(self.test_step_targets), squared=False)
        self.test_step_outputs.clear()
        self.test_step_targets.clear()
        self.log('test_rmse', rmse)

    def on_validation_epoch_end(self):
        # Calculate the average loss
        mse = torch.sum(torch.tensor(self.validation_step_losses)) / torch.sum(torch.tensor(self.validation_step_lengths))
        rmse = torch.sqrt(mse)
        # Clear the lists
        self.validation_step_losses.clear()
        self.validation_step_lengths.clear()
        # Log the results
        self.log('val_loss', mse, prog_bar=True)
        self.log('val_rmse', rmse)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        return optimizer


def train_model(
        data_dir = None,
        model_version = 'version_2',
        sequence_len = 30,
        feature_num = 14,
        transformer_encoder_head_num = 2,
        lstm_num_layers = 3,
        hidden_dim = 32,
        lstm_dropout = 0.2,
        fc_layer_dim = 32,
        fc_dropout = 0.2,
        device = 'cuda',
        batch_size = 256,
        piecewise_rul = 30,
        lr = 0.001,
        patience = 10,
        test_pct = 0.2,
        validation_rate = 0
        ):
    scores = pd.DataFrame(columns=['train_rmse', 'val_rmse', 'test_rmse'])
    model_kwargs = {
        'model_version': model_version,
        'sequence_len': sequence_len,
        'feature_num': feature_num,
        'hidden_dim': hidden_dim,
        'fc_layer_dim': fc_layer_dim,
        'lstm_num_layers': lstm_num_layers,
        'transformer_encoder_head_num': transformer_encoder_head_num,
        'fc_dropout': fc_dropout,
        'lstm_dropout': lstm_dropout,
        'device': device

    }
    train_loader, valid_loader, test_loader= MetroPTDataset.get_dataloaders(
        data_dir=data_dir,
        test_pct=test_pct,
        piece_wise_rul=piecewise_rul,
        window_size=sequence_len,
        batch_size=batch_size,
        validation_rate=validation_rate) 

    model = Module(
        lr=lr,
        **model_kwargs
    )
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=patience,
        verbose=False,
        mode='min'
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f'./checkpoints/def/model-{model_version}-10min-w60-last5days-only-analog',
        monitor='val_loss',
        filename='checkpoint-{epoch:02d}-{val_rmse:.4f}',
        save_top_k=1,
        mode='min',
    )

    trainer = pl.Trainer(
        default_root_dir='./checkpoints',
        accelerator='gpu',
        devices=1,
        max_epochs=500,
        callbacks=[early_stop_callback, checkpoint_callback],
        # checkpoint_callback=False,
        # logger=False,
        # progress_bar_refresh_rate=0
    )
    trainer.fit(model, train_loader, val_dataloaders=valid_loader or test_loader)
    t = trainer.callback_metrics
    train_rmse = t['train_rmse']
    val_rmse = t['val_rmse']
    trainer.test(dataloaders=test_loader)
    t = trainer.callback_metrics
    test_rmse = t['test_rmse']
    # Add the results to the dataframe
    scores.loc[0] = [train_rmse, val_rmse, test_rmse]
    # Save the results
    scores.to_csv(f'./results/def/last5days/{model_version}-1min-w60-only-analog_results.csv', index=False)

    # Save model predictions
    model = Module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    model.to(device)
    model.eval()
    predictions = []
    targets = []
    for x, y in test_loader:
        x = x.to(device)
        y_hat = model(x)
        predictions.extend(y_hat.cpu().detach().numpy())
        targets.extend(y.cpu().detach().numpy())
    predictions = np.array(predictions)
    targets = np.array(targets)
    predictions.tofile(f'./results/{model_version}_test_predictions.csv', sep=',')
    targets.tofile(f'./results/{model_version}_test_targets.csv', sep=',')
    predictions = []
    targets = []
    for x, y in train_loader:
        x = x.to(device)
        y_hat = model(x)
        predictions.extend(y_hat.cpu().detach().numpy())
        targets.extend(y.cpu().detach().numpy())
    predictions = np.array(predictions)
    targets = np.array(targets)
    predictions.tofile(f'./results/{model_version}_train_predictions.csv', sep=',')
    targets.tofile(f'./results/{model_version}_train_targets.csv', sep=',')

    

def leave_one_out_model_evaluation(
        model_version = 'version_2',
        data_dir = None,
        sequence_len = 144,
        feature_num = 14,
        transformer_encoder_head_num = 2,
        lstm_num_layers = 3,
        hidden_dim = 32,
        lstm_dropout = 0.2,
        fc_layer_dim = 32,
        fc_dropout = 0.2,
        device = 'cuda',
        batch_size = 32,
        combination_index = 0,
        piecewise_rul = 30,
        lr = 0.001,
        patience = 10,
        validation_rate = 0
        ):
    model_kwargs = {
        'sequence_len': sequence_len,
        'feature_num': feature_num,
        'transformer_encoder_head_num': transformer_encoder_head_num,
        'lstm_num_layers': lstm_num_layers,
        'hidden_dim': hidden_dim,
        'lstm_dropout': lstm_dropout,
        'fc_layer_dim': fc_layer_dim,
        'fc_dropout': fc_dropout,
        'device': device
    }

    num_training_subsets = 5

    score_from_splits = pd.DataFrame(columns=['split_id', 'train_rmse', 'val_rmse', 'test_rmse']) 

    for i in range(num_training_subsets):
        train_loader, val_loader, test_loader = MetroPTDataset.get_leave_one_out_dataloaders_by_index(
            data_dir=data_dir,
            piece_wise_rul=piecewise_rul,
            window_size=sequence_len,
            batch_size=batch_size,
            index=i,
            validation_rate=validation_rate
        )
        print(train_loader.dataset.__len__(), val_loader.dataset.__len__(), test_loader.dataset.__len__())
        model = Module(
            model_version=model_version,
            lr=lr,
            **model_kwargs
        )
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor='val_rmse',
            min_delta=0.00,
            patience=patience,
            verbose=False,
            mode='min'
        )
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=f'./checkpoints/def/model-{model_version}-combination-{combination_index}-1min-w60-last4days-only-analog',
            monitor='val_rmse',
            filename='checkpoint-fold-'+str(i),
            save_top_k=1,
            mode='min',
        )

        trainer = pl.Trainer(
            default_root_dir='./checkpoints',
            accelerator='gpu',
            devices=1,
            max_epochs=500,
            callbacks=[early_stop_callback, checkpoint_callback],
            # checkpoint_callback=False,
            # logger=False,
            # progress_bar_refresh_rate=0
        )
        trainer.fit(model, train_loader, val_dataloaders=val_loader or test_loader)
        t = trainer.callback_metrics
        train_rmse = t['train_rmse']
        val_rmse = t['val_rmse']
        trainer.test(dataloaders=test_loader)
        t = trainer.callback_metrics
        test_rmse = t['test_rmse']
        # Add the results to the dataframe
        score_from_splits.loc[i] = [i, train_rmse, val_rmse, test_rmse,]
    # Save the results
    score_from_splits.to_csv(f'./results/def/last5hours/{model_version}_1min-w60-last4days-only-analog_cross_validation_results.csv', index=False)



def random_search_from_dict(param_grid, n_iter):
    keys, values = zip(*param_grid.items())
    for _ in range(n_iter):
        sample = {key: np.random.choice(val) for key, val in zip(keys, values)}
        yield sample

def get_param_combinations_from_dict(param_grid):
    # Extract parameter names and values
    param_names = param_grid.keys()
    param_values = param_grid.values()

    # Generate all combinations
    combinations = list(product(*param_values))

    # Create a list of dictionaries for each combination
    combination_dicts = [dict(zip(param_names, combo)) for combo in combinations]
    return combination_dicts

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch MetroPT Training')

    parser.add_argument('--sequence-len', type=int, default=30)
    parser.add_argument('--feature-num', type=int, default=2)
    parser.add_argument('--hidden-dim', type=int, default=128, help='LSTM hidden dims')
    parser.add_argument('--fc-layer-dim', type=int, default=128)
    parser.add_argument('--rnn-num-layers', type=int, default=3)
    parser.add_argument('--lstm-dropout', type=float, default=0.2)
    parser.add_argument('--feature-head-num', type=int, default=2)
    parser.add_argument('--fc-dropout', type=float, default=0.5)
    parser.add_argument('--dataset-root', type=str, required=True, help='The dir of MetroPT dataset files')
    parser.add_argument('--max-rul', type=int, default=30, help='piece-wise RUL')
    parser.add_argument('--validation-rate', type=float, default=0.2, help='validation set ratio of train set')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--patience', type=int, default=50, help='Early Stop Patience')
    parser.add_argument('--max-epochs', type=int, default=500)
    parser.add_argument('--model-version', type=str, default='version_1', help='model version to train values: version_1, version_2, version_3, lstm')
    parser.add_argument('--cross-validation', action='store_true', help='Perform cross validation')
    args = parser.parse_args()
    
    pl.seed_everything(42, workers=True)

    if args.cross_validation:
        for model_version in ['version_1','version_2','version_3','lstm']:
            leave_one_out_model_evaluation(
                model_version=args.model_version,
                data_dir=args.dataset_root,
                sequence_len=args.sequence_len,
                feature_num=args.feature_num,
                transformer_encoder_head_num=args.feature_head_num,
                lstm_num_layers=args.rnn_num_layers,
                hidden_dim=args.hidden_dim,
                lstm_dropout=args.lstm_dropout,
                fc_layer_dim=args.fc_layer_dim,
                fc_dropout=args.fc_dropout,
                device='cuda',
                batch_size=args.batch_size,
                piecewise_rul=args.max_rul,
                lr=args.lr,
                patience=args.patience,
                validation_rate=args.validation_rate
            )
    else:
        for model_version in ['version_1','version_2','version_3','lstm']:
            train_model(
                data_dir=args.dataset_root,
                model_version=model_version,
                sequence_len=args.sequence_len,
                feature_num=args.feature_num,
                transformer_encoder_head_num=args.feature_head_num,
                lstm_num_layers=args.rnn_num_layers,
                hidden_dim=args.hidden_dim,
                lstm_dropout=args.lstm_dropout,
                fc_layer_dim=args.fc_layer_dim,
                fc_dropout=args.fc_dropout,
                device='cuda',
                batch_size=args.batch_size,
                piecewise_rul=args.max_rul,
                lr=0.001,#args.lr,
                patience=args.patience,
                #test_pct=args.validation_rate,
                validation_rate=args.validation_rate
            )

