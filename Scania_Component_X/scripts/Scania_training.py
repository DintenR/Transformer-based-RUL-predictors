import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from dataset import ScaniaDataset
from models import Simple_LSTM,TransformerEncoder_LSTM_1,TransformerEncoder_LSTM_2,TransformerEncoder_LSTM_3

import pytorch_lightning as pl
from scripts.utils import MODEL_MAP, scania_score, ScaniaLoss
from torchmetrics.functional import mean_squared_error
from torchmetrics.functional.classification import multiclass_f1_score
from itertools import product
from sklearn.model_selection import ParameterSampler

class ScaniaModule(pl.LightningModule):
    def __init__(self, lr, model_version, problem_type = 'classification', weights=None,**kwargs):
        assert problem_type in ['binary','classification'], 'Only regression or classification are supported for Scania dataset'
        super(ScaniaModule, self).__init__()
        self.problem_type = problem_type
        self.save_hyperparameters()
        self.net = MODEL_MAP[model_version](problem_type=self.problem_type,**kwargs)
        self.lr = lr
        self.validation_step_losses = []
        self.validation_step_lengths = []
        self.validation_step_outputs = []
        self.validation_targets = []
        self.test_step_outputs = []
        self.test_step_targets = []
        self.weights = weights#torch.tensor([1,5,5,5,1],dtype=torch.float32)
        #self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=weights) if self.problem_type == 'binary' else nn.CrossEntropyLoss(reduction='mean',weight=self.weights)
        self.loss_fn = ScaniaLoss()
        
        

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # print('Training step input:', x)
        x = self.net(x)
        # print('Training step output:', x)
        # print(y)
        loss = self.loss_fn(x, y.view(-1,y.shape[-1])) 
        # print('Training step loss:', loss)
        if self.problem_type == 'binary':
            self.log('train_loss', loss, prog_bar=True)
        else:
            accuracy = (x.argmax(dim=1) == y.view(-1,y.shape[-1]).argmax(dim=1)).sum().float() / float( x.size(0) )
            self.log('train_accuracy', accuracy, prog_bar=True)
            self.log('train_loss', loss, prog_bar=True)
            score = scania_score(y.view(-1,y.shape[-1]).argmax(dim=1), x.argmax(dim=1))
            f1 = multiclass_f1_score(x.argmax(dim=1), y.view(-1,y.shape[-1]).argmax(dim=1), num_classes=5, average='macro')
            self.log('train_score', score, prog_bar=True)
            self.log('train_f1', f1, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        #print('Validation step input:', x)
        x = self.net(x)
        loss = self.loss_fn(x, y.view(-1,y.shape[-1]))
        #print('Validation step output:', x)
        #print(y)
        # print('Validation step loss:', loss)
        self.validation_step_losses.append(loss)
        self.validation_step_lengths.append(len(y))
        if self.problem_type == 'classification':
            self.validation_step_outputs.extend(x)
            self.validation_targets.extend(y.view(-1,y.shape[-1]))

    def test_step(self, batch, batch_idx, reduction='sum'):
        x, y = batch
        x = self.net(x)
        self.test_step_outputs.extend(x)
        self.test_step_targets.extend(y.view(-1,y.shape[-1]))

    def on_test_epoch_end(self):
        if self.problem_type == 'binary':
            output = torch.stack(self.test_step_outputs, dim=0)
            target = torch.stack(self.test_step_targets, dim=0)
            accuracy = (output.argmax(dim=1) == target.argmax(dim=1)).sum().float() / float( target.size(0) )
            self.log('test_accuracy', accuracy)
        else:
            output = torch.stack(self.test_step_outputs, dim=0)
            target = torch.stack(self.test_step_targets, dim=0)
            accuracy = (output.argmax(dim=1) == target.argmax(dim=1)).sum().float() / float( target.size(0) )
            self.log('test_accuracy', accuracy)
            score = scania_score(target.argmax(dim=1), output.argmax(dim=1))
            f1 = multiclass_f1_score(output.argmax(dim=1), target.argmax(dim=1), num_classes=5, average='macro')
            self.log('test_score', score, prog_bar=True)
            self.log('test_f1', f1, prog_bar=True)
        self.test_step_outputs.clear()
        self.test_step_targets.clear()
        

    def on_validation_epoch_end(self):
        # Calculate the average loss
        # print('Validation step losses:', self.validation_step_losses)
        #loss = torch.sum(torch.tensor(self.validation_step_losses)) / torch.sum(torch.tensor(self.validation_step_lengths))
        loss = torch.stack(self.validation_step_losses, dim=0).mean()
        if self.problem_type == 'binary':     
            pass
        else:
            # loss = torch.stack(self.validation_step_losses, dim=0).sum()
            output = torch.stack(self.validation_step_outputs, dim=0)
            target = torch.stack(self.validation_targets, dim=0)
            accuracy = (output.argmax(dim=1) == target.argmax(dim=1)).sum().float() / float( target.size(0) )
            self.log('val_accuracy', accuracy)
            score = scania_score(target.argmax(dim=1), output.argmax(dim=1))
            f1 = multiclass_f1_score(output.argmax(dim=1), target.argmax(dim=1), num_classes=5, average='macro')
            self.log('val_score', score)
            self.log('val_f1', f1)
            self.validation_step_outputs.clear()
            self.validation_targets.clear()
        # Clear the lists
        self.log('val_loss', loss, prog_bar=True)
        self.validation_step_losses.clear()
        self.validation_step_lengths.clear()



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        return optimizer

def train_model_with_dataloaders(
        train_loader,
        valid_loader,
        test_loader,
        model_version = 'version_2',
        sequence_len = 30,
        feature_num = 106,
        transformer_encoder_head_num = 2,
        lstm_num_layers = 1,
        hidden_dim = 32,
        lstm_dropout = 0.3,
        fc_num_layers = 1,
        fc_layer_dim = 32,
        fc_dropout = 0.1,
        device = 'cuda',
        lr = 0.0002,
        patience = 10,
        problem_type = 'classification',
        weights=None
        ):
    if problem_type == 'regression':
        scores = pd.DataFrame(columns=['train_rmse', 'val_rmse', 'test_rmse'])
    elif problem_type == 'classification':
        scores = pd.DataFrame(columns=['train_accuracy','train_f1','train_score', 'val_accuracy', 'val_f1','val_score', 'test_accuracy','test_f1','test_score'])
    model_kwargs = {
        'model_version': model_version,
        'sequence_len': sequence_len,
        'feature_num': feature_num,
        'hidden_dim': hidden_dim,
        'fc_num_layers': fc_num_layers,
        'fc_layer_dim': fc_layer_dim,
        'lstm_num_layers': lstm_num_layers,
        'transformer_encoder_head_num': transformer_encoder_head_num,
        'fc_dropout': fc_dropout,
        'lstm_dropout': lstm_dropout,
        'device': device,
        'problem_type': problem_type,
        'weights': weights

    }

    model = ScaniaModule(
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
    if problem_type == 'binary':
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=f'./checkpoints/model-{model_version}-binary-definitive',
            monitor='val_loss',
            filename='checkpoint-{epoch:02d}-{val_loss:.4f}',
            save_top_k=1,
            mode='min',
        )
    elif problem_type == 'classification':
         checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=f'./checkpoints/model-{model_version}-definitive',
            monitor='val_loss',
            filename='checkpoint-{epoch:02d}-{val_score:.4f}',
            save_top_k=1,
            mode='min',
        )

    trainer = pl.Trainer(
        default_root_dir='./checkpoints',
        accelerator='gpu',
        devices=1,
        max_epochs=500,
        callbacks=[early_stop_callback, checkpoint_callback],
        # gradient_clip_val=0.1,
        detect_anomaly=False,
        # checkpoint_callback=False,
        # logger=False,
        # progress_bar_refresh_rate=0
    )
    
    trainer.fit(model, train_loader, val_dataloaders=valid_loader)
    
    if problem_type == 'classification':
        print('###############################')
        print(f'Testing on train data')
        print('###############################')
        trainer.test(dataloaders=train_loader)
        print('###############################')
        print(f'Testing on validation data')
        print('###############################')
        trainer.test(dataloaders=valid_loader)
        print('###############################')
        print(f'Testing on test data')
        print('###############################')
        trainer.test(dataloaders=test_loader)
    print(f'Best model path: {trainer.checkpoint_callback.best_model_path}')
   

def main():
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch Scania Training')

    parser.add_argument('--sequence-len', type=int, default=200)
    parser.add_argument('--hidden-dim', type=int, default=128, help='LSTM hidden dims')
    parser.add_argument('--fc-layer-dim', type=int, default=128)
    parser.add_argument('--lstm-num-layers', type=int, default=5)
    parser.add_argument('--lstm-dropout', type=float, default=0.3)
    parser.add_argument('--tranformer-head-num', type=int, default=10)
    parser.add_argument('--fc-num-layers', type=int, default=3)
    parser.add_argument('--fc-dropout', type=float, default=0.2)
    parser.add_argument('--dataset-root', type=str, required=True, help='The dir of Scania dataset files')
    parser.add_argument('--validation-rate', type=float, default=0, help='validation set ratio of train set')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--patience', type=int, default=10, help='Early Stop Patience')
    parser.add_argument('--max-epochs', type=int, default=500)
    parser.add_argument('--model-version', type=str, default='version_1', help='model version to train values: version_1, lstm')
    parser.add_argument('--cluster-specifications', action='store_true', help='Cluster the specifications')
    parser.add_argument('--only-two-classes', action='store_true', help='Only use two classes')
    parser.add_argument('--undersample', type=float, default=0.1, help='Undersample the data')
    parser.add_argument('--include-specifications', action='store_true', help='Include the specifications')
    parser.add_argument('--histogram-normalizer', action='store_true', help='Normalize the data with histogram')
    parser.add_argument('--forward-fill', action='store_true', help='Forward fill the data')
    
    args = parser.parse_args()
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("high")
    weights = None
    only_two_classes = args.only_two_classes
    problem_type = 'binary' if only_two_classes else 'classification'
    print(weights)
    window_size = args.sequence_len
    cluster_specifications = args.cluster_specifications
    train_loader, valid_loader, test_loader= ScaniaDataset.get_dataloaders(
        data_dir=args.dataset_root,
        window_size=window_size,
        batch_size=args.batch_size,
        validation_rate=args.validation_rate,
        stored_subsets=False,
        cluster_specifications=cluster_specifications,
        undersample=args.undersample,
        only_two_classes=only_two_classes,
        include_specifications=args.include_specifications,
        histogram_normalizer=args.histogram_normalizer,
        forward_fill=args.forward_fill,
        pca=False)

    label_counts = train_loader.dataset.get_label_counts()
    print(label_counts)
    if only_two_classes:
        label_counts = np.array([label_counts[0],label_counts.sum()-label_counts[0]])
        weights = torch.tensor(label_counts[0] / label_counts[1],dtype=torch.float32)
    # if one lable is missing we set the weight small so that the model will not be biased towards the other labels
    else:
        num_labels_per_class = np.array([label_counts[i] if label_counts[i] > 0 else label_counts.sum()*100 for i in range(5)])
        weights = torch.tensor(label_counts.sum() / (num_labels_per_class * 5),dtype=torch.float32)

    feature_num = train_loader.dataset.get_features()
    
    train_model_with_dataloaders(
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        model_version=args.model_version,
        sequence_len=window_size,
        feature_num=feature_num,
        transformer_encoder_head_num=args.tranformer_head_num,
        lstm_num_layers=args.lstm_num_layers,
        hidden_dim=args.hidden_dim,
        lstm_dropout=args.lstm_dropout,
        fc_num_layers=args.fc_num_layers,
        fc_layer_dim=args.fc_layer_dim,
        fc_dropout=args.fc_dropout,
        device='cuda',
        lr=args.lr,
        patience=args.patience,
        problem_type=problem_type,
        weights=weights
    )

if __name__ == '__main__':
    main()
