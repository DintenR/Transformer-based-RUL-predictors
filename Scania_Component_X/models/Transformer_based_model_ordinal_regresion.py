from torch import nn
import torch
import math
from spacecutter.models import LogisticCumulativeLink


class TransformerEncoder_LSTM_OrdinalRegressor(nn.Module):
    def __init__(self,feature_num,sequence_len,transformer_encoder_head_num,hidden_dim,lstm_num_layers,
                 lstm_dropout,fc_layer_dim,fc_dropout,device,problem_type='classification',fc_num_layers=1):
        super(TransformerEncoder_LSTM_OrdinalRegressor, self).__init__()

        self.feature_num = feature_num
        self.sequence_len = sequence_len

        self.lstm_hidden_size = hidden_dim
        self.lstm_num_layers = lstm_num_layers

        self.fc_num_layers = fc_num_layers
        self.fc_layer_dim = fc_layer_dim
        self.fc_dropout = fc_dropout
        self.output_dim = 5
        self.lstm_dropout = lstm_dropout
        self.lstm_dropout = lstm_dropout

        self.transformer_encoder_head_num = transformer_encoder_head_num
        self.device = device
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.sequence_len, nhead=self.transformer_encoder_head_num, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)
        # lstm
        self.lstm = nn.LSTM(feature_num,
                            self.lstm_hidden_size,
                            num_layers=self.lstm_num_layers,
                            dropout=self.lstm_dropout,
                            batch_first=True)
        self.norm = nn.BatchNorm1d(self.lstm_hidden_size)
        # fc layers
        self.linear = nn.Sequential()
        self.linear.add_module(f'fc_0', nn.Sequential(
                nn.Linear(self.lstm_hidden_size, self.fc_layer_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(self.fc_layer_dim), 
                nn.Dropout(self.fc_dropout),
            ))
        if self.fc_num_layers > 1:
            for layer in range(1,self.fc_num_layers):
                self.linear.add_module(f'fc_{layer}', nn.Sequential(
                    nn.Linear(self.fc_layer_dim, self.fc_layer_dim),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(self.fc_layer_dim),
                    nn.Dropout(self.fc_dropout),
                ))
        self.linear.add_module('fc_output', nn.Sequential(
            nn.Linear(self.fc_layer_dim, 1),
            LogisticCumulativeLink(self.output_dim)
        ))



    # x represents our data
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)
        # LSTM
        x, _ = self.lstm(x)
        # Raw
        x = x.contiguous()
        x = x[:, -1, :]
        self.norm(x)
        x = self.linear(x)

        return x