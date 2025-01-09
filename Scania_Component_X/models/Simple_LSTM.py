from torch import nn
import torch
import math


class Simple_LSTM(nn.Module):
    def __init__(self,feature_num,sequence_len,hidden_dim,lstm_num_layers,
                 lstm_dropout,fc_layer_dim,fc_dropout,problem_type='regression',**kwargs):
        super(Simple_LSTM, self).__init__()

        self.feature_num = feature_num
        self.sequence_len = sequence_len

        self.lstm_hidden_size = hidden_dim
        self.lstm_num_layers = lstm_num_layers

        self.fc_layer_dim = fc_layer_dim
        self.fc_dropout = fc_dropout

        if problem_type == 'regression':
            self.output_dim = 1
        else:
            self.output_dim = 5
        self.lstm_dropout = lstm_dropout

        # lstm
        self.lstm = nn.LSTM(feature_num,
                            self.lstm_hidden_size,
                            num_layers=self.lstm_num_layers,
                            dropout=self.lstm_dropout,
                            batch_first=True)
        
        self.norm = nn.BatchNorm1d(self.lstm_hidden_size)

        # fc layers
        self.linear = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, self.fc_layer_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.fc_layer_dim),
            nn.Dropout(self.fc_dropout),
            nn.Linear(self.fc_layer_dim, self.output_dim),
        )
        if problem_type == 'regression':
            self.linear.add_module('output', nn.ReLU())
        else:
            self.linear.add_module('output', nn.Softmax(dim=-1)) 


    # x represents our data
    def forward(self, x):
        # LSTM/
        if x.isnan().any():
            print('input has a nan')
        x, _ = self.lstm(x)
        
        # Raw
        x = x.contiguous()
        x = x[:, -1, :]
        if x.isnan().any():
            print('x has nan after lstm')
        x = self.norm(x)
        if x.isnan().any():
            print('x has nan after norm')
        x = self.linear(x)
        if x.isnan().any():
            print('x has nan after linear')


        return x