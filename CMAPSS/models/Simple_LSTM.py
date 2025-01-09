from torch import nn
import torch
import math


class Simple_LSTM(nn.Module):
    def __init__(self,feature_num,sequence_len,hidden_dim,lstm_num_layers,
                 lstm_dropout,fc_layer_dim,fc_dropout,**kwargs):
        super(Simple_LSTM, self).__init__()

        self.feature_num = feature_num
        self.sequence_len = sequence_len

        self.lstm_hidden_size = hidden_dim
        self.lstm_num_layers = lstm_num_layers

        self.fc_layer_dim = fc_layer_dim
        self.fc_dropout = fc_dropout

        self.output_dim = 1
        self.lstm_dropout = lstm_dropout

        # lstm
        self.lstm = nn.LSTM(feature_num,
                            self.lstm_hidden_size,
                            num_layers=self.lstm_num_layers,
                            dropout=self.lstm_dropout)

        # fc layers
        self.linear = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, self.fc_layer_dim),
            nn.ReLU(),
            nn.Dropout(self.fc_dropout),
            nn.Linear(self.fc_layer_dim, self.output_dim),
        )


    # x represents our data
    def forward(self, x):
        # LSTM/
        x, _ = self.lstm(x)
        
        # Raw
        x = x.contiguous()
        x = x[:, -1, :]
        
        x = self.linear(x)

        return x