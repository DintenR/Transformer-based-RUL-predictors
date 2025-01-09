from torch import nn
import torch
import math


class TransformerEncoder_LSTM_1(nn.Module):
    
    def __init__(self,feature_num,sequence_len,transformer_encoder_head_num,hidden_dim,lstm_num_layers,
                 lstm_dropout,fc_layer_dim,fc_dropout,device):
        super(TransformerEncoder_LSTM_1, self).__init__()

        self.feature_num = feature_num
        self.sequence_len = sequence_len

        self.lstm_hidden_size = hidden_dim
        self.lstm_num_layers = lstm_num_layers

        self.fc_layer_dim = fc_layer_dim
        self.fc_dropout = fc_dropout

        self.output_dim = 1
        self.lstm_dropout = lstm_dropout

        self.transformer_encoder_head_num = transformer_encoder_head_num

        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=self.sequence_len, nhead=self.transformer_encoder_head_num,
        )
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
        x = x.permute(0, 2, 1)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)
        # LSTM/
        x, _ = self.lstm(x)
        
        # Raw
        x = x.contiguous()
        x = x[:, -1, :]
        
        x = self.linear(x)

        return x