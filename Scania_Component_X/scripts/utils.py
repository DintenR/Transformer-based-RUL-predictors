from models import TransformerEncoder_LSTM_1, TransformerEncoder_LSTM_2, TransformerEncoder_LSTM_3, Simple_LSTM
import torch
import torch.nn as nn

MODEL_MAP = {
    'version_1': TransformerEncoder_LSTM_1,
    'lstm': Simple_LSTM,
}

def scania_score(y_true, y_pred):
    error = y_true - y_pred

    error_neg = abs(error[error < 0]) + 6
    error_pos = (error[error > 0]+1) * 100

    return torch.cat((error_neg,error_pos)).sum().float()

class ScaniaLoss(nn.Module):
    def __init__(self):
        super(ScaniaLoss, self).__init__()

    def softargmax1d(self,input, beta=100):
        n = 5
        input = nn.functional.softmax(beta * input, dim=-1)
        indices = torch.linspace(0, 1, n).to(input.device)
        result = torch.sum((n - 1) * input * indices, dim=-1)
        return result

    def forward(self, y_hat, y):
        y = self.softargmax1d(y.view(-1,y.shape[-1]))
        y_hat = self.softargmax1d(y_hat)
        
        # Calculate the error
        error = y - y_hat
        
        # Calculate the loss based on the error
        error_neg = abs(error[error < - 0.1]) + 6
        error_pos = (error[error > 0.1]+1) * 100

        return torch.cat((error_neg,error_pos)).sum().float()