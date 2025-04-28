from models import TransformerEncoder_LSTM_1, TransformerEncoder_LSTM_2, TransformerEncoder_LSTM_3, Simple_LSTM

MODEL_MAP = {
    'version_1': TransformerEncoder_LSTM_1,
    'version_2': TransformerEncoder_LSTM_2,
    'version_3': TransformerEncoder_LSTM_3,
    'lstm': Simple_LSTM,
}

def score(predict, label):
    a1 = 13
    a2 = 10
    error = predict - label
    pos_e = np.exp(-error[error < 0] / a1) - 1
    neg_e = np.exp(error[error >= 0] / a2) - 1
    return sum(pos_e) + sum(neg_e)