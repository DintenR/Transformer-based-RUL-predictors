from models import TransformerEncoder_LSTM_1, TransformerEncoder_LSTM_2, TransformerEncoder_LSTM_3, Simple_LSTM

MODEL_MAP = {
    'version_1': TransformerEncoder_LSTM_1,
    'version_2': TransformerEncoder_LSTM_2,
    'version_3': TransformerEncoder_LSTM_3,
    'lstm': Simple_LSTM,
}