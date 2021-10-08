from utils import from_input_dataframe_to_model_ready_data, load_LSTM_model

base_path = './data'


def predict(input_dataframe):
    X, Y = from_input_dataframe_to_model_ready_data(input_dataframe)
    model = load_LSTM_model()
    return model.predict(X)
