from keras.layers import SimpleRNN, LSTM, GRU, Bidirectional, Dropout, Reshape, CuDNNGRU, CuDNNLSTM
from keras import backend

def build_layer(model, layer):
    output_shape = model._keras_shape
    if len(output_shape) == 4: # If the previous layer is conv2d
        new_shape = (output_shape[1] * output_shape[2], output_shape[3])
        model = Reshape(new_shape)(model)

    layer_type = determine_layer(layer)
    recurrent_layer = layer_type(layer["units"], return_sequences=True)
    if "bidirectional" in layer and layer["bidirectional"]:
        recurrent_layer = Bidirectional(recurrent_layer)
    model = recurrent_layer(model)
    if "dropout" in layer:
        model = Dropout(layer["dropout"])(model)
    return model

def determine_layer(layer):
    gpu = len(backend.tensorflow_backend._get_available_gpus()) > 0
    if layer["name"] == "rnn":
        return SimpleRNN
    elif layer["name"] == "lstm":
        if gpu: return CuDNNLSTM
        else: return LSTM
    elif layer["name"] == "gru":
        if gpu: return CuDNNGRU
        else: return GRU
    else:
        raise ValueError(f"Unknown recurrent layer type '{layer['name']}'")