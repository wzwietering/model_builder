from keras.layers import SimpleRNN, LSTM, GRU, Bidirectional, Dropout

def build_layer(model, layer):
    layer_type = determine_layer(layer)
    recurrent_layer = layer_type(layer["units"], return_sequences=True)
    if "bidirectional" in layer and layer["bidirectional"]:
        recurrent_layer = Bidirectional(recurrent_layer)
    model = recurrent_layer(model)
    if "dropout" in layer:
        model = Dropout(layer["dropout"])(model)
    return model

def determine_layer(layer):
    if layer["name"] == "rnn":
        return SimpleRNN
    elif layer["name"] == "lstm":
        return LSTM
    elif layer["name"] == "gru":
        return GRU
    else:
        raise ValueError(f"Unknown recurrent layer type '{layer['name']}'")