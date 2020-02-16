from keras.layers import Dense, Dropout

def build_layer(model, layer):
    model = Dense(layer["units"])(model)
    if "dropout" in layer:
        model = Dropout(layer["dropout"])(model)
    return model