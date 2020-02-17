from keras.layers import Conv2D, MaxPooling2D, Dropout

def build_layer(model, layer):
    if "strides" in layer:
        strides = tuple(layer["strides"])
    else:
        strides = (1,1)
    model = Conv2D(layer["filters"], tuple(layer["kernel_size"]), strides=strides)(model)
    if "max_pooling" in layer:
        model = MaxPooling2D(tuple(layer["max_pooling"]))(model)
    if "dropout" in layer:
        model = Dropout(layer["dropout"])(model)
    return model