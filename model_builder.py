from keras.layers import Activation, Input
from keras.models import Model

from layer_builders import dense_builder, conv_builder

class ModelBuilder():
    def __init__(self):
        pass

    def build(self, config, input_shape, output_shape):
        inputs = Input(shape=input_shape)
        model = inputs
        for layer in config["layers"]:
            model = self.__build_layer(model, layer)
        model = Model(inputs=inputs, outputs=model)
        model.compile(optimizer=config["optimizer"], loss=config["loss"])
        return model
        
    def __build_layer(self, model, layer):
        if layer["name"] == "dense":
            return dense_builder.build_layer(model, layer)
        elif layer["name"] == "conv2d":
            return conv_builder.build_layer(model, layer)
        else:
            raise ValueError(f"Unknown layer type '{layer['name']}'")