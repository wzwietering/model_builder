from tensorflow.keras.layers import Activation, Input, Dense
from tensorflow.keras.models import Model

from layer_builders import dense_builder, conv_builder, rnn_builder

class ModelBuilder():
    def build(self, config, input_shape, output_shape):
        inputs = Input(shape=input_shape)
        model = inputs
        for layer in config["layers"]:
            model = self.__build_layer(model, layer)
        model = Dense(output_shape)(model)
        activation = Activation(config["activation"], name=config["activation"])(model)
        model = Model(inputs=inputs, outputs=activation)
        model.compile(optimizer=config["optimizer"], loss=config["loss"])
        return model

    def __build_layer(self, model, layer):
        if layer["name"] == "dense":
            return dense_builder.build_layer(model, layer)
        elif layer["name"] == "conv2d":
            return conv_builder.build_layer(model, layer)
        elif layer["name"] == "rnn" or layer["name"] == "lstm" or layer["name"] == "gru":
            return rnn_builder.build_layer(model, layer)
        else:
            raise ValueError(f"Unknown layer type '{layer['name']}'")