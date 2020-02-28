from model_builder.model import Model
from model_builder.layers.conv2d_layer import Conv2DLayer
from model_builder.layers.dense_layer import DenseLayer
from model_builder.layers.rnn_layer import RNNLayer
from model_builder.layers.reshape_layer import ReshapeLayer
from model_builder.layers.gaussian_noise_layer import GaussianNoiseLayer


class Parser:
    def parse_config(self, config):
        metrics = config.get("metrics", None)
        learning_rate = config.get("learning_rate", None)
        model = Model(
            config["loss"],
            config["optimizer"],
            config["activation"],
            metrics,
            learning_rate,
        )
        for layer in config["layers"]:
            model.layers.append(self.__parse_layer(layer))
        return model

    def __parse_layer(self, layer):
        if layer["name"] == "dense":
            return self.__parse_dense(layer)
        elif layer["name"] == "conv2d":
            return self.__parse_conv2d(layer)
        elif layer["name"] in self.__rnn_names():
            return self.__parse_rnn(layer)
        elif layer["name"] == "gaussian_noise":
            return GaussianNoiseLayer(None, layer["stddev"])
        else:
            raise ValueError(f"Unknown layer type '{layer['name']}'")

    def __parse_dense(self, layer):
        dropout = layer.get("dropout", None)
        return DenseLayer(None, layer["units"], dropout)

    def __parse_conv2d(self, layer):
        filters = layer["filters"]
        kernel_size = tuple(layer["kernel_size"])
        strides = self.__get_tuple(layer, "strides", (1, 1))
        max_pooling = self.__get_tuple(layer, "max_pooling", None)
        average_pooling = self.__get_tuple(layer, "average_pooling", None)
        dropout = layer.get("dropout", None)
        return Conv2DLayer(
            None, filters, kernel_size, strides, max_pooling, average_pooling, dropout
        )

    def __parse_rnn(self, layer):
        bidirectional = layer.get("bidirectional", False)
        dropout = layer.get("dropout", None)
        return RNNLayer(layer["name"], None, layer["units"], bidirectional, dropout)

    # Json has no tuples, so they are stored as list, but we want a tuple
    def __get_tuple(self, layer, key, default=None):
        if key in layer:
            return tuple(layer[key])
        else:
            return default

    def __rnn_names(self):
        return ["rnn", "lstm", "gru"]
