from models.model import Model
from models.conv2d_layer import Conv2DLayer
from models.dense_layer import DenseLayer
from models.rnn_layer import RNNLayer
from models.reshape_layer import ReshapeLayer

class Parser():
    def parse_config(self, config):
        model = Model(config["loss"], config["optimizer"], config["activation"])
        prev = ""
        for layer in config["layers"]:
            if layer["name"] in self.__rnn_names() and prev == "conv2d":
                model.layers.append(ReshapeLayer(None, 3))
            model.layers.append(self.__parse_layer(layer))
            prev = layer["name"]
        return model

    def __parse_layer(self, layer):
        if layer["name"] == "dense":
            dropout = layer.get("dropout", None)
            return DenseLayer(None, layer["units"], dropout)
        elif layer["name"] == "conv2d":
            filters = layer["filters"]
            kernel_size = tuple(layer["kernel_size"])
            strides = self.__get_tuple(layer, "strides", (1,1))
            max_pooling = self.__get_tuple(layer, "max_pooling", None)
            average_pooling = self.__get_tuple(layer, "average_pooling", None)
            dropout = layer.get("dropout", None)
            return Conv2DLayer(None, filters, kernel_size, strides, max_pooling, average_pooling, dropout)
        elif layer["name"] in self.__rnn_names():
            bidirectional = layer.get("bidirectional", False)
            dropout = layer.get("dropout", None)
            return RNNLayer(layer["name"], None, layer["units"], bidirectional, dropout)
        else:
            raise ValueError(f"Unknown layer type '{layer['name']}'")

    # Json has no tuples, so they are stored as list, but we want a tuple
    def __get_tuple(self, layer, key, default=None):
        if key in layer:
            return tuple(layer[key])
        else:
            return default

    def __rnn_names(self):
        return ["rnn", "lstm", "gru"]