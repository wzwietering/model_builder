from model_builder.model import Model


class ModelSerializer:
    def serialize(self, model):
        serialized = dict()
        serialized["loss"] = model.loss
        serialized["optimizer"] = model.optimizer.__class__.__name__.lower()
        if model.learning_rate:
            serialized["learning_rate"] = model.learning_rate
        serialized["activation"] = model.activation
        if model.metrics:
            serialized["metrics"] = model.metrics
        serialized["layers"] = self.__serialize_layers(model.layers)
        return serialized

    def __serialize_layers(self, layers):
        serialized_layers = []
        for layer in layers:
            if layer.name == "dense":
                serialized_layers.append(self.__serialize_dense(layer))
            elif layer.name == "conv2d":
                serialized_layers.append(self.__serialize_conv2d(layer))
            elif layer.name in ["rnn", "lstm", "gru"]:
                serialized_layers.append(self.__serialize_rnn(layer))
            elif layer.name == "gaussian_noise":
                serialized_layers.append(self.__serialize_gaussian_noise(layer))
            else:
                print(f"Ignored layer '{layer.name}'")
                continue
        return serialized_layers

    def __serialize_dense(self, layer):
        dense = dict()
        dense["name"] = "dense"
        dense["units"] = layer.units
        dense["dropout"] = layer.dropout
        return dense

    def __serialize_conv2d(self, layer):
        conv2d = dict()
        conv2d["name"] = "conv2d"
        conv2d["filters"] = layer.filters
        conv2d["kernel_size"] = list(layer.kernel_size)
        if layer.strides != (1, 1):
            conv2d["strides"] = list(layer.strides)
        if layer.max_pooling:
            conv2d["max_pooling"] = list(layer.max_pooling)
        if layer.average_pooling:
            conv2d["average_pooling"] = list(layer.average_pooling)
        if layer.dropout:
            conv2d["dropout"] = layer.dropout
        return conv2d

    def __serialize_rnn(self, layer):
        rnn = dict()
        rnn["name"] = layer.name
        rnn["units"] = layer.units
        if layer.bidirectional:
            rnn["bidirectional"] = layer.bidirectional
        if layer.dropout:
            rnn["dropout"] = layer.dropout
        return rnn

    def __serialize_gaussian_noise(self, layer):
        gn = dict()
        gn["name"] = "gaussian_noise"
        gn["stddev"] = layer.stddev
        return gn
