from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Bidirectional
from model_builder.layers.layer import Layer


class RNNLayer(Layer):
    def __init__(self, name, parent, units=None, bidirectional=None, dropout=None):
        super().__init__(name, parent)
        self.units = units
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.return_sequences = True

    def create(self, parent):
        layer_type = self.__determine_layer()
        layer = layer_type(
            self.units, return_sequences=self.return_sequences, dropout=self.dropout
        )
        if self.bidirectional:
            layer = Bidirectional(layer)
        model = layer(parent)
        return model

    def __determine_layer(self):
        if self.name.lower() == "rnn":
            return SimpleRNN
        elif self.name.lower() == "lstm":
            return LSTM
        elif self.name.lower() == "gru":
            return GRU
        else:
            raise ValueError(f"Unknown recurrent layer type '{self.name}'")
