from tensorflow.keras.layers import Dense, Dropout
from model_builder.layers.layer import Layer

class DenseLayer(Layer):
    def __init__(self, parent, units=None, dropout=None):
        super().__init__("dense", parent)
        self.units = units
        self.dropout = dropout

    def create(self, parent):
        layer = Dense(self.units)(parent)
        if self.dropout:
            layer = Dropout(self.dropout)(layer)
        return layer