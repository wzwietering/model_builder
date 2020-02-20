from tensorflow.keras.layers import Input
from models.layer import Layer

class InputLayer(Layer):
    def __init__(self, parent, shape):
        super().__init__("input", parent)
        self.shape = shape

    def create(self):
        return Input(shape=self.shape)