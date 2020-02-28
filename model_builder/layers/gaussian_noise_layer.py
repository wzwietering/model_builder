from tensorflow.keras.layers import GaussianNoise
from model_builder.layers.layer import Layer


class GaussianNoiseLayer(Layer):
    def __init__(self, parent, stddev):
        super().__init__("gaussian_noise", parent)
        self.stddev = stddev

    def create(self, parent):
        return GaussianNoise(self.stddev)(parent)
