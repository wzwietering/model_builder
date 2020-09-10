import pytest
from tensorflow.keras.layers import Input
from model_builder.layers.gaussian_noise_layer import GaussianNoiseLayer


def test_regular():
    parent = Input((10, 10))
    stddev = 0.2
    gn = GaussianNoiseLayer(None, stddev)
    gn.create(parent)
