import pytest
from tensorflow.keras.layers import Input
from model_builder.layers.dense_layer import DenseLayer


def test_regular():
    parent = Input((10, 10))
    units = 10
    dropout = 0.0
    dense = DenseLayer(None, units, dropout)
    dense.create(parent)


def test_dropout():
    parent = Input((10, 10))
    units = 10
    dropout = 0.3
    dense = DenseLayer(None, units, dropout)
    dense.create(parent)
