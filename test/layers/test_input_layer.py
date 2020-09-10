import pytest
from model_builder.layers.input_layer import InputLayer


def test_regular():
    shape = (10, 10)
    il = InputLayer(None, shape)
    il.create()
