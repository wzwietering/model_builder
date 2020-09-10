import pytest
from tensorflow.keras.layers import Input
from model_builder.layers.reshape_layer import ReshapeLayer


def test_reduce():
    parent = Input((10, 10, 10, 10))
    target_dimension = 2
    rs = ReshapeLayer(None, target_dimension)
    rs.create(parent)


def test_increase():
    parent = Input((10, 10))
    target_dimension = 4
    rs = ReshapeLayer(None, target_dimension)
    with pytest.raises(ValueError):
        rs.create(parent)
