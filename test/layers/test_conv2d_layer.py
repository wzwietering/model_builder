import pytest
from tensorflow.keras.layers import Input
from model_builder.layers.conv2d_layer import Conv2DLayer


def test_regular():
    parent = Input((10, 10, 10))
    filters = 10
    kernel_size = [2, 2]
    strides = [2, 2]
    max_pooling = False
    average_pooling = False
    dropout = 0.0
    conv2d = Conv2DLayer(
        None, filters, kernel_size, strides, max_pooling, average_pooling, dropout
    )

    conv2d.create(parent)


def test_max_pooling():
    parent = Input((10, 10, 10))
    filters = 10
    kernel_size = [2, 2]
    strides = [2, 2]
    max_pooling = True
    average_pooling = False
    dropout = 0.0
    conv2d = Conv2DLayer(
        None, filters, kernel_size, strides, max_pooling, average_pooling, dropout
    )

    conv2d.create(parent)


def test_avg_pooling():
    parent = Input((10, 10, 10))
    filters = 10
    kernel_size = [2, 2]
    strides = [2, 2]
    max_pooling = False
    average_pooling = True
    dropout = 0.0
    conv2d = Conv2DLayer(
        None, filters, kernel_size, strides, max_pooling, average_pooling, dropout
    )

    conv2d.create(parent)


def test_dropout():
    parent = Input((10, 10, 10))
    filters = 10
    kernel_size = [2, 2]
    strides = [2, 2]
    max_pooling = False
    average_pooling = False
    dropout = 0.2
    conv2d = Conv2DLayer(
        None, filters, kernel_size, strides, max_pooling, average_pooling, dropout
    )

    conv2d.create(parent)
