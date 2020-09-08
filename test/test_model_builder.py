import json
import pytest
from model_builder import ModelBuilder


def test_model():
    config = "examples/model.json"
    input_shape = (100, 100, 3)
    output_shape = 10
    mb = ModelBuilder()
    mb.build_model(config, input_shape, output_shape)


def test_model_crnn():
    config = "examples/model_crnn.json"
    input_shape = (10, 10, 1)
    output_shape = 10
    mb = ModelBuilder()
    mb.build_model(config, input_shape, output_shape)


def test_model_mnist():
    config = "examples/model_mnist.json"
    input_shape = (28, 28, 1)
    output_shape = 10
    mb = ModelBuilder()
    mb.build_model(config, input_shape, output_shape)


def test_model_rnn():
    config = "examples/model_rnn.json"
    input_shape = (10, 10)
    output_shape = 10
    mb = ModelBuilder()
    mb.build_model(config, input_shape, output_shape)
