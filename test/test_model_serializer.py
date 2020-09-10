import json
import pytest
from model_builder import ModelSerializer
from model_builder import ModelParser


def assert_parse_eq_serialize(config):
    with open(config, "r") as f:
        c = json.load(f)
    mp = ModelParser()
    model = mp.parse_config(c)
    ms = ModelSerializer()
    serialized = ms.serialize(model)
    assert c == serialized


def test_model():
    config = "examples/model.json"
    assert_parse_eq_serialize(config)


def test_model_crnn():
    config = "examples/model_crnn.json"
    assert_parse_eq_serialize(config)


def test_model_mnist():
    config = "examples/model_mnist.json"
    assert_parse_eq_serialize(config)


def test_model_rnn():
    config = "examples/model_rnn.json"
    assert_parse_eq_serialize(config)
