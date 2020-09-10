import json
import pytest
from model_builder import ModelParser

# Assert the examples parse right. The parser functionality is hidden, because
# the end user should be able to throw json files at the parser which just work


def test_model():
    config = "examples/model.json"
    with open(config, "r") as f:
        c = json.load(f)
    p = ModelParser()
    m = p.parse_config(c)
    assert m.loss == c["loss"]
    assert m.learning_rate == c["learning_rate"]
    assert m.activation == c["activation"]
    assert type(m.optimizer).__name__.lower() == c["optimizer"].lower()


def test_model_crnn():
    config = "examples/model_crnn.json"
    with open(config, "r") as f:
        c = json.load(f)
    p = ModelParser()
    p.parse_config(c)


def test_model_mnist():
    config = "examples/model_mnist.json"
    with open(config, "r") as f:
        c = json.load(f)
    p = ModelParser()
    p.parse_config(c)


def test_model_rnn():
    config = "examples/model_rnn.json"
    with open(config, "r") as f:
        c = json.load(f)
    p = ModelParser()
    p.parse_config(c)
