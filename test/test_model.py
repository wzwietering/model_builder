import pytest
from model_builder import Model


def get_model():
    loss = "mean_squared_error"
    optimizer = "sgd"
    activation = "relu"
    metrics = ["mae"]
    learning_rate = 0.01
    m = Model(loss, optimizer, activation, metrics, learning_rate)
    return m


def test_set_input():
    m = get_model()
    shape = 10
    m.set_input_shape(shape)
    assert m.root.shape == shape


def test_set_optimizer():
    m = get_model()
    optimizer = "Adam"
    m.set_optimizer(optimizer)
    assert type(m.optimizer).__name__ == optimizer

    optimizer = "SGD"
    lr = 0.1
    m.set_optimizer(optimizer, lr)
    assert type(m.optimizer).__name__ == optimizer
    assert m.optimizer.learning_rate == lr


def test_set_learning_rate():
    m = get_model()
    lr = 0.1
    m.set_learning_rate(lr)
    assert m.optimizer.learning_rate == lr


def test_create():
    m = get_model()
    input_shape = (10,)
    output_shape = 10
    model = m.create(input_shape, output_shape)
    assert model.input_shape == (None,) + input_shape
