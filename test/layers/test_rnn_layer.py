import pytest
from tensorflow.keras.layers import Input
from model_builder.layers.rnn_layer import RNNLayer


def permute_test():
    parent = Input((10, 10))
    units = 10
    names = ["lstm", "gru", "rnn"]
    bidirectional = [True, False]
    dropout = [0.0, 0.2]
    for n in names:
        for b in bidirectional:
            for d in dropout:
                rnn = RNNLayer(n, None, units, b, d)
                rnn.create(parent)


def test_error():
    parent = Input((10, 10))
    name = "error"
    units = 10
    bidirectional = False
    dropout = 0.0
    rnn = RNNLayer(name, None, units, bidirectional, dropout)
    with pytest.raises(ValueError):
        rnn.create(parent)
