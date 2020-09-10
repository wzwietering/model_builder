import json
from model_builder.analyzers.rnn_return_sequences import RNNReturnSequences
from model_builder.layers.rnn_layer import RNNLayer
from model_builder import ModelParser


def test_crnn_analyzer():
    config = "examples/model_rnn.json"
    with open(config, "r") as f:
        c = json.load(f)
    p = ModelParser()
    model = p.parse_config(c)
    for l in model.layers:
        if type(l) is RNNLayer:
            assert l.return_sequences

    rrs = RNNReturnSequences()
    rrs.analyze(model)
    assert not model.layers[-1].return_sequences
