import json
from model_builder.analyzers.crnn_analyzer import CRNNAnalyzer
from model_builder import ModelParser


def test_crnn_analyzer():
    config = "examples/model_crnn.json"
    with open(config, "r") as f:
        c = json.load(f)
    p = ModelParser()
    model = p.parse_config(c)
    # Assert no reshape layer
    assert "reshape" not in [l.name for l in model.layers]
    cra = CRNNAnalyzer()
    cra.analyze(model)
    # Assert reshape layer in the layers
    assert "reshape" in [l.name for l in model.layers]
