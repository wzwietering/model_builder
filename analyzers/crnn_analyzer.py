from analyzers.analyzer import Analyzer
from layers.reshape_layer import ReshapeLayer

class CRNNAnalyzer(Analyzer):
    def analyze(self, model):
        prev = None
        inserts = []
        for i, layer in enumerate(model.layers):
            if layer.name in ["rnn", "gru", "lstm"] and prev == "conv2d":
                inserts.append((i, ReshapeLayer(None, 3)))
            prev = layer.name

        for idx, (i, layer) in enumerate(inserts):
            # We have to add the index of the insert to its original index,
            # because inserts move indices
            model.layers.insert(i + idx, layer)