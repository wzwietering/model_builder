from model_builder.analyzers.analyzer import Analyzer
from model_builder.layers.rnn_layer import RNNLayer

# If the final layer of a model is a RNNLayer, it should not return its
# sequences. This analyzer ensures this.
class RNNReturnSequences(Analyzer):
    def analyze(self, model):
        if type(model.layers[-1]) is RNNLayer:
            model.layers[-1].return_sequences = False