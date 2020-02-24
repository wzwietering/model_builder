from analyzers.rnn_return_sequences import RNNReturnSequences
from analyzers.crnn_analyzer import CRNNAnalyzer

class ModelAnalyzer():
    def __init__(self):
        self.analyzers = [RNNReturnSequences(), CRNNAnalyzer()]

    def analyze_model(self, model):
        for analyzer in self.analyzers:
            analyzer.analyze(model)