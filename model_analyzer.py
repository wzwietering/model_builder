from analyzers.rnn_return_sequences import RNNReturnSequences

class ModelAnalyzer():
    def __init__(self):
        self.analyzers = [RNNReturnSequences()]

    def analyze_model(self, model):
        for analyzer in self.analyzers:
            analyzer.analyze(model)