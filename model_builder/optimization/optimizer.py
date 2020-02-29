# Optimize a model architecture given a base model, training data and
# training labels
class Optimizer():
    def __init__(self):
        self.best_model = None
        self.best_score = None

    # We assume the input data is the batch size followed by the data dimensions
    def optimize(self, model, trainX, trainY):
        pass