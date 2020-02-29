from model_builder.optimization.optimizer import Optimizer

# Optimize a model using hill climb
class HillClimb(Optimizer):
    def __init__(self, step_fraction=0.01):
        # Step fraction is the amount of change applied to a variable per step,
        # expressed as fraction of the value. Integer parameters are rounded
        self.step_fraction = step_fraction

    def optimize(self, model, trainX, trainY, hyperparameters):
        self.best_model = model
        input_shape = trainX.shape[1:]
        output_shape = trainY.shape[1:]
        compiled_model = model.create(input_shape, output_shape)
        history = compiled_model.fit(trainX, trainY, epochs=hyperparameters.epochs, validation_split=hyperparameters.validation_split)
        self.best_score = history.history['loss']