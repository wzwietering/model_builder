from model_builder.optimization.goal import Goal

# Optimize a model architecture given a base model, training data and
# training labels
class Optimizer:
    def __init__(self, goal=Goal()):
        self.goal = goal
        self.best_model = None
        self.best_score = None

    def optimize(self, model, trainX, trainY, hyperparameters):
        # Establish baseline
        self.best_model = model
        self.best_params = hyperparameters
        self.best_score = self.fit(model, trainX, trainY, hyperparameters)
        print(f"Baseline score is {self.best_score}")

        # Optimize all hyperparameters
        for name in hyperparameters.parameters():
            print(f"Optimizing {name}")
            model, hyperparameters = self.optimize_parameter(
                model, trainX, trainY, hyperparameters, name
            )
        print(
            f"Optimal score is {self.best_score * self.goal.strategy.value} with the parameters {hyperparameters.values()}"
        )
        return hyperparameters

    # We assume the input data is the batch size followed by the data dimensions
    def fit(self, model, trainX, trainY, hyperparameters):
        input_shape = trainX.shape[1:]
        output_shape = trainY.shape[-1]
        model.set_learning_rate(hyperparameters.get("learning_rate").value)
        compiled_model = model.create(input_shape, output_shape)
        history = compiled_model.fit(
            trainX,
            trainY,
            epochs=hyperparameters.get("epochs").value,
            validation_split=hyperparameters.get("validation_split").value,
        )
        return history.history[self.goal.metric][-1] * self.goal.strategy.value
