# Optimize a model architecture given a base model, training data and
# training labels
class Optimizer:
    def __init__(self):
        self.best_model = None
        self.best_score = None

    def optimize(self, model, trainX, trainY, hyperparameters):
        # Establish baseline
        self.best_model = model
        self.best_params = hyperparameters
        self.best_score = self.__fit(model, trainX, trainY, hyperparameters)

        # Optimize all hyperparameters
        for name, _ in hyperparameters.parameters():
            self.__optimize_parameter(
                model, trainX, trainY, hyperparameters, name
            )

    def __optimize_parameter(
        self, model, trainX, trainY, hyperparameters, name
    ):
        pass

    # We assume the input data is the batch size followed by the data dimensions
    def __fit(self, model, trainX, trainY, hyperparameters):
        input_shape = trainX.shape[1:]
        output_shape = trainY.shape[1:]
        compiled_model = model.create(input_shape, output_shape)
        history = compiled_model.fit(
            trainX,
            trainY,
            epochs=hyperparameters.epochs,
            validation_split=hyperparameters.validation_split,
        )
        return history["loss"]
