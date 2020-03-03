import math
from model_builder.optimization.optimizer import Optimizer

# Optimize a model using hill climb
class HillClimb(Optimizer):
    def __init__(self, step_fraction=0.01):
        # Step fraction is the amount of change applied to a variable per step,
        # expressed as fraction of the value. Integer parameters are rounded
        self.step_fraction = step_fraction

    def __optimize_parameter(
        self, model, trainX, trainY, hyperparameters, name
    ):
        flipped = False  # Did we change direction already?
        increase = True  # Direction of the climb
        while True:
            old_value = hyperparameters.get(name)
            hyperparameters.set(name, self.__next_value(old_value, increase))
            new_loss = super(HillClimb, self).__fit(
                model, trainX, trainY, hyperparameters
            )
            if new_loss >= self.best_score and not flipped:
                hyperparameters.set(name, old_value)
                increase = not (increase)
                flipped = True
            elif new_loss < self.best_score:
                self.best_params = hyperparameters
                self.best_score = new_loss
            else:
                return  # No improvement, and we changed direction already

    def __next_value(self, value, increase=True):
        if type(value) is int:
            # For integer we use floor and ceil to 'force' the direction, as
            # small fractions would cause no changes to integer values
            if increase:
                return int(math.ceil(value * (1 + self.step_fraction)))
            else:
                return int(math.floor(value * (1 - self.step_fraction)))
        elif type(value) is float:
            if increase:
                return value * (1 + self.step_fraction)
            else:
                return value * (1 - self.step_fraction)
        else:
            raise ValueError(f"Value {value} of unknown type {type(value)}")
