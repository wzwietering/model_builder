import copy
import math
from model_builder.optimization.optimizer import Optimizer
from model_builder.optimization.goal import Goal

# Optimize a model using hill climb
class HillClimb(Optimizer):
    def __init__(self, goal=Goal(), step_fraction=0.1):
        # Step fraction is the amount of change applied to a variable per step,
        # expressed as fraction of the value. Integer parameters are rounded
        super().__init__(goal)
        self.step_fraction = step_fraction

    def optimize_parameter(self, model, trainX, trainY, hyperparameters, name):
        flipped = False  # Did we change direction already?
        increase = True  # Direction of the climb
        while True:
            old_value = copy.deepcopy(hyperparameters.get(name))
            hyperparameters.set(name, self.__next_value(old_value, increase))
            new_score = super(HillClimb, self).fit(
                model, trainX, trainY, hyperparameters
            )
            if new_score <= self.best_score and not flipped:
                hyperparameters.set(name, old_value.value)
                increase = not (increase)
                flipped = True
            elif new_score > self.best_score:
                print(
                    f"Improved the score from {self.best_score} to {new_score} by changing {name} from {old_value.value} to {hyperparameters.get(name).value}"
                )
                self.best_params = hyperparameters
                self.best_score = new_score
            else:
                hyperparameters.set(name, old_value.value)
                print(f"Done optimizing {name}")
                # No improvement, and we changed direction already
                return model, hyperparameters

    def __next_value(self, parameter, increase=True):
        if type(parameter.value) is int:
            # For integer we use floor and ceil to 'force' the direction, as
            # small fractions would cause no changes to integer values
            if increase:
                return min(
                    int(math.ceil(parameter.value * (1 + self.step_fraction))),
                    parameter.max,
                )
            else:
                return max(
                    int(math.floor(parameter.value * (1 - self.step_fraction))),
                    parameter.min,
                )
        elif type(parameter.value) is float:
            if increase:
                return min(parameter.value * (1 + self.step_fraction), parameter.max)
            else:
                return max(parameter.value * (1 - self.step_fraction), parameter.min)
        else:
            raise ValueError(
                f"Value {parameter.value} of unknown type {type(parameter.value)}"
            )
