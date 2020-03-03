from enum import Enum


# Indicates whether the problem is a maximalization or minimalization problem.
# Minimize is -1 because minimalization is the same as maximalization * -1,
# so we can use the integer values of the enums to multiply the target scores,
# without having to change any of the comparision operators.
class Strategy(Enum):
    MINIMIZE = -1
    MAXIMIZE = 1


# Simple class to tell what should be optimized how
class Goal:
    def __init__(self, metric="val_loss", strategy=Strategy.MINIMIZE):
        self.metric = metric
        self.strategy = strategy
