class HyperParameters:
    def __init__(self, learning_rate, epochs, validation_split):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.validation_split = validation_split

    # Using __dict__ is good enough for the current implementation
    def parameters(self):
        return self.__dict__

    # Clean getter function
    def get(self, name):
        return getattr(self, name)

    # Clean setter function
    def set(self, name, value):
        setattr(self, name, value)
