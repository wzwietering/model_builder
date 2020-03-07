class HyperParameter:
    def __init__(self, min, max, start, value=None):
        self.min = min
        self.max = max
        self.start = start
        self.value = start if not value else value


class HyperParameters:
    def __init__(self, *args, **kwargs):
        self.learning_rate = self.__parse_parameter(kwargs.get("learning_rate"))
        self.epochs = self.__parse_parameter(kwargs.get("epochs"))
        self.validation_split = self.__parse_parameter(kwargs.get("validation_split"))

    # Using __dict__ is good enough for the current implementation
    def parameters(self):
        return self.__dict__

    def values(self):
        return {k: self.get(k).value for k in self.__dict__}

    # Clean getter function
    def get(self, name):
        return getattr(self, name)

    # Clean setter function
    def set(self, name, value):
        getattr(self, name).value = value

    def __parse_parameter(self, values):
        return HyperParameter(values["min"], values["max"], values["start"])
