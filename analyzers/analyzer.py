# Abstract class for analyzer models. Analyzers scan a model and may modify it
# in any way. This can be used for error checking for example

class Analyzer():
    def analyze(self, model):
        pass