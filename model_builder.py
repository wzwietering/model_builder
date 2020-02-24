from model_parser import Parser
from model_analyzer import ModelAnalyzer

class ModelBuilder():
    def __init__(self):
        self.parser = Parser()
        self.model_analyzer = ModelAnalyzer()

    def build_model(self, config_file, input_shape, output_shape):
        model = self.parser.parse_config(config_file)
        self.model_analyzer.analyze_model(model)
        return model.create(input_shape, output_shape)