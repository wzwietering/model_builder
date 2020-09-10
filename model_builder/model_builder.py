import json

from model_builder.model_parser import ModelParser
from model_builder.model_analyzer import ModelAnalyzer


class ModelBuilder:
    def __init__(self):
        self.parser = ModelParser()
        self.model_analyzer = ModelAnalyzer()

    def build_model(self, config_file, input_shape, output_shape):
        if type(config_file) == str:
            with open(config_file, "r") as f:
                config_file = json.load(f)
        model = self.parser.parse_config(config_file)
        self.model_analyzer.analyze_model(model)
        return model.create(input_shape, output_shape)
