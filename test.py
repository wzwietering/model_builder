import json
import model_parser

def test_config(filename, input_shape, output_shape=10):
    with open(filename, "r") as f:
        config = json.load(f)
    test_parser(config, input_shape, output_shape)

def test_parser(config, input_shape, output_shape):
    mp = model_parser.Parser()
    model = mp.parse_config(config)
    keras_model = model.create(input_shape, output_shape)
    print(keras_model.summary())

if __name__ == "__main__":
    test_config("examples/model.json", (28,28,1))
    test_config("examples/model_rnn.json", (28,28))
    test_config("examples/model_crnn.json", (28,28,1))