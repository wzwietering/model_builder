import model_builder
import json

def test_config(filename, input_shape):
    with open(filename, "r") as f:
        config = json.load(f)
    mb = model_builder.ModelBuilder()
    model = mb.build(config, input_shape, (10,))
    print(model.summary())

if __name__ == "__main__":
    test_config("examples/model.json", (28,28,1))
    test_config("examples/model_rnn.json", (28,28))
    test_config("examples/model_crnn.json", (28,28,1))