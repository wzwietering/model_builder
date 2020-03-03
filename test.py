import json
from tensorflow.keras.datasets import mnist
import tensorflow.keras.utils
from model_builder import model_builder
from model_builder import model_parser
from model_builder import model_serializer


def test_config(filename, input_shape, output_shape=10):
    with open(filename, "r") as f:
        config = json.load(f)
    test_parser(config, input_shape, output_shape)
    test_serializer(config)


def test_parser(config, input_shape, output_shape):
    mb = model_builder.ModelBuilder()
    mb.build_model(config, input_shape, output_shape)


def test_serializer(config):
    mp = model_parser.Parser()
    model = mp.parse_config(config)
    ms = model_serializer.ModelSerializer()
    serialized = ms.serialize(model)
    assert config == serialized


def test_mnist():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainX = trainX.reshape(
        trainX.shape[0], trainX.shape[1], trainX.shape[2], 1
    )
    testX = testX.reshape(testX.shape[0], testX.shape[1], testX.shape[2], 1)
    num_classes = 10
    with open("examples/model_mnist.json", "r") as f:
        config = json.load(f)
    mb = model_builder.ModelBuilder()
    trainY = tensorflow.keras.utils.to_categorical(trainY, num_classes)
    testY = tensorflow.keras.utils.to_categorical(testY, num_classes)
    model = mb.build_model(config, trainX[0].shape, num_classes)
    model.fit(trainX, trainY, epochs=10)
    predictions = model.predict(testX)


if __name__ == "__main__":
    test_config("examples/model.json", (28, 28, 1))
    test_config("examples/model_rnn.json", (28, 28))
    test_config("examples/model_crnn.json", (28, 28, 1))
    test_config("examples/model_mnist.json", (28, 28, 1))
    test_mnist()
