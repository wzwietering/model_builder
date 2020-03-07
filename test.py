import json
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import tensorflow.keras.utils
from model_builder import ModelBuilder
from model_builder import Parser
from model_builder import ModelSerializer
from model_builder import HillClimb
from model_builder import HyperParameters
from model_builder import Goal, Strategy


def test_config(filename, input_shape, output_shape=10):
    with open(filename, "r") as f:
        config = json.load(f)
    test_parser(config, input_shape, output_shape)
    test_serializer(config)


def test_parser(config, input_shape, output_shape):
    mb = ModelBuilder()
    mb.build_model(config, input_shape, output_shape)


def test_serializer(config):
    mp = Parser()
    model = mp.parse_config(config)
    ms = ModelSerializer()
    serialized = ms.serialize(model)
    assert config == serialized


def get_mnist(num_classes):
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], trainX.shape[2], 1)
    testX = testX.reshape(testX.shape[0], testX.shape[1], testX.shape[2], 1)
    trainY = tensorflow.keras.utils.to_categorical(trainY, num_classes)
    testY = tensorflow.keras.utils.to_categorical(testY, num_classes)
    return trainX, trainY, testX, testY


def test_mnist():
    num_classes = 10
    trainX, trainY, testX, testY = get_mnist(num_classes)
    with open("examples/model_mnist.json", "r") as f:
        config = json.load(f)
    mb = ModelBuilder()
    model = mb.build_model(config, trainX[0].shape, num_classes)
    model.fit(trainX, trainY, epochs=10)


def test_optimization(config):
    num_classes = 10
    trainX, trainY, testX, testY = get_mnist(num_classes)
    with open("examples/model_mnist.json", "r") as f:
        config = json.load(f)
    mp = Parser()
    model = mp.parse_config(config)

    hc = HillClimb(Goal("val_accuracy", Strategy.MAXIMIZE))
    hyp_dict = {
        "learning_rate": {"min": 0.0001, "max": 0.001, "start": 0.0001},
        "epochs": {"min": 1, "max": 100, "start": 2},
        "validation_split": {"min": 0.1, "max": 0.3, "start": 0.2},
    }
    hyp = HyperParameters(**hyp_dict)
    hc.optimize(model, trainX, trainY, hyp)


if __name__ == "__main__":
    test_config("examples/model.json", (28, 28, 1))
    test_config("examples/model_rnn.json", (28, 28))
    test_config("examples/model_crnn.json", (28, 28, 1))
    test_config("examples/model_mnist.json", (28, 28, 1))
    test_mnist()
    test_optimization("examples/model_mnist.json")
