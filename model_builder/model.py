from tensorflow.keras.layers import Activation, Dense
import tensorflow.keras.optimizers
import tensorflow.keras.models
from model_builder.layers.input_layer import InputLayer


class Model:
    def __init__(self, loss, optimizer, activation, metrics=None, learning_rate=None):
        self.loss = loss
        self.activation = activation
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.root = None
        self.layers = []
        self.set_optimizer(optimizer, learning_rate)

    def set_input_shape(self, input_shape):
        root = InputLayer(None, input_shape)
        # Prevent losing all children when updating the shape
        if self.root:
            root.children = self.root.children
        self.root = root

    def create(self, input_shape, output_shape):
        self.set_input_shape(input_shape)
        input_layer = self.root.create()
        model = input_layer
        for layer in self.layers:
            model = layer.create(model)
        model = Dense(output_shape)(model)
        activation = Activation(self.activation, name=self.activation)(model)
        model = tensorflow.keras.models.Model(inputs=input_layer, outputs=activation)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

    def set_optimizer(self, name, learning_rate=None):
        self.optimizer = tensorflow.keras.optimizers.get(name)
        if learning_rate:
            self.optimizer.learning_rate = learning_rate
