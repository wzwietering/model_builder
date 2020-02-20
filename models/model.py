from tensorflow.keras.layers import Activation, Dense
import tensorflow.keras.models
from models.input_layer import InputLayer

class Model():
    def __init__(self, loss, optimizer, activation):
        self.loss = loss
        self.optimizer = optimizer
        self.activation = activation
        self.root = None
        self.layers = []

    def set_input_shape(self, input_shape):
        root = InputLayer(None, input_shape)
        # Prevent losing all children when updating the shape
        if self.root:
            root.children = self.root.children
        self.root = root

    def create(self, output_shape):
        if not self.root:
            raise ValueError("Model has no input, use 'Model.set_input_shape' before creating the model")
        input_layer = self.root.create()
        model = input_layer
        for layer in self.layers:
            model = layer.create(model)
        model = Dense(output_shape)(model)
        activation = Activation(self.activation, name=self.activation)(model)
        model = tensorflow.keras.models.Model(inputs=input_layer, outputs=activation)
        model.compile(optimizer=self.optimizer, loss=self.loss)
        return model