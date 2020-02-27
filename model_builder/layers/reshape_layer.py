from tensorflow.keras.layers import Reshape
from model_builder.layers.layer import Layer

# Reshape automatically determines how to reshape. Reshaping is done by
# multiplying the original dimensions from left to right with eachother.
# Reshaping to a higher dimension is not supported yet, though with some
# integer factorization this should be easy, but probably inaccurate
class ReshapeLayer(Layer):
    def __init__(self, parent, target_dimension):
        super().__init__("reshape", parent)
        self.target_dimension = target_dimension

    def create(self, parent):
        output_shape = parent.shape
        new_shape = self.__calculate_new(output_shape)
        return Reshape(new_shape)(parent)

    def __calculate_new(self, output_shape):
        if len(output_shape) <= self.target_dimension:
            raise ValueError(f"Reshape only supported for reduction of dimensions, attempting to go from {len(output_shape)} to {self.target_dimension}")
        reductions = len(output_shape) - self.target_dimension
        new_shape = list(output_shape)[1:] # first dimension is None for the batch size
        for _ in range(reductions):
            new_shape = self.__reduce_one(new_shape)
        return tuple(new_shape)

    def __reduce_one(self, shape):
        reduced = shape[1:]
        reduced[0] = reduced[0] * shape[0]
        return reduced
