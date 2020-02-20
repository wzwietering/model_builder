from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, AveragePooling2D
from models.layer import Layer

class Conv2DLayer(Layer):
    def __init__(self,
                 parent,
                 filters=None,
                 kernel_size=None,
                 strides=None,
                 max_pooling=None,
                 average_pooling=None,
                 dropout=None):
        super().__init__("conv2d", parent)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.max_pooling = max_pooling
        self.average_pooling = average_pooling
        self.dropout = dropout

    def create(self, parent):
        layer = Conv2D(self.filters, self.kernel_size, strides=self.strides)(parent)

        if self.max_pooling:
            layer = MaxPooling2D(self.max_pooling)(layer)
        elif self.average_pooling:
            layer = AveragePooling2D(self.average_pooling)(layer)

        if self.dropout:
            layer = Dropout(self.dropout)(layer)
        return layer