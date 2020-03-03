# Keras model builder

## What is this?

This is a tool build Keras models a lot faster than by coding. No coding is required, all can be done from a configuration file. The ModelBuilder not only builds a model from your configuration, but it also tries to make this process easier. Using analyzers your model is analyzed to prevent errors. For completeness, a serializer is also included to convert your model to a configuration file. There is also a framework present for hyperparameter optimalization, but this is currently limited to [Hill Climbing](https://en.wikipedia.org/wiki/Hill_climbing).

## Configuration parameters

Currently the code supports the following parameters:

* Model parameters
  * Loss
  * Activation
  * Optimizer
  * Metrics (optional)
  * Learning rate (optional)
* Dense layers
  * Units
  * Dropout (optional)
* Conv2D layers
  * Filters
  * Kernel size
  * Stride (optional)
  * Max Pooling (optional, can not be combined with Average Pooling)
  * Average Pooling (optional, can not be combined with Max Pooling)
  * Dropout (optional)
* RNN, GRU and LSTM layers
  * Units
  * Dropout (optional)
  * Bidirectional (optional)
* Gaussian Noise layers
  * Stddev

## Why did you make this software?

Creating machine learning models in Keras is quite easy, but I think it can be easier. Editing layers, adding layers, removing layers, it all requires coding. Coding takes time, and code has to be debugged. That is why I created this tool, were the structure of a machine learning model is merely configuration, a set of variables in a file. It also makes it easier to create machine learning models from other code, and to use libraries different than Keras if wanted.

## But Keras has a model format already

Yes, and you can do more with it than with this tool. The API of this tool is more important than the json format, which is still a reason to create this tool. The ModelBuilder also tries to help you, making the creation of models easier. Input, Activation and Reshape layers are automatically created where necessary.

## Requirements

* Python >= 3.6
* Tensorflow
