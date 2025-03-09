import tensorflow as tf

#arxiv 180500794
class ResidualLayer(tf.keras.layers.Layer):
  """
  A custom layer for a Convolutional Neural Network (CNN) residual connection.

  Args:
      filters: The number of filters/channels for the convolutional layers.
      kernal_size: The size of the kernel for the convolutional layers.

  Returns:
      A tensor with processed features.
  """
  def __init__(self, filters, kernal_size):
    super(ResidualLayer, self).__init__()

    self.Layer1 = tf.keras.layers.Convolution1D(filters,kernal_size, padding='same')
    self.Relu1 = tf.keras.layers.ReLU()
    self.Layer2 = tf.keras.layers.Convolution1D(filters,kernal_size, padding='same')
    self.Relu2 = tf.keras.layers.ReLU()
    self.Pool = tf.keras.layers.MaxPool1D(kernal_size,2, padding='same')

  def call(self, x):

    y = self.Layer1(x)
    y = self.Relu1(y)
    y = self.Layer2(y)
   
    y = y+x

    y = self.Relu2(y)
    y = self.Pool(y)

    return y



class ResidualCNN(tf.keras.Model):
  """
  A Convolutional Neural Network (CNN) model for timeseries.

  Args:
      model_output_size: The number of output units.
      filters: The number of filters/channels for the convolutional layers.
      kernal_size: The size of the kernel for the convolutional layers.
      num_residual_layers: The number of residual layers.
      feedforward: The number of units for the feedforward layers.
      feedforward_layer: The number of feedforward layers.
      classification: A boolean indicating whether the model is used for classification.

  Returns:
      A tensor with model outputs.
  """        
  def __init__(self, model_output_size:int, filters:int=32, kernal_size:int=5, num_residual_layers:int=5, feedforward:int=32, feedforward_layer:int = 2, classification:bool=True):
    super(ResidualCNN, self).__init__()

    self.input_layer = tf.keras.layers.Convolution1D(filters,kernal_size, strides = 1, padding='same',name="input")
    self.ConvLayers = [ ResidualLayer(filters, kernal_size) for _ in range(num_residual_layers)]

    self.flatten = tf.keras.layers.Flatten()

    self.feed_forward = [ tf.keras.layers.Dense(feedforward, activation="relu") for _ in range(feedforward_layer)]

    if classification:
      self.output_layer = tf.keras.layers.Dense(model_output_size, activation="softmax")
    else:
      self.output_layer = tf.keras.layers.Dense(model_output_size)


  def call(self, x):

    y = self.input_layer(x)
    for i in range(len(self.ConvLayers)):
      y = self.ConvLayers[i](y)

    y = self.flatten(y)

    for i in range(len(self.feed_forward)):
      y = self.feed_forward[i](y)

    y = self.output_layer(y)    

    return y
  


