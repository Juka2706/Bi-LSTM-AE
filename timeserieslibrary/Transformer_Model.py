#Base of implementation:
#https://www.tensorflow.org/text/tutorials/transformer; 
#https://keras.io/examples/timeseries/timeseries_transformer_classification/

import tensorflow as tf
import numpy as np
import TimeSeries.Transformer_Visualization as tv
import gc

class PositionalEncoding(tf.keras.layers.Layer):
  """Positional Encoding Layer for the Encoder Transformer model.

  This layer adds positional encoding to the input sequences.

  Args:
      model_dimension (int): The dimensionality of the model.

  Attributes:
      d_model (int): The dimensionality of the model.
      pos_encoding (tf.Tensor): Positional encoding tensor.

  """
  def __init__(self, model_dimension, length = 2048):
    super().__init__()

    self.d_model = model_dimension

    depth = model_dimension/2
    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 

    self.pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)

  def call(self, x):

    length = tf.shape(x)[1]   
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x


class SegmentationProjektion(tf.keras.layers.Layer):
  """Segmentation and Projection Layer.

  This layer performs segmentation (to segments of size segment_sizes with segment_sizes) and projection (to model dimension) of input data using 1D convolution.
  
  Input dim: (Batchsize, seqenz len, variables)
  Output dim: (Batchsize, number of segments, model dim)

  (number of segments = seqenz len / segment stride)

  Args:
      model_dimension (int): The dimensionality of the model.
      segment_sizes (int): The size of segments for segmentation.
      segment_stride (int, optional): The stride for segmentation. If None, stride is set equal to `segment_sizes` -> No overlapping segments
          Default is None. 

  Attributes:
      seq (tf.keras.Sequential): Sequential model containing a 1D convolutional layer followed by layer normalization.

  """
     
  def __init__(self, model_dimension, segment_sizes, segment_stride = None):
    super().__init__()

    if segment_stride is None:
       segment_stride=segment_sizes #Keine überlappung

    self.seq = tf.keras.Sequential([   
      tf.keras.layers.Conv1D(filters=model_dimension, kernel_size=segment_sizes, strides = segment_stride, activation="tanh", padding = "same", name="input1"),
      tf.keras.layers.LayerNormalization(epsilon=1e-6)
    ])

  def call(self, x):
  
    y = self.seq(x)
    return y


class SelfAttention(tf.keras.layers.Layer):
  """Self-Attention Layer for the Encoder Transformer model.

  This layer contains multi-head self-attention mechanism followed by layer normalization and residual connection.

  Args:
      **kwargs: Additional keyword arguments to be passed to the `tf.keras.layers.MultiHeadAttention` layer.

  Attributes:
      mha (tf.keras.layers.MultiHeadAttention): Multi-head self-attention mechanism.
      layernorm (tf.keras.layers.LayerNormalization): Layer normalization.
      add (tf.keras.layers.Add): Layer for adding the residual connection.

  """
  def __init__(self, **kwargs):
    super().__init__()

    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    #self.mha_spartial = tf.keras.layers.MultiHeadAttention(**kwargs)

    self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.add = tf.keras.layers.Add()

  def call(self, x):

    attn_output, attn_score = self.mha(query=x, key=x, value=x, return_attention_scores=True)
    

    #Add spartial attention
    #attn_output = tf.transpose(attn_output, (1,0,2))
    #attn_output = self.mha_spartial(attn_output,attn_output,attn_output)
    #attn_output = tf.transpose(attn_output, (1,0,2))

    x = self.add([x, attn_output])
    x = self.layernorm(x)
    
    return x, attn_score




class FeedForward(tf.keras.layers.Layer):
  """Feed-Forward Layer for the Encoder Transformer model.

  This layer contains two 1D convolutional layers followed by dropout, residual connection, and layer normalization.

  Args:
      model_dimension (int): The dimensionality of the model.
      dff (int): The dimensionality of the feed-forward layer.
      dropout_rate (float, optional): Dropout rate for regularization. Default is 0.1.

  Attributes:
      seq (tf.keras.Sequential): Sequential model containing two 1D convolutional layers followed by dropout.
      add (tf.keras.layers.Add): Layer for adding the residual connection.
      layer_norm (tf.keras.layers.LayerNormalization): Layer normalization.

  """
  def __init__(self, model_dimension, dff, dropout_rate=0.1, **kwargs):
    super().__init__()

    self.seq = tf.keras.Sequential([
      tf.keras.layers.Conv1D(filters=dff, kernel_size=1, activation="relu", padding = "same"),
      tf.keras.layers.Conv1D(filters=model_dimension, kernel_size=1, padding = "same"),
      tf.keras.layers.Dropout(dropout_rate)
    ])

    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

  def call(self, x):
    tmp = self.seq(x)
    x = self.add([x, tmp])
    x = self.layer_norm(x) 
    return x


class EncoderLayer(tf.keras.layers.Layer):
  """Encoder Layer of the Transformer model.

  This layer contains self-attention mechanism followed by feed-forward neural network.

  Args:
      model_dimension (int): The dimensionality of the model.
      num_heads (int): The number of attention heads.
      dff (int): The dimensionality of the feed-forward layer.
      dropout_rate (float, optional): Dropout rate for regularization. Default is 0.1.

  Attributes:
      self_attention (SelfAttention): Self-attention mechanism.
      ffn (FeedForward): Feed-forward neural network.

  """
  def __init__(self,*, model_dimension, num_heads, dff, dropout_rate=0.1,**kwargs):
    super().__init__()

    self.self_attention = SelfAttention(
        num_heads=num_heads,
        key_dim=model_dimension,
        value_dim=model_dimension,
        dropout=dropout_rate)
    
    self.ffn = FeedForward(model_dimension, dff, dropout_rate)

  def call(self, x):
    x, attn_scores = self.self_attention(x)
    x = self.ffn(x)
    return x, attn_scores



class CNN_Block(tf.keras.layers.Layer):
  """CNN Block for the Encoder Transformer model.

  One block contains two 1D convolutional layers followed by max pooling.
  Sequenz reduction of 2 by each CNN Block

  Args:
      model_dimension (int): The dimensionality of the model.

  Attributes:
      cnn1 (tf.keras.layers.Conv1D): First 1D convolutional layer.
      cnn2 (tf.keras.layers.Conv1D): Second 1D convolutional layer.
      maxpool (tf.keras.layers.MaxPool1D): Max pooling layer.

  """
  def __init__(self, model_dimension, **kwargs):
    super().__init__()

    self.cnn1 = tf.keras.layers.Conv1D(filters=model_dimension, kernel_size=5, strides = 1, activation="relu", padding = "same")
    self.cnn2 = tf.keras.layers.Conv1D(filters=model_dimension, kernel_size=5, strides = 1, activation="relu", padding = "same") 
    self.maxpool = tf.keras.layers.MaxPool1D(5,2, padding='same')

  def call(self, x):
    x = self.cnn1(x)
    x = self.cnn2(x)
    x = self.maxpool(x)
    #x has dim of q
    return x

    
  
class EncoderTransformer(tf.keras.Model):
  """Encoder Transformer Model implementation for time series with additional features.

  Args:
  ----------
  model_dimension : int
      The dimensionality of the model.
  num_layers : int
      The number of encoder layers.
  num_heads : int
      The number of attention heads in each encoder layer.
  encoder_feed_forward : int
      The dimensionality of the feed-forward layer in each encoder layer.
  model_output_size : int
      The dimensionality of the output.
  encoder_dropout_rate : float, optional
      The dropout rate for the encoder layers. Default is 0.2.
  output_feed_forward_layer : list, optional
      A list specifying the dimensionality of additional feed-forward layers after the encoder.
      Default is [0], indicating no additional feed-forward layers.
  dropout_rate_after_enocder : float, optional
      The dropout rate after the encoder layers. Default is 0.0.
  cnn_layer_after_encoder : int, optional
      The number of CNN layers applied after the encoder layers. Default is 0.
  classification : bool, optional
      Whether the model is used for classification. Default is False.
  segment_sizes : int, optional
      The size of segments for segmentation. Default is 1.
  segment_stride : int or None, optional
      The stride for segmentation. If None, stride is set equal to `segment_sizes`. Default is None.
  cls_token : bool, optional
      Whether to use a classification token. Default is False.
  add_positional_encoding : bool, optional
      Add sinusoidal positional encoding. No improvement in performance when time stamps are used. Default is False.
  rCNN_reconstruktion: bool, optional
      Reverse CNN reconstruction (for rekonstruktion of input data from Encoder or CNN-block output for multivariant data (Rekonstruct multiple variables at once)) Beta
  Returns
  -------
  outputs : {ndarray, sparse matrix} of shape (n_samples, output lenght (number of classes or output sequence lenght))
  
  See Also
  -----
  - SegmentationProjektion: A class for segmenting and projecting input data.
  - PositionalEncoding: A class for adding positional encoding to input sequences.
  - EncoderLayer: A class representing a single layer of the Transformer encoder.
  - CNN_Block: A class representing a single block of CNN layers.

  Notes
  -----
  This implementation allows for flexible configuration of the Transformer encoder architecture for time series
  with optional additional layers and features.
  """

  def __init__(self, model_dimension:int, num_layers:int, num_heads:int, encoder_feed_forward:int,
               model_output_size:int, model_input_size:int=None, encoder_dropout_rate : float=0.2, output_feed_forward_layer : list = [0] ,  dropout_rate_after_enocder : float = 0.0, cnn_layer_after_encoder:int = 0, 
               classification:bool=False,  segment_sizes:int=1, segment_stride:int=None, cls_token:bool=False, add_positional_encoding:bool=False, 
               transpose_cnn_for_rekonstruction:bool=False, output_dim:int=None, transpose_cnn_kernal_size:int=16, transpose_cnn_layer_number:int=1):
    super().__init__()


    if cls_token and (output_feed_forward_layer[0] > 0 or cnn_layer_after_encoder>0 or dropout_rate_after_enocder>0):
       raise Exception("If cls token is activatet the following parameter will be ignored: output_feed_forward_layer, dropout_rate_after_enocder, cnn_layer_after_encoder") 

    self.num_layers=num_layers
    self.cnn_layer_after_encoder=cnn_layer_after_encoder
    self.output_feed_forward_layer = output_feed_forward_layer
    self.cls_token = cls_token
    self.segment_size = segment_sizes
    self.segment_stride = segment_stride
    self.add_positional_encoding = add_positional_encoding
    self.transpose_cnn_for_rekonstruction = transpose_cnn_for_rekonstruction
    self.output_dim=output_dim
    self.model_dimension=model_dimension
    self.encoder_feed_forward=encoder_feed_forward
    self.encoder_dropout_rate=encoder_dropout_rate
    if self.output_dim is None:
      self.output_dim=1
  
    #Segmentation of input and projektion in model dimension with CNN
    self.segmenattion_projektion = SegmentationProjektion(model_dimension, segment_sizes, segment_stride) 
          
    #Positional encoding
    if self.add_positional_encoding:
      self.pos_encoding = PositionalEncoding(model_dimension)

    #Encoder
    self.enc_layers = [EncoderLayer(model_dimension=model_dimension, num_heads=num_heads, dff=encoder_feed_forward, dropout_rate=encoder_dropout_rate) for _ in range(num_layers)]

    #CNN Layer
    self.dropout = tf.keras.layers.Dropout(dropout_rate_after_enocder)
    self.cnn_blocks = [ CNN_Block(model_dimension=model_dimension) for _ in range(cnn_layer_after_encoder)]


    if self.transpose_cnn_for_rekonstruction and not classification:

      if not model_input_size is None:
        if model_input_size % model_output_size == 0:
          input_output_diff = model_input_size // model_output_size
        else:
          raise Exception("model output must be dividble by model input")
      else: 
        raise Exception("if output size is differnet to input size -> input size have to be defined")
      

      input_window_reduction_faktor = (2 **(cnn_layer_after_encoder) * segment_stride) // input_output_diff # = stride1 * stride2

      if int(np.sqrt(input_window_reduction_faktor)) == (np.sqrt(input_window_reduction_faktor)):
        stride1=int(np.sqrt(input_window_reduction_faktor))
        stride2=stride1
      elif int(np.sqrt(input_window_reduction_faktor//2)) == (np.sqrt(input_window_reduction_faktor//2)):
        stride1= int(np.sqrt(input_window_reduction_faktor//2)*2)
        stride2= int(np.sqrt(input_window_reduction_faktor//2))
      elif int(np.sqrt(input_window_reduction_faktor//3)) == (np.sqrt(input_window_reduction_faktor//3)):
        stride1= int(np.sqrt(input_window_reduction_faktor//3)*3)
        stride2= int(np.sqrt(input_window_reduction_faktor//3))
      else: 
        print("Warning transposed CNN stride error")


      self.ff_sig_bevor_reconstruction_beta = tf.keras.Sequential([
      tf.keras.layers.Conv1D(filters=self.encoder_feed_forward, kernel_size=1, activation="relu", padding = "same"),
      tf.keras.layers.Conv1D(filters=model_dimension, kernel_size=1, padding = "same", activation="sigmoid"),
      tf.keras.layers.Dropout(self.encoder_dropout_rate),
      ], name="feed_forward_with_sigmoid_activation")

      print("stride", stride1)
      print("stride", stride2)

      if transpose_cnn_layer_number== 2:
        self.transpose_cnn_for_rekonstruction = [tf.keras.layers.Conv1DTranspose(filters=(model_dimension+output_dim)//2, kernel_size=transpose_cnn_kernal_size//2, strides = stride2, padding="same", activation='relu'), tf.keras.layers.Conv1DTranspose(filters=output_dim, kernel_size=transpose_cnn_kernal_size, strides = stride1, padding="same")]
      elif transpose_cnn_layer_number== 1:
        self.transpose_cnn_for_rekonstruction = [tf.keras.layers.Conv1DTranspose(filters=output_dim, kernel_size=transpose_cnn_kernal_size, strides = (stride1*stride2), padding="same")] #One CNN layer
      else:
        raise Exception("transpose_cnn_layer_number param error. Value must be 1 or 2")
    else:

      self.flatten = tf.keras.layers.Flatten()    
      self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

      if classification:      

        if self.output_feed_forward_layer[0] > 0:
          self.output_dense_layer = [tf.keras.layers.Dense(fc, activation='relu', name = "feed_forward_layer") for fc in output_feed_forward_layer ]
        self.final_layer = [tf.keras.layers.Dense(model_output_size, activation='softmax', name = "classification_layer") for dim in range(self.output_dim)] 
      else:        
        if self.output_feed_forward_layer[0] > 0:
          
          self.output_dense_layer = [tf.keras.layers.Dense(fc, activation='relu', name = "feed_forward_layer") for fc in output_feed_forward_layer ]
        self.final_layer = [tf.keras.layers.Dense(model_output_size, name = "output_layer") for dim in range(self.output_dim)] 

    
  #Input (batchsize, sequence lenght, variables)
  def call(self, inputs, return_scores_and_feature_vectore=False):

    #Segmentation into segments (Batchsize, seqenz len, variables) -> (Batchsize, number of segments (seqenz len / segment stride), model dim)
    x = self.segmenattion_projektion(inputs)  

    #Concat random intzilased klassification token at the beginn
    if self.cls_token == True:     
      cls_token=x[:,0:1,:]
      cls_token = tf.random.uniform(tf.shape(cls_token), minval=0, maxval=1,dtype=tf.dtypes.float32)
      x = tf.keras.layers.Concatenate(axis = 1)([cls_token, x])

    #Positional encoding
    if self.add_positional_encoding:
      x = self.pos_encoding(x)

    #Generate list to store the attention heads for each encoder layer  
    attn_scores = [0 for _ in range(self.num_layers)]
    #Prozess Encoder Layer
    for i in range(0, self.num_layers):
      x, attn_scores[i] = self.enc_layers[i](x)

    #store row encoder output
    encoder_feature_vectors = x

    #Separate cls token for klassifikation layer or take the all tokens for klassifikation
    if self.cls_token == True:
      x = x[:,0:1,:] #just cls token
    else:
      #dropout after encoder
      x = self.dropout(x)
      #Process CNN blocks if defined
      for i in range(0,  self.cnn_layer_after_encoder):
        x = self.cnn_blocks[i](x)

       
    #reconstruction with transposed cnn
    if self.transpose_cnn_for_rekonstruction:

      x = self.ff_sig_bevor_reconstruction_beta(x)
      for i in range(0,  len(self.transpose_cnn_for_rekonstruction)):
        x = self.transpose_cnn_for_rekonstruction[i](x)
      outputs=x

    else:

      x = self.flatten(x)
      x = self.layer_norm(x)

      if self.output_feed_forward_layer[0] > 0 and self.cls_token == False:
        for fc in self.output_dense_layer:
          x = fc(x)


      dense_per_layer=[]
      for i in range(self.output_dim):
        dense_per_layer += [self.final_layer[i](x)]

        outputs = tf.convert_to_tensor(dense_per_layer)
        outputs = tf.transpose(outputs, perm=[1, 2, 0])
    

    if return_scores_and_feature_vectore:
       return outputs, attn_scores, encoder_feature_vectors
    else:
      return outputs



  #return: prediction, attention score, encoder output feature vector
  def predict_return_attention_scores_and_feature_vectore(self, x, batch_size = 16):
    """predict data and return attention score and encoder output feature vectors (output tokens)
    """

    prediction=[]    
    attention_score=[]
    encoder_output_feature_vector=[]

    rest = len(x)%batch_size

    for i in np.arange(0,len(x)-rest,batch_size):

        pred, attn, feature_vector = self.call(x[i:i+batch_size], return_scores_and_feature_vectore=True)
        prediction += [(pred)]
        attention_score += [(attn)]
        encoder_output_feature_vector += [(feature_vector)]
        print("Sample:",i)
        gc.collect()


    '''
    for i in np.arange(len(x)-rest, len(x),1):

        pred, attn, feature_vector = self.call(x[i:i+1], return_scores_and_feature_vectore=True)
        prediction += [(pred)]
        attention_score += [(attn)]
        encoder_output_feature_vector += [(feature_vector)]

        print("Sample:",pred.shape)
        gc.collect()
    '''
 

    shape=np.array(prediction).shape
    if len(shape)==4:
      prediction = np.reshape(prediction,(shape[0]*shape[1],shape[2], shape[3]))
    if len(shape)==3:
      prediction = np.reshape(prediction,(shape[0]*shape[1],shape[2]))

    attention_score = np.transpose(attention_score, (0, 2, 1, 3, 4, 5)) # (num batches ,batchsize , num layer, num heads, attention, attention)
    shape=np.array(attention_score).shape
    attention_score = np.reshape(attention_score,(shape[0]*shape[1],shape[2],shape[3], shape[4], shape[5]))

    shape=np.array(encoder_output_feature_vector).shape
    encoder_output_feature_vector = np.reshape(encoder_output_feature_vector,(shape[0]*shape[1],shape[2],shape[3]))

    return prediction, attention_score, encoder_output_feature_vector



  def predict_return_last_token_after_encoder(self, x, batch_size = 16, number_of_token=1):
    """predict data and return attention score and encoder output feature vectors (output tokens)
    """

    encoder_output_feature_vector=[]

    rest = len(x)%batch_size

    for i in np.arange(0,len(x)-rest,batch_size):

        _, _, feature_vector = self.call(x[i:i+batch_size], return_scores_and_feature_vectore=True)

        step = feature_vector.shape[1]//number_of_token
        encoder_output_feature_vector += [(feature_vector[:,::step,:])]
        if not (i//batch_size) % 10:
          print("Sample:",i)
        
        gc.collect()

    shape=np.array(encoder_output_feature_vector).shape
    encoder_output_feature_vector = np.reshape(encoder_output_feature_vector,(shape[0]*shape[1],shape[2],shape[3]))

    for i in np.arange(len(x)-rest, len(x),1):

        _, _, feature_vector = self.call(x[i:i+1], return_scores_and_feature_vectore=True)
        encoder_output_feature_vector = np.append(encoder_output_feature_vector,feature_vector[:,::step,:], axis=0)

        print("Sample:",i)
        gc.collect()

    return encoder_output_feature_vector



  
  def visualize_attention_score_within_timeseries(self, x_test, y_test, number_of_samples_to_plot_from_each_class = 5, x_test_variable_to_plot = 0, batch_size=64 ):
    """Visualizes encoder output feature vectors within a time series with PCA to RGB.
       Time stamp is important for good visualization result.
    """
    pred, attention_score, _ = self.predict_return_attention_scores_and_feature_vectore(x_test, batch_size=batch_size)
    attention_scores_1d = tv.attention_score_to_1D(attention_score, remove_first_token=self.cls_token)
    if len(x_test.shape) > 2:
      ts = np.squeeze(x_test[:,:,x_test_variable_to_plot])
    tv.visualize_attention_score_within_timeseries(ts, attention_scores_1d, self.segment_size, self.segment_stride, number_of_samples_to_plot_from_each_class = number_of_samples_to_plot_from_each_class, label = y_test, prediction_to_print_with_plot=pred)




  def visualize_encoder_output_feature_vector_within_timeseries_with_PCA_to_RRG(self, x_test, y_test, number_of_samples_to_plot_from_each_class = 5, x_test_variable_to_plot= 0):
    """Visualizes encoder output feature vectors within a time series with PCA to RGB (PCA reduction to 3 dim and normalization -> RGB values for each segments in time series)
    """
    pred, _, encoder_output_feature_vector = self.predict_return_attention_scores_and_feature_vectore(x_test, batch_size=batch_size)
    if len(x_test.shape) > 2:
      ts = np.squeeze(x_test[:,:,x_test_variable_to_plot])
    tv.visualize_encoder_output_feature_vector_within_timeseries_with_PCA_to_RRG(ts, encoder_output_feature_vector, self.segment_size, self.segment_stride, number_of_samples_to_plot_from_each_class = number_of_samples_to_plot_from_each_class, label = y_test, prediction_to_print_with_plot=pred)

