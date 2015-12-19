"""Builds the MNIST network.
Built to duplicate Hinton 2012 and Srivastava 2014 method of dropout feed forward 
network with max norm regularisaton.  A few learnings:

  * dropout not applied to softmax layer (input and hidden layers only )
  * max norm not applied to softmax layer - though it could be applied at a different rate
  * max norm applied by hidden unit, 
     - for example h1 has 784 weights (input size) at each hidden unit (w is 784 x h_units),  
       so the norm of interest is the sqrt( sum of squares by columns)
  * adam optimiser works as well and faster than elaborate annealed learning and increasing momentum
  * loss function tried with raw cross_entropy rather than mean but needed very small learning rates
  * 

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.

TensorFlow install instructions:
https://tensorflow.org/get_started/os_setup.html

MNIST tutorial:
https://tensorflow.org/tutorials/mnist/tf/index.html
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from batchnormalizer import BatchNormalizer

import tensorflow.python.platform
import tensorflow as tf


# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
SEED = 25

def p_relu_layer(x, w, b, a, name=None):
  with tf.op_scope([x, w, b, a], name, "p_relu") as scope:
    t = tf.matmul(x,w) + b
    return tf.maximum(t,0) + tf.mul(a, tf.minimum(t,0))

def max_out(t, size, name=None):
  with tf.op_scope([t, size], name, "max_out") as scope:
    t_size = tf.shape(t)
    m = tf.reshape(t,[t_size[0], t_size[1] // size, size])
    m = tf.reduce_max(m,2)
    return m

def gaussian_dropout(t, p, name=None, seed=None):
  with tf.op_scope([t, p], name, "gaussian_dropout") as scope:
    sd = tf.sqrt(tf.div(tf.sub(1.0, p), p))
    noise = tf.random_normal( tf.shape(t), mean=1.0, stddev=sd, seed=seed)
    return t * noise

def clip_weight_norm(t, clip_norm, name=None):
  with tf.op_scope([t, clip_norm], name, "clip_weight_norm") as scope:
    l2norm_inv = tf.rsqrt(
      tf.reduce_sum(t * t, 0))
    tclip = tf.identity(t * clip_norm * tf.minimum(
      l2norm_inv, tf.constant(1.0 / clip_norm)))

    return tclip

def inference(images, hidden_units, num_layers, wd, keep_prob=1.0, keep_input=1.0, max_norm=100.0):
  """Build the MNIST model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    hidden1: Size of the first hidden layer.
    hidden2: Size of the second hidden layer.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  
  initial_a = 0.0
  def hidden_layer(data, input_size, layer_size, keep_prob_prior, name=None):
    with tf.variable_op_scope([data, input_size, layer_size], name, "hidden_layer") as scope:
      ewma = tf.train.ExponentialMovingAverage(decay=0.99, name='ema_' + name)                  
      bn = BatchNormalizer(layer_size, 0.001, ewma, True, keep_prob_prior,'bn_'+name)                                      
               
      weights = tf.get_variable('weights', 
        [input_size, layer_size],
        initializer=tf.truncated_normal_initializer(0,
                              stddev=math.sqrt(2.0 / ((1.0 + initial_a ** 2.0) * float(input_size)))))
      

      
      #weights = clip_weight_norm(weights, max_norm, name='clipped_weights')
      if not scope.reuse:
        tf.histogram_summary(weights.name, weights)            
      x = bn.normalize(tf.matmul(data,weights), train=keep_prob < 1.0)
      mean, variance = tf.nn.moments(x, [0])
      c = tf.div(tf.matmul(x-mean, x-mean, transpose_a=True), tf.to_float(tf.shape(x)[0]))
      weight_decay = 0.0
      if (keep_prob < 1.0):
        weight_decay = tf.nn.l2_loss(c) - tf.nn.l2_loss(variance)#tf.mul(tf.nn.l2_loss(weights), wd, name='weight_loss')
      
      tf.add_to_collection('losses', weight_decay)

      hidden = tf.nn.elu(x)
      #tf.scalar_summary('sparsity_'+hidden.name, tf.nn.zero_fraction(hidden))
      hidden_dropout = tf.nn.dropout(hidden, keep_prob)
      return hidden_dropout, bn
  
  images = gaussian_dropout(images, keep_input, name='input_dropout')
  # Hidden 1
  hidden_layers = [None for num in range(num_layers)]
  batch_normalizations = [None for num in range(num_layers)]
  hidden_layers[0], batch_normalizations[0] = hidden_layer(images, IMAGE_PIXELS, hidden_units, 1.0 / ( 1.0 + tf.sqrt(tf.div(tf.sub(1.0, keep_input), keep_input))), 'hidden1')
  for layer in range(1,num_layers):
    hidden_layers[layer], batch_normalizations[layer] = hidden_layer(hidden_layers[layer - 1], hidden_units, hidden_units, keep_prob, 'hidden'+str(layer + 1))
  # Linear
  with tf.variable_scope('softmax_linear') as scope:
    weights = tf.get_variable("weights",
      [hidden_units, NUM_CLASSES],
      initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / ((1.0 + initial_a ** 2.0) * float(hidden_units)))))
                            
    biases = tf.get_variable('biases',
      [NUM_CLASSES],
      initializer=tf.constant_initializer(0.0))
    #weight_decay = tf.mul(tf.nn.l2_loss(weights), wd, name='weight_loss')
    #tf.add_to_collection('losses', weight_decay)
    #weights = clip_weight_norm(weights, max_norm, name='clipped_weights')
    if not scope.reuse:
      tf.histogram_summary(weights.name, weights)
      tf.histogram_summary(biases.name, biases)
    logits = tf.matmul(hidden_layers[-1], weights) + biases
  return logits, batch_normalizations

def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  # Convert from sparse integer labels in the range [0, NUM_CLASSSES)
  # to 1-hot dense float vectors (that is we will have batch_size vectors,
  # each with NUM_CLASSES values, all of which are 0.0 except there will
  # be a 1.0 in the entry corresponding to the label).
  batch_size = tf.size(labels)
  labels = tf.expand_dims(labels, 1)
  indices = tf.expand_dims(tf.range(0,batch_size,1), 1)
  concated = tf.concat(1, [indices, labels])
  onehot_labels = tf.sparse_to_dense(
      concated, tf.pack([batch_size, NUM_CLASSES]), 1.0, 0.0)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                          onehot_labels,
                                                          name='xentropy')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses')) + cross_entropy_mean 

def training(loss, initial_learning_rate=0.1, initial_momentum=0.9, beta2=0.999):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Create a variable to track the global step.
  step = tf.Variable(0.0, trainable=False, name='global_step')
  
  learning_rate = tf.train.exponential_decay(
    initial_learning_rate,        # Base learning rate.
    step,  # Current index into the dataset.
    5000,          # Decay step.
    0.6,       # Decay rate.
    staircase=True)
  
  final_momentum = 0.999
  momentum_steps = 25000.0
  momentum = tf.minimum( final_momentum, 
    tf.add(initial_momentum,tf.mul(step, tf.div(tf.sub(final_momentum, initial_momentum), momentum_steps))))
  #momentum = tf.identity( min( 0.95, initial_momentum + tf.to_float(global_step) * (0.95 - initial_momentum)/ 20000.0))
  tf.scalar_summary('model_momentum', momentum)
  tf.scalar_summary('model_learning_rate', learning_rate)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.AdamOptimizer(learning_rate, initial_momentum)
  #optimizer = tf.train.AdamOptimizer(initial_learning_rate, initial_momentum)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  #max_norm_tensor = tf.Variable(max_norm, name='max_norm')

  train_op = optimizer.minimize(loss, global_step=step)
  
  return train_op

def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label's is was in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))
