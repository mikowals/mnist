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

import tensorflow.python.platform
import tensorflow as tf


# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
SEED = 25

def add_noise(grad, var, noise=0.01):
  grad += tf.random_normal(stddev=noise)
  return (grad, var)

def inference(images, hidden_units, hidden2_units):
  """Build the MNIST model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    hidden1: Size of the first hidden layer.
    hidden2: Size of the second hidden layer.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  
  def hidden_layer(data, input_size, layer_size, name=None):
    with tf.variable_scope( name) as scope:
                                
               
      weights = tf.get_variable('weights', 
        [input_size, layer_size],
        initializer=tf.random_normal_initializer(stddev=tf.sqrt(2.0 / float(input_size))))
      biases = tf.get_variable( "biases", 
        [layer_size], 
        initializer=tf.constant_initializer(0.0))
      
      if not scope.reuse:
        tf.histogram_summary(weights.name, weights)
        #tf.histogram_summary(biases.name, biases)
     
      return tf.nn.relu(tf.matmul(data, weights) + biases)
  
  
  # Hidden 1
  hidden1= hidden_layer(images, IMAGE_PIXELS, hidden_units, 'hidden1')
  hidden2 = hidden_layer(hidden1, hidden_units, hidden_units, 'hidden2')
  
  with tf.variable_scope('softmax_linear') as scope:
    weights = tf.get_variable("weights",
      [hidden_units, NUM_CLASSES],
      initializer=tf.random_normal_initializer(stddev=tf.sqrt(2.0 / float(hidden_units))))
                            
    biases = tf.get_variable('biases',
      [NUM_CLASSES],
      initializer=tf.constant_initializer(0.0))
    #weights = clip_weight_norm(weights, max_norm, name='clipped_weights')
    if not scope.reuse:
      tf.histogram_summary(weights.name, weights)
      tf.histogram_summary(biases.name, biases)
    
  return tf.add(tf.matmul(hidden2, weights), biases, name="logits")


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
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

  return loss


def training(loss, initial_learning_rate=0.05):
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
  global_step = tf.Variable(0.0, name='global_step', trainable=False)
  
  
  # Create the gradient descent optimizer with the given learning rate.
  opt = tf.train.GradientDescentOptimizer(initial_learning_rate)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  #max_norm_tensor = tf.Variable(max_norm, name='max_norm')
  # Compute the gradients for a list of variables.
  
  train_op = opt.minimize(loss, global_step=global_step)
  
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
