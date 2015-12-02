"""Builds the MNIST network.
Built to duplicate Hinton 2012 and Srivastava 2014 method of dropout feed forward 
network with max norm regularisaton.  A few learnings:

  * dropout not applied to softmax layer (input and hidden layers only )
  * max norm not applied to softmax layer - though it could be applied at a different rate
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
import numpy as np
import math

import tensorflow.python.platform
import tensorflow as tf

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def p_relu(t, a, name=None):
  with tf.op_scope([t, a], name, "p_relu") as scope:
    return tf.maximum(t,0.0) + tf.mul(a, tf.minimum(t, 0.0))

def gaussian_dropout(t, p, name=None):
  with tf.op_scope([t, p], name, "gaussian_dropout") as scope:
    sd = tf.sqrt(tf.div(tf.sub(1.0, p), p))  # ((1 - p) / p) ** (1/2)
    noise = tf.random_normal( tf.shape(t), mean=1.0, stddev=sd)
    return t * noise

def clip_weights_by_norm(t, clip_norm, name=None):
  with tf.op_scope([t, clip_norm], name, "max_norm") as scope:
    l2norm_inv = tf.rsqrt(
        tf.reduce_sum(t * t, 0))
    tclip = tf.identity(t * clip_norm * tf.minimum(
        l2norm_inv, tf.constant(1.0 / clip_norm)), name=name)

  return tclip 

def inference(images, hidden1_units, hidden2_units, hidden3_units, keep_prob=1.0, keep_input=1.0, max_norm=100.0):
  """Build the MNIST model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    hidden1: Size of the first hidden layer.
    hidden2: Size of the second hidden layer.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """

  def hidden_layer(data, input_size, layer_size, name):
    with tf.name_scope(name) as scope:
      #a = tf.Variable(0.25, name="a_" + name)
      weights = tf.Variable(
          tf.random_normal([input_size, layer_size],
                              stddev=math.sqrt(2.0 / (( 1.0 + 0.0 ** 2) * float(input_size)))),
          name='weights')
      biases = tf.Variable( tf.zeros([layer_size]),
                           name='biases')
      weights =clip_weights_by_norm( weights, max_norm)
      tf.histogram_summary('w_'+name, weights)
      tf.histogram_summary('b_'+name, biases)
      #tf.scalar_summary('a_' + name, a)
      hidden = tf.nn.relu(tf.matmul(data, weights) + biases)
      hidden_dropout = gaussian_dropout(hidden, keep_prob)
      return hidden_dropout

  images = gaussian_dropout(images, keep_input)
  # Hidden 1
  hidden1 = hidden_layer(images, IMAGE_PIXELS, hidden1_units, 'hidden1')
  # near working attempt at visualisation.  See 3 images when expecting 10
  #tf.image_summary('images', tf.expand_dims(tf.reshape(tf.slice(images,[0,0],[5,784]), [5, 28, 28]), 3))
  hidden2 = hidden_layer(hidden1, hidden1_units, hidden2_units, 'hidden2')
  #hidden3 = hidden_layer(hidden2, hidden2_units, hidden3_units, 'hidden3')  
  
  # Linear
  with tf.name_scope('softmax_linear') as scope:
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
                            name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    tf.histogram_summary('w_softmax', weights)
    tf.histogram_summary('b_softmax', biases)
    logits = tf.matmul(hidden2, weights) + biases
  return logits


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


def training(loss, initial_learning_rate, initial_momentum, beta2=0.999):
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
  learning_rate = tf.train.exponential_decay(
      initial_learning_rate,        # Base learning rate.
      global_step,  # Current index into the dataset.
      300,          # Decay step.
      0.993,                # Decay rate.
      staircase=False)
  final_momentum = tf.Variable(0.97)
  momentum_steps = tf.Variable(20000.0)
  momentum = tf.minimum( final_momentum, 
      tf.add(initial_momentum,tf.mul(global_step, tf.div(tf.sub(final_momentum, initial_momentum), momentum_steps))))
  tf.scalar_summary(loss.op.name, loss)
  tf.scalar_summary('model_momentum', momentum)
  tf.scalar_summary('model_learning_rate', learning_rate)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.AdamOptimizer(initial_learning_rate, initial_momentum, beta2=beta2)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  #max_norm_tensor = tf.Variable(max_norm, name='max_norm')
  #tvars = tf.trainable_variables()
  #grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
  #                                    5.0)
  #train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
  train_op = optimizer.minimize(loss, global_step=global_step)
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
