"""Trains and Evaluates the MNIST network using a feed dictionary.

TensorFlow install instructions:
https://tensorflow.org/get_started/os_setup.html

MNIST tutorial:
https://tensorflow.org/tutorials/mnist/tf/index.html

"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time

import tensorflow.python.platform
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import mnist

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.05, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 300000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1_units', 800, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2_units', 800, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('adversarial_noise', 0.08, 'additive noise for fast gradient sign')
flags.DEFINE_integer('init_std', 0.01, 'initialization stddev')
flags.DEFINE_integer('noise_std', 0.3, 'additive noise for hidden units in training')
flags.DEFINE_integer('batch_size', 10, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_string('experiment-name', 'spearmint', 'Spearmint expiriement name')
flags.DEFINE_string('database-address', 'localhost', 'mongodb host')
flags.DEFINE_string('job-id', '001', 'Spearmint job id.')

def make_adversarial_inputs(inputs, grad, noise):
  return tf.add(inputs, noise * tf.sign(grad), name='adversarial_inputs')
  
def run_training(  
  learning_rate=FLAGS.learning_rate,
  hidden1_units=FLAGS.hidden1_units,
  hidden2_units=FLAGS.hidden2_units,
  init_std=FLAGS.init_std,
  noise_std = FLAGS.noise_std,
  adversarial_noise=FLAGS.adversarial_noise):
  """Train MNIST for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.
  data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    tf.set_random_seed(10)
    # Generate placeholders for the images and labels.
    images_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                         mnist.IMAGE_PIXELS), name='images')
    labels_placeholder = tf.placeholder(tf.int32, shape=[None], name='labels')
    noise_std_pl = tf.placeholder(tf.float32, name='noise_std')
    
    def fill_feed_dict(data_set, batch_size=FLAGS.batch_size):
      # Create the feed_dict for the placeholders filled with the next
      # `batch size ` examples.
      images_feed, labels_feed = data_set.next_batch(batch_size,
                                                     FLAGS.fake_data)
      feed_dict = {
          images_placeholder: images_feed,
          labels_placeholder: labels_feed
      }
      return feed_dict
    
    def fill_feed_dict_eval(data_set):
      return {
        images_placeholder: data_set._images,
        labels_placeholder: data_set._labels,
        noise_std_pl: 0.0
      }

    # Build a Graph that computes predictions from the inference model.
    with tf.variable_scope('feed_forward_model') as scope:
      logits = mnist.inference(images_placeholder,
                         hidden1_units,
                         hidden2_units,
                         init_std,
                         noise_std_pl)

    # Add to the Graph the Ops for loss calculation.
    loss = mnist.loss(logits, labels_placeholder)
    
    #generate adversarial examples
    input_gradient = tf.gradients(loss, images_placeholder)[0]
    adversarial_inputs = tf.stop_gradient(make_adversarial_inputs(images_placeholder, input_gradient, adversarial_noise))
    
    train_op = mnist.training(loss, learning_rate)
    
    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = mnist.evaluation(logits, labels_placeholder)
    
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()
  
    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.

    first_step = 0
    sess = tf.Session()
    restore_path = tf.train.latest_checkpoint("/home/ubuntu/mnist")
    if restore_path:
      saver.restore(sess, restore_path)
      first_step = int(restore_path.split('/')[-1].split('-')[-1])

    else:
      # Run the Op to initialize the variables.
      init = tf.initialize_all_variables()
      sess.run(init)

    train_loss = test_loss = 0
    train_cor = test_cor = 0.97
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph_def)
    # And then after everything is built, start the training loop.
    for step in xrange(first_step,FLAGS.max_steps):
      start_time = time.time()
      
      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = fill_feed_dict(data_sets.train)
      feed_dict[noise_std_pl] = 0.0
      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      adv_inputs = sess.run(adversarial_inputs, feed_dict=feed_dict)
      feed_dict[images_placeholder] = adv_inputs
      feed_dict[noise_std_pl] = noise_std
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)
      
      duration = time.time() - start_time
      # Write the summaries and print an overview fairly often.
      
      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        #saver.save(sess, FLAGS.train_dir, global_step=step)
        # Evaluate against the training set.
        
        # Evaluate against the validation set.
        print('training Data Eval:')
        feed_dict = fill_feed_dict_eval(data_sets.train)
        train_cor, train_loss = sess.run([eval_correct, loss], feed_dict=feed_dict)
        train_cor = train_cor / data_sets.train.num_examples
        print(train_cor, train_loss)
  
        print('Validation Data Eval:')
        feed_dict = fill_feed_dict_eval(data_sets.validation)
        test_cor, test_loss = sess.run([eval_correct, loss], feed_dict=feed_dict)
        test_cor = test_cor / data_sets.validation.num_examples
        print (test_cor, test_loss )
        

      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        

  return -test_cor 

def main(job_id='a-1', params={}):
  params.setdefault("learning_rate", np.array([FLAGS.learning_rate]))
  params.setdefault("hidden1_units", np.array([FLAGS.hidden1_units]))
  params.setdefault("hidden2_units", np.array([FLAGS.hidden2_units]))
  params.setdefault("init_std", np.array([FLAGS.init_std]))
  params.setdefault("noise_std", np.array([FLAGS.noise_std]))
  params.setdefault("adversarial_noise", np.array([FLAGS.adversarial_noise]))
  
  return run_training(
        learning_rate=params['learning_rate'][0].item(),
        hidden1_units=params['hidden1_units'][0].item(),
        hidden2_units=params['hidden2_units'][0].item(),
        init_std=params['init_std'][0].item(),
        noise_std=params['noise_std'][0].item(),
        adversarial_noise=params['adversarial_noise'][0].item())
        
if __name__ == '__main__':
  tf.app.run()
