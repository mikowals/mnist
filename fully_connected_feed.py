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
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import mnist

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.2, 'Initial learning rate.')
flags.DEFINE_float('momentum', 0.5, 'Initial momentum.')
flags.DEFINE_float('beta2', 0.999, 'second moment for gradient in Adam.')
flags.DEFINE_float('max_norm', 2.0,'max norm of weights')
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 1000, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 1000, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 2048, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('keep_prob', 0.50, 'dropout ratio for hidden layers')
flags.DEFINE_integer('keep_input', 0.80, 'dropout ratio for input layer')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('eval_batch_size', 10000, 'Batch size for eval.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_string('experiment-name', 'spearmint', 'Spearmint expiriement name')
flags.DEFINE_string('database-address', 'localhost', 'mongodb host')
flags.DEFINE_string('job-id', '001', 'Spearmint job id.')


def run_training(learning_rate=0.1,
        momentum=0.8,
        max_norm=2.0,
        keep_prob=0.5,
        beta2=0.999):
  """Train MNIST for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.
  data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder = tf.placeholder(tf.float32, shape=None, name='images')
    labels_placeholder = tf.placeholder(tf.int32, shape=None, name='labels')
    keep_prob_placeholder = tf.placeholder("float", name='keep_prob')
    keep_input_placeholder = tf.placeholder("float", name='keep_input')
    def fill_feed_dict(data_set, keep_prob, keep_input, batch_size = FLAGS.batch_size):
      # Create the feed_dict for the placeholders filled with the next
      # `batch size ` examples.
      images_feed, labels_feed = data_set.next_batch(batch_size,
                                                     FLAGS.fake_data)
      feed_dict = {
          images_placeholder: images_feed,
          labels_placeholder: labels_feed,
          keep_prob_placeholder: keep_prob,
          keep_input_placeholder: keep_input
      }
      return feed_dict

    def do_eval(sess,
            eval_correct,
            data_set,
            label):
      """Runs one evaluation against the full epoch of data.

      Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
        images_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of images and labels to evaluate, from
          input_data.read_data_sets().
      """
      # And run one epoch of eval.
     
      true_count = 0  # Counts the number of correct predictions.
      true_loss = 0
      steps_per_epoch = data_set.num_examples // FLAGS.eval_batch_size
      num_examples = steps_per_epoch * FLAGS.eval_batch_size
      for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   1.0, 1.0, FLAGS.eval_batch_size)
        true_count, true_loss += sess.run([eval_correct, loss], feed_dict=feed_dict)
      precision = true_count / num_examples
      avg_loss = true_loss / steps_per_epoch
      print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
            (num_examples, true_count, precision))
    
      return precision, avg_loss

    # Build a Graph that computes predictions from the inference model.
    logits = mnist.inference(images_placeholder,
                         FLAGS.hidden1,
                         FLAGS.hidden2,
                         FLAGS.hidden3,
                         keep_prob_placeholder,
                         keep_input_placeholder,
                         max_norm)

    # Add to the Graph the Ops for loss calculation.
    loss = mnist.loss(logits, labels_placeholder)
    
    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = mnist.training(loss, learning_rate, momentum, beta2)
    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = mnist.evaluation(logits, labels_placeholder)
    precision_labels = tf.constant(["correct_train", "loss_train", "correct_test", "loss_test])
    precision_placeholder = tf.placeholder(tf.float32, [4])
    summarize_precision = tf.scalar_summary(precision_labels, precision_placeholder)
    
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()
  
    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    
    sess = tf.Session()
    #restore_path = tf.train.latest_checkpoint('/Users/mikowals/projects/tensorflow/tensorflow/g3doc/tutorials/mnist/', 'checkpoint')
    #print(restore_path)
    #saver.restore(sess, restore_path)

    # Run the Op to initialize the variables.
    init = tf.initialize_all_variables()
    sess.run(init)
    # saver.restore(sess, 'data-3999')
    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph_def)
    train_cor = test_cor = 0.97
    train_loss = test_loss = 2.0
    # And then after everything is built, start the training loop.
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      
      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = fill_feed_dict(data_sets.train, keep_prob, FLAGS.keep_input)
      
      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        feed_dict[precision_placeholder] = [train_cor, train_loss, test_cor, test_loss]
        sess.run(summarize_train, summarize_test, feed_dict=feed_dict)
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        #saver.save(sess, FLAGS.train_dir, global_step=step)
        # Evaluate against the training set.
        
        # Evaluate against the validation set.
        print('training Data Eval:')
        train_cor, train_loss = do_eval(sess,
                eval_correct,
                data_sets.train,
                'train')
  #      sess.run(logValue, feed_dict={out_pl: val_cor, label_pl: 'validation_correct'})
        # Evaluate against the test set.
        print('Test Data Eval:')
        test_cor, test_loss = do_eval(sess,
                eval_correct,
                data_sets.validation,
                'test')
        if (step > 5000) and (test_cor < 0.9):
          return 1.0 - test_loss        
        if (step > 15000) and (test_cor < 0.95):
          return test_loss

  return test_loss

def main(job_id, params):
  return run_training(
        learning_rate=params['learning_rate'][0].item(),
        momentum=params['momentum'][0].item(),
        max_norm=params['max_norm'][0].item(),
        keep_prob=params['keep_prob'][0].item(),
        beta2=params['beta2'][0].item())


if __name__ == '__main__':
  tf.app.run()
