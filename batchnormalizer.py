"""A helper class for managing batch normalization state.                   

This class is designed to simplify adding batch normalization               
(http://arxiv.org/pdf/1502.03167v3.pdf) to your model by                    
managing the state variables associated with it.                            

Important use note:  The function get_assigner() returns                    
an op that must be executed to save the updated state.                      
A suggested way to do this is to make execution of the                      
model optimizer force it, e.g., by:                                         

  update_assignments = tf.group(bn1.get_assigner(),                         
                                bn2.get_assigner())                         
  with tf.control_dependencies([optimizer]):                                
    optimizer = tf.group(update_assignments)                                

"""

import tensorflow as tf

class BatchNormalizer(object):
  """Helper class that groups the normalization logic and variables.        

  Use:                                                                      
      ewma = tf.train.ExponentialMovingAverage(decay=0.99)                  
      bn = BatchNormalizer(depth, 0.001, ewma, True)           
      update_assignments = bn.get_assigner()                                
      x = bn.normalize(y, train=training?)                                  
      (the output x will be batch-normalized).                              
  """

  def __init__(self, depth, epsilon, ewma_trainer, scale_after_norm, keep_prob_prior=1.0, name=None):
    with tf.variable_op_scope([self, depth, ewma_trainer, epsilon], name, 'batch_normalizer') as scope:
      self.mean = tf.get_variable('mean', 
        shape=[depth],
        initializer=tf.constant_initializer(0.0),
        trainable=False)
      self.variance = tf.get_variable('variance', 
        shape=[depth],
        initializer=tf.constant_initializer(1.0),
        trainable=False)
      self.beta = tf.get_variable('beta', 
        shape=[depth],
        initializer=tf.constant_initializer(0.0))
      self.gamma = tf.get_variable('gamma', 
        shape=[depth],
        initializer=tf.constant_initializer(1.0))
      print (scope.name)
      self.ewma_trainer = ewma_trainer
      self.epsilon = epsilon
      self.keep_prob_prior = keep_prob_prior

  def get_assigner(self):
    """Returns an EWMA apply op that must be invoked after optimization."""
    return self.ewma_trainer.apply([self.mean, self.variance])

  def normalize(self, x, train=True):
    """Returns a batch-normalized version of x."""
    if train:
      mean, variance = tf.nn.moments(x, [0])
      assign_mean = self.mean.assign(mean)
      assign_variance = self.variance.assign(tf.mul(variance, self.keep_prob_prior))
      with tf.control_dependencies([assign_mean, assign_variance]):
        act_bn = tf.mul((x - mean), tf.rsqrt(variance + self.epsilon), name="act_bn")
        return tf.add(tf.mul(act_bn, self.gamma), self.beta)
      
    else:
      mean = self.ewma_trainer.average(self.mean) or self.epsilon
      variance = self.ewma_trainer.average(self.variance) or self.epsilon
      local_beta = tf.identity(self.beta)
      local_gamma = tf.identity(self.gamma)
      act_bn = tf.mul((x-mean), tf.rsqrt(variance + self.epsilon), name="act1_bn")
      return tf.add(tf.mul(act_bn, local_gamma), local_beta)
      