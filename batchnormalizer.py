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

  def __init__(self, depth, epsilon, ewma_trainer, scale_after_norm):
    
    self.mean = tf.Variable(tf.constant(0.0000, shape=[depth]),
                            trainable=False)
    self.variance = tf.Variable(tf.constant(1.0, shape=[depth]),
                                trainable=False)
    self.beta = tf.Variable(tf.constant(0.0000, shape=[depth]))
    self.gamma = tf.Variable(tf.constant(1.0, shape=[depth]))
    self.ewma_trainer = ewma_trainer
    self.epsilon = epsilon
    self.scale_after_norm = scale_after_norm

  def get_assigner(self):
    """Returns an EWMA apply op that must be invoked after optimization."""
    return self.ewma_trainer.apply([self.mean, self.variance])

  def normalize(self, x, train=True):
    """Returns a batch-normalized version of x."""
    if train:
      mean, variance = tf.nn.moments(x, [0])
      assign_mean = self.mean.assign(mean)
      assign_variance = self.variance.assign(variance)
      with tf.control_dependencies([assign_mean, assign_variance]):
        act_bn = tf.mul((x - mean), tf.rsqrt(variance + self.epsilon), name="act_bn")
        return tf.add(tf.mul(act_bn, self.gamma), self.beta)
      
    else:
      mean = self.ewma_trainer.average(self.mean) or tf.reduce_mean(x, [0])
      variance = self.ewma_trainer.average(self.variance) or tf.reduce_sum(tf.square(x),[0])
      local_beta = tf.identity(self.beta)
      local_gamma = tf.identity(self.gamma)
      act_bn = tf.mul((x-mean), tf.rsqrt(variance + self.epsilon), name="act1_bn")
      return tf.add(tf.mul(act_bn, local_gamma), local_beta)
      