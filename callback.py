import tensorflow as tf

class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, initial_learning_rate, max_lr, warmup_steps, update_every):
      self.initial_learning_rate = initial_learning_rate
      self.max_lr = max_lr
      self.warmup_steps = warmup_steps
      self.update_every = update_every
      self.real_step = 0
      self.curr = 0
  
  def call_helper(self):
      self.real_step += 1
      self.curr = self.max_lr * 100 * min(self.real_step ** (-0.5), self.real_step * self.warmup_steps**(-1.5))

  def __call__(self, step):
      tf.cond(
        tf.math.logical_and(tf.equal(step % self.update_every, 0), tf.greater(step, 0)),
        true_fn = lambda: self.call_helper(),
        false_fn = lambda: None,
      )
      return self.curr