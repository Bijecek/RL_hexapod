import tensorflow as tf
import numpy as np
class RunningNormalizer:
    def __init__(self, shape):
        # Initialize variables with float32 dtype
        self.mean = tf.Variable(tf.zeros(shape, dtype=tf.float32), trainable=False)
        self.var = tf.Variable(tf.ones(shape, dtype=tf.float32), trainable=False)
        self.count = tf.Variable(1e-4, trainable=False, dtype=tf.float32)

    def update(self, batch):
        # Convert batch to float32 Tensor
        if not isinstance(batch, tf.Tensor):
            batch = tf.convert_to_tensor(batch, dtype=tf.float32)
        else:
            batch = tf.cast(batch, tf.float32)  # Ensure dtype is float32

        batch_mean = tf.reduce_mean(batch, axis=0)
        batch_var = tf.math.reduce_variance(batch, axis=0)
        batch_count = tf.cast(tf.shape(batch)[0], tf.float32)

        # Update statistics (all in float32)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / total_count)
        self.mean.assign(new_mean)

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + tf.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count
        self.var.assign(new_var)

        self.count.assign(total_count)

    def normalize(self, x):
        # Convert input to float32 Tensor
        if not isinstance(x, tf.Tensor):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
        else:
            x = tf.cast(x, tf.float32)

        return (x - self.mean) / tf.sqrt(self.var + 1e-8)