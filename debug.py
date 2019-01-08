import tensorflow as tf
import numpy as np

vals_np = np.random.rand(4, 5)
print(vals_np)
vals = tf.constant(vals_np, dtype=tf.float32)
svals = tf.gather(vals, [0, 2], axis=0)
sess = tf.Session()
r = sess.run(svals)
print(r)
