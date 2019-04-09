#encoding=utf-8
import numpy as np
import tensorflow as tf

#W = tf.Variable(np.arange(6).reshape(2,3), dtype=tf.float32)
#b = tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32)
W = tf.Variable(np.arange(6).reshape(2,3), dtype=tf.float32,name="W")
b = tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32, name='b')
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'save_net.ckpt')
    print("Weight:\n", sess.run(W))
    print("biase:\n", sess.run(b))