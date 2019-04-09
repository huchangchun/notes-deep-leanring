#encoding=utf-8
import numpy as np
import tensorflow as tf
#save sess
#W = tf.Variable([[1,2,3],[4,5,6]], dtype=tf.float32)
#b = tf.Variable([[1,2,3]], dtype=tf.float32)
#如果加入name变量，restore的时候也要加name
W = tf.Variable([[1,2,3],[4,5,6]], dtype=tf.float32, name="W")
b = tf.Variable([[1,2,3]], dtype=tf.float32, name="b")
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess,'save_net.ckpt')
    
