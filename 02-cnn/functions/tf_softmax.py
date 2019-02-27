import tensorflow as tf
A = tf.constant([[1.0,2.0,3.0,4.0,5.0],[2.0,2.0,3.0,4.0,7.0]])
with tf.Session() as sess:
    print(sess.run(tf.nn.softmax(A)))