#encoding=utf-8
import tensorflow as tf
import numpy as np
A = [[1, 3, 4, 5, 6]]
B = [[1, 3, 4, 3, 2]]
with tf.Session() as sess:
    argmax = sess.run(tf.argmax(A,axis=1))
    argmax2 =sess.run(tf.argmax(A))
    eq = sess.run(tf.equal(A,B))
    cast = sess.run(tf.cast(eq,tf.float32))
    av = sess.run(tf.reduce_mean(cast))
    print("argmax:",argmax)
    print("argmax2:",argmax2)
    print("eq:",eq)
    print("cast:",cast)
    print("av:",av)
    print(sess.run(tf.shape(A)[0]))