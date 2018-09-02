# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 13:24:39 2018

@author: hcc
"""
"""
tf.concat 是连接两个矩阵的操作
tf.concat(concat_dim,values,name='concat')
concat_dim:必须是一个数，表明在哪一维上连接

"""
#encoding=utf8
import tensorflow as tf
t1 = [[[1, 1, 1],[2, 2, 2]],[[3, 3, 3],[4, 4, 4]]]
t2 = [[[5, 5, 5],[6, 6, 6]],[[7, 7, 7],[8, 8, 8]]]

with tf.Session() as sess:
    print("=====-1========")
    a = tf.concat(t1,-1)
    print(sess.run(a))
    print("=======-2======")
    b = tf.concat(t1,-2)
    print(sess.run(b))
    print("=============")
    c = tf.concat([t1,t2],0)
    print(sess.run(c))
    print("=============")
    d = tf.concat([t1,t2],1)
    print(sess.run(d))
    print("=============")
    d = tf.concat([t1,t2],2)
    print(sess.run(d))
"""
=============
[[1 1 1 3 3 3]
 [2 2 2 4 4 4]]
=============
[[1 1 1]
 [2 2 2]
 [3 3 3]
 [4 4 4]]
=============
[[[1 1 1]
  [2 2 2]]

 [[3 3 3]
  [4 4 4]]

 [[5 5 5]
  [6 6 6]]

 [[7 7 7]
  [8 8 8]]]
=============
[[[1 1 1]
  [2 2 2]
  [5 5 5]
  [6 6 6]]

 [[3 3 3]
  [4 4 4]
  [7 7 7]
  [8 8 8]]]
=============
[[[1 1 1 5 5 5]
  [2 2 2 6 6 6]]

 [[3 3 3 7 7 7]
  [4 4 4 8 8 8]]]
"""