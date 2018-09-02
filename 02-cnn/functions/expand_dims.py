# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 13:49:41 2018

@author: hcc
"""
"""
假设维度为1x3,在第二个维度上增加一维，即将3/3
"""
import tensorflow as tf
import numpy as np
t1 = [[[1, 1, 1],[2, 2, 2]],[[3, 3, 3],[4, 4, 4]]]
t2=[1,2,3]
a = np.asarray(t2)

print("a:\n",a)
print(a.shape)
b = np.reshape(a,[-1,3])  #-1是占位符，表示Unknown的维度，通过维度乘积不变性可以计算出来
print("b.shape:\n",b.shape)
print("b:\n",b)
c = np.reshape(a,[3,-1])
print("c.shape:\n",c.shape)
print("c:\n",c)
with tf.Session() as sess:
    print("------------")
    print("t1:\n",t1)
    print("----t1 + 1--------")
    a = tf.expand_dims(t1,1)
    print(sess.run(a))
    print("t2:\n",t2)
    print("----t2+1--------")
    b = tf.expand_dims(t2,1)
    print(sess.run(b))
    
"""
a:
 [1 2 3]
(3,)
b.shape:
 (1, 3)
b:
 [[1 2 3]]
c.shape:
 (3, 1)
c:
 [[1]
 [2]
 [3]]
------------
t1:
 [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
----t1 + 1--------
[[[[1 1 1]
   [2 2 2]]]


 [[[3 3 3]
   [4 4 4]]]]
t2:
 [1, 2, 3]
----t2+1--------
[[1]
 [2]
 [3]]

"""