#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
input_ids = tf.placeholder(dtype=tf.int32, shape=None)
#input_ids = tf.placeholder(dtype=tf.int32, shape=[3,2])
embedding = tf.Variable(np.identity(5,dtype = np.int32))
input_embedding = tf.nn.embedding_lookup(embedding, input_ids)
init=tf.global_variables_initializer()
        
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)#对变量进行初始化，执行（run）init语句
    input_embedding= sess.run(input_embedding, feed_dict={input_ids:[1,2,3,0,3,2,1]}) 
    print(embedding.eval())
    print('-------------')
    print(input_embedding)
    
#------------------------------------    
"""
    [[1 0 0 0 0]
     [0 1 0 0 0]
     [0 0 1 0 0]
     [0 0 0 1 0]
     [0 0 0 0 1]]
    -------------
    [[0 1 0 0 0]
     [0 0 1 0 0]
     [0 0 0 1 0]
     [1 0 0 0 0]
     [0 0 0 1 0]
     [0 0 1 0 0]
     [0 1 0 0 0]]
    
从上面输出可以知道，就是根据input_ids中的id寻找embedding中对应元素，如果input_ids = [1,2,3]就是找对应
下标为1,2,3的向量组成一个矩阵返回
"""
#如果改成下面就生成另一种了
#with tf.Session() as sess:
    #sess.run(init)#对变量进行初始化，执行（run）init语句
    #input_embedding= sess.run(input_embedding, feed_dict={input_ids:[[1,2],[3,0],[3,3]]}) 
    #print(embedding.eval())
    #print('-------------')
    #print(input_embedding)
"""
[[1 0 0 0 0]
 [0 1 0 0 0]
 [0 0 1 0 0]
 [0 0 0 1 0]
 [0 0 0 0 1]]
-------------
[[[0 1 0 0 0]
  [0 0 1 0 0]]

 [[0 0 0 1 0]
  [1 0 0 0 0]]

 [[0 0 0 1 0]
  [0 0 0 1 0]]]
"""