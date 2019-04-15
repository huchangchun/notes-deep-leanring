#-*- coding:utf-8 -*-
"""
embedding本质上就是一个全连接层，可以看例子2：
在nlp中一般会有一个word_to_id的字典，在数据预处理过程中，会将输入的字符转化为id,
假设word_to_id的长度为features_num
因此输入x是字对应的id的序列，假设它的shape是(batch_size,seq_length)
embedding_lookup就是通过x中的id在embedding中查找对应id的向量
embedding对应的shape就是(features_num, embbeding_size) :embedding_size对应的是字向量的维度
x在经过embedding_lookup之后输出的就是( batch_size, seq_length, embbeding_size)
如果我们把embedding表看成是一个由{w11,w12,w23...w1n
                                w21,w22,w23...w2n,
                                .
                                .
                                .
                                wt1,wt2,wt3...wtn
                               }组成的参数表，那x embedding_lookup之后实际上是得到一堆可以待训练的参数。
所以embedding本质上是一个全连接层

"""
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
#例2
import tensorflow as tf
import numpy as np
input_ids = tf.placeholder(tf.int32, shape=None)
embedding = tf.get_variable('emb1', [5,5])

input_embedding = tf.nn.embedding_lookup(embedding, input_ids)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    input_embedding = sess.run(input_embedding, feed_dict={input_ids:[0,1,2,3]})
    embedding = sess.run(embedding)
    print("input_embedding:\n",input_embedding)
    print("embedding:\n", embedding)
    
"""
input_embedding:
 [
 [-0.02526045  0.01002508  0.13484949 -0.29325667 -0.74569196]
 [ 0.5865232   0.635286    0.19724154 -0.7158668  -0.6938371 ]
 [ 0.56464374  0.21008295  0.5433301  -0.5832288  -0.6354758 ]
 [ 0.66462064 -0.690945    0.06116873 -0.36609322  0.25676024]
 ]
embedding:
 [
 [-0.02526045  0.01002508  0.13484949 -0.29325667 -0.74569196]
 [ 0.5865232   0.635286    0.19724154 -0.7158668  -0.6938371 ]
 [ 0.56464374  0.21008295  0.5433301  -0.5832288  -0.6354758 ]
 [ 0.66462064 -0.690945    0.06116873 -0.36609322  0.25676024]
 [-0.4204422   0.17029124  0.47239923 -0.7612795  -0.22816002]
 ]   
"""