#encoding=utf-8
import numpy as np 
import tensorflow as tf
#x = np.array([[1.,2.,3.],[4.,5.,6.]]) 注意np和constant得出的结果不一样
x = tf.constant([[1.,2.,3.], [4.,5.,6.]])
sess = tf.Session()
mean_none = sess.run(tf.reduce_mean(x))
mean_0 = sess.run(tf.reduce_mean(x,0)) #axis = 0
mean_1 = sess.run(tf.reduce_mean(x,1)) #axis = 1
print(x)#Tensor("Const:0", shape=(2, 3), dtype=float32)
print(mean_none)#3.5  #取所有维度的平均
print(mean_0)#[2.5 3.5 4.5]#取列的平均
print(mean_1)#[2. 5.] #取每行的平均
sess.close()