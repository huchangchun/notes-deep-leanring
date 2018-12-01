#coding:utf-8
import tensorflow as tf
"""
#reduce表示降维1/0、分别表示横向纵向降维，不传则横纵都降维，_fun表示降维的方式，求和或求均值等
tf.reduce_max函数的作用：计算张量的各个维度上的元素的最大值
tf.sequence_mask函数的作用：构建序列长度的mask标志
"""
max_value = tf.reduce_max([1,3,2]) 
mask = tf.sequence_mask([1,3,2],max_value)
with tf.Session() as sess:
	max_value = sess.run(max_value)
	mask = sess.run(mask)
	
	print(max_value)
	print(mask)
	
"""     
3
[[ True False False]
 [ True  True  True]
 [ True  True False]]


"""