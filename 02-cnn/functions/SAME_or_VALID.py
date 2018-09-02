# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 21:46:12 2018

@author: hcc
 
 If padding == "SAME": 
      output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])

    If padding == "VALID":
      output_spatial_shape[i] =
        ceil((input_spatial_shape[i] -
              (spatial_filter_shape[i]-1) * dilation_rate[i])
              / strides[i]).
dilation_rate为一个可选的参数，默认为1，这里我们可以先不管它。 
SAME
 out_height = cell(float(in_height)/float(stride[1]))
 out_width = cell(float(in_width)/float(strdes[2]))
VALID
out_height = cell(float(in_height-filter_height+1)/float(strides1))
out_width = cell(float(in_width -filter_width +1)/float(strides2))

整理一下，对于“VALID”，输出的形状计算如下： 
new_height=new_width=⌈(W–F+1)/S⌉

对于“SAME”，
能保持输出的高宽和输入的一直，比如：输入[2,2],filter[3,3]，则
w + p -f + 1 = w
p = 1 + f = f-1=2,所以需要padding两个，这时卷积后刚好还是[2,2]
输出的形状计算如下： 
new_height=new_width=⌈W/S⌉
其中，W为输入的size，F为filter为size，S为步长，⌈⌉为向上取整符号。

"""
import tensorflow as tf
x_image = tf.Variable(tf.random_normal([1,28,28,1]))
kernel = tf.Variable(tf.random_normal([5,5,1,1]))
#conv = tf.nn.conv2d(x_image,kernel,strides = [1,1,1,1],padding='SAME')#(1, 28, 28, 1)
conv = tf.nn.conv2d(x_image,kernel,strides = [1,1,1,1],padding='VALID')#(1, 24, 24, 1) (28-5+1)/1
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = (sess.run(conv))
    print(result.shape)
    
x = tf.reshape(tf.range(24),[1,3,4,2])
print(x)
out = tf.transpose(x,[0,3,1,2])
print(x)
print(x.shape)
print(out.shape)

 