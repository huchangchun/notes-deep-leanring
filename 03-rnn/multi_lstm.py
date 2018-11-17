#-*-coding:utf-8 -*-
"""
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data 

#set GPU increase on demand
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config = config)

mnist = input_data.read_data_sets('.\\data\\', one_hot=True)
batch_size = tf.placeholder(tf.int32,[])

lr = 1e-3
input_size = 28
timestep_size = 28

# nodes per hidden_layer
hidder_size = 256
#LSTM layers
layer_num = 2
#num of classes to predict
class_num = 10

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32,[None, class_num])
keep_prob = tf.placeholder(tf.float32)
 
#step1: shape =(batch_size,timestep_size,input_size)
input_data= tf.reshape(x, [-1,28,28])

#step2: define lstm cell 
lstm_cell = rnn.BasicLSTMCell(num_units = hidder_size)
#step3:set dropout
lstm_cell = rnn.DropoutWrapper(cell=lstm_cell,input_keep_prob=1.0,output_keep_prob=keep_prob)
#step4:multi layers 
multi_lstm_cell = rnn.MultiRNNCell([lstm_cell] * layer_num ,state_is_tuple=True)
#step5:init state with zeros
init_state = multi_lstm_cell.zero_state(batch_size,dtype=tf.float32)

#step6:通过dynamic_rnn()将网络运行起来得到输出
outputs, state = tf.nn.dynamic_rnn(cell=multi_lstm_cell,inputs=input_data,initial_state=init_state,time_major=False)
h_state = output[:,-1,:] #or h_state = state[-1][1]

## step6 also can be this
#outputs = list()
#state = init_state
#with tf.variable_scope("RNN"):
    #for timestep in range(timestep_size):
        #if timestep_size > 0:
            #tf.get_variable_scope().reuse
        #(cell_output,state) = multi_lstm_cell(input_data[:, timestep, :],state)
        #outputs.append(cell_output)
#h_state = outputs[-1]

# LSTM 部分的输出会是一个 [hidden_size] 的tensor，我们要分类的话，还需要接一个 softmax 层
out_W =tf.Variable(tf.truncated_normal(shape=[hidden_size,class_num],stddev=0.1),dtype=tf.float32)
out_B = tf.Variable(tf.constant(0.1,shape=[class_num,]),dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state,out_W) + bias) #matmul:矩阵相乘
cross_entropy = -tf.reduce_mean(y *tf.log(y_pre))
train_op =tf.train.AdamOptimizer(lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

sess.run(tf.global_variables_initializer())
for i in range(2000):
    _batch_size = 128
    batch = mnist.train.next_batch(_batch_size)
    if (i+1)%200 == 0:
        train_accuracy = sess.run(accuracy,feed_dict={x:batch[0],
                                                      y:batch[1],
                                                      keep_prob:1.0,
                                                      batch_size:_batch_size
                                                      }
                                  )
    print("Iter {0},Step:{1} ,train_accuracy:".format(mnist.train.epochs_completed,(i+1),train_accuracy))
    
    sess.run(train_op,feed_dict={x:batch[0],y:batch[1],keep_prob:0.5,batch_size:_batch_size})
print("test accuracy :{0}".format(sess.run(accuracy,feed_dict={x:mnist.test.images,
                                                               y:mnist.test.labels,
                                                               keep_prob:1.0,
                                                               batch_size:mnist.test.images.shape[0]
                                                               }
                                           )
                                  )
      )
      """
#encoding=utf8
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
sess = tf.Session(config=config)
mnist = input_data.read_data_sets('.\\data\\', one_hot=True)
#http://m.blog.csdn.net/jerr__y/article/details/61195257
batch_size = tf.placeholder(tf.int32,[])
lr=0.001
hidden_unit=256
predict_class=10
input_dim=28
input_step=28
layer_num=2
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,predict_class])
keep_prob=0.5

input_data=tf.reshape(x,[-1,28,28])
lstm_cell=rnn.BasicLSTMCell(num_units=hidden_unit, forget_bias=1.0, state_is_tuple=True)
lstm_cell=rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)

two_lstm_cell=rnn.MultiRNNCell([lstm_cell]*layer_num, state_is_tuple=True)
init_state=two_lstm_cell.zero_state(batch_size, tf.float32)
#two_lstm_cell.zero_state(batch_size, dtype)
#outputs,state=tf.nn.dynamic_rnn(two_lstm_cell,inputs=input_data,initial_state=init_state,time_major=False)
outputs=list()
state=init_state
with tf.variable_scope('RNN'):
    for ii in range(input_step):
        if ii >0:
            tf.get_variable_scope().reuse_variables()
            cell_output,state=two_lstm_cell(input_data[:,ii,:],state)
            outputs.append(cell_output)
h_state=outputs[-1]


wout=tf.Variable(tf.truncated_normal([hidden_unit,predict_class], mean=0.0, stddev=1.0))
biase_out=tf.Variable(tf.constant(0.1,shape=[predict_class,]),tf.float32)
predict_out=tf.nn.softmax(tf.matmul(h_state,wout)+biase_out)
loss=y*tf.log(predict_out)
cross_entropy_loss=-tf.reduce_mean(loss)
train=tf.train.AdamOptimizer(lr).minimize(cross_entropy_loss)
correct=tf.equal(tf.argmax(predict_out,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct,"float"))
sess.run(tf.global_variables_initializer())
for iterate in range(10000):
    _batch_size=128
    print("%d iterate "%(iterate))
    batch=mnist.train.next_batch(_batch_size)
    sess.run(accuracy,feed_dict={x:batch[0],y:batch[1]})
    if iterate % 500 ==0:
        accuracy1=sess.run(accuracy,feed_dict={x:batch[0],y:batch[1],batch_size:_batch_size})
        print("%d iterate ,train accuracy %g"%(iterate+1,accuracy1))
test_accuracy=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
print("test accuracy is %g"%(test_accuracy))