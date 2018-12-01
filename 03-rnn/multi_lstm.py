#-*-coding:utf-8 -*-
 
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data 
import os
#set GPU increase on demand
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config = config)

mnist = input_data.read_data_sets('.\\data\\', one_hot=True)
batch_size = tf.placeholder(tf.int32,[])

lr = 1e-3 #0.001
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

def lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(num_units=hidder_size,state_is_tuple=True)
def dropout():
    cell = lstm_cell()
    return tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=keep_prob)
def multi_lstm():
    cells = [dropout() for _ in range(layer_num)]
    MultiRNN_cell = tf.contrib.rnn.MultiRNNCell(cells,state_is_tuple=True)
    return MultiRNN_cell

two_lstm_cell = multi_lstm()
init_state=two_lstm_cell.zero_state(batch_size, tf.float32)

#outputs,state=tf.nn.dynamic_rnn(two_lstm_cell,inputs=input_data,initial_state=init_state,time_major=False)
outputs = list()
state = init_state
with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        # 这里的state保存了每一层 LSTM 的状态
        (cell_output, state) = two_lstm_cell(input_data[:, timestep, :], state)
        outputs.append(cell_output)
h_state = outputs[-1] # h_state: 只取了outputs的最后一状态 <tf.Tensor 'RNN/RNN/multi_rnn_cell/cell_1_27/dropout/mul:0' shape=(?, 256) dtype=float32>]
#lstm模型最终输出是最后一个时序的隐层维度，因此是256维，
# LSTM 部分的输出会是一个 [hidden_size] 的tensor，我们要分类的话，还需要接一个 softmax 层
out_W =tf.Variable(tf.truncated_normal(shape=[hidder_size,class_num],stddev=0.1),dtype=tf.float32)
out_B = tf.Variable(tf.constant(0.1,shape=[class_num,]),dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state,out_W) + out_B) #matmul:矩阵相乘
cross_entropy = -tf.reduce_mean(y *tf.log(y_pre)) #reduce表示降维1/0、分别表示横向纵向降维，不传则横纵都降维，_fun表示降维的方式，求和或求均值等
train_op =tf.train.AdamOptimizer(lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(y,1))#argmax取vector中最大值的索引号
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

sess.run(tf.global_variables_initializer())
tensorboard_dir = "./logs"
if not os.path.exists(tensorboard_dir):
    os.mkdir(tensorboard_dir)
writer = tf.summary.FileWriter(tensorboard_dir,sess.graph) 
tf.summary.scalar(name="cross_entropy", tensor = cross_entropy)
tf.summary.scalar(name="accuracy",tensor=accuracy)
merged_summary = tf.summary.merge_all()
 
save_per_step = 100
for i in range(2000):
    _batch_size = 128
    batch = mnist.train.next_batch(_batch_size)
    if (i+1) % save_per_step == 0:
        train_summary = sess.run(merged_summary,feed_dict={x:batch[0],
                                                      y:batch[1],
                                                      keep_prob:1.0,
                                                      batch_size:_batch_size
                                                      }
                                  )
        writer.add_summary(train_summary,(i+1))
    if (i+1)%200 == 0:
        train_accuracy = sess.run(accuracy,feed_dict={x:batch[0],
                                                      y:batch[1],
                                                      keep_prob:1.0,
                                                      batch_size:_batch_size
                                                      }
                                  )
        print("Iter {0},Step:{1} ,train_accuracy:{2}".format(mnist.train.epochs_completed,(i+1),train_accuracy))
    
    sess.run(train_op,feed_dict={x:batch[0],y:batch[1],keep_prob:0.5,batch_size:_batch_size})
print("test accuracy :{0}".format(sess.run(accuracy,feed_dict={x:mnist.test.images,
                                                               y:mnist.test.labels,
                                                               keep_prob:1.0,
                                                               batch_size:mnist.test.images.shape[0]
                                                               }
                                           )
                                  )
      )