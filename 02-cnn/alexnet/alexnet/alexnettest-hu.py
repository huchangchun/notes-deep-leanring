#encoding=utf8
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
#读取整个输入样本的训练集,测试集及标签文件
mnist = input_data.read_data_sets('data\\',ont_hot=True)
#28*28*1
import tensorflow as tf
#import tensorflow.contrib.eager as tfe
#pip install tf-nightly
import os 
import numpy as np
lr = 0.001
training_iters = 100
batch_size = 16
display_step=1

input_dim = 784
mnist_class = 10 #类别,10中图像类别
dropout = 0.5

x = tf.placeholder(tf.float32, [None,input_dim])
y = tf.placeholder(tf.float32,[None,mnist_class])

drop_prob = tf.placeholder(tf.float32)

def conv2d(name,input_data,input_filter,bias):
    x = tf.nn.conv2d(input_data,input_filter,strides=[1,1,1,1], padding = 'SAME', use_cudnn_on_gpu=False,data_format='NHWC', name=None)
    x = tf.nn.bias_add(x, bias, data_format=None, name=None)
    return tf.nn.relu(x,name=name)
def max_pooling(name,input_data,k):
    #ksize:池化窗口的大小,取一个4维向量,一般是[1,height,width,1]
    #strides:和卷积类似,窗口在每一个维度上滑动的步长,一般也是[1,stride,stride,1]
    return tf.nn.max_pool(input_data, ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME',name=name)
def norm(name,input_data,lsize=4):
    return tf.nn.lrn(input_data,depth_radius=lsize,bias=1,alpha=1,beta=0.5,name=name)

weights={
    #卷积核filter大小11*11输入层为1个feature maps,输出层有64个feature maps
    'wc1': tf.Variable(tf.random_normal([11,11,1,48])),
    'wc2': tf.Variable(tf.random_normal([5,5,48,128])),
    'wc3': tf.Variable(tf.random_normal([3,3,128,192])),
    'wc4': tf.Variable(tf.random_normal([3,3,192,192])),
    'wc5': tf.Variable(tf.random_normal([3,3,192,128])),
    'wd1': tf.Variable(tf.random_normal([4*4*128,4096])),
    'wd2': tf.Variable(tf.random_normal([4096, 4096])),
    'out': tf.Variable(tf.random_normal([4096, 10]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([48])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([192])),
    'bc4': tf.Variable(tf.random_normal([192])),
    'bc5': tf.Variable(tf.random_normal([128])),
    'bd1': tf.Variable(tf.random_normal([4096])),
    'bd2': tf.Variable(tf.random_normal([4096])),
    'out': tf.Variable(tf.random_normal([mnist_class]))    
}

def alexnet(_input_img,_weights, _biases,_dropout):
    _input_img = tf.reshape(_input_img,shape=[-1,28,28,1])
    #第一层
    conv1 = conv2d("conv1_layer", _input_img,_weights['wc1'], _biases['bc1'])
    pooling1 = max_pooling("pooling1", conv1, k=2)
    #第二层
    with tf.device('/cpu'):
        conv2 = conv2d("conv2_layer", norm1,_weights['wc2'], _biases['bc2'])
        pooling2 = max_pooling("pooling2", conv2,k=2)
        norm2=norm("norm2",pooling2,lsize=4)
    #第三层
    conv3 = conv2d("conv3_layer", _input_img,_weights['wc3'], _biases['bc3'])
    norm3=norm("norm3",pooling2,lsize=4)
    
    #第四层
    conv4 = conv2d("conv4_layer", _input_img,_weights['wc4'], _biases['bc4'])
    norm3=norm("norm3",pooling2,lsize=4)
    
    #第五层
    conv5 = conv2d("conv5_layer", _input_img,_weights['wc5'], _biases['bc5'])
    pooling5 = max_pooling("pooling5", conv5,k=2)
    norm5=norm("norm5",pooling5,lsize=4)
    
    #第六层
    dense1 = tf.reshape(norm5, [-1,_weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.matmul(dense1,_weights['wd1']) + _biases['bd1'],name='fc1')
    dense1 = tf.nn.dropout(dense1, keep_prob=_dropout)
    
    #第七层
    dense2 = tf.reshape(dense1, [-1,_weights['wd2'].get_shape().as_list()[0]])
    dense2 = tf.nn.relu(tf.matmul(dense2,_weights['wd2']) + _biases['bd2'],name='fc2')
    dense2 = tf.nn.dropout(dense2, keep_prob=_dropout)
    out = tf.matmul(dense2,_weights['out']) + _biases['out']  #x^T * W + b
    return out

#构建模型
feature = alexnet(x,weights,biases,drop_prob)

gloable_step = tf.constant(0,tf.int64)
decay_rate = tf.constant(0.9,tf.float64)
learn_rate = tf.train.exponential_decay(lr, gloable_step,10000,decay_rate)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= feature, labels=y))

#反向传播
optimizer = tf.train.AdadeltaOptimizer(learning_rate=learn_rate).minimize(cost)
correct_feature = tf.equal(tf.argmax(feature,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_feature, tf,float32))

init = tf.global_variables_initializer()

#开启训练
def train():
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        while step * batch_size < training_iters:
            batch_xs, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_xs, y:batch_ys, drop_prob:dropout})
            if step % display_step == 0:
                acc = sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys,drop_prob:1.})
                #计算损失
                loss = sess.run(cost, feed_dict={x: batch_xs, y:batch_ys, drop_prob:1.})
                print("Iter" + str(step*batch_size)+ ",Minibatch Loss =" + "{:.6f}".format(loss)+ ",Training Accuracy" + "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")
        
        #保存模型
        saver = tf.train.Saver()
        save_path = 'ckpt'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        model_name = save_path + os.sep + "alexnet.ckpt"
        saver.save(sess,model_name)
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256],y:mnist.test.labels[:256], drop_prob: 1.}))
if __name__ =="__main__":
    if os.path.exists("ckpt"):
        count = 0
        ls = os.listdir("ckpt")
        for a in ls:
            count+=1
        if count == 4:
            init = tf.global_variables_initializer()
            restore = tf.train.Saver()
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state("ckpt")
                if ckpt and ckpt.model_checkpoint_path:
                    restore.restore(sess, ckpt.model_checkpoint_path)
                    #制定读取测试文件的索引号
                    img_index=10 #  
                    #把mnist.test.images测试集里面所有样本转化成图片格式，-1代表文件个数，28,28,1代表一个图片文件
                    test_imgs=tf.reshape(mnist.test.images,[-1,28,28,1])#读入测试集所有文件，并且改变图像的形状
                    #取索引号为img_index=10的一张图片
                    a_img=sess.run(test_imgs)[img_index,:,:,0]
                    #取出索引号为img_index的一张图片，img_index:img_index+1表示取一张图片，而且是索引号从16到17的
                    img_input=mnist.test.images[img_index:img_index+1,:]
                    #表示把feature依赖的输入传给feature变量，drop_prob:1.0表示全部保留神经元，不进行dropout操作
                    #feature = alexnet(x, weights, biases, drop_prob)
                    featureict=sess.run(feature,feed_dict={x:img_input,drop_prob:1.0})
                    #tf.argmax表示取featureict变量最大元素的索引号，1表示取每一行最大的索引值
                    result=tf.argmax(featureict,1)
                    
                    a_img=tf.reshape(img_input,[1,28,28])
                    #把预测结果打印出来
                    print('featureiction is:',sess.run(result))                
                    import matplotlib.pyplot as plt
                    import pylab
                    plt.imshow(a_img)      
                    pylab.show()
    
              
                    ###get_ipython().magic('matplotlib inline')
                    print (sess.run(weights['wc1']).shape)
                    f, axarr = plt.subplots(4,figsize=[10,10])
                    axarr[0].imshow(sess.run(weights['wc1'])[:,:,0,0])
                axarr[1].imshow(sess.run(weights['wc2'])[:,:,23,12])
                
                axarr[2].imshow(sess.run(weights['wc3'])[:,:,41,44])
                axarr[3].imshow(sess.run(weights['wc4'])[:,:,45,55])  
                pylab.show()
    else:
        #训练
        train()