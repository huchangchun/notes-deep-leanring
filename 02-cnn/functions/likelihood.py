# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 07:45:27 2018
link:https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/crf
@author: hcc
"""
 
import numpy as np
import tensorflow as tf
num_examples = 10
num_words = 20
num_features = 300 #100->200->300准确率都有提高
num_tags = 5
x = np.random.rand(num_examples,num_words,num_features).astype(np.float32)
y = np.random.randint(num_tags, size=[num_examples, num_words]).astype(np.int32)

sequence_lengths = np.full(shape = num_examples, fill_value=num_words-1,dtype=np.int32)

#Train and evaluate the model
with tf.Graph().as_default():
    with tf.Session() as session:
        x_t = tf.constant(x)
        y_t = tf.constant(y)
        sequence_lengths_t = tf.constant(sequence_lengths)
        
        #设置一个无偏置的线性层
        """
        weights:       (100,5)
        matricized_x_t (200,100)
        matricized_unary_scores:xw:(200,5)
        unary_scores:  reshape (10,20,5)
        """
        weigths = tf.get_variable(name="weights",
                                  shape=[num_features,num_tags])
        matricized_x_t = tf.reshape(tensor=x_t, shape=[-1,num_features])
        
        matricized_unary_scores = tf.matmul(a=matricized_x_t, b=weigths)
        
        unary_scores = tf.reshape(tensor=matricized_unary_scores,
                                  shape=[num_examples,num_words,num_tags])
        #crf_log_likelihood:
        """
        return:
        log_likelihood: A[batch_size] ‘Tensor’ containing the log_likelihood of each example
        given the sequence of tag indices
        transition_params: A[num_tags,num_tags] transition matrix,This is either provided by caller or created in this function    
        """
        log_likelihood,transition_params = tf.contrib.crf.crf_log_likelihood(
            inputs=unary_scores, 
            tag_indices=y_t, 
            sequence_lengths=sequence_lengths_t) #sequence_lengths_t:(batch_size)
        #通过条件随机场取到最大似然和转移概率矩阵
        #通过viterbi解码得到最可能隐藏序列和分数 viterbi_sequence shape=(10, 20)
        viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
            potentials=unary_scores,
            transition_params = transition_params,
            sequence_length = sequence_lengths_t)
        loss = tf.reduce_mean(input_tensor=-log_likelihood)
        train_op = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss)
        session.run(tf.global_variables_initializer()) #mask shape(10,20) (batch_size,num_words)
        mask = (np.expand_dims(np.arange(num_words), axis = 0)< np.expand_dims(sequence_lengths, axis=1))
        total_labels = np.sum(a=sequence_lengths) #190
        
        #Train
        for i in range(1001):
            tf_viterbi_sequence, _ = session.run(fetches =[viterbi_sequence,train_op])
            if i % 100 == 0:
                correct_labels = np.sum((y==tf_viterbi_sequence)*mask)
                accuracy = 100.0 * correct_labels /float(total_labels)
                print("i: %d,accuracy: %.2f%%" %(i,accuracy))
            if i == 1000:
                print("Finnished")