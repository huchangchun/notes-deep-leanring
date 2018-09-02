# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 07:45:27 2018

@author: hcc
"""
#coding=utf8
import numpy as np
improt tensorflow as tf
#data settings
num_examples = 10
num_words=20
num_features = 100
num_tags =5

#5 tags
# x shape=[10,20,100]
#random features.
x = np.random.rand(num_examples, num_words, num_features).astype(np.float32)

#y  shape=[10,20]
#random tag indices representing the gold sequence
y = np.random.randint(num_tags, size=[num_examples, num_words]).astype(np.iint32)

#序列的长度