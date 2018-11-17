#coding:utf-8

import tensorflow as tf

"""
1.在tf.name_scope下时
tf.get_variable()创建的变量名不受name_scope的影响，
而在未指定共享变量时，如果重名就会报错
tf.Variable() :有自动检测变量名的机制,在同一个name_scope下，创建变量如存在相同的变量，会自动引入别名
"""
#with tf.name_scope('name_scope_x'):
	#var1 = tf.get_variable(name='var1',shape=[1],dtype=tf.float32)
	#var3 = tf.Variable(name='var2',initial_value=[2],dtype=tf.float32)
	#var4 = tf.Variable(name='var2',initial_value=[2],dtype=tf.float32)

#with tf.Session() as sess:
	#sess.run(tf.global_variables_initializer())
	#print(var1.name,sess.run(var1))
	#print(var3.name,sess.run(var3))
	#print(var4.name,sess.run(var4))
"""
var1:0 [0.60180557]
name_scope_x/var2:0 [2.]  
name_scope_x/var2_1:0 [2.]
可以看出var3/var4分别变成var2:0 var2_1:0

"""


#---------------------------------------

"""
2.使用tf.get_variable()创建变量，且没有设置共享变量，重名会报错
"""
#with tf.name_scope('name_scope_1'):
	#var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
	#var2 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
#with tf.Session() as sess:
	#sess.run(tf.global_variables_initializer())
	#print(var1.name, sess.run(var1))
	#print(var2.name, sess.run(var2))
	
"""
 Variable var1 already exists, disallowed. 
 Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? 	

"""

"""
3.共享变量,使用tf.get_variable()
共享变量方式1：通过reuse_variables()函数显示说明该变量作用域中的变量可以共享变量
"""
with tf.variable_scope('variable_scope_y') as scope:
	var1 = tf.get_variable(name='var1',shape=[1],dtype=tf.float32)
	scope.reuse_variables() #设置共享变量
	
	var1_reuse = tf.get_variable(name='var1')
	var2 = tf.Variable(initial_value=[2], name='var2',dtype=tf.float32)
	var2_reuse = tf.Variable(initial_value=[2],name='var2',dtype=tf.float32)
	
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(var1.name,sess.run(var1))
	print(var1_reuse.name,sess.run(var1_reuse))
	print(var2.name,sess.run(var2))
	print(var2_reuse.name,sess.run(var2_reuse))
"""
variable_scope_y/var1:0 [-0.6294124]
variable_scope_y/var1:0 [-0.6294124]
variable_scope_y/var2:0 [2.]
variable_scope_y/var2_1:0 [2.]
"""
"""
共享变量方式2：
"""

with tf.variable_scope('variable_scope_2') as scope:
	v = tf.get_variable(name='v',shape=[1],dtype=tf.float32)

with tf.variable_scope('variable_scope_2' ,reuse=True):
	
	v1 = tf.get_variable(name='v')

assert v == v1
	
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(v.name,sess.run(v))
	print(v1.name,sess.run(v1))
"""
variable_scope_2/v:0 [1.3348032]
variable_scope_2/v:0 [1.3348032]
"""

