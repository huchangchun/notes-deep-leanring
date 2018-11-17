#coding:utf-8
import tensorflow as tf

"""
tf.name_scope()主要用于管理一个图里的各种操作(op)返回的是一个以scope_name命名的context manager，
一个graph会维护一个name_space的堆，每一个namespace下面可以定义各种op，实现层次化的管理，避免op之间冲突
tf.variable_scope() 一般会维护一个name_scope配合使用，用于管理一个graph中变量的名字，避免变量
之间的冲突，允许在一个variable_scope下面共享变量，variable_scope默认reuse=False

note:name_scope只能管住ops的名字，对Variales是没有约束的
"""


with tf.variable_scope('foo') :
	with tf.name_scope('bar'):
		v = tf.get_variable('v',[1])
		x = 1 + v
assert v.name == "foo/v:0"
assert x.op.name == "foo/bar/add"

with tf.variable_scope("foo",reuse=True):#对Variales是没有约束的
	v_reuse = tf.get_variable(name='v')
	v_reuse2 = tf.Variable(initial_value=[2],name='v',dtype=tf.int32)
	
"""这会报错"""
#with tf.variable_scope("tom", reuse=True):
	#v_tom = tf.get_variable(name='v')

assert v.name==v_reuse.name
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(v.name,sess.run(v))
	print(x.name,sess.run(x))
	print(v_reuse.name,sess.run(v_reuse))
	print(v_reuse2.name,sess.run(v_reuse2))
"""
foo/v:0 [0.50807774]
foo/bar/add:0 [1.5080777]
foo/v:0 [0.50807774]
foo_1/v:0 [2]

"""