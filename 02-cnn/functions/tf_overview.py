import tensorflow as tf
##two params
#a = tf.constant(3.2)
#b = tf.constant(4.8)
#c = a + b
#print(a)
#print(b)
#print(c)
#sess = tf.Session()
#print(sess.run(c))


"""
Tensor("Const:0", shape=(), dtype=float32)
Tensor("Const_1:0", shape=(), dtype=float32)
Tensor("add:0", shape=(), dtype=float32)
8.0
"""


#-----------------------------------------
"""
constant表示常量，创建后不能改变
有时我们需要从外部输入数据，可以用tf.placeholder()
创建占位Tensor,Tensor的值可以在运行的时候输入
"""
#a = tf.placeholder(tf.float32)
#b = tf.placeholder(tf.float32)
#c = a + b
#print(a) 
#print(b)
#print(c)
#sess = tf.Session()
#print(sess.run(c,{a:3,b:4.5}))
#print(sess.run(c,{a:[1,3],b:[2,4]}))
"""
Tensor("Placeholder:0", dtype=float32)
Tensor("Placeholder_1:0", dtype=float32)
Tensor("add_1:0", dtype=float32)
7.5
[3. 7.]
"""

"""
y=W×x+by=W×x+b  中，输入xx可以用占位 Tensor 表示，输出yy可以用线性模型的输出表示，
我们需要不断的改变WW和bb的值，来找到一个使lossloss最小的值。
这里WW和bb可以用变量 Tensor 表示。使用tf.Variable()可以创建一个变量 Tensor，如下就是我们模型的实现代码：

"""
W = tf.Variable([1], dtype = tf.float32)
b = tf.Variable([-1], dtype = tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
loss = tf.reduce_sum(tf.square(linear_model - y))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(W))
print(sess.run(linear_model,{x:[1,2,3,4,6,7]}))
print(sess.run(loss, {x: [1, 2, 3, 6, 8], y: [4.8, 8.5, 10.4, 21.0, 25.3]}))
"""
[1.]
[0. 1. 2. 3. 5. 6.]

"""