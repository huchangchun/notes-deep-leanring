#coding:utf-8
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
constant琛ㄧず甯搁噺锛屽垱寤哄悗涓嶈兘鏀瑰彉
鏈夋椂鎴戜滑闇��浠庡�閮ㄨ緭鍏ユ暟鎹�紝鍙�互鐢╰f.placeholder()
鍒涘缓鍗犱綅Tensor,Tensor鐨勫�鍙�互鍦ㄨ繍琛岀殑鏃跺�杈撳叆
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
y=W脳x+by=W脳x+b  涓�紝杈撳叆xx鍙�互鐢ㄥ崰浣�Tensor 琛ㄧず锛岃緭鍑簓y鍙�互鐢ㄧ嚎鎬фā鍨嬬殑杈撳嚭琛ㄧず锛�
鎴戜滑闇��涓嶆柇鐨勬敼鍙榃W鍜宐b鐨勫�锛屾潵鎵惧埌涓�釜浣縧ossloss鏈�皬鐨勫�銆�
杩欓噷WW鍜宐b鍙�互鐢ㄥ彉閲�Tensor 琛ㄧず銆備娇鐢╰f.Variable()鍙�互鍒涘缓涓�釜鍙橀噺 Tensor锛屽�涓嬪氨鏄�垜浠�ā鍨嬬殑瀹炵幇浠ｇ爜锛�

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