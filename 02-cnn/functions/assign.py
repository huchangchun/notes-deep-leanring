#-*-coding:UTF-8-*-
#这句话是指定*.py的编码方式，如果文件中涉及到中文汉字的话，有必要写一下这句话。当然也可以这样写：encoding:UTF-8
import tensorflow as tf

state=tf.Variable(0,name="a")#使用tensorflow在默认的图中创建节点，这个节点是一个变量
one = tf.constant(1)#此处调用了tf的一个函数，用于创建常量
new_value = tf.add(state,one)#对常量与变量进行简单的加法操作，
#这点需要说明的是： 在TensoorFlow中，所有的操作op，变量都视为节点，tf.add() 
#的意思就是在tf的默认图中添加一个op，这个op是用来做加法操作的。

#update = tf.assign(state,new_value)#这个操作是：赋值操作。将new_value的值赋值给state变量,update只是一个用于sess
#的变量

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)#对变量进行初始化，执行（run）init语句
    for i in range(3):
        #sess.run(state.assign(new_value))
        sess.run(update)
        print(sess.run(state))