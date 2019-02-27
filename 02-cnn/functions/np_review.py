#encoding=utf-8
import numpy as np
print(np.array([1,2,3,5]))
"""
1.
标准Python的列表(list)中，元素本质是对象。如：L = [1, 2, 3]，需要3个指针和三个整数对象，对于数值运算比较浪费内存和CPU。
因此，Numpy提供了ndarray(N-dimensional array object)对象：存储单一数据类型的多维数组。
"""

"""
2.如何强制生成一个float类型的数组
"""
d = np.array([[1,2,3,4],[5,6,7,8]],dtype=np.float)
print(d)

"""
3.使用astype(int)对上述array进行类型转换
"""
e = d.astype(int)
print(e)
"""
4.dtype和type的区别：
type(d) 和 d.dtype 一个返回的是d 的数据类型 nd.array 另一个返回的是数组中内容的数据类型 
"""

"""
5.arange
"""
print(np.arange(0,10)) #[0,10) 左闭右开区间范围[0 1 2 3 4 5 6 7 8 9]
print(np.arange(0,10,dtype=float)) #[0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
"""
6.arange和python的range区别
arange可以生成float,range只能生成整数
"""

"""
7.reshape
"""
print(np.arange(1,10).reshape((3,3)))

"""
8.构造等差数列
start,end,totalNumber
"""
print(np.linspace(1,10,10)) #[ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
print(np.linspace(1,10,10,endpoint=False)) #[1.  1.9 2.8 3.7 4.6 5.5 6.4 7.3 8.2 9.1]
"""
9.构造等比数列
"""
print(np.logspace(1,4,4,base=2,endpoint=True)) #[ 2.  4.  8. 16.]

"""
10.随机生成
"""
print(np.random.randint(1,5,size=(4,4))) #生成二维数组

"""
11.切片 np.array
规则：[start:end:step]
"""
a =np.arange(1,10)
print(a) #[1 2 3 4 5 6 7 8 9]
print(a[-1]) #开始为-1
print(a[2:-1]) #结尾为-1，开始必须大于等于0的数 #[3 4 5 6 7 8]
print(a[::-1]) #步长-1 #[9 8 7 6 5 4 3 2 1]反转
print(a[::-2]) #[9 7 5 3 1]
print(a[::-3]) #[9 6 3]

"""
12.特殊矩阵生成
"""
print(np.ones((3,3)))
print(np.zeros((3,3)))
print(np.diag([1,2,3]))
"""
[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]
[[1 0 0]
 [0 2 0]
 [0 0 3]]
"""

"""
13.运算
np数组相加，可不不用行列一致
"""
a = np.array([1,2,3,34])
b = a.reshape(-1,1)
print(b)
print(a+b)
"""
[[ 1]
 [ 2]
 [ 3]
 [34]]
[[ 2  3  4 35]
 [ 3  4  5 36]
 [ 4  5  6 37]
 [35 36 37 68]]
"""
"""
矩阵加法比须行列相同
"""
a = np.matrix(np.array([[1,2,3],[2,1,3]]))

b = a.T
print(a)
print(b)
#print(a+b) error
 
print("multiply:")
print(np.multiply(2,4))
x1 = np.arange(9.0).reshape((3,3))
x2 = np.arange(3.0)
print(x1)
print(x2)
print(np.multiply(x1,x2)) #广播
print("-----")
import tensorflow as tf
#矩阵相乘，[m,n]*[n,k] = [m,k]
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2]) 
c = tf.matmul(a, b)
with tf.Session() as sess:
    print(sess.run([a,b,c]))
"""
[array([[1, 2, 3],
       [4, 5, 6]]),
array([[ 7,  8],
       [ 9, 10],
       [11, 12]]), 
array([[ 58,  64],
       [139, 154]])]

"""