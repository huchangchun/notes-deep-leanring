"""
namedtuple() 使用工厂方法创建带有命名的字段的元组的子类	 
deque	     类似列表的容器，能够快速响应在任何一端进行pop	 
Counter	     字典子类，为可以进行哈希的对象计数	 
OrderedDict	 字典子类，记录了字典的添加次序	 
defaultdict	 字典子类，调用一个工厂方法来提供缺失的值
"""
#encoding:utf8
from collections import Counter
cnt = Counter()
for word in ['red','blue','red','green','blue','blue']:
    cnt[word]+=1
print(cnt)
#Counter({'blue': 3, 'red': 2, 'green': 1})