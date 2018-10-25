"""
namedtuple() ʹ�ù����������������������ֶε�Ԫ�������	 
deque	     �����б���������ܹ�������Ӧ���κ�һ�˽���pop	 
Counter	     �ֵ����࣬Ϊ���Խ��й�ϣ�Ķ������	 
OrderedDict	 �ֵ����࣬��¼���ֵ����Ӵ���	 
defaultdict	 �ֵ����࣬����һ�������������ṩȱʧ��ֵ
"""
#encoding:utf8
from collections import Counter
cnt = Counter()
for word in ['red','blue','red','green','blue','blue']:
    cnt[word]+=1
print(cnt)
#Counter({'blue': 3, 'red': 2, 'green': 1})