#-*- coding: cp936 -*-
"""
���ո��⣬ȫ��/��ǰ�unicode����������˳�����Ƕ�Ӧ�ģ���� + 0x7e= ȫ�ǣ�,���Կ���ֱ��ͨ����+-��������ǿո����ݣ��Կո񵥶�����
1. ����������Զ��ȫ�ǣ�ֻ��Ӣ����ĸ�����ּ������ż�����ȫ�ǰ�ǵĸ���,һ����ĸ������ռһ�����ֵ�λ�ý�ȫ�ǣ�ռ������ֵ�λ�ýа�ǡ�

2. ��������Ӣ�ġ�ȫ���������ǲ�ͬ��
"""
 
def strFullToHalf(string):
    """ȫ��ת���"""
    result =""
    for uchar in string:
        inside_code = ord(uchar)
        if inside_code == 0x3000:#12288 ȫ�ǿո�ֱ��ת��   
            inside_code = 32
        elif 0xFF01 <=inside_code <=0xFF5E:#ȫ���ַ������ո񣩸��ݹ�ϵת��
            inside_code -= 0xfee0
        result += chr(inside_code)
    return result

def strHalfToFull(string):
    """���תȫ��"""
    result = ""
    for uchar in string:
        inside_code = ord(uchar)
        if inside_code == 32:
            inside_code = 0x3000
        elif 32<=inside_code<=126:#����ַ������ո񣩸��ݹ�ϵת��
            inside_code += 0xfee0
        result +=chr(inside_code)
    return result
b = strFullToHalf("���fas����")
print(b)
#mnfas����
c = strHalfToFull("mnfas����")
print(c)
#���������