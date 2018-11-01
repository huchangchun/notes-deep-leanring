#-*- coding: cp936 -*-
"""
除空格外，全角/半角按unicode编码排序在顺序上是对应的（半角 + 0x7e= 全角）,所以可以直接通过用+-法来处理非空格数据，对空格单独处理
1. 中文文字永远是全角，只有英文字母、数字键、符号键才有全角半角的概念,一个字母或数字占一个汉字的位置叫全角，占半个汉字的位置叫半角。

2. 引号在中英文、全半角情况下是不同的
"""
 
def strFullToHalf(string):
    """全角转半角"""
    result =""
    for uchar in string:
        inside_code = ord(uchar)
        if inside_code == 0x3000:#12288 全角空格直接转换   
            inside_code = 32
        elif 0xFF01 <=inside_code <=0xFF5E:#全角字符（除空格）根据关系转化
            inside_code -= 0xfee0
        result += chr(inside_code)
    return result

def strHalfToFull(string):
    """半角转全角"""
    result = ""
    for uchar in string:
        inside_code = ord(uchar)
        if inside_code == 32:
            inside_code = 0x3000
        elif 32<=inside_code<=126:#半角字符（除空格）根据关系转化
            inside_code += 0xfee0
        result +=chr(inside_code)
    return result
b = strFullToHalf("ｍｎfas测试")
print(b)
#mnfas测试
c = strHalfToFull("mnfas测试")
print(c)
#ｍｎｆａｓ测试