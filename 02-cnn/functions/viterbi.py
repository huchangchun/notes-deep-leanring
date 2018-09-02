#-*- coding:utf-8 -*-
#Filename:viterbi.py
#Author:hcc
#Data:2018-09-02

#Day 0:计算Sunny、Rainy的概率
#观察到的是walk，
#V[0][Sunny] = start_p[Sunny]*emit_p[Sunny][walk]
#V[0][Rainy] = start_p[Rainy]*emit_p[Rainy][walk]
#path[y]=[y]
# 

#Day 1:计算Sunny，Rainy的概率
#观察到的是shop
#newpath = {}
#V[1][Sunny] = V[0][Sunny] *tran_p[V[0][Sunny]][Sunny] *emit_p[Sunny][shop]
#V[1][Sunny] = V[0][Sunny] *tran_p[V[0][Sunny]][Rainy] *emit_p[Sunny][shop]
#取最大的得到Pro，PreState ,curState,更新newpath[y] = path[PreState]+[curState]
#V[1][Rainy] = V[0][Rainy] *tran_p[V[0][Rainy]][Sunny] *emit_p[Rainy][shop]
#V[1][Rainy] = V[0][Rainy] *tran_p[V[0][Rainy]][Rainy] *emit_p[Rainy][shop]
#取最大的得到Pro，PreState ,curState,更新newpath[y] = path[PreState]+[curState]
#path = newPath

#Day 2:计算Sunny、Rainy的概率
#观察到的是clean
#V[2][Sunny] = V[1][Sunny]* tran_p[V[1][Sunny]][Sunny] *emit_p[Sunny][clean]
#V[2][Sunny] = V[1][Sunny]* tran_p[V[1][Sunny]][Rainy] *emit_p[Sunny][clean]
#取最大的得到Pro，PreState ,curState,更新newpath[y] = path[PreState]+[curState]
#V[2][Rainy] = V[1][Rainy]* tran_p[V[1][Rainy]][Sunny] *emit_p[Rainy][clean]
#V[2][Rainy] = V[1][Rainy]* tran_p[V[1][Rainy]][Rainy] *emit_p[Rainy][clean]
#path = newPath
#最后一个取概率最大pro对应的state，序列即path[state]
 
def print_dptable(V):
    
    for i in range(len(V)): print("\t%12d" % i,end='')
    print()
    for y in V[0].keys():
        print()
        print("%7s: " % y,end='')
        for t in range(len(V)):
            print ("\t%.12s\t" % ("%f" % V[t][y]),end='')
        print()
    print("\n")
def viterbi(obs,states,start_p,trans_p,emit_p):
    
    V = [{}]
    path = {}
    #初始化初始状态t=0
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]
    #t>0时刻之后   
    for t in range(1,len(obs)):
        V.append({})
        newPath ={}
        for y1 in states:
            #隐状态 = 前状态是y0的概率 * y0转移到y的概率 * y表现为当前状态的概率
            (MaxPro,PreState) = max([(V[t-1][y0]*trans_p[y0][y1] * emit_p[y1][obs[t]],y0) for y0 in states])
            V[t][y1] = MaxPro
            newPath[y1] = path[PreState] + [y1] #从前向后的路径都不断的更新，分两条线更新：分别是初始状态为Rainy和Sunny，朝着概率最大的方向
        path = newPath
    
    (prob,state) = max([(V[len(obs)-1][y],y) for y in states])
    print_dptable(V)
    return (prob,path[state])
def main():
    states = ('Rainy','Sunny')
    obs = ('walk','shop','clean')
    start_p = {'Rainy':0.6,'Sunny':0.4}
    trans_p = {
        
        'Rainy':{'Rainy':0.7,'Sunny':0.3},
        'Sunny':{'Rainy':0.4,'Sunny':0.6},
    }
    emit_p = {
        
        'Rainy':{'walk':0.1,'shop':0.4,'clean':0.5},
        'Sunny':{'walk':0.6,'shop':0.3,'clean':0.1},
    }    
    return viterbi(obs, states, start_p, trans_p, emit_p)
if __name__ =="__main__":
    pro,seq = main()
    print("Most Prob Seq:\n")
    print(pro,seq)