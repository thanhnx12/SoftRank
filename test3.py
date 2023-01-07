import DataProcess
from scipy.stats import norm
import math
import torch
import numpy as np

q = DataProcess.getQuery("D:\\data.txt",10)
lst = []
for x in q[0].documents :
    lst.append(x.label/4)
q[0].score = lst
PI = q[0].claculate_pi()
P = q[0].prob_all(PI)
print(q[0].G_derivative_by_score(PI,P))
print(PI)
for i in range(0,10):
    for j in range(0,10):
        print(PI[i][j] + PI[j][i],end=' ')
    print('\n')