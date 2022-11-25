import DataProcess
from scipy.stats import norm
import math
import torch
import numpy as np
LR = 0.01
class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = torch.nn.Linear(136,1)
        self.layer2 = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
model = Model()
model.layer1.weight.requires_grad = True
#readData
q = DataProcess.getQuery("U:\\MSLR-WEB10K\Fold1\\test.txt",10)
#q = DataProcess.getQuery("U:\\data.txt",10)
datasize = len(q)
#print(datasize) #22379
acc = []
for i in range(1000):
    #model predictions
    ds_dw = []
    score = []
    for x in q[i].documents:
        temp = model.forward(x.feature)
        score.append(temp)
        temp.backward()
        gradw = model.layer1.weight.grad
        ds_dw.append(gradw)
        model.layer1.weight.grad.zero_()
    q[i].score = score
    # loss

    # gradient
    PI = q[i].claculate_pi()
    P = q[i].prob_all(PI)
    # print("------------------------")
    # print(P)
    # print("----------------------")
    dG_ds = q[i].G_derivative_by_score(PI,P)
    dG_ds = torch.from_numpy(np.array(dG_ds))
    ds_dw = torch.cat(ds_dw)
    grad = torch.matmul(dG_ds,ds_dw.double())
    #Training
    with torch.no_grad():
        model.layer1.weight += LR*grad
    softndcg = q[i].SoftNDCG(P)
    acc.append(softndcg)

print(model.layer1.weight)
res = np.array(acc)
print(f"Mean of SoftNDCG = {res.mean()*100} %")