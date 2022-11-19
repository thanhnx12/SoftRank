import DataProcess
from scipy.stats import norm
import math
import torch
import numpy as np
LR = 0.01
class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = torch.nn.Linear(136, 1)

    def forward(self, x):
        x = self.layer1(x)
        return x
model = Model()
model.layer1.weight.requires_grad = True
#readData
q = DataProcess.getQuery("U:\\MSLR-WEB10K\Fold1\\test.txt",10)
datasize = len(q)
for i in range(datasize):
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
    dG_ds = q[i].G_derivative_by_score(PI,P)
    dG_ds = torch.from_numpy(np.array(dG_ds))
    ds_dw = torch.cat(ds_dw)
    grad = torch.matmul(dG_ds,ds_dw.double())
    #Training
    with torch.no_grad():
        model.layer1.weight += LR*grad
    print(q[i].SoftNDCG(P))

print(model.layer1.weight)