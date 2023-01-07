import DataProcess
from scipy.stats import norm
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
import optuna
from sklearn.model_selection import train_test_split
EPOCHES = 5
BATCH_SIZE = 500
class Model(nn.Module):
    in_features = 136
    def __init__(self, params):
        super().__init__()
        self.layer1_linear = nn.Linear(self.in_features, params['n1']),
        self.layer1_activation = nn.ReLU(),
        self.layer2_linear = nn.Linear(params['n1'], params['n2']),
        self.layer2_activation = nn.ReLU(),
        self.layer3_linear = nn.Linear(params['n2'], 1),
        self.layer3_activation = nn.Sigmoid()
    def forward(self, x):
        x = self.layer1_linear(x)
        x = self.layer1_activation(x)
        x = self.layer2_linear(x)
        x = self.layer2_activation(x)
        x = self.layer3_linear(x)
        x = self.layer3_activation(x)
        return x

q = DataProcess.getQuery("D:\\MSLR-WEB10K\Fold1\\train.txt",10)
test_set = DataProcess.getQuery("D:\\MSLR-WEB10K\Fold1\\test.txt",10)

def train_and_evaluate(param, model):
    model.layer1_linear.requires_grad_(requires_grad=True)
    model.layer2_linear.requires_grad_(requires_grad=True)
    model.layer3_linear.requires_grad_(requires_grad=True)
    opt = getattr(optim, param['optimizer'])(model.parameters(), lr=param['learning_rate'])
    #train
    for i in range(len(q)):
        # model predictions
        score = []
        for x in q[i].documents:
            temp = model.forward(x.feature)
            score.append(temp)
            temp.backward(retain_graph=True)
        model.layer1_linear.weight.grad /= 10
        model.layer1_linear.bias.grad /= 10
        model.layer2_linear.weight.grad /= 10
        model.layer2_linear.bias.grad /= 10
        model.layer3_linear.weight.grad /= 10
        model.layer3_linear.bias.grad /= 10
        q[i].score = score
        # loss

        # gradient
        PI = q[i].claculate_pi()
        P = q[i].prob_all(PI)
        dG_ds = q[i].G_derivative_by_score(PI, P)
        dG_ds = torch.from_numpy(np.array(dG_ds))
        model.layer2_linear.weight.grad = torch.matmul(dG_ds, model.layer2_linear.weight.grad)
        model.layer1_linear.weight.grad = torch.matmul(dG_ds, model.layer1_linear.weight.grad)
        model.layer3_linear.weight.grad = torch.matmul(dG_ds, model.layer3_linear.weight.grad)
        # Update Parameters
        opt.zero_grad()
        opt.step()
        model.layer1_linear.weight.grad.zero_()
        model.layer1_linear.bias.grad.zero_()
        model.layer2_linear.weight.grad.zero_()
        model.layer2_linear.bias.grad.zero_()
        model.layer3_linear.weight.grad.zero_()
        model.layer3_linear.bias.grad.zero_()
    #test
    res = []
    with torch.no_grad():
        for i in range(len(test_set)):
            # model predictions
            score = []
            for x in test_set[i].documents:
                temp = model.forward(x.feature)
                score.append(temp)
            test_set[i].score = score
            res.append(test_set[i].NDCG())

    print(f"Mean of Test set NDCG :{np.mean(res)} %")
    return np.mean(res)



def objective(trial):
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-7, 1e-1),
        'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        'n1': trial.suggest_int("n_unit", 50, 70),
        'n2': trial.suggest_int("n_unit",50,70)
    }

    model = Model(params)

    meanNDCG = train_and_evaluate(params, model)

    return meanNDCG

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=30)


best_trial = study.best_trial

for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))
