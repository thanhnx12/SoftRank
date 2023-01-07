import optuna
from sklearn.model_selection import train_test_split

import DataProcess
from scipy.stats import norm
import math
import torch.nn as nn
import numpy as np

study = optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler())


def build_model(params):
    in_features = 136

    return nn.Sequential(
        nn.Linear(in_features, params['n1']),
        nn.ReLU(),
        nn.Linear(params['n1'], params['n2']),
        nn.ReLU(),
        nn.Linear(params['n2'],1),
        nn.Sigmoid()
    )


def train_and_evaluate(param, model):
    q = DataProcess.getQuery("D:\\MSLR-WEB10K\Fold1\\test.txt", 10)
    datasize = len(q)

    train_data, val_data = train_test_split(q, test_size=0.2, random_state=42)


    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, param['optimizer'])(model.parameters(), lr=param['learning_rate'])

    for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in train_dataloader:
            train_label = train_label.to(device)
            train_input = train_input.to(device)

            output = model(train_input.float())

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                val_input = val_input.to(device)

                output = model(val_input.float())

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        accuracy = total_acc_val / len(val_data)

    return accuracy
def objective(trial):
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-7, 1e-1),
        'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        'n1': trial.suggest_int("n_unit", 50, 70)
        'n2': trial.suggest_int("n_unit",50,70)
    }

    model = build_model(params)

    accuracy = train_and_evaluate(params, model)

    return accuracy