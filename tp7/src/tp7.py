import logging
logging.basicConfig(level=logging.INFO)

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import click
from utils import *
import numpy as np


# Ratio du jeu de train à utiliser
TRAIN_RATIO = 0.05
NB_EPOCHS = 1000
INPUT_SIZE = 28*28
HIDDEN_SIZE = 100
OUTPUT_SIZE = 10



data = MNIST(ratio=TRAIN_RATIO)
train_size = int(len(data)*0.6)
test_size = len(data) - train_size
train_set, test_set = random_split(data, [train_size, test_size])
data_train = DataLoader(train_set, batch_size=300, shuffle=True, drop_last=True)
data_test = DataLoader(test_set, batch_size=300, shuffle=True, drop_last=True)

print('\ndata uploaded\n')


# Parametres lambda pour les normalisation l1 et l2
lbd1 = 1.
lbd2 = 0.
# Propotions d'unites neuronales ignorees
p_dropout = 0.


learning_rate = 1e-3
citerion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(
        params=model.parameters(), 
        lr=learning_rate, 
        weight_decay=decay
    )



writer = SummaryWriter('runs')

for t in range(NB_EPOCHS):

    train_loss, train_grad = [], []

    for x, y in data_train:
        optim.zero_grad()
        yhat, grad = model.forward(x)
        loss = criterion(yhat, y) + lbd1*l1_reg(model.parameters())
        loss.backward()
        optim.step()
        train_loss.append(loss.item())
        train_grad.append(grad.mean().item())

    """with torch.no_grad():

        for x, y in data_test:"""
            

    print(f"Iteration {t+1}: train/loss={np.mean(train_loss)}   mean/grad={np.mean(train_grad)}")


