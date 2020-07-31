#!/usr/bin/env python3

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.magic_num = 1103908 #234395 * 4
        self.conv1 = nn.Conv1d(2, 4, 1600, stride=10)
        self.pool = nn.MaxPool1d(5)
        # self.conv2 = nn.Conv1d(4, 8, 400, stride=10)
        self.fc1 = nn.Linear(self.magic_num, 4)
        # self.fc2 = nn.Linear(10000, 100)
        # self.fc3 = nn.Linear(100, 4)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.magic_num)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        x = self.fc1(x)
        return x