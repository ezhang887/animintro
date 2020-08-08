#!/usr/bin/env python3

import math
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, max_length, batch_size):
        super(Net, self).__init__()
        conv1_kernel_size = 1600
        conv1_stride = 10
        conv1_out_channels = 4
        self.conv1_pool_size = 5
        self.conv1 = nn.Conv1d(
            2, conv1_out_channels, conv1_kernel_size, stride=conv1_stride
        )
        self.fc1_input_size = self.__conv_pool_output_dim(
            max_length,
            conv1_out_channels,
            conv1_kernel_size,
            conv1_stride,
            self.conv1_pool_size,
            batch_size,
        )
        self.fc1 = nn.Linear(self.fc1_input_size, 4)

    def forward(self, x):
        # Input (x) shape: [batch_size, input_channels (2), max_length]
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, self.conv1_pool_size)
        x = x.view(-1, self.fc1_input_size)
        x = self.fc1(x)
        return x

    def __conv_pool_output_dim(
        self,
        input_size,
        num_channels,
        conv_kernel_size,
        conv_stride,
        pool_kernel_size,
        batch_size,
    ):
        conv_output_dim = math.floor(
            (input_size - (conv_kernel_size - 1) - 1) / conv_stride + 1
        )
        pool_output_dim = math.floor(
            (conv_output_dim - (pool_kernel_size - 1) - 1) / pool_kernel_size + 1
        )
        return pool_output_dim * num_channels * batch_size
