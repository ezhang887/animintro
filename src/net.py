#!/usr/bin/env python3

import math
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, max_length, mean=None, stddev=None):
        super(Net, self).__init__()
        self.mean = mean
        self.stddev = stddev
        self.max_length = max_length

        conv1_kernel_size = 1600
        conv1_stride = 320
        conv1_out_channels = 4
        self.conv1_pool_size = 50
        self.conv1 = nn.Conv1d(
            2, conv1_out_channels, conv1_kernel_size, stride=conv1_stride
        )
        self.fc1_input_size = self.__conv_pool_output_dim(
            max_length,
            conv1_out_channels,
            conv1_kernel_size,
            conv1_stride,
            self.conv1_pool_size,
        )
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.fc1_input_size, 250)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(250, 4)
    
    def forward(self, x):
        # Input (x) shape: [input_channels (2), max_length]
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, self.conv1_pool_size)
        x = x.view(-1, self.fc1_input_size)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def __conv_pool_output_dim(
        self,
        input_size,
        num_channels,
        conv_kernel_size,
        conv_stride,
        pool_kernel_size,
    ):
        """
        Calculate the size of the flatted conv/pool layer going into the fully connected layer.
        See https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html and
            https://pytorch.org/docs/master/generated/torch.nn.MaxPool1d.html for the formulas.

        Parameters
        ----------
        input_size: int
            number of datapoints in the input of the final conv layer
        num_channels: int
            number of channels of the data
        conv_kernel_size: int
            size of the kernel for the final conv layer
        conv_stride: int
            stride for the final conv layer
        pool_kernel_size: int
            size of the kernel for the max pool layer after the conv layer
        
        """
        conv_output_dim = math.floor(
            (input_size - (conv_kernel_size - 1) - 1) / conv_stride + 1
        )
        pool_output_dim = math.floor(
            (conv_output_dim - (pool_kernel_size - 1) - 1) / pool_kernel_size + 1
        )
        return pool_output_dim * num_channels

    def predict(self, data):
        output = self.forward(data)
        label = self.mean + (output * self.stddev)
        return label
