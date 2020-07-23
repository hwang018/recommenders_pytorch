import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.init as weight_init

class Encoder(nn.Module):
    """
    Class for Symmetrical AutoEncoder Network
    :param L: List of int, contains sizes of encoding layers and starts with feature size
    For example: [500, 20, 10] will result in:
      - encoder 2 layers: 500x20 and 20x10. Representation layer (z) will be 10
      - decoder 2 layers: 10x20 and 20x500, output size is 500, reconstructed.
    :param activation_fn: (default 'sigmoid') Type of activation function
    :param drop_prob: (default: 0.0) Dropout probability
    """
    def __init__(self, L, activation_fn='sigmoid', drop_prob=0.0):
        super(Encoder, self).__init__()
        layers = self.create_nn_structure(L)
        self.num_layers = len(L)
        # create controller for activation function
        self.activation_fn_nm = activation_fn
        # create dropout module
        self._drop_prob = drop_prob
        if drop_prob > 0.0:
            self.dropout = nn.Dropout(drop_prob)
        #initialize with empty list to store layers
        self.linears = nn.ModuleList([])
        self.linears.extend([nn.Linear(i[0], i[1]) for i in layers])
        
    def get_activation_fn(self):
        # user selected activation function at layers except for last layer
        if self.activation_fn_nm == 'relu':
            return nn.ReLU()
        elif self.activation_fn_nm == 'lrelu':
            return nn.LeakyReLU()
        elif self.activation_fn_nm == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError('Activation function type not defined')
    
    def forward(self, x):
        for i,layer in enumerate(self.linears):
            if i <= self.num_layers-1:
                # create instance of activation function
                act_fn = self.get_activation_fn()
                # pass in the input
                x = act_fn(self.linears[i](x))
                if self._drop_prob > 0.0 and i <= int(self.num_layers/2): 
                    # apply dropout only on encode layer by control of i
                    x = self.dropout(x)
        # No activation on the last decoding layer
        x = self.linears[-1](x)
        return x

    def create_nn_structure(self, L):
        max_ind = len(L)-1
        layers = []
        for i,v in enumerate(L):
            if i < max_ind:
                #still have i+1 available, create layer tuple
                layer = [v,L[i+1]]
                layers.append(layer)
        #then inverse the layers for decoder size
        encoder_layers = layers[:]
        for l in encoder_layers[::-1]:
            decoder_layer = l[::-1]
            layers.append(decoder_layer)
        return layers