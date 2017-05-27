import numpy as np
import torch.nn
import torch.nn.functional


import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, eta):
        super(BiRNN, self).__init__()
        self.eta = eta
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstmAb = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)

        self.lstmTi = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 4, 2)  # 2 for bidirection

    def forward(self, ab, ti):

        # Set initial states

        #Abstract
        #h0Ab = Variable(torch.zeros(self.num_layers*2, ab.size(0), self.hidden_size)).cuda() # 2 for bidirection
        #c0Ab = Variable(torch.zeros(self.num_layers*2, ab.size(0), self.hidden_size)).cuda()

        #Title
        #h0Ti = Variable(torch.zeros(self.num_layers*2, ti.size(0), self.hidden_size)).cuda() # 2 for bidirection
        #c0Ti = Variable(torch.zeros(self.num_layers*2, ti.size(0), self.hidden_size)).cuda()

        # Forward propagate RNN Layers
        outAb, _ = self.lstmAb(ab)#, (h0Ab, c0Ab))

        outTi, _ = self.lstmTi(ab)#, (h0Ti, c0Ti))

        #Combine final steps
        out = torch.cat([outAb[:, -1, :], outTi[:, -1, :]], dim = 1)

        #Combine in FC
        out = self.fc(out)
        return out

    def __repr__(self):
        """Misusing and overwriting repr"""
        return r"BiRNN-{}-{}".format(self.num_layers, self.hidden_size)

    def save(self, saveDir):
        with open("{}/{}.pt".format(saveDir, repr(self)), 'wb') as f:
            torch.save(Net, f)
