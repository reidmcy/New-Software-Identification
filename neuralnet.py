import torch
import torch.nn

class BiRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, eta, saveDir):
        super(BiRNN, self).__init__()
        self.eta = eta
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epoch = 0
        self.saveDir = saveDir

        self.lstmAb = torch.nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first = True,
                            bidirectional = True)

        self.lstmTi = torch.nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first = True,
                            bidirectional = True)

        self.fc = torch.nn.Linear(hidden_size * 4, 2)

    def forward(self, ab, ti):
        #Reading abstract and title
        outAb, _ = self.lstmAb(ab)
        outTi, _ = self.lstmTi(ab)

        #Combine final steps
        out = torch.cat([outAb[:, -1, :], outTi[:, -1, :]], dim = 1)

        #Combine in FC
        out = self.fc(out)
        return out

    def __repr__(self):
        """Misusing and overwriting repr"""
        return r"BiRNN-{}-{}-{}".format(self.num_layers, self.hidden_size, self.epoch)

    def save(self, saveDir = None):
        if saveDir is None:
            fname = "{}/{}.pt".format(self.saveDir, repr(self))
        else:
            fname = "{}/{}.pt".format(saveDir, repr(self))
        with open(fname, 'wb') as f:
            torch.save(self, f)
