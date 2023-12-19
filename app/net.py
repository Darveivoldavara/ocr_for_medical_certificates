from torch import nn


class Net(nn.Module):
    def __init__(self, n_hidden_neurons=138, n_in_neurons=768, n_out_neurons=1):
        super().__init__()

        self.fc1 = nn.Linear(n_in_neurons, n_hidden_neurons)
        self.bn = nn.BatchNorm1d(n_hidden_neurons)
        self.act = nn.ELU()
        self.fc2 = nn.Linear(n_hidden_neurons, n_out_neurons)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
