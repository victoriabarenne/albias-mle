import torch.nn as nn
import torch
import torch.nn.functional as F
def MLP(n_hidden_layers, width=50, n_in=1, n_out=1):
    modules=[]
    modules.append(nn.Linear(n_in, width))
    modules.append(nn.Tanh())
    for i in range(n_hidden_layers-1):
        modules.append(nn.Linear(width, width))
        modules.append(nn.Tanh())
    modules.append(nn.Linear(width, n_out))

    network = nn.Sequential(*modules)
    return network


class MLP_test(nn.Module):
    def __init__(self, n_layers, n_in, n_out, n_neurons=50):
        super().__init__()
        self.n_layers= n_layers
        self.n_in= n_in
        self.n_out= n_out
        self.in_layer = nn.Linear(n_in, n_neurons)
        self.hidden_layer= nn.Linear(n_neurons, n_neurons)
        self.out_layer = nn.Linear(n_neurons, n_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = nn.Tanh()(self.in_layer(x))
        for _ in range(self.n_layers-1):
            x = nn.Tanh()(self.hidden_layer(x))
        x = self.out_layer(x)
        return F.log_softmax(x, dim=1)


# Define CNN and ResNet

def CNN(n_hidden_layers, width, n_in, n_out):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


