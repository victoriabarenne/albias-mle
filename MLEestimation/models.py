import torch.nn as nn

def MLP(n_hidden_layers, n_neurons=50, n_in=1, n_out=1):
    modules=[]
    modules.append(nn.Linear(n_in, n_neurons))
    modules.append(nn.Tanh())
    for i in range(n_hidden_layers-1):
        modules.append(nn.Linear(n_neurons,n_neurons))
        modules.append(nn.Tanh())
    modules.append(nn.Linear(n_neurons,n_out))

    network = nn.Sequential(*modules)
    return network

# Define CNN and ResNet

