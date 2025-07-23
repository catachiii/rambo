import torch.nn as nn
from crl2.utils.get_activation import get_activation


class Value(nn.Module):
    def __init__(self,
                 num_obs,
                 hidden_dims=None,
                 activation='relu',
                 device='cpu'):

        super(Value, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim = num_obs

        # Value function
        layers = [nn.Linear(mlp_input_dim, hidden_dims[0]).to(device), activation]
        for la in range(len(hidden_dims)):
            if la == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[la], 1).to(device))
            else:
                layers.append(nn.Linear(hidden_dims[la], hidden_dims[la + 1]).to(device))
                layers.append(activation)
        self.value = nn.Sequential(*layers)

    def forward(self, input_x):
        return self.value(input_x)

