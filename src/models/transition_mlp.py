import torch
import torch.nn as nn


class TransitionMLP(nn.Module):
    """
    Generic transition network:

        (component_t, params) → component_{t+1}

    Input dimension:
        component_dim + param_dim

    Output dimension:
        component_dim

    Uses residual formulation:
        x_{t+1} = x_t + f(x_t, params)
    """

    def __init__(
        self,
        component_dim=3,
        param_dim=3,
        hidden_dim=64,
        num_layers=3,
    ):
        super().__init__()

        input_dim = component_dim + param_dim

        layers = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, component_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x_t, params):
        """
        x_t:     (batch, component_dim)
        params:  (batch, param_dim)
        """

        # concatenate component and parameters
        inp = torch.cat([x_t, params], dim=-1)

        delta = self.net(inp)

        # residual formulation
        x_tp1 = x_t + delta

        return x_tp1