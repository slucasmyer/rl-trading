from torch import nn

class PolicyNetwork(nn.Module):
    """A simple feedforward neural network for policy approximation."""
    def __init__(self, dimensions):
        super(PolicyNetwork, self).__init__() # Call the superclass constructor
        self.dims = dimensions # The dimensions of the network
        self.network = nn.Sequential(
            nn.Linear(self.dims[0], self.dims[1]),
            nn.ReLU(),
            nn.Linear(self.dims[1], self.dims[2]),
            nn.ReLU(),
            nn.Linear(self.dims[2], self.dims[3]),
            nn.ReLU(),
            nn.Linear(self.dims[3], self.dims[4]),
            nn.ReLU(),
            nn.Linear(self.dims[4], self.dims[5]),
            nn.ReLU(),
            nn.Linear(self.dims[5], self.dims[6]),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        """Forward pass of the neural network."""
        return self.network(x)