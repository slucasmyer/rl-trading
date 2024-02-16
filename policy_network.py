from torch import nn

class PolicyNetwork(nn.Module):
    def __init__(self, dimensions):
        super(PolicyNetwork, self).__init__()
        self.dims = dimensions
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
        return self.network(x)