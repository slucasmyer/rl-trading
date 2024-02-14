import numpy as np
import pandas as pd
import torch
from pathlib import Path
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from torch import nn
import matplotlib.pyplot as plt

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