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


class TradingEnvironment:
    def __init__(self, data, model):
        self.data = data
        self.model = model
        self.features = preprocess_data(self.data)
        self.balance = 100_000.00
        
    def simulate_trading(self):
      profit = 0
      max_profit = 0
      drawdown = 0
      # Simulate trading over the dataset
      for i in range(len(self.features)):
          # Get the feature vector for the current day
          feature_vector = self.features[i:i+1]  # model expects shape [1, num_features]
          
          # Make a decision based on the policy network
          decision = self.model(feature_vector).argmax().item()  # 0=buy, 1=hold, 2=sell
          if decision == 0:  # Buy
              profit -= self.data['close'][i]
          elif decision == 2:  # Sell
              profit += self.data['close'][i]
          
          # Update max_profit and drawdown
          max_profit = max(max_profit, profit)
          drawdown = min(drawdown, profit - max_profit)
      
      return profit, drawdown


def preprocess_data(data, columns_to_drop=[]):

    # Drop columns
    if columns_to_drop:
        data = data.drop(columns=columns_to_drop)
    
    # Fill NaN values
    data_filled = data.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    # Convert the DataFrame to a tensor
    features_tensor = torch.tensor(data_filled.values, dtype=torch.float32)
    
    return features_tensor

