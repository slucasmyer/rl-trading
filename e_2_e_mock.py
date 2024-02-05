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


# Policy Network (generic example)
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.dims = [input_size, 128, 256, 512, 256, 128, output_size]
        self.n_vars = sum([(self.dims[i] + 1) * self.dims[i + 1] for i in range(len(self.dims) - 1)])
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

# Problem definition
class TradingProblem(ElementwiseProblem):
    def __init__(self, input_size, output_size):
        self.dims = [input_size, 128, 256, 512, 256, 128, output_size]
        self.n_vars = sum([(self.dims[i] + 1) * self.dims[i + 1] for i in range(len(self.dims) - 1)])
        super().__init__(n_var=self.n_vars, n_obj=2, xl=-1.0, xu=1.0)
        self.model_template = PolicyNetwork(input_size, output_size)
        self.data_path = Path("./testing_tqqq.csv")
        self.data = pd.read_csv(self.data_path)
        
    def _evaluate(self, x, out, *args, **kwargs):
        model = self.decode_model(x) # Decode the individual's parameters into the policy network
        profit, drawdown = simulate_trading(model, self.data)  # Simulate trading
        print(f"Profit: {profit}, Drawdown: {drawdown}")
        out["F"] = np.array([profit, -drawdown])

    def decode_model(self, params):
        model = self.model_template # `self.model_template` is our PolicyNetwork instance
        idx = 0 # Starting index in the parameter vector
        new_state_dict = {} # New state dictionary to load into the model
        for name, param in model.named_parameters(): # Iterate over each layer's weights and biases in the model
            num_param = param.numel() # Compute the number of elements in this layer
            param_values = params[idx:idx + num_param] # Extract the corresponding part of `params`
            param_values = param_values.reshape(param.size()) # Reshape the extracted values into the correct shape for this layer
            param_values = torch.Tensor(param_values) # Convert to the appropriate tensor
            new_state_dict[name] = param_values # Add to the new state dictionary
            idx += num_param # Update the index
        model.load_state_dict(new_state_dict) # Load the new state dictionary into the model
        return model


def preprocess_data(data, feature_columns=['close'], lookback=5):
    """
    Preprocesses the stock data by normalizing and creating a simple feature set for the PolicyNetwork.

    Parameters:
    - data: pd.DataFrame, the stock price data.
    - feature_columns: list of str, the column names to be used as features.
    - lookback: int, the number of previous timesteps to include as features.

    Returns:
    - A torch.Tensor containing the preprocessed features for each timestep.
    """
    # Normalize the features
    normalized_data = (data[feature_columns] - data[feature_columns].min()) / (data[feature_columns].max() - data[feature_columns].min())
    features = [] # Will hold the processed feature vectors

    # Create feature vectors with lookback
    for i in range(lookback, len(normalized_data)):
        # Get the lookback window of features
        window = normalized_data.iloc[i-lookback:i].values.flatten()  # Flatten to create a single feature vector
        features.append(window) # Add to the feature list
    features_tensor = torch.tensor(features, dtype=torch.float32) # Convert to a PyTorch tensor
    return features_tensor


def simulate_trading(model, data):
    features = preprocess_data(data) # `features` is a tensor of shape [num_days, num_features]
    # Initialize variables to track profit and drawdown
    profit = 0
    max_profit = 0
    drawdown = 0
    
    # Simulate trading over the dataset
    for i in range(len(features)):
        # Get the feature vector for the current day
        feature_vector = features[i:i+1]  # model expects shape [1, num_features]
        
        # Make a decision based on the policy network
        decision = model(feature_vector).argmax().item()  # 0=buy, 1=hold, 2=sell
        if decision == 0:  # Buy
            profit -= data['close'][i]
        elif decision == 2:  # Sell
            profit += data['close'][i]
        
        # Update max_profit and drawdown
        max_profit = max(max_profit, profit)
        drawdown = min(drawdown, profit - max_profit)
    
    return profit, drawdown


if __name__ == '__main__':

    problem = TradingProblem(input_size=5, output_size=3) # Optimization setup

    algorithm = NSGA2(
        pop_size=100,
        sampling = FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.1, eta=20),
        eliminate_duplicates=True
    )


    res = minimize(problem, algorithm, ('n_gen', 100), verbose=True) # Run optimization
    scatter = Scatter()
    scatter.add(res.F, color="red")
    scatter.show()
    plt.show()