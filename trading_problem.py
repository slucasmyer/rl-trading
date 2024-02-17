import numpy as np
import pandas as pd
from torch import Tensor
from torch.nn import DataParallel
from pymoo.core.problem import ElementwiseProblem

from trading_environment import TradingEnvironment
from policy_network import PolicyNetwork


class TradingProblem(ElementwiseProblem):
    def __init__(self, data: pd.DataFrame, network: DataParallel[PolicyNetwork] | PolicyNetwork, environment: TradingEnvironment, *args, **kwargs):
        self.data = data
        self.network = network
        self.environment = environment
        self.dims = network.dims
        self.n_vars = sum([(self.dims[i] + 1) * self.dims[i + 1] for i in range(len(self.dims) - 1)])
        super().__init__(n_var=self.n_vars, n_obj=2, xl=-1.0, xu=1.0)
        self.data = data
        
    def _evaluate(self, x, out, *args, **kwargs):
        self.decode_model(x) # Decode the individual's parameters into the policy network
        profit, drawdown = self.environment.simulate_trading()  # Simulate trading
        # print(f"Profit: {profit}, Drawdown: {drawdown}")
        out["F"] = np.array([profit, -drawdown])

    def decode_model(self, params):
        model = self.network # `self.network` is our PolicyNetwork instance
        idx = 0 # Starting index in the parameter vector
        new_state_dict = {} # New state dictionary to load into the model
        for name, param in model.named_parameters(): # Iterate over each layer's weights and biases in the model
            num_param = param.numel() # Compute the number of elements in this layer
            param_values = params[idx:idx + num_param] # Extract the corresponding part of `params`
            param_values = param_values.reshape(param.size()) # Reshape the extracted values into the correct shape for this layer
            param_values = Tensor(param_values) # Convert to the appropriate tensor
            new_state_dict[name] = param_values # Add to the new state dictionary
            idx += num_param # Update the index
        model.load_state_dict(new_state_dict) # Load the new state dictionary into the model

