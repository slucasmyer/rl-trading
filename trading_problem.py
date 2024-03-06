import numpy as np
import pandas as pd
from torch import Tensor
from torch.nn import DataParallel
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback

from trading_environment import TradingEnvironment
from policy_network import PolicyNetwork
from plotter import Plotter


class TradingProblem(ElementwiseProblem):
    """
    The trading problem class for the multi-objective optimization.
    Takes a dataset, a policy network, and a trading environment.
    Calculates the number of variables based on the policy network's parameters.
    Calls the superclass constructor with the number of variables, objectives, and lower/upper bounds for x (gene) values.

    *** Still need to add 1 to n_vars for stop-loss gene ***
    """
    def __init__(self, data: Tensor, network: DataParallel[PolicyNetwork] | PolicyNetwork, environment: TradingEnvironment, *args, **kwargs):
        self.data = data # The dataset
        self.network: DataParallel[PolicyNetwork] | PolicyNetwork = network # The policy network
        self.environment = environment # The trading environment
        self.n_vars = sum([(self.network.dims[i] + 1) * self.network.dims[i + 1] for i in range(len(self.network.dims) - 1)]) # The number of variables
        super().__init__(n_var=self.n_vars, n_obj=3, xl=-1.0, xu=1.0) # Call the superclass constructor
        self.data = data # The dataset. Still not sure why I need to do this again, but it doesn't work otherwise.
        self.chromosome: int = 0 # Int representing he individual's chromosome for mapping/visualization/selection purposes

    def get_chromosome(self):
        """Updates the chromosome and returns it."""
        self.chromosome %= 100
        self.chromosome += 1
        return self.chromosome
        
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Method to evaluate the individual (x).
        Called by the optimization algorithm.
        Profit and drawdown are calculated based on the trading decisions agent makes in environment.
        The objectives are set to the profit and the negative drawdown.
        """
        self.decode_model(x) # Decode the individual's parameters into the policy network
        profit, drawdown, num_trades = self.environment.simulate_trading(self.get_chromosome())  # Simulate trading
        out["F"] = np.array([profit, -drawdown, num_trades]) # Set the objectives

    def decode_model(self, params):
        """
        The most important method in this class.
        Decodes (i.e. maps) the genes of an individual (x) into the policy network.
        
        *** When stop-loss added we'll need to pop the last gene and return it to set the stop-loss value in the environment ***
        """
        idx = 0 # Starting index in the parameter vector
        new_state_dict = {} # New state dictionary to load into the model
        for name, param in self.network.named_parameters(): # Iterate over each layer's weights and biases in the model
            num_param = param.numel() # Compute the number of elements in this layer
            param_values = params[idx:idx + num_param] # Extract the corresponding part of `params`
            param_values = param_values.reshape(param.size()) # Reshape the extracted values into the correct shape for this layer
            param_values = Tensor(param_values) # Convert to the appropriate tensor
            new_state_dict[name] = param_values # Add to the new state dictionary
            idx += num_param # Update the index
        self.network.load_state_dict(new_state_dict) # Load the new state dictionary into the model




class PerformanceLogger(Callback):
    def __init__(self):
        super().__init__()
        self.history = []
        self.plotter = Plotter()

    def notify(self, algorithm):
        F = algorithm.pop.get("F") # The objective values
        X = algorithm.pop.get("X") # The decision variables
        
        # Log the objective values (and any additional information)
        self.history.append({
            "generation": algorithm.n_gen,
            "objectives": F.copy(),
            "decision_variables": X.copy(),
            "best": F.min(),
        })
        
        # Plot objective data for each generation
        profits, drawdowns, num_trades = zip(*F)
        self.plotter.update_interactive_convergence_scatter(profits, drawdowns, num_trades, algorithm.n_gen)