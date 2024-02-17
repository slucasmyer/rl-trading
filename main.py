import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn import DataParallel
from pathlib import Path
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization
from pymoo.visualization.scatter import Scatter


from data_preparation import DataCollector
from trading_problem import TradingProblem
from policy_network import PolicyNetwork
from trading_environment import TradingEnvironment


if __name__ == '__main__':
    """
    Basic multi-objective optimization using NSGA-II.
    This main script is used to run the optimization.
    Can be run in a python notebook or as a standalone script.
    """

    # Load the data
    data_collector = DataCollector(pd.read_csv(Path("./training_tqqq.csv")))

    # Prepare and calculate the data
    processed_data = data_collector.prepare_and_calculate_data()

    # Get the input shape
    input_shape = processed_data.shape[1]

    # Define the dimensions of the policy network
    dimensions = [input_shape, 10, 5, 10, 4, 10, 3]

    # Create the policy network
    network = PolicyNetwork(dimensions)

    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs available. Using DataParallel.")
        network = DataParallel(network) # Use DataParallel to use multiple GPUs
    else:
        print("Using a single GPU because you have a sad compute env.")

    # Move the model to GPU if available
    network.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Create the trading environment
    trading_env = TradingEnvironment(processed_data, network)

    # initialize the thread pool and create the runner for ElementwiseProblem parallelization
    n_threads = 4
    pool = ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)

    # Create the trading problem
    problem = TradingProblem(processed_data, network, trading_env, elementwise_runner = runner) # Optimization setup

    # Create the algorithm
    algorithm = NSGA2(
        pop_size=100,
        sampling = FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.1, eta=20),
        eliminate_duplicates=True
    )

    # Run the optimization
    res = minimize(problem, algorithm, ('n_gen', 50), verbose=True, seed=1)

    # Plot the results
    scatter = Scatter()
    scatter.add(res.F, color="red")
    scatter.show()
    plt.show()

    # We will want to save the best policy network to disk
    # We might use the following code to do that, but I wouldn't know as it hasn't been reached :(
    best_policy_network = None if res.pop is None else res.pop.get("X")[0]