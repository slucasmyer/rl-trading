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
from trading_problem import TradingProblem, PerformanceLogger
from policy_network import PolicyNetwork
from trading_environment import TradingEnvironment


if __name__ == '__main__':
    """
    Basic multi-objective optimization using NSGA-II.
    This main script is used to run the optimization.
    Can be run in a python notebook or as a standalone script.
    """

    # Hyperparameters
    n_pop = 100
    n_gen = 50
    
    # Load the data
    data_collector = DataCollector(data_df=pd.read_csv(Path("./training_tqqq.csv")))

    # Prepare and calculate the data, columns_to_drop listed just to highlight where that ability is
    data_collector.prepare_and_calculate_data(columns_to_drop=['close'])
    print("processed_data (main)", data_collector.data_df.head())

    # Get the input shape
    input_shape = data_collector.data_tensor.shape[1]

    print("input_shape (main)", input_shape)

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
    trading_env = TradingEnvironment(data_collector.data_tensor, network, data_collector.closing_prices, n_gen, n_pop)

    # initialize the thread pool and create the runner for ElementwiseProblem parallelization
    n_threads = 4
    pool = ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)

    # Create the trading problem
    problem = TradingProblem(data_collector.data_tensor, network, trading_env, elementwise_runner = runner) # Optimization setup

    # Create the algorithm
    algorithm = NSGA2(
        pop_size = n_pop,
        sampling = FloatRandomSampling(),
        crossover = SBX(prob=0.9, eta=15),
        mutation = PM(prob=0.1, eta=20),
        eliminate_duplicates = True
    )

    performance_logger = PerformanceLogger()

    # Run the optimization
    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        callback=performance_logger,
        verbose=True,
        seed=1
    )

    # Plot the results
    results_plot = Scatter()
    results_plot.add(res.F, color="red")
    results_plot.show()
    plt.show()

    history = pd.DataFrame(performance_logger.history)
    date_time = pd.to_datetime("today").strftime("%Y-%m-%d_%H-%M-%S")
    history.to_csv(f"Figures/optimization_history_{date_time}.csv")

    ojbectives_history = Scatter()
    ojbectives_history.add(history["objectives"].values, color="blue")
    ojbectives_history.show()
    plt.show()

    best_policy_history = Scatter()
    best_policy_history.add(history["best"].values, color="green")
    best_policy_history.show()
    plt.show()


    # We will want to save the best policy network to disk
    # We might use the following code to do that, but I wouldn't know as it hasn't been reached :(
    top_10 = None if res.pop is None else res.pop.get("X")[10]