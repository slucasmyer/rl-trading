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

    data_collector = DataCollector(pd.read_csv(Path("./training_tqqq.csv")))

    processed_data = data_collector.prepare_and_calculate_data()

    input_shape = processed_data.shape[1]

    dimensions = [input_shape, 10, 5, 10, 4, 10, 3]

    network = PolicyNetwork(dimensions)

    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs available. Using DataParallel.")
        network = DataParallel(network)

    # Move the model to GPU if available
    network.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    trading_env = TradingEnvironment(processed_data, network)

    # initialize the thread pool and create the runner
    n_threads = 4
    pool = ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)

    problem = TradingProblem(processed_data, network, trading_env, elementwise_runner = runner) # Optimization setup

    algorithm = NSGA2(
        pop_size=100,
        sampling = FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.1, eta=20),
        eliminate_duplicates=True
    )

    res = minimize(problem, algorithm, ('n_gen', 10000), verbose=True, seed=1) # Run optimization
    scatter = Scatter()
    scatter.add(res.F, color="red")
    scatter.show()
    plt.show()