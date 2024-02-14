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

from data_preparation import DataCollector
from trading_problem import TradingProblem
from policy_network import PolicyNetwork
from trading_environment import TradingEnvironment



if __name__ == '__main__':

    data_collector = DataCollector(pd.read_csv(Path("./training_tqqq.csv")))

    processed_data = data_collector.prepare_and_calculate_data()

    input_shape = processed_data.shape[1]

    dimensions = [input_shape, 128, 256, 512, 256, 128, 3]

    network = PolicyNetwork(dimensions)

    trading_env = TradingEnvironment(processed_data, network)

    problem = TradingProblem(processed_data, network, trading_env) # Optimization setup

    algorithm = NSGA2(
        pop_size=100,
        sampling = FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.1, eta=20),
        eliminate_duplicates=True
    )

    res = minimize(problem, algorithm, ('n_gen', 10000), verbose=True) # Run optimization
    scatter = Scatter()
    scatter.add(res.F, color="red")
    scatter.show()
    plt.show()