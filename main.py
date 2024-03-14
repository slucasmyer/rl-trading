import os
from pathlib import Path
import argparse
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
import multiprocessing as mp
from pymoo.core.problem import StarmapParallelization
from pymoo.visualization.scatter import Scatter
from data_preparation import DataCollector
from trading_problem import TradingProblem, PerformanceLogger
from policy_network import PolicyNetwork
from trading_environment import TradingEnvironment
import sys
from yahoo_fin_data import get_data
from plotter import Plotter

def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="Run trading optimization with optional settings.")

    # Add arguments
    parser.add_argument('--pop_size', type=int, default=100, help='Population size for the genetic algorithm')
    parser.add_argument('--n_gen', type=int, default=100, help='Number of generations for the genetic algorithm')
    parser.add_argument('--profit_threshold', type=float, default=100.0, help='Profit threshold for considering a model worth saving')
    parser.add_argument('--drawdown_threshold', type=float, default=40.0, help='Drawdown threshold for considering a model worth saving')

    # Parse the arguments
    args = parser.parse_args()

    return args

def set_path(script_path: Path, dir_path: str, file_path: str) -> Path:
    """Set output path."""
    output_dir = script_path / Path(dir_path) 
    output_dir.mkdir(parents=True, exist_ok=True)
    new_path = output_dir / file_path
    return new_path
    

def map_params_to_model(model, params):
        """
        Decodes (i.e. maps) the genes of an individual (x) into the policy network.
        """
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

def train_and_validate(queue, n_pop, n_gen):

    SCRIPT_PATH = Path(__file__).parent

    # Get and load data
    stock_df = pd.DataFrame(get_data("TQQQ"))
    data_collector = DataCollector(data_df=stock_df)

    # Prepare and calculate the data, columns_to_drop listed just to highlight where that ability is
    data_collector.prepare_and_calculate_data(columns_to_drop=['close'])

    # Get the input shape
    input_shape = data_collector.data_tensor.shape[1]

    print("input_shape (main)", input_shape)

    # Define the dimensions of the policy network
    dimensions = [input_shape, 64, 32, 16, 8, 4, 3]

    # Create the policy network
    network = PolicyNetwork(dimensions)

    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs available. Using DataParallel.")
        # Use DataParallel to use multiple GPUs
        network = DataParallel(network)
    else:
        print("Using a single GPU because you have a sad compute env.")

    # Move the model to GPU if available
    network.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Create the trading environment
    trading_env = TradingEnvironment(
        data_collector.training_tensor, network, data_collector.training_prices)

    # initialize the thread pool and create the runner for ElementwiseProblem parallelization
    n_threads = 4
    pool = mp.pool.ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)

    # Create the trading problem
    problem = TradingProblem(data_collector.training_tensor, network,
                             trading_env, elementwise_runner=runner)  # Optimization setup

    # Create the algorithm
    algorithm = NSGA2(
        pop_size=n_pop,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.2, eta=20),
        eliminate_duplicates=True
    )

    performance_logger = PerformanceLogger(queue)

    # Run the optimization
    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        callback=performance_logger,
        verbose=True,
        save_history=True
    )
    date_time = pd.to_datetime("today").strftime("%Y-%m-%d_%H-%M-%S")
    history: pd.DataFrame = pd.DataFrame(performance_logger.history)
    history.to_csv(set_path(SCRIPT_PATH, f"Output/performance_log/ngen_{n_gen}", f"{date_time}.csv"))

    generations = history["generation"].values
    objectives = history["objectives"].values
    decisions = history["decision_variables"].values
    best = history["best"].values

    historia = []
    
    for i in range(len(generations)):
        avg_profit, avg_drawdown, avg_trades = 0, 0, 0
        objs = objectives[i]
        for row in objs:
            avg_profit += row[0]
            avg_drawdown += row[1]
            avg_trades += row[2]
        avg_profit /= len(objs)
        avg_drawdown /= len(objs)
        avg_trades /= len(objs)
        row = [generations[i], avg_profit, avg_drawdown, avg_trades, best[i]]
        historia.append(row)
    
    history_df: pd.DataFrame = pd.DataFrame(
        columns=["generation", "avg_profit", "avg_drawdown", "num_trades", "best"],
        data=historia
    )
    history_df.to_csv(set_path(SCRIPT_PATH, f"Output/performance_log/ngen_{n_gen}", f"{date_time}_avg.csv"))

    trading_env.set_features(data_collector.testing_tensor)
    trading_env.set_closing_prices(data_collector.testing_prices)
    population = None if res.pop is None else res.pop.get("X")

    validation_results = []
    max_ratio = 0.0
    best_network = None
    if population is not None:
        for i, x in enumerate(population):
            map_params_to_model(network, x)
            # torch.save(network.state_dict(), f"Output/policy_networks/{date_time}_ngen_{n_gen}_top_{i}.pt")
            trading_env.reset()
            profit, drawdown, num_trades = trading_env.simulate_trading()
            
            ratio = profit / drawdown if drawdown != 0 else profit / 0.0001

            if ratio > max_ratio and drawdown < 55.0:
                best = ratio
                best_network = network.state_dict()
                
            print(f"Profit: {profit}, Drawdown: {drawdown}, Num Trades: {num_trades}, Ratio: {ratio}")
            validation_results.append([profit, drawdown, num_trades, ratio, str(x)])
        
        torch.save(best_network, set_path(SCRIPT_PATH, f"Output/policy_networks/ngen_{n_gen}", f"{date_time}_best.pt"))
        
        queue.put(validation_results)
        
        validation_results_df = pd.DataFrame(
            columns=["profit", "drawdown", "num_trades", "ratio", "chromosome"],
            data=validation_results
        )

        # sort by ratio
        validation_results_df = validation_results_df.sort_values(by="ratio", ascending=False)
        
        validation_results_df.to_csv(set_path(SCRIPT_PATH, f"Output/validation_results/ngen_{n_gen}", f"{date_time}.csv"))

        # plot in crude manner
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # # Plotting data
        # ax.scatter(validation_results_df["profit"], validation_results_df["drawdown"], validation_results_df["num_trades"])

        # ax.set_xlabel('Profit')
        # ax.set_ylabel('Drawdown')
        # ax.set_zlabel('Number of Trades')
        
        # plt.savefig(set_path(SCRIPT_PATH, f"Output/validation_results/ngen_{n_gen}", f"{date_time}_validation.png"))
        # plt.show()

    # use the video writer as a resource
    
    # with Recorder(Video(f"Assets/videos/n_gen_{n_gen}_{date_time}.mp4")) as rec:

    #     # for each algorithm object in the history
    #     for entry in res.history:
    #         sc = Scatter(title=("Gen %s" % entry.n_gen), labels=["Profit", "Drawdown", "Trade Count"])
    #         sc.add(entry.pop.get("F"))
    #         sc.add(entry.problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    #         sc.do()

    #         # finally record the current visualization to the video
    #         rec.record()
    #     rec.close()

    pool.close()


if __name__ == '__main__':
    """
    Basic multi-objective optimization using NSGA-II.
    This main script is used to run the optimization.
    Can be run in a python notebook or as a standalone script.
    """

    args = parse_args()

    # Access arguments like this
    print(f"Population size: {args.pop_size}")
    print(f"Number of generations: {args.n_gen}")
    print(f"Profit threshold: {args.profit_threshold}")
    print(f"Drawdown threshold: {args.drawdown_threshold}")

    # NSGA-II parameters
    n_pop = args.pop_size
    n_gen = args.n_gen
    profit_threshold = args.profit_threshold
    drawdown_threshold = args.drawdown_threshold

    # Start training and validation in new process, create visualizations with data from queue
    queue = mp.Queue()
    plotter = Plotter(queue, n_gen)
    train_and_validate_process = mp.Process(target=train_and_validate, args=(queue, n_pop, n_gen))

    train_and_validate_process.start()

    plotter.update_while_training()

    train_and_validate_process.join()

    train_and_validate_process.close()

    queue.close()

    print("Training and validation process finished.")
