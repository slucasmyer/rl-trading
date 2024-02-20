import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def multi_objective_scatter(profit_data: dict, drawdown_data: dict, show: bool = True) -> None:
    """
    Expects dict of lists of profit and drawdown data indexed by generation. 
    """
    sorted_gens = sorted(profit_data.keys())
    gens_covered = str(sorted_gens[0])
    plt.xlabel("Profit")
    plt.ylabel("Drawdown")
    plt.title("Profit vs. Drawdown")
    for gen in sorted_gens:
        plt.scatter(profit_data[gen], drawdown_data[gen], label="Gen %s" % gen)
    if len(sorted_gens) > 1:
        gens_covered += "-%s" % max(sorted_gens)
    plt.savefig("Figures/%s_multi_obj_scatter_gen_%s.png" %
                (timestamp(), gens_covered))
    if show:
        if len(profit_data) > 1:
            plt.legend()
        plt.show()


def univariate_time_series(step_data: list, variable: str, gen_pop: tuple, show: bool = True) -> None:
    """
    Expects list of daily measurements for variable associated with solution identified by gen_pop. 
    """
    days = range(1, len(step_data) + 1)
    x = days
    y = step_data
    plt.plot(x, y)
    plt.xlabel("Day")
    plt.ylabel(variable)
    plt.xticks(days)
    # Add gen and individual number?
    plt.savefig("Figures/%s_%s_time_series_gen_%s_pop_%s.png" %
                (timestamp(), variable, gen_pop[0], gen_pop[1]))
    if show:
        plt.show()


def stop_loss_triggered(step_data: list, identifier: list) -> None:
    """
    Takes list of stop loss triggered step data for individual. 
    """
    pass


def timestamp():
    return str(int(time.time()))


if __name__ == '__main__':

    profit1_data = {1: [12000, 15000, 2000, 30, 10000]}
    drawdown1_data = {1: [58000, 20000, 32000, 2000, 1000]}

    multi_objective_scatter(profit1_data, drawdown1_data)

    profit2_data = {1: [12000, 15000, 2000, 30, 10000],
                    2: [19000, 18000, 12000, 3000, 8000]}
    drawdown2_data = {1: [58000, 20000, 32000, 2000, 1000],
                      2: [20000, 98000, 1000, 5000, 52000]}

    multi_objective_scatter(profit2_data, drawdown2_data)

    univariate_time_series(profit1_data[1], "profit", (1, 2), False)
