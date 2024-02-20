import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def multi_objective_scatter(profit_data: list, drawdown_data: list, show: bool = True):
    """
    Expects lists of lists of profit and drawdown data for variable generations of individuals.
    """
    plt.xlabel("Profit")
    plt.ylabel("Drawdown")
    plt.title("Multi-objective Scatter")
    for i in range(len(profit_data)):
        plt.scatter(profit_data[i], drawdown_data[i], label="Gen %s" % i)
    # Add timestamp and gen numbers?
    plt.savefig("Figures/multi_obj_scatter.png")
    if show:
        plt.legend()
        plt.show()

def univariate_time_series(step_data: list, variable: str, show: bool = True):
    """
    Expects list of daily measurements for identified variable. 
    """
    days = range(1, len(step_data) + 1)
    x = days
    y = step_data
    plt.plot(x, y)
    plt.xlabel("Day")
    plt.ylabel(variable)
    plt.xticks(days)
    plt.savefig("Figures/" + variable + "_time_series.png")
    if show:
        plt.show()


if __name__ == '__main__':
    profit1_data = [[12000, 15000, 2000, 30, 10000]]
    drawdown1_data = [[58000, 20000, 32000, 2000, 1000]]
    multi_objective_scatter(profit1_data, drawdown1_data)
    profit2_data = [[12000, 15000, 2000, 30, 10000],
                   [19000, 18000, 12000, 3000, 8000]]
    drawdown2_data = [[58000, 20000, 32000, 2000, 1000],
                     [20000, 98000, 1000, 5000, 52000]]
    multi_objective_scatter(profit2_data, drawdown2_data)
    univariate_time_series(profit1_data[0], "Profit")
