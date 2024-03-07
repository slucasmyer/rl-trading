import matplotlib.pyplot as plt
import matplotlib
import datetime as dt
from mpl_toolkits.mplot3d import Axes3D

# May be easiest to use this in performance logger? Maybe add a general scatter method with annotations?
# May need better way to handle percentages since we're introducing another objective.


class Plotter():

    def __init__(self, queue, n_gen):
        matplotlib.use('TkAgg')
        plt.ion()
        plt.style.use('ggplot')
        self.queue = queue
        self.scatter = None
        self.interactive_convergence_scatter = self._create_interactive_scatter()
        self.n_gen = n_gen

    def update_while_training(self):
        current_generation = 1
        while current_generation < self.n_gen:
            plt.pause(1)
            if not self.queue.empty():
                x_data, y_data, z_data = self.queue.get()
                self.update_interactive_convergence_scatter(
                    x_data, y_data, z_data, current_generation)
                current_generation += 1
        plt.show(block=True)

    def _create_fig_ax(self, title: str = "Generation", xlabel: str = "Profit", ylabel: str = "Drawdown", zlabel: str = "Trade Count", x_percentage: bool = True, y_percentage: bool = True, z_percentage: bool = True) -> tuple:
        """
        Returns base fig/ax for plots.
        """
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(projection='3d')
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['right'].set_color((.8, .8, .8))
        ax.spines['top'].set_color((.8, .8, .8))
        ax.set_xlabel(xlabel, fontsize='large', fontstyle='italic')
        ax.set_ylabel(ylabel, fontsize='large', fontstyle='italic')
        ax.set_zlabel(zlabel, fontsize='large', fontstyle='italic')
        if x_percentage:
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
        if y_percentage:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
        if z_percentage:
            ax.zaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
        return (fig, ax)

    def _create_interactive_scatter(self) -> tuple:
        """
        Creates a figure appropriate for use as an interactive scatter plot (can click plotted points for individual ID and info).
        """
        fig, ax = self._create_fig_ax(z_percentage=False)
        self.scatter = ax.scatter([], [])
        return (fig, ax)

    def update_interactive_convergence_scatter(self, x_data: list, y_data: list, z_data: list, n_gen: int) -> None:
        """
        Updates and redraws scatter plot with data for a color-coded generation of chromosomes.
        """
        self.scatter.remove()
        fig, ax = self.interactive_convergence_scatter
        ax.set_title(f"Generation {n_gen}", fontsize='x-large', weight='bold')
        # plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1)
        colormap = matplotlib.cm.plasma
        self.scatter = ax.scatter(x_data, y_data, z_data,
                                  color=colormap((n_gen % 5) / 5))

    def create_standard_scatter(self, x_data: list, y_data: list, title: str = "Profit vs. Drawdown", xlabel: str = "Profit", ylabel: str = "Drawdown") -> None:
        """
        Creates a scatter for a single undifferentiated group of chromosomal data (has no annotations).
        """
        fig, ax = self._create_fig_ax(title, xlabel, ylabel)
        scatter = ax.scatter(x_data, y_data)
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"Figures/{timestamp}_{title}_scatter.png")

    # def stop_losses_triggered(stop_loss_data: list, network_decision_data: list, gen_id: int, pop_id: int) -> None:
    #     """
    #     Delete?
    #     """
    #     days = range(1, len(stop_loss_data) + 1)
    #     plt.scatter(days, network_decision_data,
    #                 c=stop_loss_data, cmap='RdYlGn')
    #     plt.xlabel("Day")
    #     plt.ylabel("Network Decision")
    #     plt.xticks(days)
    #     plt.yticks([0, 1, 2], ["buy", "hold", "sell"])
    #     timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #     plt.savefig(
    #         f"Figures/{timestamp}_stop_losses_triggered_gen{gen_id}_pop_{pop_id}.png")


if __name__ == '__main__':

    # Scatter plotting testing data
    profit_data_1 = [12000, 15000, 2000, 20000, 55000]
    drawdown_data_1 = [58000, 20000, 32000, 80000, 55000]

    profit_data_2 = [1000, 2000, 5000, 8000, 10000, 15000, 18000,
                     30000, 35000, 40000, 62000, 90000, 95000, 99000, 100000]
    drawdown_data_2 = [5000, 10000, 15000, 20000, 35000, 40000,
                       50000, 62000, 65000, 70000, 75000, 80000, 85000, 92000, 100000]

    profit_data_3 = [5000, 6000, 7000, 1000, 12000]
    drawdown_data_3 = [24000, 2000, 3000, 6000, 2500]

    # Stop loss trigger plotting testing data
    stop_loss_step_data = [0, 1, 0, 1, 0]
    network_decision_data = [0, 1, 2, 1, 0]

    # Test instantiation
    plotter = Plotter()

    # Test methods
    plotter.update_interactive_convergence_scatter(
        profit_data_1, drawdown_data_1, 1)
    plotter.update_interactive_convergence_scatter(
        profit_data_2, drawdown_data_2, 2)
    plotter.update_interactive_convergence_scatter(
        profit_data_3, drawdown_data_3, 3)

    plotter.create_standard_scatter(profit_data_1, drawdown_data_1)
    plotter.create_standard_scatter(profit_data_2, drawdown_data_2)
    plotter.create_standard_scatter(profit_data_3, drawdown_data_3)

    # plotter.stop_losses_triggered(stop_loss_step_data, network_decision_data, (1, 2))

    print("Done")
