import matplotlib.pyplot as plt
import matplotlib
import datetime as dt
matplotlib.use('TkAgg')

# May be easiest to use this in performance logger?

class Plotter():

    def __init__(self):
        plt.ion()
        plt.style.use('ggplot')
        self.interactive_scatter = self._create_interactive_scatter()
        self.scatters = []
        self.annotations = []
        self.scatter_counter = 0

    def _create_fig_ax(self, title: str = "Profit vs. Drawdown", xlabel: str = "Profit", ylabel: str = "Drawdown", are_percentages: bool = True) -> tuple:
        """
        Returns base fig/ax for plots.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['right'].set_color((.8, .8, .8))
        ax.spines['top'].set_color((.8, .8, .8))
        ax.set_title(title, fontsize='x-large', weight='bold')
        ax.set_xlabel(xlabel, fontsize='large', fontstyle='italic')
        ax.set_ylabel(ylabel, fontsize='large', fontstyle='italic')
        if are_percentages:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
        plt.pause(0.1)
        return (fig, ax)

    def _create_interactive_scatter(self) -> tuple:
        """
        Creates a figure appropriate for use as an interactive scatter plot (can click plotted points for individual ID and info).
        """
        fig, ax = self._create_fig_ax()
        plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1)
        def on_pick(event):
            counter = -1
            for scatter in self.scatters:
                counter += 1
                if event.artist == scatter:
                    ind = event.ind[0]
                    annotation = self.annotations[counter][ind]
                    visibility = not annotation.get_visible()
                    annotation.set_visible(visibility)
                    break

        fig.canvas.mpl_connect("pick_event", on_pick)
        return (fig, ax)

    def update_interactive_scatter(self, profit_data: list, drawdown_data: list, gen_id: int) -> None:
        """
        Updates and redraws scatter plot with final profit and drawdown data for a color-coded generation of chromosomes.
        """
        fig, ax = self.interactive_scatter
        colormap = matplotlib.cm.plasma
        scatter = ax.scatter(profit_data, drawdown_data,
                             color=colormap((self.scatter_counter % 5) / 5), picker=True)
        self.scatter_counter += 1
        self.scatters.append(scatter)
        self.annotations.append([])

        for i, (x_coord, y_coord) in enumerate(zip(profit_data, drawdown_data)):
            annotation = ax.annotate(f"Gen ID: {gen_id}\nPop ID: {i+1}\nProfit: {int(profit_data[i])}\nDrawdown: {int(drawdown_data[i])}", xy=(x_coord, y_coord), xytext=(
                38, 20), textcoords="offset points", bbox=dict(boxstyle="round", fc="w", fill=True), arrowprops=dict(arrowstyle="fancy"))
            annotation.set_visible(False)
            self.annotations[-1].append(annotation)
        plt.pause(0.1)

    def create_standard_scatter(self, profit_data: list, drawdown_data: list, title: str = "Profit vs. Drawdown", xlabel: str = "Profit", ylabel: str = "Drawdown") -> None:
        """
        Creates a standard non-interactive scatter for an undifferentiated group of chromosomal data. 
        """
        fig, ax = self._create_fig_ax(title, xlabel, ylabel)
        scatter = ax.scatter(profit_data, drawdown_data)
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
    plotter.update_interactive_scatter(profit_data_1, drawdown_data_1, 1)
    plotter.update_interactive_scatter(profit_data_2, drawdown_data_2, 2)
    plotter.update_interactive_scatter(profit_data_3, drawdown_data_3, 3)

    plotter.create_standard_scatter(profit_data_1, drawdown_data_1)
    plotter.create_standard_scatter(profit_data_2, drawdown_data_2)
    plotter.create_standard_scatter(profit_data_3, drawdown_data_3)

    # plotter.stop_losses_triggered(stop_loss_step_data, network_decision_data, (1, 2))

    print("Done")
