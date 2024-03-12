import matplotlib.pyplot as plt
import matplotlib
import datetime as dt
import time
from mpl_toolkits.mplot3d import Axes3D

# TO-DO: Add points to existing scatters rather than creating new ones. Replace naive pareto algo.
# Put all the axes in one figure at this point? Rename axes/functions for clarity. Clean stuff up.


class Plotter():

    def __init__(self, queue: object, n_gen: int):
        matplotlib.use('TkAgg')
        plt.ion()
        self.queue = queue
        self.max_gen = n_gen
        self.cmap = matplotlib.cm.viridis_r
        self.compl_pop_obj_data = []
        self.pareto_gen_data = []
        self.final_pareto_frontier = []
        self.previous_frontier = None
        self.convergence_figures = []

    def _create_fig_ax(self, title: str, dimensions: int = 2, xlabel: str = "Profit",
                       ylabel: str = "Drawdown", zlabel: str = "Trade Count", x_percentage: bool = True,
                       y_percentage: bool = True, z_percentage: bool = False) -> tuple:
        """
        Configures and returns base fig/ax for plots.
        """
        fig = plt.figure(figsize=(8, 6))
        if dimensions == 2:
            with plt.style.context('ggplot'):
                ax = fig.add_subplot()
                fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)
        else:
            ax = fig.add_subplot(projection='3d')
            fig.subplots_adjust(left=0, right=0.9, top=0.9, bottom=0.1)

        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['right'].set_color((.8, .8, .8))
        ax.spines['top'].set_color((.8, .8, .8))
        ax.set_title(title, fontsize='x-large', weight='bold')
        ax.set_xlabel(xlabel, fontsize='large',
                      fontstyle='italic', labelpad=10)
        ax.set_ylabel(ylabel, fontsize='large', fontstyle='italic', labelpad=5)

        if x_percentage:
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
        if y_percentage:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())

        sm = matplotlib.cm.ScalarMappable(
            cmap=self.cmap, norm=matplotlib.colors.Normalize(vmin=1, vmax=self.max_gen))
        sm.set_array([])
        fig.colorbar(sm, ticks=[1, self.max_gen], aspect=12, pad=0.1, fraction=0.1, shrink=0.6,
                     orientation='vertical', ax=ax, label='Generation')

        if dimensions == 3:
            ax.set_zlabel(zlabel, fontsize='large', fontstyle='italic')
            if z_percentage:
                ax.zaxis.set_major_formatter(
                    matplotlib.ticker.PercentFormatter())

        return (fig, ax)

    def _create_convergence_figures(self):
        self.convergence_figures.append(self._create_fig_ax(
            title="Profit vs. Drawdown vs. Trade Count", dimensions=3))
        self.convergence_figures.append(self._create_fig_ax(
            title="Profit vs. Drawdown", dimensions=2))
        self.convergence_figures.append(
            self._create_fig_ax(title="Current Pareto Front (Gen 0)", dimensions=3))
        self.previous_frontier = self.convergence_figures[2][1].scatter(
            [], [])  # For removal logic

    def _update_interactive_convergence_scatter(self, current_gen: int) -> None:
        """
        Updates convergence scatters with objective performance data for generation.
        """
        x_data, y_data, z_data = self.compl_pop_obj_data[-1]
        x_par, y_par, z_par = zip(*self.final_pareto_frontier)
        x_par = [-x for x in x_par]
        z_par = [-z for z in z_par]
        normalized_gen = (current_gen-1) / self.max_gen
        fig_3d, ax_3d = self.convergence_figures[0]
        fig_2d, ax_2d = self.convergence_figures[1]
        fig_par, ax_par = self.convergence_figures[2]
        ax_3d.scatter(x_data, y_data, z_data, color=self.cmap(normalized_gen))
        ax_2d.scatter(x_data, y_data, color=self.cmap(
            normalized_gen), alpha=0.6)
        self.previous_frontier.remove()
        self.previous_frontier = ax_par.scatter(x_par, y_par, z_par, color=self.cmap(
            normalized_gen))
        ax_par.set_title(
            f'Current Pareto Front (Gen {current_gen})', fontsize='x-large', weight='bold')

    def calc_pareto_set(self, gen_data: list):
        """
        Returns a list of non-dominated objective data points.
        """
        non_dominated = []
        for point in gen_data:
            if not any(other_point is not point and
                       other_point[0] <= point[0] and
                       other_point[1] <= point[1] and
                       other_point[2] <= point[2] and
                       (other_point[0] < point[0] or other_point[1]
                           < point[1] or other_point[2] < point[2])
                       for other_point in gen_data):
                non_dominated.append(point)
        return non_dominated

    def update_while_training(self):
        """
        Plots performance data on objectives, generated by the training process, on convergence scatters.
        """
        current_generation = 0
        self._create_convergence_figures()
        while current_generation < self.max_gen:
            if not self.queue.empty():
                current_generation += 1
                gen_data = self.queue.get()

                # Update pareto set data
                self.pareto_gen_data.append(self.calc_pareto_set(gen_data))
                self.final_pareto_frontier.extend(self.pareto_gen_data[-1])
                self.final_pareto_frontier = self.calc_pareto_set(
                    self.final_pareto_frontier)
                # Transform data for plotting since trade count/profit negated for min optimization
                x_data, y_data, z_data = zip(*gen_data)
                x_data = [-x for x in x_data]
                z_data = [-z for z in z_data]
                self.compl_pop_obj_data.append((x_data, y_data, z_data))
                self._update_interactive_convergence_scatter(
                    current_generation)

            for figure in self.convergence_figures:
                figure[0].canvas.flush_events()
            time.sleep(0.1)
            
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        counter = 0 
        for fig_ax in self.convergence_figures:
            counter += 1
            fig_ax[0].savefig(f"Assets/Images/{timestamp}_scatter_{counter}.png")
        plt.show(block=True)
        plt.ioff()

    def create_gen_scatter(self, title: str, dimensions: int, gen: int) -> None:
        """
        Generates scatter of performance data on objectives for passed generation. 
        """
        fig, ax = self._create_fig_ax(title, dimensions)
        x_data, y_data, z_data = self.compl_pop_obj_data[gen-1]
        normalized_gen = gen / self.max_gen
        if dimensions == 2:
            ax.scatter(x_data, y_data, color=self.cmap(normalized_gen))
        else:
            ax.scatter(x_data, y_data, z_data, color=self.cmap(normalized_gen))
        fig.canvas.draw()
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fig.savefig(f"Assets/Images/{timestamp}_scatter.png")

    # def stop_losses_triggered(stop_loss_data: list, network_decision_data: list, gen_id: int,
    # pop_id: int) -> None:
    #     """
    #     Delete? we shall see
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
