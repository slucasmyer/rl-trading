import matplotlib.pyplot as plt
import matplotlib
import datetime as dt
import time
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from pyrecorder.recorder import Recorder
from pyrecorder.writers.video import Video


class Plotter():

    def __init__(self, queue: object, n_gen: int):
        self.queue = queue  # For IPC
        self.max_gen = n_gen  # Num of gens to run NSGA-II for
        self.cmap = matplotlib.cm.viridis_r  # Colormap
        self.obj_outcomes = []  # Objective outcomes for training population across generations
        self.pareto_by_gen = []  # Pareto front for each generation's outcomes taken alone
        self.final_pareto_frontier = []  # Pareto front for training population's outcomes across generations
        self.previous_frontier = None  # Previous frontier scatter
        self.training_figs_axs = []  # Collection of plots to update while training/validating
        self.script_path = Path(__file__).parent  # Path script is run from

    def _create_fig_ax(self, title: str, dimensions: int = 2, xlabel: str = "Profit",
                       ylabel: str = "Drawdown", zlabel: str = "Trade Count", x_percentage: bool = True,
                       y_percentage: bool = True, z_percentage: bool = False, colorbar: bool = True) -> tuple:
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
            ax.set_zlabel(zlabel, fontsize='large', fontstyle='italic')
            if z_percentage:
                ax.zaxis.set_major_formatter(
                    matplotlib.ticker.PercentFormatter())

        ax.set_title(title, fontsize='x-large', weight='bold')
        ax.set_xlabel(xlabel, fontsize='large',
                      fontstyle='italic', labelpad=12)
        ax.set_ylabel(ylabel, fontsize='large', fontstyle='italic', labelpad=5)

        if x_percentage:
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
        if y_percentage:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())

        if colorbar:
            sm = matplotlib.cm.ScalarMappable(
                cmap=self.cmap, norm=matplotlib.colors.Normalize(vmin=1, vmax=self.max_gen))
            sm.set_array([])
            fig.colorbar(sm, ticks=[1, self.max_gen], aspect=12, pad=0.1, fraction=0.1, shrink=0.6,
                         orientation='vertical', ax=ax, label='Generation')

        return (fig, ax)

    def _create_training_plots(self):
        """
        Creates training plots. 
        """
        self.training_figs_axs.append(self._create_fig_ax(
            title="Population Outcomes", dimensions=3))
        self.training_figs_axs.append(self._create_fig_ax(
            title="Population Outcomes"))
        self.training_figs_axs.append(
            self._create_fig_ax(title="Current Pareto Front (Gen 0)", dimensions=3))
        self.previous_frontier = self.training_figs_axs[2][1].scatter(
            [], [])  # For removal logic

    def _create_validation_plots(self, x_data: list, y_data: list, z_data: list):
        """
        Generates scatter of validation outcomes for candidate solutions. 
        """
        fig_3d, ax_3d = self._create_fig_ax(
            "Validation Outcomes", dimensions=3, colorbar=False)
        fig_2d, ax_2d = self._create_fig_ax(
            "Validation Outcomes", colorbar=False)
        ax_2d.scatter(x_data, y_data)
        ax_3d.scatter(x_data, y_data, z_data)
        fig_3d.canvas.draw()
        fig_2d.canvas.draw()
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fig_3d.savefig(self.set_path(self.script_path, f"Output/validation_results/ngen_{self.max_gen}",
                                     f"{timestamp}_validation_3D.png"))
        fig_2d.savefig(self.set_path(self.script_path, f"Output/validation_results/ngen_{self.max_gen}",
                                     f"{timestamp}_validation_2D.png"))

    def _update_training_plots(self, current_gen: int) -> None:
        """
        Updates training plots with outcomes from current gen's solutions.   
        """
        x_data, y_data, z_data = self.obj_outcomes[-1]
        x_par, y_par, z_par = zip(*self.final_pareto_frontier)
        x_par = [-x for x in x_par]
        z_par = [-z for z in z_par]
        normalized_gen = (current_gen-1) / self.max_gen
        self.training_figs_axs[0][1].scatter(
            x_data, y_data, z_data, color=self.cmap(normalized_gen))
        self.training_figs_axs[1][1].scatter(
            x_data, y_data, color=self.cmap(normalized_gen), alpha=0.6)
        self.previous_frontier.remove()
        self.previous_frontier = self.training_figs_axs[2][1].scatter(x_par, y_par, z_par, color=self.cmap(
            normalized_gen))
        self.training_figs_axs[2][1].set_title(
            f'Current Pareto Front (Gen {current_gen})', fontsize='x-large', weight='bold')

    def _create_training_outcomes_video(self):
        """
        Records video of each generation's performance on objectives.
        """
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with Recorder(Video(f"Assets/Videos/n_gen_{self.max_gen}_{timestamp}.mp4")) as rec:

            # For each training generation
            for i in range(self.max_gen):
                x_data, y_data, z_data = self.obj_outcomes[i]
                fig, ax = self._create_fig_ax(
                    f"Generation {i+1} Outcomes", colorbar=False)
                ax.scatter(x_data, y_data)
                # finally record the current visualization to the video
                rec.record()

    def calc_pareto_front(self, outcomes: list) -> list:
        """
        Returns the pareto front of the outcomes. 
        """
        pareto_front = []
        for point in outcomes:
            if not any(other_point is not point and
                       other_point[0] <= point[0] and
                       other_point[1] <= point[1] and
                       other_point[2] <= point[2] and
                       (other_point[0] < point[0] or other_point[1]
                           < point[1] or other_point[2] < point[2])
                       for other_point in outcomes):
                pareto_front.append(point)
        return pareto_front

    def set_path(self, script_path: Path, dir_path: str, file_path: str) -> Path:
        """
        Sets output path.
        """
        output_dir = script_path / Path(dir_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        new_path = output_dir / file_path
        return new_path

    def update_while_training(self):
        """
        Plots solutions' performance on objectives, generated by the training process.
        """
        try:
            matplotlib.use('TkAgg')
            plt.ion()
        except:
            print("\nContinuing without tkinter backend...\n")
        current_generation = 0
        self._create_training_plots()
        while current_generation <= self.max_gen:
            if not self.queue.empty():
                current_generation += 1
                if current_generation > self.max_gen:
                    continue
                gen_data = self.queue.get()

                # Update pareto collections
                self.pareto_by_gen.append(self.calc_pareto_front(gen_data))
                self.final_pareto_frontier.extend(self.pareto_by_gen[-1])
                self.final_pareto_frontier = self.calc_pareto_front(
                    self.final_pareto_frontier)
                # Transform data for plotting since trade count/profit negated for min optimization
                x_data, y_data, z_data = zip(*gen_data)
                x_data = [-x for x in x_data]
                z_data = [-z for z in z_data]
                self.obj_outcomes.append((x_data, y_data, z_data))
                self._update_training_plots(current_generation)

            for figure in self.training_figs_axs:
                figure[0].canvas.flush_events()
            time.sleep(0.1)

        # Auto-save training figures at the end of training and record video
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        counter = 0
        for fig_ax in self.training_figs_axs:
            counter += 1
            fig_ax[0].savefig(self.set_path(self.script_path,
                                            f"Output/performance_log/ngen_{self.max_gen}",
                                            f"{timestamp}_training_{counter}.png"))
        try:
            self._create_training_outcomes_video()
        except:
            print("\nContinuing without recording training video...\n")

        # Generate validation scatters
        validation_results = self.queue.get()
        x_data, y_data, z_data = zip(*[(x, y, z)
                                     for x, y, z, *r in validation_results])
        self._create_validation_plots(x_data, y_data, z_data)

        print("\nClose figures to continue...\n")
        plt.show(block=True)
        plt.ioff()
