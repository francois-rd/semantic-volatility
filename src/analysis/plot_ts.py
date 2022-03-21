import matplotlib.pyplot as plt
import numpy as np
import random
import pickle

from utils.pathing import (
    makepath,
    ExperimentPaths,
    EXPERIMENT_DIR,
    TIME_SERIES_DIR,
    PLOT_TS_DIR,
    SURVIVING_FILE,
    DYING_FILE
)
from model.time_series import TimeSeriesTypes
from utils.config import CommandConfigBase


class PlotTimeSeriesConfig(CommandConfigBase):
    def __init__(self, **kwargs):
        """
        Configs for the PlotTimeSeries class. Accepted kwargs are:

        experiment_dir: (type: Path-like, default: utils.pathing.EXPERIMENT_DIR)
            Directory (either relative to utils.pathing.EXPERIMENTS_ROOT_DIR or
            absolute) representing the currently-running experiment.

        input_dir: (type: Path-like, default: utils.pathing.TIME_SERIES_DIR)
            Directory (either absolute or relative to 'experiment_dir') from
            which to read the time series data.

        surviving_file: (type: str, default: utils.pathing.SURVIVING_FILE)
            Path (relative to 'input_dir') of the surviving new word time series
            file.

        dying_file: (type: str, default: utils.pathing.DYING_FILE)
            Path (relative to 'input_dir') of the dying new word time series
            file.

        output_dir: (type: Path-like, default: utils.pathing.PLOT_TS_DIR)
            Directory (either absolute or relative to 'experiment_dir') in which
            to store all the output files.

        num_anecdotes: (type: int, default: 5)
            Number of anecdotal words to randomly sample for plotting.

        :param kwargs: optional configs to overwrite defaults (see above)
        """
        self.experiment_dir = kwargs.pop('experiment_dir', EXPERIMENT_DIR)
        self.input_dir = kwargs.pop('input_dir', TIME_SERIES_DIR)
        self.surviving_file = kwargs.pop('surviving_file', SURVIVING_FILE)
        self.dying_file = kwargs.pop('dying_file', DYING_FILE)
        self.output_dir = kwargs.pop('output_dir', PLOT_TS_DIR)
        self.num_anecdotes = kwargs.pop('num_anecdotes', 5)
        super().__init__(**kwargs)

    def make_paths_absolute(self):
        paths = ExperimentPaths(
            experiment_dir=self.experiment_dir,
            time_series_dir=self.input_dir,
            plot_ts_dir=self.output_dir
        )
        self.experiment_dir = paths.experiment_dir
        self.input_dir = paths.time_series_dir
        self.surviving_file = makepath(self.input_dir, self.surviving_file)
        self.dying_file = makepath(self.input_dir, self.dying_file)
        self.output_dir = paths.plot_ts_dir
        return self


class PlotTimeSeries:
    def __init__(self, config: PlotTimeSeriesConfig):
        """
        Plots the computed time series representation of new word volatility in
        various ways.

        :param config: see PlotTimeSeriesConfig for details
        """
        self.config = config

    def run(self) -> None:
        surviving = self._do_run("Surviving", self.config.surviving_file)
        #dying = self._do_run("Dying", self.config.dying_file)
        #self._plot(surviving, dying)

    def _do_run(self, word_type, input_path):
        with open(input_path, 'rb') as file:
            all_time_series_by_word = pickle.load(file)
        self._plot_anecdotes(word_type, all_time_series_by_word)
        with_word_type = word_type, self._swap_keys(all_time_series_by_word)
        self._plot(with_word_type)
        return with_word_type

    def _plot_anecdotes(self, word_type, all_time_series_by_word):
        anecdotes = {k: all_time_series_by_word[k] for k in random.sample(
            list(all_time_series_by_word), self.config.num_anecdotes)}
        for ts_type in TimeSeriesTypes.ALL_TYPES:
            ts_type_name = ts_type['main_short_name']
            # All time series plotted in one graph.
            plt.figure()
            for word, time_series in anecdotes.items():
                for id_ in ['main', 'control']:
                    ts = time_series[ts_type[f'{id_}_id']]
                    label = word + (" (C)" if id_ == 'control' else "")
                    plt.plot(np.arange(len(ts)), ts, label=label)
            plt.title(f"{ts_type_name} Time Series for Randomly-Selected "
                      f"{word_type} Words")
            plt.xlabel("Time Index")
            plt.ylabel(ts_type['main_full_name'])
            plt.legend()
            filename = f"anecdotal-{word_type}-{ts_type_name}-all.pdf"
            plt.savefig(makepath(self.config.output_dir, filename))
            plt.close()

            # Each time series in its own graph.
            for word, time_series in anecdotes.items():
                plt.figure()
                for id_ in ['main', 'control']:
                    ts = time_series[ts_type[f'{id_}_id']]
                    label = ts_type[f'{id_}_short_name']
                    plt.plot(np.arange(len(ts)), ts, label=label)
                plt.title(f"{ts_type_name} Time Series for '{word}' "
                          f"({word_type})")
                plt.xlabel("Time Index")
                plt.ylabel(ts_type['main_full_name'])
                plt.legend()
                filename = f"anecdotal-{word_type}-{ts_type_name}-{word}.pdf"
                plt.savefig(makepath(self.config.output_dir, filename))
                plt.close()

    @staticmethod
    def _swap_keys(all_time_series_by_word):
        swapped = {}
        for time_series_for_word in all_time_series_by_word.values():
            for time_series_name, time_series in time_series_for_word.items():
                swapped.setdefault(time_series_name, []).append(time_series)
        return swapped

    def _plot(self, *args):
        # Mostly follow 'old', but looping args and looping type.
        for ts_type in TimeSeriesTypes.ALL_TYPES:
            ts_type_name = ts_type['main_short_name']
            plt.figure()
            for word_type, swapped_ts in args:
                for id_ in ['main', 'control']:
                    # Variable length input, so can't vectorize easily.
                    ts_list = swapped_ts[ts_type[f'{id_}_id']]
                    index, means, stds = 0, [], []
                    while True:
                        pool = []
                        for ts in ts_list:
                            if index < len(ts):
                                pool.append(ts[index])
                        if not pool:
                            break
                        means.append(np.mean(pool))
                        stds.append(np.std(pool))
                        index += 1
                    means = np.array(means)
                    stds = np.array(stds)

                    # Plot means and stds.
                    x = np.arange(len(means))
                    label = word_type + (" (C)" if id_ == 'control' else "")
                    mean_line, = plt.plot(x, means, label=label)
                    plt.fill_between(x, means - stds, means + stds,
                                     color=mean_line.get_color(), alpha=0.2)
            plt.title(f"{ts_type_name} Time Series")
            plt.xlabel("Time Index")
            plt.ylabel(ts_type['main_full_name'])
            plt.legend()
            filename = f"{'-'.join(wt for wt, _ in args)}-{ts_type_name}.pdf"
            plt.savefig(makepath(self.config.output_dir, filename))
            plt.close()
