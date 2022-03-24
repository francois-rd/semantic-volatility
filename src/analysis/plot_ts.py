from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle

from utils.pathing import (
    makepath,
    ensure_path,
    ExperimentPaths,
    EXPERIMENT_DIR,
    TIME_SERIES_DIR,
    PLOT_TS_DIR,
    SURVIVING_FILE,
    DYING_FILE,
    EXISTING_FILE
)
from utils.timeline import TimelineConfig, Timeline
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

        existing_file: (type: str, default: utils.pathing.EXISTING_FILE)
            Path (relative to 'input_dir') of the existing word time series
            file.

        output_dir: (type: Path-like, default: utils.pathing.PLOT_TS_DIR)
            Directory (either absolute or relative to 'experiment_dir') in which
            to store all the output files.

        num_anecdotes: (type: int, default: 5)
            Number of anecdotal words to randomly sample for plotting.

        timeline_config: (type: dict, default: {})
            Timeline configurations to use. Any given parameters override the
            defaults. See utils.timeline.TimelineConfig for details.

        :param kwargs: optional configs to overwrite defaults (see above)
        """
        self.experiment_dir = kwargs.pop('experiment_dir', EXPERIMENT_DIR)
        self.input_dir = kwargs.pop('input_dir', TIME_SERIES_DIR)
        self.surviving_file = kwargs.pop('surviving_file', SURVIVING_FILE)
        self.dying_file = kwargs.pop('dying_file', DYING_FILE)
        self.existing_file = kwargs.pop('existing_file', EXISTING_FILE)
        self.output_dir = kwargs.pop('output_dir', PLOT_TS_DIR)
        self.num_anecdotes = kwargs.pop('num_anecdotes', 5)
        self.timeline_config = kwargs.pop('timeline_config', {})
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
        self.existing_file = makepath(self.input_dir, self.existing_file)
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
        tl_config = TimelineConfig(**self.config.timeline_config)
        self.max_time_slice = Timeline(tl_config).slice_of(tl_config.end)
        self.max_time_slice -= tl_config.early
        self.slice_size = tl_config.slice_size
        self.style = None

    def run(self) -> None:
        styles = ['seaborn-colorblind', 'seaborn-deep', 'dark_background']
        for style in styles:
            self.style = style   # Can't get this programmatically from context.
            with plt.style.context(style):
                surv = self._do_run("Surviving", self.config.surviving_file)
                #dying = self._do_run("Dying", self.config.dying_file)
                existing = self._do_run("Existing", self.config.existing_file)
                #self._plot(surv, dying, existing)
                self._plot(surv, existing)

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
        yticks = np.arange(0, 1.2, 0.2)
        for ts_type in TimeSeriesTypes.ALL_TYPES:
            # Define common variables once.
            ts_type_name = ts_type['main_short_name']
            ylabel = ts_type['main_full_name']
            filename = f"anecdotal-{word_type}-{ts_type_name}-"

            # All time series plotted in one graph.
            plt.figure()
            for word, time_series in anecdotes.items():
                c = None
                for id_ in ['main', 'control']:
                    ts = time_series[ts_type[f'{id_}_id']]
                    label = word + (" (C)" if id_ == 'control' else "")
                    x = np.arange(len(ts))
                    if id_ == 'main':
                        c = plt.plot(x, ts, label=label)[0].get_color()
                    else:
                        plt.plot(x, ts, label=label, color=c, linestyle='--')
            self._finalize_plot(
                yticks=yticks,
                ylabel=ylabel,
                title=f"{ts_type_name} Time Series for Randomly-Selected "
                      f"{word_type} Words",
                filename=filename + "all.pdf"
            )

            # Each time series in its own graph.
            for word, time_series in anecdotes.items():
                plt.figure()
                for id_ in ['main', 'control']:
                    ts = time_series[ts_type[f'{id_}_id']]
                    label = ts_type[f'{id_}_short_name']
                    plt.plot(np.arange(len(ts)), ts, label=label)
                self._finalize_plot(
                    yticks=yticks,
                    ylabel=ylabel,
                    title=f"{ts_type_name} Time Series for '{word}' "
                          f"({word_type})",
                    filename=filename + f"{word}.pdf"
                )

    @staticmethod
    def _swap_keys(all_time_series_by_word):
        swapped = {}
        for time_series_for_word in all_time_series_by_word.values():
            for time_series_name, time_series in time_series_for_word.items():
                swapped.setdefault(time_series_name, []).append(time_series)
        return swapped

    def _finalize_plot(self, *, yticks, ylabel, title, filename):
        plt.xticks(np.arange(self.max_time_slice + 1))
        plt.gca().xaxis.set_major_locator(MultipleLocator(5))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
        plt.yticks(yticks)
        plt.xlabel(f"Time Index ({self.slice_size}s since first appearance)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        sub_dir = ensure_path(makepath(self.config.output_dir, self.style))
        plt.savefig(makepath(sub_dir, filename))
        plt.close()

    def _plot(self, *args):
        yticks = np.arange(-0.2, 1.4, 0.2)
        for ts_type in TimeSeriesTypes.ALL_TYPES:
            ts_type_name = ts_type['main_short_name']
            ylabel = ts_type['main_full_name']
            title = f"{ts_type_name} Time Series"
            filename = f"{'-'.join(wt for wt, _ in args)}-{ts_type_name}.pdf"
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

                    # Plot means, stds, and linear regression line.
                    x = np.arange(len(means))
                    label = word_type + (" (C)" if id_ == 'control' else "")
                    color = plt.plot(x, means, label=label)[0].get_color()
                    plt.fill_between(x, means - stds, means + stds,
                                     color=color, alpha=0.2)
                    poly1d_fn = np.poly1d(np.polyfit(x, means, 1))
                    plt.plot(x, poly1d_fn(x), color=color, linestyle='dashed')
            self._finalize_plot(yticks=yticks, ylabel=ylabel,
                                title=title, filename=filename)
