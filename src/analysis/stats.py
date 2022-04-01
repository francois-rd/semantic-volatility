from scipy.stats import spearmanr, ttest_ind_from_stats
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import logging
import pickle
import math

from utils.pathing import (
    makepath,
    ensure_path,
    ExperimentPaths,
    EXPERIMENT_DIR,
    TIME_SERIES_DIR,
    STATS_DIR,
    SURVIVING_FILE,
    DYING_FILE,
    EXISTING_FILE
)
from utils.timeline import TimelineConfig, Timeline
from model.time_series import TimeSeriesTypes
from utils.config import CommandConfigBase


class PlotStatsConfig(CommandConfigBase):
    def __init__(self, **kwargs):
        """
        Configs for the PlotStats class. Accepted kwargs are:

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

        output_dir: (type: Path-like, default: utils.pathing.STATS_DIR)
            Directory (either absolute or relative to 'experiment_dir') in which
            to store all the output files.

        drop_last: (type: bool, default: True)
            Whether to drop the last time slice or not.

        major_x_ticks: (type: int, default: 0)
            If positive, the value to set for the major xticks in matplotlib.
            Otherwise, leave xticks to the default layout.

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
        self.output_dir = kwargs.pop('output_dir', STATS_DIR)
        self.drop_last = kwargs.pop('drop_last', True)
        self.major_x_ticks = kwargs.pop('major_x_ticks', 0)
        self.timeline_config = kwargs.pop('timeline_config', {})
        super().__init__(**kwargs)

    def make_paths_absolute(self):
        paths = ExperimentPaths(
            experiment_dir=self.experiment_dir,
            time_series_dir=self.input_dir,
            stats_dir=self.output_dir
        )
        self.experiment_dir = paths.experiment_dir
        self.input_dir = paths.time_series_dir
        self.surviving_file = makepath(self.input_dir, self.surviving_file)
        self.dying_file = makepath(self.input_dir, self.dying_file)
        self.existing_file = makepath(self.input_dir, self.existing_file)
        self.output_dir = paths.stats_dir
        return self


class PlotStats:
    def __init__(self, config: PlotStatsConfig):
        """
        Plots the computed time series representation of new word volatility in
        various ways.

        :param config: see PlotStatsConfig for details
        """
        self.config = config
        tl_config = TimelineConfig(**self.config.timeline_config)
        self.max_time_slice = Timeline(tl_config).slice_of(tl_config.end)
        self.max_time_slice -= tl_config.early
        if config.drop_last:
            self.max_time_slice -= 1
        self.slice_size = tl_config.slice_size
        self.n_tests = 0
        self.style = None

    def run(self) -> None:
        logging.getLogger().setLevel(logging.INFO)
        surv = self._do_run("Surviving", self.config.surviving_file)
        dying = self._do_run("Dying", self.config.dying_file)
        existing = self._do_run("Existing", self.config.existing_file)
        args = [surv, dying, existing]
        for ts_type in TimeSeriesTypes.ALL_TYPES:
            n_combs = len(list(itertools.combinations(range(len(args)), 2)))
            n_rows = (len(args) * 2 + n_combs)
            n_cols = self.max_time_slice + ts_type['offset'] - 1
            self.n_tests += n_rows * n_cols
        for ts_type in TimeSeriesTypes.ALL_TYPES:
            self._table(ts_type, *args)
        styles = ['seaborn-colorblind', 'seaborn-deep', 'dark_background']
        for style in styles:
            self.style = style  # Can't get this programmatically from context.
            with plt.style.context(style):
                for word_type in args:
                    self._plot(word_type)
                self._plot(*args)

    def _do_run(self, word_type, input_path):
        with open(input_path, 'rb') as file:
            all_time_series_by_word = pickle.load(file)
        self._maybe_drop_last(all_time_series_by_word)
        rhos = self._spearman(all_time_series_by_word)
        with_word_type = word_type, self._stats(rhos)
        return with_word_type

    def _maybe_drop_last(self, all_time_series_by_word):
        if self.config.drop_last:
            for time_series_for_word in all_time_series_by_word.values():
                for ts_type in TimeSeriesTypes.ALL_TYPES:
                    for id_ in ['main', 'control']:
                        t = time_series_for_word[ts_type[f'{id_}_id']]
                        if len(t) > self.max_time_slice + ts_type['offset'] + 1:
                            del t[-1]

    def _spearman(self, all_time_series_by_word):
        rhos, swapped = {}, self._swap_keys(all_time_series_by_word)
        for time_series_name, time_series_list in swapped.items():
            rhos[time_series_name] = []
            for time_series in time_series_list:
                rhos_for_k = []
                for k in range(1, len(time_series)):
                    rho = spearmanr(np.arange(k + 1), time_series[:k + 1])[0]
                    if math.isnan(rho):
                        logging.warning("Treating NaN rho as 0.0")
                        rho = 0.0
                    rhos_for_k.append(rho)
                rhos[time_series_name].append(rhos_for_k)
        return rhos

    @staticmethod
    def _stats(rhos):
        # Variable length input, so can't vectorize easily.
        stats = {}
        for ts_name, time_series_list in rhos.items():
            index, means, stds, nobs = 0, [], [], []
            while True:
                pool = []
                for time_series in time_series_list:
                    if index < len(time_series):
                        pool.append(time_series[index])
                if not pool:
                    break
                means.append(np.mean(pool))
                stds.append(np.std(pool))
                nobs.append(len(pool))
                index += 1
            stats[ts_name] = [np.array(means), np.array(stds), np.array(nobs)]
        return stats

    @staticmethod
    def _swap_keys(all_time_series_by_word):
        swapped = {}
        for time_series_for_word in all_time_series_by_word.values():
            for time_series_name, time_series in time_series_for_word.items():
                swapped.setdefault(time_series_name, []).append(time_series)
        return swapped

    def _finalize_plot(self, *, ylabel, title, filename, offset):
        plt.xticks(np.arange(1, self.max_time_slice + offset + 1))
        if self.config.major_x_ticks > 0:
            ax = plt.gca().xaxis
            ax.set_major_locator(MultipleLocator(self.config.major_x_ticks))
            ax.set_minor_locator(MultipleLocator(1))
        plt.yticks(np.arange(-1, 1.001, 0.5))
        plt.ylim(-1, 1)
        plt.xlabel(f"Time Index ({self.slice_size}s since first appearance)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        sub_dir = ensure_path(makepath(self.config.output_dir, self.style))
        plt.savefig(makepath(sub_dir, filename))
        plt.close()

    def _plot(self, *args):
        for ts_type in TimeSeriesTypes.ALL_TYPES:
            ts_type_name = ts_type['main_short_name']
            filename = f"{'-'.join(wt for wt, _ in args)}-{ts_type_name}.pdf"
            plt.figure()
            for word_type, stat_ts in args:
                for id_ in ['main', 'control']:
                    means, stds, _ = stat_ts[ts_type[f'{id_}_id']]
                    x = np.arange(1, len(means) + 1)
                    label = word_type + (" (C)" if id_ == 'control' else "")
                    color = plt.plot(x, means, label=label)[0].get_color()
                    plt.fill_between(x, means - stds, means + stds,
                                     color=color, alpha=0.2)
                    poly1d_fn = np.poly1d(np.polyfit(x, means, 1))
                    plt.plot(x, poly1d_fn(x), color=color, linestyle='dashed')
            self._finalize_plot(ylabel="Spearman's Rho", title=ts_type_name,
                                filename=filename, offset=ts_type['offset'])

    def _table(self, ts_type, *args):
        # Compare each word type's main stats to 0.
        index = []
        offset = ts_type['offset']
        data, all_index = [], []
        for word_type, stats_ts in args:
            index.append(word_type[0])
            self._ttest(stats_ts[ts_type['main_id']], offset, data)
        all_index.extend([("Zero", i) for i in index])

        # Compare each word type's main stats and control stats.
        for word_type, stats_ts in args:
            all_index.append(("Control", f"{word_type[0]} - {word_type[0]}(C)"))
            self._ttest(stats_ts[ts_type['main_id']], offset, data,
                        second_stats_ts=stats_ts[ts_type['control_id']])

        # Compare each pair of word types' main stats to each other.
        args_dict = {k[0]: v for k, v in args}
        for a, b in list(itertools.combinations(index, 2)):
            all_index.append(("Pair", f"{a} - {b}"))
            self._ttest(args_dict[a][ts_type['main_id']], offset, data,
                        second_stats_ts=args_dict[b][ts_type['main_id']])

        # Save table.
        filename = f"{ts_type['main_short_name']}.txt"
        cols = [(f"Time Index ({self.slice_size}s since first appearance)",
                 str(i)) for i in range(1, self.max_time_slice + offset + 1)]
        all_index = pd.MultiIndex.from_tuples(all_index)
        cols = pd.MultiIndex.from_tuples(cols)
        df = pd.DataFrame(data, index=all_index, columns=cols)
        self._save(df, offset, filename)

    def _ttest(self, stats_ts, offset, data, second_stats_ts=None):
        row_data = []
        if second_stats_ts is None:
            reps = len(stats_ts[0])
            second_stats_ts = [[0.0] * reps, [0.0] * reps, [2] * reps]
        for m1, s1, n1, m2, s2, n2 in zip(*stats_ts, *second_stats_ts):
            t = ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
            pval = t[1] * self.n_tests  # Bonferroni Correction.
            if pval < 0.001:
                star = "^{***}"
            elif pval < 0.01:
                star = "^{**}"
            elif pval < 0.05:
                star = "^{*}"
            else:
                star = ""
            row_data.append(f"{m1 - m2:.2f}{star}")
        padding = self.max_time_slice - len(row_data) + offset
        row_data.extend(["-"] * padding)
        data.append(row_data)

    def _save(self, df, offset, filename):
        df.style.applymap_index(
            lambda v: "rotatebox:{90}--rwrap;", level=0
        ).applymap_index(
            lambda v: "multicolumn:{1}{c}--rwrap;", level=1, axis=1
        ).applymap_index(
            lambda v: "textbf:--rwrap;", axis=1
        ).applymap_index(
            lambda v: "textbf:--rwrap;"
        ).to_latex(
            buf=makepath(self.config.output_dir, filename),
            column_format="cc|" + "d{2.5}" * (self.max_time_slice + offset),
            position="h",
            position_float="centering",
            hrules=True,
            multirow_align="c",
            multicol_align="c"
        )
