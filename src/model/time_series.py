from sklearn.metrics.pairwise import cosine_similarity as cosim
import numpy as np
import random
import pickle

from utils.pathing import (
    makepath,
    ExperimentPaths,
    EXPERIMENT_DIR,
    EMBEDDINGS_DIR,
    TIME_SERIES_DIR,
    SURVIVING_FILE,
    DYING_FILE
)
from utils.timeline import TimelineConfig, Timeline
from utils.config import CommandConfigBase


class ShuffleOptions:
    FULL = 'full'
    NUM_SLICES = 'num_slices'
    SAME_SLICES = 'same_slices'

    @staticmethod
    def __options__():
        attr = ShuffleOptions.__dict__
        return [attr[k] for k in attr.keys() if not k.startswith('__')]


class TimeSeriesConfig(CommandConfigBase):
    def __init__(self, **kwargs):
        """
        Configs for the TimeSeries class. Accepted kwargs are:

        experiment_dir: (type: Path-like, default: utils.pathing.EXPERIMENT_DIR)
            Directory (either relative to utils.pathing.EXPERIMENTS_ROOT_DIR or
            absolute) representing the currently-running experiment.

        input_dir: (type: Path-like, default: utils.pathing.EMBEDDINGS_DIR)
            Directory (either absolute or relative to 'experiment_dir') from
            which to read the new word BERT embeddings.

        surviving_input_file: (type: str, default: utils.pathing.SURVIVING_FILE)
            Path (relative to 'input_dir') of the surviving new word embeddings
            file.

        dying_input_file: (type: str, default: utils.pathing.DYING_FILE)
            Path (relative to 'input_dir') of the dying new word embeddings
            file.

        output_dir: (type: Path-like, default: utils.pathing.TIME_SERIES_DIR)
            Directory (either absolute or relative to 'experiment_dir') in which
            to store all the output files.

        surviving_output_file: (type: str, default:
                utils.pathing.SURVIVING_FILE)
            Path (relative to 'output_dir') of the surviving new word time
            series output file.

        dying_output_file: (type: str, default: utils.pathing.DYING_FILE)
            Path (relative to 'output_dir') of the dying new word time series
            output file.

        shuffle_option: (type: str, default: 'same_slices')
            'full': All embeddings are randomly shuffled any time between the
                start and end points (see 'timeline_config'), and time slices
                are then computed from the result (see 'timeline_config').
            'num_slices': The number of slices are kept the same as in the
                chronological version, but both the slices' relative positions
                and the embeddings assigned to each slice are randomly changed.
            'same_slices': The slices are as in the chronological version, but
                the embeddings assigned to each slice are randomly changed.

        timeline_config: (type: dict, default: {})
            Timeline configurations to use. Any given parameters override the
            defaults. See utils.timeline.TimelineConfig for details.

        :param kwargs: optional configs to overwrite defaults (see above)
        """
        self.experiment_dir = kwargs.pop('experiment_dir', EXPERIMENT_DIR)
        self.input_dir = kwargs.pop('input_dir', EMBEDDINGS_DIR)
        self.surviving_input_file = kwargs.pop(
            'surviving_input_file', SURVIVING_FILE)
        self.dying_input_file = kwargs.pop('dying_input_file', DYING_FILE)
        self.output_dir = kwargs.pop('output_dir', TIME_SERIES_DIR)
        self.surviving_output_file = kwargs.pop(
            'surviving_output_file', SURVIVING_FILE)
        self.dying_output_file = kwargs.pop('dying_output_file', DYING_FILE)
        self.shuffle_option = kwargs.pop(
            'shuffle_option', ShuffleOptions.SAME_SLICES)
        self.timeline_config = kwargs.pop('timeline_config', {})
        super().__init__(**kwargs)

    def make_paths_absolute(self):
        paths = ExperimentPaths(
            experiment_dir=self.experiment_dir,
            embeddings_dir=self.input_dir,
            time_series_dir=self.output_dir
        )
        self.experiment_dir = paths.experiment_dir
        self.input_dir = paths.embeddings_dir
        self.surviving_input_file = makepath(
            self.input_dir, self.surviving_input_file)
        self.dying_input_file = makepath(self.input_dir, self.dying_input_file)
        self.output_dir = paths.time_series_dir
        self.surviving_output_file = makepath(
            self.output_dir, self.surviving_output_file)
        self.dying_output_file = makepath(
            self.output_dir, self.dying_output_file)
        return self


class TimeSeries:
    def __init__(self, config: TimeSeriesConfig):
        """
        Computes 'average pairwise cosine distance' and 'cosine distance between
        consecutive average embeddings' time series representations from the
        BERT embeddings of all novel words over the time slice specified by a
        Timeline. Both time series also have a control condition where the word
        occurrences are randomly shuffled, rather than sorted chronologically.

        :param config: see TimeSeriesConfig for details
        """
        assert config.shuffle_option in ShuffleOptions.__options__(), \
            "Unsupported shuffle option."

        self.config = config
        self.timeline = Timeline(TimelineConfig(**self.config.timeline_config))

    def run(self) -> None:
        config = self.config
        self._do_run(config.surviving_input_file, config.surviving_output_file)
        self._do_run(config.dying_input_file, config.dying_output_file)

    def _do_run(self, input_path, output_path):
        with open(input_path, 'rb') as file:
            embs = pickle.load(file)

        time_series = {word: self._process(embs[word]) for word in embs}
        with open(output_path, 'wb') as file:
            pickle.dump(time_series, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _process(self, word_embs):
        emb_length = word_embs[0][0].shape[0]
        time_slices = self._time_slices(word_embs)
        time_slices_random = self._shuffle(word_embs, time_slices)
        return {
            'apcd': self._process_apcd(time_slices),
            'apcd_control': self._process_apcd(time_slices_random),
            'cdcae': self._process_cdcae(time_slices, emb_length),
            'cdcae_control': self._process_cdcae(time_slices_random, emb_length)
        }

    def _time_slices(self, word_embs):
        time_slices = {}
        for emb, timestamp in word_embs:
            time_slice = self.timeline.slice_of(timestamp)
            time_slices.setdefault(time_slice, []).append(emb)
        return time_slices

    def _shuffle(self, word_embs, time_slices):
        start = self.config.timeline_config['start']
        end = self.config.timeline_config['end']
        if self.config.shuffle_option == ShuffleOptions.FULL:
            return self._time_slices(
                [(e, random.randint(start, end)) for e, _ in word_embs])
        elif self.config.shuffle_option == ShuffleOptions.NUM_SLICES:
            first = self.timeline.slice_of(start)
            last = self.timeline.slice_of(end)
            slices = random.sample(range(first, last + 1), len(time_slices))
            return self._random_assignment(word_embs, slices)
        elif self.config.shuffle_option == ShuffleOptions.SAME_SLICES:
            return self._random_assignment(word_embs, list(time_slices.keys()))
        else:
            raise NotImplementedError

    @staticmethod
    def _random_assignment(word_embs, slice_list):
        time_slices_rand = {k: [] for k in slice_list}

        # Make sure that each new random slice gets at least one embedding.
        rand_indices = random.sample(range(len(word_embs)), len(slice_list))
        for rand, time_slice in zip(rand_indices, time_slices_rand.values()):
            time_slice.append(word_embs[rand][0])

        # Randomly assign the remaining embeddings.
        for i, (emb, _) in enumerate(word_embs):
            if i not in rand_indices:  # Don't duplicate pre-assigned indices.
                time_slices_rand[random.choice(slice_list)].append(emb)
        return time_slices_rand

    @staticmethod
    def _process_apcd(time_slices):
        offset = min(time_slices)
        time_series = [0.0] * (max(time_slices) - offset + 1)
        for slice_id, slice_embs in time_slices.items():
            if len(slice_embs) == 1:
                continue  # Can't take pairwise distance of 1 item.
            upper_indices = np.triu_indices(len(slice_embs), 1)
            pairwise_cosims = cosim(slice_embs)[upper_indices]
            time_series[slice_id - offset] = pairwise_cosims.mean()
        return time_series

    @staticmethod
    def _process_cdcae(time_slices, emb_length):
        offset = min(time_slices)
        time_series = [[0.0] * emb_length] * (max(time_slices) - offset + 1)
        for slice_id, slice_embs in time_slices.items():
            time_series[slice_id - offset] = np.mean(slice_embs, axis=0)
        off_by_1 = zip(time_series, time_series[1:])
        return [cosim([i], [j])[0, 0] for i, j in off_by_1]
