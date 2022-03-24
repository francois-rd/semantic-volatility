import pickle

from utils.pathing import (
    makepath,
    ExperimentPaths,
    EXPERIMENT_DIR,
    USAGES_DATA_DIR,
    EXIST_DATA_DIR,
    CAP_DATA_DIR,
    NEO_DATA_DIR,
    USAGE_DICT_FILE,
    CAP_FREQ_FILE,
    SURVIVING_FILE,
    DYING_FILE,
    EXISTING_FILE
)
from utils.timeline import TimelineConfig, Timeline
from utils.config import CommandConfigBase


class BasicDetectorConfig(CommandConfigBase):
    def __init__(self, **kwargs):
        """
        Configs for the BasicDetector class. Accepted kwargs are:

        experiment_dir: (type: Path-like, default: utils.pathing.EXPERIMENT_DIR)
            Directory (either relative to utils.pathing.EXPERIMENTS_ROOT_DIR or
            absolute) representing the currently-running experiment.

        input_dir: (type: Path-like, default: utils.pathing.USAGES_DATA_DIR)
            Directory (either absolute or relative to 'experiment_dir') from
            which to read all the word usage data.

        usage_file: (type: str, default: utils.pathing.USAGE_DICT_FILE)
            Path (relative to 'input_dir') of the usage dictionary input file.

        exist_data_dir: (type: Path-like, default: utils.pathing.EXIST_DATA_DIR)
            Directory (either absolute or relative to 'experiment_dir') from
            which to read the existing words sample auxiliary input file.

        existing_aux_file: (type: str, default: utils.pathing.EXISTING_FILE)
            Path (relative to 'exist_data_dir') of the randomly-sampled existing
            words auxiliary input file.

        cap_data_dir: (type: Path-like, default: utils.pathing.CAP_DATA_DIR)
            Directory (either absolute or relative to 'experiment_dir') from
            which to read the capitalization frequency auxiliary input file.

        cap_freq_file: (type: str, default: utils.pathing.CAP_FREQ_FILE)
            Path (relative to 'cap_data_dir') of the capitalization frequency
            auxiliary input file.

        output_dir: (type: Path-like, default: utils.pathing.NEO_DATA_DIR)
            Directory (either absolute or relative to 'experiment_dir') in which
            to store all the output files.

        surviving_file: (type: str, default: utils.pathing.SURVIVING_FILE)
            Path (relative to 'output_dir') of the detected surviving new words
            output file.

        dying_file: (type: str, default: utils.pathing.DYING_FILE)
            Path (relative to 'output_dir') of the detected dying new words
            output file.

        existing_output_file: (type: str, default: utils.pathing.EXISTING_FILE)
            Path (relative to 'output_dir') of the randomly-sampled existing
            words output file.

        min_usage_cutoff: (type: int, default: 1)
            The minimum number of occurrences of a word to be considered valid.
            Words occurring less often are considered noise.

        timeline_config: (type: dict, default: {})
            Timeline configurations to use. Any given parameters override the
            defaults. See utils.timeline.TimelineConfig for details.

        :param kwargs: optional configs to overwrite defaults (see above)
        """
        self.experiment_dir = kwargs.pop('experiment_dir', EXPERIMENT_DIR)
        self.input_dir = kwargs.pop('input_dir', USAGES_DATA_DIR)
        self.usage_file = kwargs.pop('usage_file', USAGE_DICT_FILE)
        self.exist_data_dir = kwargs.pop('exist_data_dir', EXIST_DATA_DIR)
        self.existing_aux_file = kwargs.pop('existing_aux_file', EXISTING_FILE)
        self.cap_data_dir = kwargs.pop('cap_data_dir', CAP_DATA_DIR)
        self.cap_freq_file = kwargs.pop('cap_freq_file', CAP_FREQ_FILE)
        self.output_dir = kwargs.pop('output_dir', NEO_DATA_DIR)
        self.surviving_file = kwargs.pop('surviving_file', SURVIVING_FILE)
        self.dying_file = kwargs.pop('dying_file', DYING_FILE)
        self.existing_output_file = kwargs.pop(
            'existing_output_file', EXISTING_FILE)
        self.min_usage_cutoff = kwargs.pop('min_usage_cutoff', 1)
        self.timeline_config = kwargs.pop('timeline_config', {})
        super().__init__(**kwargs)

    def make_paths_absolute(self):
        paths = ExperimentPaths(
            experiment_dir=self.experiment_dir,
            usages_data_dir=self.input_dir,
            exist_data_dir=self.exist_data_dir,
            cap_data_dir=self.cap_data_dir,
            neo_data_dir=self.output_dir
        )
        self.experiment_dir = paths.experiment_dir
        self.input_dir = paths.usages_data_dir
        self.usage_file = makepath(self.input_dir, self.usage_file)
        self.exist_data_dir = paths.exist_data_dir
        self.existing_aux_file = makepath(
            self.exist_data_dir, self.existing_aux_file)
        self.cap_data_dir = paths.cap_data_dir
        self.cap_freq_file = makepath(self.cap_data_dir, self.cap_freq_file)
        self.output_dir = paths.neo_data_dir
        self.surviving_file = makepath(self.output_dir, self.surviving_file)
        self.dying_file = makepath(self.output_dir, self.dying_file)
        self.existing_output_file = makepath(
            self.output_dir, self.existing_output_file)
        return self


class BasicDetector:
    def __init__(self, config: BasicDetectorConfig):
        """
        Detects novel words based on earliness and usage cutoffs and separates
        dying and surviving words based on a lateness cutoff. Also separates out
        the previously randomly-sampled existing words for later comparison.

        :param config: see BasicDetectorConfig for details
        """
        self.config = config
        self.timeline = Timeline(TimelineConfig(**self.config.timeline_config))

    def run(self) -> None:
        with open(self.config.usage_file, 'rb') as file:
            usage_dict = pickle.load(file)
        with open(self.config.cap_freq_file, 'rb') as file:
            cap_freq = pickle.load(file)
        with open(self.config.existing_aux_file, 'rb') as file:
            existing_words = pickle.load(file)

        # Detect new words.
        neologisms = dict((word, usage) for word, usage in usage_dict.items()
                          if not self.timeline.is_early(usage[0])
                          and len(usage[2]) >= self.config.min_usage_cutoff
                          and cap_freq[word] > 0  # Not >=
                          and word not in existing_words)

        # Split surviving vs dying new words.
        surviving, dying = {}, {}
        for word, usage in neologisms.items():
            if self.timeline.is_late(usage[1]):
                surviving[word] = usage
            else:
                dying[word] = usage
        self._save(surviving, self.config.surviving_file)
        self._save(dying, self.config.dying_file)

        # Store usages for randomly-sampled existing words.
        existing = dict((word, usage) for word, usage in usage_dict.items()
                        if len(usage[2]) >= self.config.min_usage_cutoff
                        and cap_freq[word] > 0  # Not >=
                        and word in existing_words)
        self._save(existing, self.config.existing_output_file)

    @staticmethod
    def _save(words, filename):
        with open(filename, 'wb') as file:
            pickle.dump(words, file, protocol=pickle.HIGHEST_PROTOCOL)
