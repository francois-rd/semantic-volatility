from nltk.corpus import words
import random
import pickle

from utils.pathing import (
    makepath,
    ExperimentPaths,
    EXPERIMENT_DIR,
    EXIST_DATA_DIR,
    EXISTING_FILE
)
from utils.config import CommandConfigBase


class ExistingWordSamplerConfig(CommandConfigBase):
    def __init__(self, **kwargs):
        """
        Configs for the ExistingWordSampler class. Accepted kwargs are:

        experiment_dir: (type: Path-like, default: utils.pathing.EXPERIMENT_DIR)
            Directory (either relative to utils.pathing.EXPERIMENTS_ROOT_DIR or
            absolute) representing the currently-running experiment.

        output_dir: (type: Path-like, default: utils.pathing.EXIST_DATA_DIR)
            Directory (either absolute or relative to 'experiment_dir') in which
            to store the randomly-sampled existing words.

        existing_file: (type: str, default: utils.pathing.EXISTING_FILE)
            Path (relative to 'output_dir') of the randomly-sampled existing
            words output file.

        num_existing: (type: int, default: 1000)
            The number of existing words to randomly sample.

        :param kwargs: optional configs to overwrite defaults (see above)
        """
        self.experiment_dir = kwargs.pop('experiment_dir', EXPERIMENT_DIR)
        self.output_dir = kwargs.pop('output_dir', EXIST_DATA_DIR)
        self.existing_file = kwargs.pop('existing_file', EXISTING_FILE)
        self.num_existing = kwargs.pop('num_existing', 1000)
        super().__init__(**kwargs)

    def make_paths_absolute(self):
        paths = ExperimentPaths(
            experiment_dir=self.experiment_dir,
            exist_data_dir=self.output_dir
        )
        self.experiment_dir = paths.experiment_dir
        self.output_dir = paths.exist_data_dir
        self.existing_file = makepath(self.output_dir, self.existing_file)
        return self


class ExistingWordSampler:
    def __init__(self, config: ExistingWordSamplerConfig):
        """
        Randomly sample a few existing words from a dictionary.

        :param config: see ExistingWordSamplerConfig for details
        """
        self.config = config
        self.words = set(word.lower() for word in words.words())

    def run(self) -> None:
        sample = set(random.sample(list(self.words), self.config.num_existing))
        with open(self.config.existing_file, 'wb') as file:
            pickle.dump(sample, file, protocol=pickle.HIGHEST_PROTOCOL)
