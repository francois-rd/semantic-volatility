import pandas as pd
import pickle
import math
import os

from utils.pathing import (
    makepath,
    ExperimentPaths,
    EXPERIMENT_DIR,
    PREPROC_DATA_DIR,
    USAGES_DATA_DIR,
    USAGE_DICT_FILE,
    ID_MAP_FILE
)
from utils.config import CommandConfigBase
from utils.data_management import RowFileMapper


class WordUsageFinderConfig(CommandConfigBase):
    def __init__(self, **kwargs):
        """
        Configs for the WordUsageFinder class. Accepted kwargs are:

        experiment_dir: (type: Path-like, default: utils.pathing.EXPERIMENT_DIR)
            Directory (either relative to utils.pathing.EXPERIMENTS_ROOT_DIR or
            absolute) representing the currently-running experiment.

        input_dir: (type: Path-like, default: utils.pathing.PREPROC_DATA_DIR)
            Directory (either absolute or relative to 'experiment_dir') from
            which to read all the preprocessed Reddit data.

        output_dir: (type: Path-like, default: utils.pathing.USAGES_DATA_DIR)
            Directory (either absolute or relative to 'experiment_dir') in which
            to store all the output files.

        usage_file: (type: str, default: utils.pathing.USAGE_DICT_FILE)
            Path (relative to 'output_dir') of the usage dictionary output file.

        map_file: (type: str, default: utils.pathing.ID_MAP_FILE)
            Path (relative to 'output_dir') of the usage ID map output file.

        :param kwargs: optional configs to overwrite defaults (see above)
        """
        self.experiment_dir = kwargs.pop('experiment_dir', EXPERIMENT_DIR)
        self.input_dir = kwargs.pop('input_dir', PREPROC_DATA_DIR)
        self.output_dir = kwargs.pop('output_dir', USAGES_DATA_DIR)
        self.usage_file = kwargs.pop('usage_file', USAGE_DICT_FILE)
        self.map_file = kwargs.pop('map_file', ID_MAP_FILE)
        super().__init__(**kwargs)

    def make_paths_absolute(self):
        paths = ExperimentPaths(
            experiment_dir=self.experiment_dir,
            preproc_data_dir=self.input_dir,
            usages_data_dir=self.output_dir
        )
        self.experiment_dir = paths.experiment_dir
        self.input_dir = paths.preproc_data_dir
        self.output_dir = paths.usages_data_dir
        self.usage_file = makepath(self.output_dir, self.usage_file)
        self.map_file = makepath(self.output_dir, self.map_file)
        return self


class WordUsageFinder:
    def __init__(self, config: WordUsageFinderConfig):
        """
        Creates a word-usage dictionary from the preprocessed Reddit data.

        :param config: see WordUsageFinderConfig for details
        """
        self.config = config
        self.word_usage = {}
        self.mapper = RowFileMapper()

    def run(self) -> None:
        for root, _, files in os.walk(self.config.input_dir):
            for file in files:
                self.mapper.new_file(file)
                df = pd.read_csv(makepath(root, file))
                for b, c in zip(df['body'], df['created_utc']):
                    self._process(b, c)
        with open(self.config.usage_file, 'wb') as file:
            pickle.dump(self.word_usage, file, protocol=pickle.HIGHEST_PROTOCOL)
        self.mapper.save(self.config.map_file)

    def _process(self, body, created):
        comment_id = self.mapper.new_row_id()  # Must happen before isnan().
        if isinstance(body, float) and math.isnan(body):
            return
        for word in body.split():
            usage = self.word_usage.setdefault(word, [float('inf'), 0, []])
            usage[0] = min(created, usage[0])  # First usage.
            usage[1] = max(created, usage[1])  # Last usage.
            usage[2].append(comment_id)  # List of all usages.
