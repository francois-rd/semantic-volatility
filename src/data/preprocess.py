from nltk.corpus import words
import pandas as pd
import spacy
import os
import re

from utils.pathing import makepath, ExperimentPaths
from utils.pathing import EXPERIMENT_DIR, RAW_DATA_DIR, PREPROC_DATA_DIR
from utils.config import CommandConfigBase


class RedditPreprocessorConfig(CommandConfigBase):
    def __init__(self, **kwargs):
        """
        Configs for the RedditPreprocessor class. Accepted kwargs are:

        experiment_dir: (type: Path-like, default: utils.pathing.EXPERIMENT_DIR)
            Directory (either relative to utils.pathing.EXPERIMENTS_ROOT_DIR or
            absolute) representing the currently-running experiment.

        input_dir: (type: Path-like, default: utils.pathing.RAW_DATA_DIR)
            Directory (either absolute or relative to 'experiment_dir') from
            which to read all the downloaded Reddit data.

        output_dir: (type: Path-like, default: utils.pathing.PREPROC_DATA_DIR)
            Directory (either absolute or relative to 'experiment_dir') in which
            to store all the output files.

        :param kwargs: optional configs to overwrite defaults (see above)
        """
        self.experiment_dir = kwargs.pop('experiment_dir', EXPERIMENT_DIR)
        self.input_dir = kwargs.pop('input_dir', RAW_DATA_DIR)
        self.output_dir = kwargs.pop('output_dir', PREPROC_DATA_DIR)
        super().__init__(**kwargs)

    def make_paths_absolute(self):
        paths = ExperimentPaths(
            experiment_dir=self.experiment_dir,
            raw_data_dir=self.input_dir,
            preproc_data_dir=self.output_dir
        )
        self.experiment_dir = paths.experiment_dir
        self.input_dir = paths.raw_data_dir
        self.output_dir = paths.preproc_data_dir
        return self


class RedditPreprocessor:
    def __init__(self, config: RedditPreprocessorConfig):
        """
        Preprocesses the (body of text from the) Reddit data using Spacy.

        :param config: see RedditPreprocessorConfig for details
        """
        self.config = config
        self.nlp = spacy.load("en_core_web_sm")
        self.words = set(word.lower() for word in words.words())

    def run(self) -> None:
        for root, _, files in os.walk(self.config.input_dir):
            for file in files:
                df = pd.read_csv(makepath(root, file))
                df['body'] = df['body'].map(self._clean)
                path = makepath(self.config.output_dir, "cleaned-" + file)
                df.dropna().to_csv(path, index=False, columns=list(df.axes[1]))

    def _clean(self, body):
        kept = []
        for token in self.nlp(body):
            if token.is_alpha and not token.is_stop:
                if token.lemma_ not in self.words:
                    # Collapse repeating letters to a maximum of 3.
                    kept.append(re.sub(r'(.)\1\1+', r'\1\1\1', token.lower_))
        return " ".join(kept) if kept else float('NaN')
