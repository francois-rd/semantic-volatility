from nltk.corpus import words
import pandas as pd
import logging
import pickle
import spacy
import os
import re

from utils.pathing import (
    makepath,
    ExperimentPaths,
    EXPERIMENT_DIR,
    RAW_DATA_DIR,
    PREPROC_DATA_DIR,
    EXIST_DATA_DIR,
    CAP_DATA_DIR,
    EXISTING_FILE
)
from utils.config import CommandConfigBase
import utils.data_management as dm


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
            to store all the preprocessing output files.

        exist_data_dir: (type: Path-like, default: utils.pathing.EXIST_DATA_DIR)
            Directory (either absolute or relative to 'experiment_dir') from
            which to read the existing words sample auxiliary input file.

        existing_file: (type: str, default: utils.pathing.EXISTING_FILE)
            Path (relative to 'exist_data_dir') of the randomly-sampled existing
            words auxiliary input file.

        cap_data_dir: (type: Path-like, default: utils.pathing.CAP_DATA_DIR)
            Directory (either absolute or relative to 'experiment_dir') in which
            to store the capitalization frequency auxiliary output files.

        subreddits: (type: list, default: ["news"])
            A string list of subreddits to preprocess.

        :param kwargs: optional configs to overwrite defaults (see above)
        """
        self.experiment_dir = kwargs.pop('experiment_dir', EXPERIMENT_DIR)
        self.input_dir = kwargs.pop('input_dir', RAW_DATA_DIR)
        self.output_dir = kwargs.pop('output_dir', PREPROC_DATA_DIR)
        self.exist_data_dir = kwargs.pop('exist_data_dir', EXIST_DATA_DIR)
        self.existing_file = kwargs.pop('existing_file', EXISTING_FILE)
        self.cap_data_dir = kwargs.pop('cap_data_dir', CAP_DATA_DIR)
        self.subreddits = kwargs.pop('subreddits', ["news"])
        super().__init__(**kwargs)

    def make_paths_absolute(self):
        paths = ExperimentPaths(
            experiment_dir=self.experiment_dir,
            raw_data_dir=self.input_dir,
            preproc_data_dir=self.output_dir,
            exist_data_dir=self.exist_data_dir,
            cap_data_dir=self.cap_data_dir
        )
        self.experiment_dir = paths.experiment_dir
        self.input_dir = paths.raw_data_dir
        self.output_dir = paths.preproc_data_dir
        self.exist_data_dir = paths.exist_data_dir
        self.existing_file = makepath(self.exist_data_dir, self.existing_file)
        self.cap_data_dir = paths.cap_data_dir
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
        self.cap_freq = None
        with open(self.config.existing_file, 'rb') as file:
            self.existing = pickle.load(file)

    def run(self) -> None:
        for root, _, files in os.walk(self.config.input_dir):
            for file in files:
                if dm.parts(file)['subreddit'] in self.config.subreddits:
                    self.cap_freq = {}  # Can't use defaultdict. Need to pickle.
                    df = pd.read_csv(makepath(root, file))
                    df['body'] = df['body'].map(self._clean)
                    proc_path = makepath(self.config.output_dir, file)
                    df.to_csv(proc_path, index=False, columns=list(df.axes[1]))
                    cap_file = os.path.splitext(file)[0] + ".pickle"
                    cap_path = makepath(self.config.cap_data_dir, cap_file)
                    with open(cap_path, 'wb') as f:
                        pickle.dump(
                            self.cap_freq, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _clean(self, body):
        if isinstance(body, float):
            logging.warning(f"Body is float: '{body}'. Treating as NaN.")
            return float('NaN')
        kept = []
        for token in self.nlp(body):
            if token.is_alpha and not token.is_stop:
                lemma = token.lemma_
                if lemma not in self.words or lemma in self.existing:
                    value = -1 if token.shape_.startswith("Xx") else 1
                    self.cap_freq.setdefault(token.lower_, 0)
                    self.cap_freq[token.lower_] += value
                    # Collapse repeating letters to a maximum of 3.
                    kept.append(re.sub(r'(.)\1\1+', r'\1\1\1', token.lower_))
        return " ".join(kept) if kept else float('NaN')
