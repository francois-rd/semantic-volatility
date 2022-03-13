from redditcleaner import clean
from pmaw import PushshiftAPI
import pandas as pd
import logging
import re

from utils.pathing import EXPERIMENT_DIR, CACHE_DIR, RAW_DATA_DIR
from utils.pathing import makepath, ExperimentPaths
from utils.config import CommandConfigBase
from utils.timeline import TimelineConfig


class RedditDownloaderConfig(CommandConfigBase):
    def __init__(self, **kwargs):
        """
        Configs for the RedditDownloader class. Accepted kwargs are:

        experiment_dir: (type: Path-like, default: utils.pathing.EXPERIMENT_DIR)
            Directory (either relative to utils.pathing.EXPERIMENTS_ROOT_DIR or
            absolute) representing the currently-running experiment.

        cache_dir: (type: Path-like, default: utils.pathing.CACHE_DIR)
            Directory (either absolute or relative to 'experiment_dir') in which
            to store all the temporarily-cached download files.

        output_dir: (type: Path-like, default: utils.pathing.RAW_DATA_DIR)
            Directory (either absolute or relative to 'experiment_dir') in which
            to store all the output files.

        num_workers: (type: int, default: 1)
            The number of current threads to use for download.

        subreddits: (type: list, default: ["news"])
            A string list of subreddits to download.

        timeline_config: (type: dict, default: {})
            Timeline configurations to use. Any given parameters override the
            defaults. See utils.timeline.TimelineConfig for details.

        :param kwargs: optional configs to overwrite defaults (see above)
        """
        self.experiment_dir = kwargs.pop('experiment_dir', EXPERIMENT_DIR)
        self.cache_dir = kwargs.pop('cache_dir', CACHE_DIR)
        self.output_dir = kwargs.pop('output_dir', RAW_DATA_DIR)
        self.num_workers = kwargs.pop('num_workers', 1)
        self.subreddits = kwargs.pop('subreddits', ["news"])
        self.timeline_config = kwargs.pop('timeline_config', {})
        super().__init__(**kwargs)

    def make_paths_absolute(self):
        paths = ExperimentPaths(
            experiment_dir=self.experiment_dir,
            cache_dir=self.cache_dir,
            raw_data_dir=self.output_dir
        )
        self.experiment_dir = paths.experiment_dir
        self.cache_dir = paths.cache_dir
        self.output_dir = paths.raw_data_dir
        return self


class RedditDownloader:
    def __init__(self, config: RedditDownloaderConfig):
        """
        Downloads (the subset of) Reddit from PushShift.io specified by the
        given configs.

        :param config: see RedditDownloaderConfig for details
        """
        self.config = config
        self.timeline_config = TimelineConfig(**self.config.timeline_config)
        logging.getLogger().setLevel(logging.INFO)

    def run(self) -> None:
        for subreddit in self.config.subreddits:
            comments = self._download_comments(subreddit)
            subreddit_id = comments[0]['subreddit_id']
            comments = [self._prune_fields(c) for c in comments]
            self._save_comments(comments, subreddit, subreddit_id)

    def _download_comments(self, subreddit):
        count = 0
        while True:
            try:
                comments = PushshiftAPI(
                    num_workers=self.config.num_workers,
                    jitter='full'
                ).search_comments(
                    subreddit=subreddit,
                    after=self.timeline_config.start,
                    before=self.timeline_config.end,
                    mem_safe=True,
                    safe_exit=True,
                    filter_fn=self._filter_deleted_removed,
                    cache_dir=makepath(self.config.cache_dir, subreddit)
                )
            except Exception as e:
                count += 1
                logging.warning(f"Download failed. Count={count}. "
                                f"Exception type={type(e)}")
            else:
                break
        return list(comments)

    def _save_comments(self, comments, subreddit, subreddit_id):
        filename = "start={}-end={}-subreddit={}-subreddit_id={}.csv".format(
            self.timeline_config.start, self.timeline_config.end,
            subreddit, subreddit_id)
        filepath = makepath(self.config.output_dir, filename)
        df = pd.DataFrame(comments)
        df = df[df['body'].str.strip().astype(bool)]  # Remove empty strings.
        df.to_csv(filepath, index=False, columns=list(df.axes[1]))

    @staticmethod
    def _filter_deleted_removed(comment):
        if comment['author'] in ['[removed]', '[deleted]']:
            return False
        if comment['body'] in ['[removed]', '[deleted]']:
            return False
        return True

    @staticmethod
    def _prune_fields(comment):
        return {
            'author_fullname': comment['author_fullname'],
            'body': RedditDownloader._clean_body(comment['body']),
            'created_utc': comment['created_utc']
        }

    @staticmethod
    def _clean_body(body):
        # Quote has a bug. Matches "gt" within a word, not just "&gt;". The
        # simple fix employed here (not sure how robust it really is) is to
        # ensure there isn't a word character before it.
        cleanish = clean(body, quote=False)
        return re.sub(r'\W\"?\\?&?gt;?', '', cleanish)
