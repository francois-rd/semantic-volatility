from commands.core import CommandBase
from data.download import RedditDownloader, RedditDownloaderConfig


class RedditDownloaderCommand(CommandBase):
    @property
    def config_class(self):
        return RedditDownloaderConfig

    def start(self, config: RedditDownloaderConfig, parser_args):
        RedditDownloader(config).run()
