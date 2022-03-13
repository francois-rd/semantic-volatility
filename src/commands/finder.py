from commands.core import CommandBase
from data.find import WordUsageFinder, WordUsageFinderConfig


class WordUsageFinderCommand(CommandBase):
    @property
    def config_class(self):
        return WordUsageFinderConfig

    def start(self, config: WordUsageFinderConfig, parser_args):
        WordUsageFinder(config).run()
