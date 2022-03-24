import random

from commands.core import CommandBase
from model.time_series import TimeSeries, TimeSeriesConfig


class TimeSeriesCommand(CommandBase):
    @property
    def config_class(self):
        return TimeSeriesConfig

    def start(self, config: TimeSeriesConfig, parser_args):
        random.seed(parser_args.seed)
        TimeSeries(config).run()
