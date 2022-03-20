from commands.core import CommandBase
from model.time_series import TimeSeries, TimeSeriesConfig


class TimeSeriesCommand(CommandBase):
    @property
    def config_class(self):
        return TimeSeriesConfig

    def start(self, config: TimeSeriesConfig, parser_args):
        TimeSeries(config).run()
