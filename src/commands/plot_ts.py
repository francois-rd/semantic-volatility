from commands.core import CommandBase
from analysis.plot_ts import PlotTimeSeries, PlotTimeSeriesConfig


class PlotTimeSeriesCommand(CommandBase):
    @property
    def config_class(self):
        return PlotTimeSeriesConfig

    def start(self, config: PlotTimeSeriesConfig, parser_args):
        PlotTimeSeries(config).run()
