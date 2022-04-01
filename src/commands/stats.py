from commands.core import CommandBase
from analysis.stats import PlotStats, PlotStatsConfig


class PlotStatsCommand(CommandBase):
    @property
    def config_class(self):
        return PlotStatsConfig

    def start(self, config: PlotStatsConfig, parser_args):
        PlotStats(config).run()
