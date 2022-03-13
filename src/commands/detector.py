from commands.core import CommandBase
from data.detect import BasicDetector, BasicDetectorConfig


class BasicDetectorCommand(CommandBase):
    @property
    def config_class(self):
        return BasicDetectorConfig

    def start(self, config: BasicDetectorConfig, parser_args):
        BasicDetector(config).run()
