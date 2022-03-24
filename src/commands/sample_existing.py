import random

from commands.core import CommandBase
from data.sample_existing import ExistingWordSampler, ExistingWordSamplerConfig


class ExistingWordSamplerCommand(CommandBase):
    @property
    def config_class(self):
        return ExistingWordSamplerConfig

    def start(self, config: ExistingWordSamplerConfig, parser_args):
        random.seed(parser_args.seed)
        ExistingWordSampler(config).run()
