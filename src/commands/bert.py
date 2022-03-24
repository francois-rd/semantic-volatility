from transformers import set_seed

from commands.core import CommandBase
from model.bert import Bert, BertConfig


class BertCommand(CommandBase):
    @property
    def config_class(self):
        return BertConfig

    def start(self, config: BertConfig, parser_args):
        set_seed(parser_args.seed)
        Bert(config).run()
