from typing import Optional, List, Type  # Python 3.6 backwards compatibility.
import logging
import json

from utils.config import CommandConfigBase


class CommandBase:
    def __init__(self, subparser):
        """
        Base command from which all other commands derive. Serves as a main
        entry point into the program.

        :param subparser: an argparse subparser from which to build this command
        """
        subparser.add_argument(
            '--log', type=str, metavar='FILE', default="main.log",
            help="The log file to use. Defaults to 'main.log'."
        )
        subparser.add_argument(
            '--config', type=str, metavar='FILE',
            help="JSON-style config file. If this option is not given, the "
                 "default config values are used. If option is given, but FILE "
                 "does not already exist, the default config values are used "
                 "and also written to FILE for future reference. If this "
                 "option is given and FILE already exists, FILE is read and "
                 "its contents are used in place of the default configs."
        )
        subparser.add_argument(
            '--seed', type=int, metavar='SEED', default=314159,
            help="The random seed to use. Defaults to 314159."
        )
        subparser.add_argument(
            '--dry-run', action='store_true', default=False,
            help="If present, load config and then exit."
        )
        if self.other_parser_args is not None:
            for _args, _kwargs in self.other_parser_args:
                subparser.add_argument(*_args, **_kwargs)
        subparser.set_defaults(func=self._start_wrapper)

    @property
    def config_class(self) -> Type[CommandConfigBase]:
        """
        The command's configuration class.
        """
        raise NotImplementedError

    @property
    def other_parser_args(self) -> Optional[List[tuple]]:
        """
        Any other parser args to add that aren't the default. This should be a
        list of (*args, **kwargs) tuples, where the args and kwargs map to the
        parameters of the 'argparse.parser.add_argument()' method.
        """
        return None

    def start(self, config: CommandConfigBase, parser_args):
        """
        Start running this command from the given configs and parser args.

        :param config: an instance of this command's configuration class
        :param parser_args: all parser args for this command
        """
        raise NotImplementedError

    def _start_wrapper(self, parser_args):
        logging.basicConfig(filename=parser_args.log, level=logging.DEBUG,
                            format='%(asctime)s    %(levelname)s:%(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p')
        logging.info("====================== NEW RUN =========================")

        if parser_args.config is None:
            logging.info("No config. Using default.")
            config = self.config_class()
        else:
            try:
                with open(parser_args.config, 'r') as f:
                    logging.info("Config found. Attempting to load...")
                    config = self.config_class(**json.load(f))
                    logging.info("Loading successful. Config = {}".format(
                        json.dumps(config.__dict__)))
            except FileNotFoundError:
                logging.info("Provided config file doesn't exist. Provided "
                             "file name will be used to output default config.")
                config = self.config_class()
                with open(parser_args.config, 'w') as f:
                    # TODO: This is a bit of a hack, because __dict__ won't
                    #  necessarily only contain serializable values.
                    json.dump(config.__dict__, f, indent=4)

        if parser_args.dry_run:
            logging.info("Dry run. Exiting.")
        else:
            self.start(config.make_paths_absolute(), parser_args)
