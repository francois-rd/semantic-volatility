import logging


def warn_not_empty(kwargs):
    """
    Logs a warning if the given kwargs is not empty.
    """
    if kwargs:
        logging.warning("Unexpected kwargs: {}".format(list(kwargs.keys())))


class CommandConfigBase:
    def __init__(self, **kwargs):
        """
        Base class for configurations relating to top-level program commands.

        :param kwargs: optional configs to overwrite defaults
        """
        warn_not_empty(kwargs)

    def make_paths_absolute(self) -> 'CommandConfigBase':
        """
        Resolves and replaces any relative paths in this configuration with the
        equivalent absolute path.

        :return: this object
        """
        return self
