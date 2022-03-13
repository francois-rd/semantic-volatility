from datetime import timedelta, datetime

from utils.config import warn_not_empty


class TimelineConfig:
    slice_size_options = {
        'minute': timedelta(minutes=1),
        'hour': timedelta(hours=1),
        'day': timedelta(days=1),
        'week': timedelta(weeks=1),
        'month': timedelta(days=30),
        'year': timedelta(days=365)
    }

    def __init__(self, **kwargs):
        """
        Configs for the Timeline class. Accepted kwargs are:

        start: (type: int, default: 0)
            Start of the time window in POSIX time (time since Epoch).

        end: (type: int, default: 1)
            End of the time window in POSIX time (time since Epoch).

        early: (type: int, default: 1)
            The number of time slices considered close to the start time.

        late: (type: int, default: 1)
            The number of time slices (counting back from the last) considered
            close to the end time.

        slice_size: (type: str, default: 'week')
            The time interval defining the size of each time slice.

        :param kwargs: optional configs to overwrite defaults (see above)
        """
        self.start = kwargs.pop('start', 0)
        self.end = kwargs.pop('end', 1)
        self.early = kwargs.pop('early', 1)
        self.late = kwargs.pop('late', 1)
        self.slice_size = kwargs.pop('slice_size', 'week')
        warn_not_empty(kwargs)

        if self.start >= self.end or self.start < 0:
            raise ValueError("must have 0 <= start < end")
        if self.early < 0 or self.late < 0:
            raise ValueError("both early and late must be non-negative")
        if self.slice_size not in self.slice_size_options.keys():
            msg = "`slice_size' must be one of: {}"
            raise ValueError(msg.format(self.slice_size_options.keys()))


class Timeline:
    def __init__(self, config: TimelineConfig):
        """
        Utilities for managing a time line and slices thereof.

        :param config: see TimelineConfig for details
        """
        self.config = config
        self.slice_size = self.config.slice_size_options[self.config.slice_size]
        self.early_cutoff = int((
            datetime.fromtimestamp(self.config.start)
            + self.config.early * self.slice_size
        ).timestamp())
        self.late_cutoff = int((
            datetime.fromtimestamp(self.config.end)
            - self.config.late * self.slice_size
        ).timestamp())
        self.start_datetime = datetime.fromtimestamp(self.config.start)

    def is_early(self, timestamp):
        return self.config.start <= timestamp <= self.early_cutoff

    def is_late(self, timestamp):
        return self.late_cutoff <= timestamp <= self.config.end

    def slice_of(self, timestamp):
        if timestamp < self.config.start or timestamp > self.config.end:
            raise ValueError("timestamp out of range")
        td = datetime.fromtimestamp(timestamp) - self.start_datetime
        return int(td / self.slice_size)
