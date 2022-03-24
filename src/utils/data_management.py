from collections import defaultdict
import pickle
import os


def parts(filename):
    parts_ = os.path.splitext(filename)[0].split("-")
    return dict(tuple(p.split("=")) for p in parts_)


def make(start, end, subreddit, subreddit_id):
    return "start={}-end={}-subreddit={}-subreddit_id={}.csv".format(
        start, end, subreddit, subreddit_id)


class RowFileMapper:
    def __init__(self):
        """
        Compactly creates a unique mapping between consecutive row indices of a
        sequence of filenames, and vice versa. Useful for when a direct 1:1 map
        would result in OOM errors.

        Use pattern:

        # Setting up the mapping.
        mapper = RowFileMapper()
        for filename in file_list:
            mapper.new_file(filename)
            for row in get_rows_from(filename):
                row_id = mapper.new_row_id()
                # Business logic involving row_id
                ...
        mapper.save('my_id_map.pickle')  # Always pickles.

        # Later reverse mapping.
        mapper = RowFileMapper.load('my_id_map.pickle')
        for row_id in row_id_list:
            filename, row_index = mapper.reverse(row_id)
            row = get_rows_from(filename)[row_index]
            # Business logic involving row
            ...
        """
        self.id = 0
        self.id_map = {}

    def new_file(self, filename):
        self.id_map[self.id] = filename

    def new_row_id(self):
        current_id = self.id
        self.id += 1
        return current_id

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.id_map, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        mapper = RowFileMapper()
        with open(filename, 'rb') as file:
            mapper.id_map = pickle.load(file)
        return mapper

    def reverse(self, row_id):
        for file_id in reversed(self.id_map.keys()):  # Dicts ordered in Py>3.6
            if file_id <= row_id:
                return self.id_map[file_id], row_id - file_id


def make_file_row_map(usage_dict_file, id_map_file):
    with open(usage_dict_file, 'rb') as file:
        usage_dict = pickle.load(file)
    file_row_map = defaultdict(lambda: defaultdict(list))
    mapper = RowFileMapper.load(id_map_file)
    for word, usage in usage_dict.items():
        for comment_id in usage[2]:
            file, row = mapper.reverse(comment_id)
            file_row_map[file][row].append(word)
    return file_row_map
