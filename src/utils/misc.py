import pickle
import os


def parts_from_filename(file):
    parts = os.path.splitext(file)[0].split("-")
    if parts[0] == 'cleaned':
        parts[0] = "cleaned=True"
    else:
        parts.insert(0, "cleaned=False")
    return dict(part.split("=") for part in parts)


class ItemBlockMapper:
    def __init__(self):
        """
        Compactly maps consecutive Item IDs to (Block IDs, block index) pairs,
        and vice versa. Useful for when a direct 1:1 map would result in OOM.

        Use pattern:

        # Setting up the mapping.
        mapper = ItemBlockMapper()
        for block in block_list:
            mapper.new_block(block.id)
            for item in block:
                item_id = mapper.new_item_id()
                # Business logic involving item_id
                ...
        mapper.save('my_id_map.pickle')  # Always pickles.

        # Later reverse mapping.
        mapper = ItemBlockMapper.load('my_id_map.pickle')
        for item_id in item_id_list:
            block_id, block_index = mapper.get_block_id(item_id)
            item = block_list[block_id][block_index]
            # Business logic involving item
            ...
        """
        self.id = 0
        self.id_map = {}

    def new_block(self, block_id):
        self.id_map[self.id] = block_id

    def new_item_id(self):
        current_id = self.id
        self.id += 1
        return current_id

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.id_map, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        mapper = ItemBlockMapper()
        with open(filename, 'rb') as file:
            mapper.id_map = pickle.load(file)
        return mapper

    def get_block_id(self, item_id):
        for block_id in reversed(self.id_map.keys()):  # Dicts ordered in Py>3.6
            if block_id <= item_id:
                return self.id_map[block_id], item_id - block_id
