from transformers import DistilBertModel, DistilBertTokenizerFast
import pandas as pd
import numpy as np
import logging
import pickle
import torch
import re

from utils.pathing import (
    makepath,
    ExperimentPaths,
    EXPERIMENT_DIR,
    RAW_DATA_DIR,
    NEO_DATA_DIR,
    USAGES_DATA_DIR,
    EMBEDDINGS_DIR,
    SURVIVING_FILE,
    DYING_FILE,
    EXISTING_FILE,
    ID_MAP_FILE
)
from utils.data_management import make_file_row_map
from utils.config import CommandConfigBase

DEFAULT_TOK_AGG = 'mean'
DEFAULT_MODEL_NAME = 'distilbert-base-uncased'


def default_tok_agg(x):
    return np.mean(x, axis=0)


class BertConfig(CommandConfigBase):
    def __init__(self, **kwargs):
        """
        Configs for the Bert class. Accepted kwargs are:

        experiment_dir: (type: Path-like, default: utils.pathing.EXPERIMENT_DIR)
            Directory (either relative to utils.pathing.EXPERIMENTS_ROOT_DIR or
            absolute) representing the currently-running experiment.

        raw_data_dir: (type: Path-like, default: utils.pathing.RAW_DATA_DIR)
            Directory (either absolute or relative to 'experiment_dir') from
            which to read all the raw Reddit data.

        neo_data_dir: (type: Path-like, default: utils.pathing.NEO_DATA_DIR)
            Directory (either absolute or relative to 'experiment_dir') from
            which to read all the detected (surviving and dying) new words.

        surviving_neo_file: (type: str, default: utils.pathing.SURVIVING_FILE)
            Path (relative to 'neo_data_dir') to the detected surviving new
            words file.

        dying_neo_file: (type: str, default: utils.pathing.DYING_FILE)
            Path (relative to 'neo_data_dir') to the detected dying new words
            file.

        existing_neo_file: (type: str, default: utils.pathing.EXISTING_FILE)
            Path (relative to 'neo_data_dir') to the randomly-sampled existing
            words file.

        usages_dir: (type: Path-like, default: utils.pathing.USAGES_DATA_DIR)
            Directory (either absolute or relative to 'experiment_dir') from
            which to read 'map_file'.

        map_file: (type: str, default: utils.pathing.ID_MAP_FILE)
            Path (relative to 'usages_dir') to the usage ID map file.

        output_dir: (type: Path-like, default: utils.pathing.EMBEDDINGS_DIR)
            Directory (either absolute or relative to 'experiment_dir') in which
            to store all the output files.

        surviving_output_file: (type: str, default:
                utils.pathing.SURVIVING_FILE)
            Path (relative to 'output_dir') of the surviving new word BERT
            embeddings output file.

        dying_output_file: (type: str, default: utils.pathing.DYING_FILE)
            Path (relative to 'output_dir') of the dying new word BERT
            embeddings output file.

        existing_output_file: (type: str, default: utils.pathing.EXISTING_FILE)
            Path (relative to 'output_dir') of the existing words BERT
            embeddings output file.

        remove_urls: (type: bool, default: False)
            Whether to remove URLs (starting with "http") from the input before
            embedding.

        token_aggregator: (type: str, default: 'mean')
            Which aggregation function to apply to the aggregate the (possibly
            numerous) individual WordPiece token embeddings making up a word.
            Currently, only the default is supported.

        model_name: (type: str, default: 'distilbert-base-uncased')
            Which BERT model to use. Currently, only the default is supported.

        :param kwargs: optional configs to overwrite defaults (see above)
        """
        self.experiment_dir = kwargs.pop('experiment_dir', EXPERIMENT_DIR)
        self.raw_data_dir = kwargs.pop('raw_data_dir', RAW_DATA_DIR)
        self.neo_data_dir = kwargs.pop('neo_data_dir', NEO_DATA_DIR)
        self.surviving_neo_file = kwargs.pop(
            'surviving_neo_file', SURVIVING_FILE)
        self.dying_neo_file = kwargs.pop('dying_neo_file', DYING_FILE)
        self.existing_neo_file = kwargs.pop('existing_neo_file', EXISTING_FILE)
        self.usages_dir = kwargs.pop('usages_dir', USAGES_DATA_DIR)
        self.map_file = kwargs.pop('map_file', ID_MAP_FILE)
        self.output_dir = kwargs.pop('output_dir', EMBEDDINGS_DIR)
        self.surviving_output_file = kwargs.pop(
            'surviving_output_file', SURVIVING_FILE)
        self.dying_output_file = kwargs.pop('dying_output_file', DYING_FILE)
        self.existing_output_file = kwargs.pop(
            'existing_output_file', EXISTING_FILE)
        self.remove_urls = kwargs.pop('remove_urls', False)

        self.token_aggregator = kwargs.pop('token_aggregator', DEFAULT_TOK_AGG)
        if self.token_aggregator != DEFAULT_TOK_AGG:
            logging.warning(f"Unsupported value '{self.token_aggregator}' "
                            f"for 'token_aggregator'. Reverting to default.")
            self.token_aggregator = DEFAULT_TOK_AGG

        self.model_name = kwargs.pop('model_name', DEFAULT_MODEL_NAME)
        if self.model_name != DEFAULT_MODEL_NAME:
            logging.warning(f"Unsupported value '{self.model_name}' "
                            f"for 'model_name'. Reverting to default.")
            self.model_name = DEFAULT_MODEL_NAME

        super().__init__(**kwargs)

    def make_paths_absolute(self):
        paths = ExperimentPaths(
            experiment_dir=self.experiment_dir,
            raw_data_dir=self.raw_data_dir,
            neo_data_dir=self.neo_data_dir,
            usages_data_dir=self.usages_dir,
            embeddings_dir=self.output_dir
        )
        self.experiment_dir = paths.experiment_dir
        self.raw_data_dir = paths.raw_data_dir
        self.neo_data_dir = paths.neo_data_dir
        self.surviving_neo_file = makepath(
            self.neo_data_dir, self.surviving_neo_file)
        self.dying_neo_file = makepath(self.neo_data_dir, self.dying_neo_file)
        self.existing_neo_file = makepath(
            self.neo_data_dir, self.existing_neo_file)
        self.usages_dir = paths.usages_data_dir
        self.map_file = makepath(self.usages_dir, self.map_file)
        self.output_dir = paths.embeddings_dir
        self.surviving_output_file = makepath(
            self.output_dir, self.surviving_output_file)
        self.dying_output_file = makepath(
            self.output_dir, self.dying_output_file)
        self.existing_output_file = makepath(
            self.output_dir, self.existing_output_file)
        return self


class Bert:
    def __init__(self, config: BertConfig):
        """
        Computes BERT embeddings for each usage of each detected new word.

        :param config: see BertConfig for details
        """
        name = config.model_name
        assert name == DEFAULT_MODEL_NAME, "Unsupported model name."
        assert config.token_aggregator == DEFAULT_TOK_AGG, \
            "Unsupported token aggregator."

        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")

        self.config = config
        self.device = device
        self.token_aggregator_fn = default_tok_agg
        self.bert = DistilBertModel.from_pretrained(name)
        self.model_max_size = 512  # Not sure how to get this programmatically.
        self.bert.to(self.device)
        self.bert.eval()
        self.tok = DistilBertTokenizerFast.from_pretrained(name)

    def run(self) -> None:
        config = self.config
        self._do_run(config.surviving_neo_file, config.surviving_output_file)
        self._do_run(config.dying_neo_file, config.dying_output_file)
        self._do_run(config.existing_neo_file, config.existing_output_file)

    def _do_run(self, input_path, output_path):
        file_row_map = make_file_row_map(input_path, self.config.map_file)
        embs = {}  # Can't use defaultdict because we need to pickle after.
        for file, row_map in file_row_map.items():
            self._process_file(file, row_map, embs)
            logging.debug(f"Finished processing '{file}'")
        with open(output_path, 'wb') as file:
            pickle.dump(embs, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _process_file(self, file, row_map, embs):
        # NOTE: Instead of processing row-by-row, it would in principle be
        # faster to process in mini-batches. However, the logic to do some
        # become significantly more complex, and the GPU being used can only
        # handle batches sizes of 1-2 anyways, so it's not a significant
        # benefit. If you find this to be a bottleneck, feel free to optimize.
        df = pd.read_csv(makepath(self.config.raw_data_dir, file))
        for row_id, word_list in row_map.items():
            row = df.iloc[[row_id]]
            body, created_utc = row['body'].item(), row['created_utc'].item()
            if self.config.remove_urls:
                body = re.sub(r'http*\S+', ' ', body)  # Very basic URL check.
            slices = self._find_word_piece_token_slices(body, word_list)
            emb = self._embed(body)
            agg_emb = self._filter_and_aggregate_word_piece_emb(emb, slices)
            for word, word_agg_emb in zip(word_list, agg_emb):
                if word_agg_emb is None:
                    continue
                embs.setdefault(word, []).append((word_agg_emb, created_utc))

    def _find_word_piece_token_slices(self, body, word_list):
        tokens = self.tok.convert_ids_to_tokens(self.tok(body)['input_ids'])
        slices, indices = [], list(range(len(tokens)))
        for word in word_list:
            add, delete = None, None
            for start_idx_idx in range(len(indices)):
                start_index = indices[start_idx_idx]

                # First, we check the special case of being at the end of the
                # sequence of tokens, as we get an off-by-one error otherwise.
                if start_idx_idx == len(indices) - 1:
                    sub_token = tokens[start_index]
                    token = self.tok.convert_tokens_to_string(sub_token)
                    if word == token.lower():
                        # If it's a full match, we are done.
                        add = slice(start_index, start_index + 1)
                        delete = slice(start_idx_idx, start_idx_idx + 1)

                # Next, we consider matching just one token, but also lookahead
                # to the next tokens to try to find a full match. However, the
                # tokens in any found match must be consecutive in order to be
                # valid. We check these JIT.
                consecutive_check = start_index
                for end_idx_idx in range(start_idx_idx + 1, len(indices)):
                    # TODO: This part causes an off-by-one error if two words
                    #  are consecutive and the second word is in wordlist before
                    #  the first. Test example:
                    #    body = "I hope the plandemic scamdemic ends soon"
                    #    wordlist = ['scamdemic', 'plandemic']
                    #  Solution is not obvious. If we leave extra indices in,
                    #  then we might get multi-matches on the same spot for
                    #  single-word tokens. On top of that, it's not clear what
                    #  happens if we have three+ consecutive words in particular
                    #  orders. One approach is in the partial match below, we
                    #  can lookahead one extra position. But then it's not clear
                    #  how to record the 'add' and 'delete' correctly. There's
                    #  also an increased chance of missing even more corner
                    #  cases (eg. end of string) and accidental greedy matching.
                    #  So, an alternative solution (since these are rare events)
                    #  is to skip these cases to prevent further off-by-one
                    #  errors or corner cases later (including in other  files).

                    # TODO: Another issue is that telescoping words won't match.
                    #  This is impossible to fix: if we match the word against
                    #  a collapsed version of the string, we will not go far
                    #  enough (early stopping) if the end of the word is where
                    #  the repeat is. We might also get other strange false
                    #  matches. The real solution is to discount telescoping
                    #  words in preprocess. But it's too late for that at this
                    #  stage (the preprocessor takes too long to run).

                    # TODO: In both cases above, the solutions are difficult and
                    #  error prone, and therefore time consuming. So, we will
                    #  just implement the 'ignore' method instead and log the
                    #  ignoring in the log file for later inspection.
                    end_index = indices[end_idx_idx]
                    # Only continue if consecutive.
                    if consecutive_check + 1 == end_index:
                        consecutive_check = end_index
                    else:
                        break

                    # Get the string corresponding to the consecutive tokens.
                    sub_tokens = tokens[start_index:end_index]
                    string = self.tok.convert_tokens_to_string(sub_tokens)
                    string = string.lower()

                    if word == string:
                        # If it's a full match, we are done.
                        add = slice(start_index, end_index)
                        delete = slice(start_idx_idx, end_idx_idx)
                        break
                    elif word.startswith(string):
                        # If we get a partial match, we keep going.
                        continue
                    else:
                        # If we get no amount of match at all, we move on.
                        break
                if add is not None:
                    # A match was found. Break to exit loop and change list.
                    break
            if add is not None:
                # A match was found. Transfer indices and move to next word.
                slices.append(add)
                del indices[delete]
            else:
                # No match was found for the current word. Problem!
                logging.error(f"Cannot construct word '{word}' from tokens: "
                              f"'{tokens}'")
                slices.append(None)
        return slices

    def _embed(self, body):
        # NOTE: Instead of processing chunk-by-chunk, it would in principle be
        # faster to process in mini-batches. However, the the GPU being used can
        # only handle batches sizes of 1-2 anyways, so it's not a significant
        # benefit. If you find this to be a bottleneck, feel free to optimize.
        stride = int(self.model_max_size / 4)
        input_id_chunks = self.tok(
            body,
            max_length=self.model_max_size,  # These arguments are
            truncation=True,  # needed to force
            padding=True,  # the tokenizer to
            return_overflowing_tokens=True,  # chunk long inputs
            stride=stride,  # with an overlap.
            return_tensors='pt'
        )['input_ids']

        # Only 1 chunk. Optimize.
        if input_id_chunks.shape[0] == 1:
            with torch.no_grad():
                input_ids = input_id_chunks.to(self.device)
                return self.bert(input_ids).last_hidden_state.squeeze(0)

        # Otherwise, deal with chunks.
        with torch.no_grad():
            emb_chunks = []

            # Do all but the last.
            for i in range(input_id_chunks.shape[0] - 1):
                input_ids = input_id_chunks[i].to(self.device).unsqueeze(0)
                emb_chunk = self.bert(input_ids).last_hidden_state.squeeze(0)
                emb_chunks.append(emb_chunk)

            # Last one is padded. Remove padding.
            input_ids = input_id_chunks[-1].to(self.device)
            input_ids = input_ids[input_ids.nonzero(as_tuple=True)].unsqueeze(0)
            emb_chunk = self.bert(input_ids).last_hidden_state.squeeze(0)
            emb_chunks.append(emb_chunk)

            # Merge back into a single sequence.
            emb_avg_chunks = [emb_chunks[0]]
            for i in range(1, len(emb_chunks)):
                prev = emb_avg_chunks[i - 1][:-1, :]  # Get, then remove '[SEP]'
                curr = emb_chunks[i][1:, :]  # Get, then remove '[CLS]'
                last, first = prev[-stride:, :], curr[:stride, :]  # Overlap.
                prev[-stride:, :] = (last + first) / 2  # Average them.
                emb_avg_chunks[i - 1] = prev  # Replace with updated chunk.
                emb_avg_chunks.append(curr[stride:, :])  # Add new chunk's end.
            return torch.cat(emb_avg_chunks)

    def _filter_and_aggregate_word_piece_emb(self, emb, slices):
        agg_emb = []
        with torch.no_grad():
            for slice_ in slices:
                if slice_ is None:
                    agg = None
                else:
                    agg = self.token_aggregator_fn(emb[slice_, :].numpy())
                agg_emb.append(agg)
            return agg_emb
