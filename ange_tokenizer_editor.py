import os
import sys
import json
import time
import shutil
import functools
import statistics
import nltk
import regex as re
import numpy as np
import multiprocessing
from functools import wraps
from multiprocessing import Pool, cpu_count, Manager
from datasets import load_dataset, get_dataset_config_info, SplitInfo, Dataset, IterableDataset
from typing import (
    List, Set, Dict, DefaultDict, Generator, Any, Tuple, Optional, Iterable, Counter as CounterType, Callable
)
from collections import Counter, defaultdict, OrderedDict
from copy import deepcopy
import unicodedata
import itertools
import socket
import requests
import logging
import tempfile
import unittest
import pickle
import gzip
import math

# Try importing transformers for the benchmark section
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# --- Constants and Special Tokens ---
SUBWORD_PREFIX = "#"
PAD = "<|pad|>"
UNK = "<|unk|>"
BOS = "<|bos|>"
EOS = "<|eos|>"
MASK = "<|mask|>"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS, MASK]
WORD_PATTERN = r'\p{Alphabetic}+'
SPACE_PATTERN = r'\p{White_Space}'
CASE_SPLIT_PATTERN = re.compile(r'\p{Lu}\p{Ll}+|\p{Ll}+|\p{Lu}+(?=\p{Lu}\p{Ll}{2,})|\p{Lu}+|\p{Alphabetic}+', re.UNICODE)
UNICODE_PATTERN = re.compile(r'\p{Alphabetic}+|\p{White_Space}+|\p{M}|\p{N}|\p{P}|\p{S}|\p{C}|.', re.UNICODE)
WHITE_SPACE_CHARS = {" ", "\n", "\t", "\r", "\f", "\v"}
SUPPORTED_LANGUAGES = {
    "eng": "retrieve_valid_english_words",
}

# Robust filtering thresholds for new morphemes
# 1. A morpheme's total frequency must be at least this high.
FREQUENCY_LIMIT = 1000
MIN_TOTAL_FREQUENCY = 100
MIN_STEM_VARIETY = 4
WORD_SIZE_LIMIT = 6
MIN_SUBWORD_LEN = 4
MMAP_MIN_WORD_LENGTH = 9
CORPUS_FREQ_SCALE = 100000  # The max value to use when scaling the corpus values
MIN_WORD_FREQUENCY = CORPUS_FREQ_SCALE * 0.25
MIN_WORD_FREQ = CORPUS_FREQ_SCALE * 0.35  # Words with frequency greater than 35% of corpus max frequency
HIGH_WORD_FREQ = CORPUS_FREQ_SCALE * 0.425
DEFAULT_TOP_K_WORDS = 2000000  # The number of most frequent words not in vocab to pre-tokenize.
MAX_VOCAB_BATCH_SIZE = 300000  # 1000000 examples is about 2GB of text data per batch
MAX_FREQUENCY_SPLIT = 20480
MAX_LANG_VOCAB_SIZE = 75000
SUBWORD_FREQ_LIMIT = CORPUS_FREQ_SCALE * 0.40
MAX_DICT_WORD_LEN = 20
MAX_WORD_LEN = 50
# Flow Control
MAX_CONCURRENT_DOWNLOADS = 10
LINES_PER_CHUNK = 50000

# --- Load Initial Morphemes (Global Scope) ---
PREFIXES, SUFFIXES, ROOT_WORDS = set(), set(), set()

# --- Pre-computed constants for efficiency ---
# Build a comprehensive list of characters to DELETE.
# This includes BOM, Zero-Width-Spaces/Joiners, BiDi markers,
# variation selectors, and other 'Cf' (Format) characters.
DELETION_CHARS = (
    '\u00ad'  # Soft Hyphen
    '\u180e'  # Mongolian Vowel Separator
    '\u200b'  # Zero Width Space
    '\u200c'  # Zero Width Non-Joiner
    '\u200d'  # Zero Width Joiner
    '\u200e'  # Left-to-Right Mark
    '\u200f'  # Right-to-Left Mark
    '\u202a'  # Left-to-Right Embedding
    '\u202b'  # Right-to-Left Embedding
    '\u202c'  # Pop Directional Formatting
    '\u202d'  # Left-to-Right Override
    '\u202e'  # Right-to-Left Override
    '\u2060'  # Word Joiner
    '\u2061'  # Function Application
    '\u2062'  # Invisible Times
    '\u2063'  # Invisible Separator
    '\u2064'  # Invisible Plus
    '\u206A\u206B\u206C\u206D\u206E\u206F' # Deprecated format chars
    '\ufeff'  # Byte Order Mark (BOM)
    '\ufff9'  # Interlinear Annotation Anchor
    '\ufffa'  # Interlinear Annotation Separator
    '\ufffb'  # Interlinear Annotation Terminator
    # Range of variation selectors
    '\ufe00\ufe01\ufe02\ufe03\ufe04\ufe05\ufe06\ufe07\ufe08\ufe09\ufe0a\ufe0b\ufe0c\ufe0d\ufe0e\ufe0f'
)

# Build a comprehensive list of characters to REPLACE with a space (U+0020).
# This includes all C0 and C1 control characters (except TAB U+0009,
# which is handled by .replace()) and Unicode line/paragraph separators.

# C0 Controls (U+0000 to U+001F)
# Excludes U+0009 (TAB)
C0_CONTROLS_TO_SPACE = (
    '\u0000\u0001\u0002\u0003\u0004\u0005\u0006\u0007\u0008'
    '\u000A\u000B\u000C\u000D\u000E\u000F'
    '\u0010\u0011\u0012\u0013\u0014\u0015\u0016\u0017'
    '\u0018\u0019\u001A\u001B\u001C\u001D\u001E\u001F'
)

# C1 Controls (U+007F to U+009F)
C1_CONTROLS_TO_SPACE = (
    '\u007F'
    '\u0080\u0081\u0082\u0083\u0084\u0085\u0086\u0087\u0088\u0089\u008A\u008B\u008C\u008D\u008E\u008F'
    '\u0090\u0091\u0092\u0093\u0094\u0095\u0096\u0097\u0098\u0099\u009A\u009B\u009C\u009D\u009E\u009F'
)

# Unicode line breaks not in C0/C1
UNICODE_BREAKS_TO_SPACE = (
    '\u2028'  # Line Separator
    '\u2029'  # Paragraph Separator
)

REPLACEMENT_CHARS = C0_CONTROLS_TO_SPACE + C1_CONTROLS_TO_SPACE + UNICODE_BREAKS_TO_SPACE

# Create the single, efficient translation map (once, at module level)
_ROBUST_TRANSLATION_MAP = str.maketrans(
    REPLACEMENT_CHARS,
    ' ' * len(REPLACEMENT_CHARS),
    DELETION_CHARS
)
# Get the directory where the script is located
try:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for interactive environments (like notebooks)
    ROOT_DIR = os.getcwd()

LOG_DIR = os.path.join(ROOT_DIR, "logs")
STORAGE_DIR = os.path.join(ROOT_DIR, "vocab_files")
VOCAB_FILE = os.path.join(STORAGE_DIR, 'vocab.json')
GREEDY_FILE = os.path.join(STORAGE_DIR, 'greedy_splits.json')
PERFECT_FILE = os.path.join(STORAGE_DIR, 'perfect_splits.json')
MORPHEME_FILE = os.path.join(STORAGE_DIR, 'morphemes.json')
FILTERED_WORD_FREQ_FILE = os.path.join(STORAGE_DIR, 'dict_word_frequencies.json')
MORPHEME_FREQ_FILE_NAME = 'morpheme_frequencies.json'
WORD_REGISTRY_FILE_NAME = 'word_registry.json'
WORD_FREQ_FILE_NAME = 'word_frequencies.json'
CONFIG_FILE_NAME = 'tokenizer_config.json'
VOCAB_FILE_NAME = 'vocab.json'
DICTIONARIES = "dictionaries"
PREFIX_SIZE = "prefix_size"
SUFFIX_SIZE = "suffix_size"
BASE_SIZE = "base_size"
PREFIX = "prefix"
SUFFIX = "suffix"
BASE = "base"
DICT = "dict"
MMAP = "mmap"

# --- Basic Configuration ---
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout
)

class LRUCache(OrderedDict):
    def __init__(self, capacity=0):
        super().__init__()
        self.capacity = capacity

    def get(self, key):
        if key not in self:
            return None
        self.move_to_end(key)  # O(1) operation
        return self[key]

    def put(self, key, value):
        self[key] = value
        if 0 < self.capacity < len(self):
            self.popitem(last=False)  # O(1) removes the LRU item

def log_execution_time(func):
    """A decorator that logs the execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds.")
        return result

    return wrapper

def add_to_running_average(new_value: float, running_average: float, num_of_values: int):
    """
    Calculates the cumulative average (running average) for a list of numbers
    using the recursive formula, which avoids needing to compute a running total (sum).
    """
    running_average += (new_value - running_average) / num_of_values

    return running_average

# --- Centralized Logging Setup ---
def setup_logging(file_name="dataset_processing.log"):
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, date_format)

    logger = logging.getLogger("WordPieceTokenizer")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, file_name)

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.propagate = False
    return logger


# Setup logging for the main process
logger = setup_logging()


def retrieve_valid_english_words(min_length: int = 1) -> Set[str]:
    """
    Retrieves a reliable set of valid English words using the NLTK corpus.

    This replaces the Hugging Face implementation to ensure higher
    linguistic accuracy and filter out non-English or invalid entries.
    """
    try:
        # 1. Ensure the 'words' corpus is downloaded
        # This only downloads if not already present
        nltk.download('words', quiet=True)

        # 2. Load the words from the corpus
        from nltk.corpus import words
        word_list = words.words()

        # 3. Clean and filter
        # We use a set comprehension for efficiency
        valid_words = {
            word.lower() for word in word_list
            if len(word) >= min_length and word.isalpha()
        }

        return valid_words

    except Exception as e:
        logger.error("Failed to retrieve words from NLTK corpus.", exc_info=e)
        return set()

def retrieve_english_words_from_hf(
        dataset_name: str = "Maximax67/English-Valid-Words",
        split: str = "train",
        word_column: str = "a",
        min_length: int = 1) -> Set[str]:
    """
    Retrieves a list of valid English words from a specified Hugging Face dataset.

    This function loads a dataset, extracts the column containing the words,
    and returns them as a simple Python list.

    Args:
        dataset_name (str): The name of the dataset on the Hugging Face Hub.
                            Defaults to a common English word list.
        split (str): The dataset split to load (e.g., 'train', 'test', 'validation').
        word_column (str): The name of the column that contains the word strings.
        min_length (int): The minimum length of a word to include in the list.

    Returns:
        Set[str]: A set of valid vocabulary words.
    """
    valid_words = set()

    try:
        # 1. Load the dataset from the Hugging Face Hub
        # configs: ['sorted_by_frequency', 'sorted_alphabetically', 'valid_words']
        dataset = load_dataset(dataset_name, 'valid_words', split=split)

        # 2. Extract the column of words
        # The 'Maximax67/English-Valid-Words' dataset uses a 'Word' column.
        words = dataset[word_column]

        # 3. Clean and filter the list
        # We filter for non-empty strings and enforce the minimum length.
        for word in words:
            word_str = str(word).strip().lower()
            if word_str and len(word_str) >= min_length:
                valid_words.add(word_str)

        return valid_words

    except Exception as e:
        logger.error("An error occurred while loading or processing the dataset.", exc_info=e)
        return valid_words


# --- File I/O Functions ---
def load_json_file(file_path):
    conf = {}
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            conf = json.load(f)

    return conf


def save_json_file(save_directory, file_name, json_data):
    os.makedirs(save_directory, exist_ok=True)
    file_path = os.path.join(save_directory, file_name)
    logger.info(f"Saving data to {file_path}...")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)


def _load_tokenizer_config(load_directory):
    config_file = os.path.join(load_directory, CONFIG_FILE_NAME)
    conf = {
        "unk_token": UNK,
        "subword_prefix": SUBWORD_PREFIX,
        DICTIONARIES: defaultdict(set)
    }
    if os.path.exists(config_file):
        start = time.time()
        with open(config_file, "r", encoding="utf-8") as f:
            conf = json.load(f)
            base = time.time() - start
            # Convert loaded dictionary lists back to sets
            for lang, word_list in conf.get(DICTIONARIES, {}).items():
                conf[DICTIONARIES][lang] = set(word_list)
                total_time = (time.time() - start) + base
                logger.info(f"Loaded '{lang}' dictionary with {len(word_list)} words in {total_time} seconds.")
    else:
        for lang in SUPPORTED_LANGUAGES:
            start = time.time()
            words = load_default_dictionary(lang)
            conf[DICTIONARIES][lang] = words
            total_time = time.time() - start
            logger.info(f"Created new '{lang}' dictionary with: {len(words)} words in {total_time} seconds.")

    return conf


def _load_tokenizer_vocab(load_directory: str):
    """Loads a tokenizer from a directory."""
    vocab = {v: idx for idx, v in enumerate(SPECIAL_TOKENS)}
    if not os.path.exists(load_directory):  # Load dictionaries and default configs
        os.makedirs(load_directory)

    vocab_file = os.path.join(load_directory, VOCAB_FILE_NAME)
    if os.path.exists(vocab_file):
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = json.load(f)

    return vocab


def _save_tokenizer_config(save_directory: str, config: Dict):
    """Saves a tokenizer vocab to a directory."""
    os.makedirs(save_directory, exist_ok=True)
    config_file = os.path.join(save_directory, CONFIG_FILE_NAME)

    # Convert sets to lists for JSON serialization
    for lang, word_set in config.get(DICTIONARIES, {}).items():
        words_list = sorted([x for x in word_set if len(x) > 1])
        config[DICTIONARIES][lang] = words_list
        logger.info(f"Saving '{lang}' dictionary with {len(words_list)} words to {config_file}...")

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def _save_tokenizer_vocab(save_directory: str, vocab: Dict):
    """Saves a tokenizer configurations to a directory."""
    os.makedirs(save_directory, exist_ok=True)
    vocab_file = os.path.join(save_directory, VOCAB_FILE_NAME)
    logger.info(f"Saving {len(vocab)} vocab tokens to {vocab_file}...")
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)


def load_morphemes(
    lang_code: str, filepath: str
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Loads prefixes, suffixes, and roots."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                all_data = json.load(f)

            data = all_data.get(lang_code, {})
            root_words = set(data.get("root_words", []))
            prefixes = set(data.get("prefixes", []))
            suffixes = set(data.get("suffixes", []))

            logger.info(
                f"Successfully loaded morphemes for '{lang_code}' from {filepath}"
            )

            return prefixes, suffixes, root_words

        except json.JSONDecodeError:
            return set(), set(), set()
    else:
        return set(), set(), set()


def save_morphemes(
    lang_code: str,
    filepath: str,
    prefixes: Set[str],
    suffixes: Set[str],
    root_words: Set[str],
):
    """Saves the updated morpheme lists to a JSON file under the lang_code key."""
    all_data = {}
    # Load existing data first to not overwrite other languages
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                all_data = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Warning: {filepath} was corrupted. Overwriting.")
            all_data = {}

    # Create the new data structure for the specified language
    lang_data = {
        "prefixes": sorted(list(prefixes)),
        "suffixes": sorted(list(suffixes)),
        "root_words": sorted(list(root_words)),
    }
    # Update the data for that language
    all_data[lang_code] = lang_data

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)


def load_word_frequencies_json(file_name: str, lang_code: str='') -> CounterType[str]:
    """
    Loads a word frequency file from JSON and returns it as a Counter.
    """
    if lang_code:
        file_name = f'{lang_code}_{file_name}'

    filepath = os.path.join(STORAGE_DIR, file_name)
    logger.info(f"Loading word frequencies from {filepath}...")
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}. Returning empty counter.")
        return Counter()

    try:
        with open(filepath, 'r', encoding="utf-8") as f:
            word_freq_dict = json.load(f)

        # Convert the loaded dict back into a Counter
        word_counter = Counter(word_freq_dict)
        logger.info(f"Successfully loaded {len(word_counter)} word frequencies.")
        return word_counter

    except json.JSONDecodeError:
        logger.error(f"Error: {filepath} is corrupted. Returning empty counter.")
        return Counter()
    except Exception as e:
        logger.error(f"An unexpected error occurred loading {filepath}: {e}", exc_info=True)
        return Counter()


def save_word_frequencies_json(file_name: str, word_counter: CounterType[str] | Dict[str, int], lang_code: str=''):
    """
    Saves a word frequency Counter to a JSON file.
    Converts Counter to a standard dict for serialization.
    """
    if lang_code:
        file_name =  f'{lang_code}_{file_name}'

    filepath = os.path.join(STORAGE_DIR, file_name)
    logger.info(f"Saving {len(word_counter)} word frequencies to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Convert Counter to a plain dictionary for JSON serialization
    if isinstance(word_counter, CounterType):
        word_counter = OrderedDict(word_counter)

    with open(filepath, 'w', encoding="utf-8") as f:
        json.dump(word_counter, f, indent=2)
    logger.info("Save complete.")


def get_initial_str_spaces(word) -> Tuple[str, int]:
    """Returns the lowercased, left-stripped word and the number of spaces removed."""
    word_str = word.lower()  # First lower to get true size before strip
    size = len(word_str)
    word_str = word_str.lstrip()
    spaces = size - len(word_str)

    return word_str, spaces


def is_word_str_equal(spaces: int, word: str, word_str: str) -> bool:
    """
    Checks if the original word matches the normalized word string, accounting for leading spaces.

    Args:
        spaces (int): Number of leading spaces.
        word (str): The original raw word.
        word_str (str): The normalized (usually lowercase) word.

    Returns:
        bool: True if the content matches, False otherwise.
    """
    word_str_equal = word[spaces:] == word_str if spaces > 0 else word == word_str
    return word_str_equal


def _group_by_length(morpheme_set: Set[str]) -> Dict[int, Set[str]]:
    """Groups morphemes by their length, sorted from largest to smallest length."""
    groups = defaultdict(set)
    for m in morpheme_set:
        groups[len(m)].add(m)
    return dict(sorted(groups.items(), key=lambda item: item[0], reverse=True))


def _chunk_iterable(iterable: Iterable, chunk_size: int) -> Generator[List[Any], None, None]:
    """Yields successive chunks from an iterable."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def calculate_pair_score(subword_pairs_tuples: List[Tuple[int, str]],
                         morphemes_freq: Counter[str]) -> float:
    """
    Calculates a score for a proposed list of subword splits.

    Uses a **Cubic Length Weighting** strategy ($length^3$) multiplied by log-smoothed
    frequency. This heavily favors retaining long root words (e.g., 'establish') over
    breaking them into smaller, high-frequency affixes.

    Args:
        subword_pairs_tuples: A list of tuples (start_index, subword_string).
        morphemes_freq: A Counter containing frequency data for known morphemes.

    Returns:
        float: The calculated score. Higher is better.
    """
    subwords = [x[1] for x in subword_pairs_tuples]

    if not subwords:
        return 0.0

    try:
        word_str_equal = subwords[0][0].islower()
    except IndexError:
        word_str_equal = subwords[1].islower()

    score = 0.0

    # Check for "orphaned" single characters which are usually bad splits
    # (e.g. 's' is fine, but 'a' or 'i' in the middle of a word is usually wrong)
    single_char_penalty = 0

    for idx, subword in enumerate(subwords):
        # 1. Frequency Component
        freq = morphemes_freq.get(subword, 1)
        if not word_str_equal:
            freq = max(freq, morphemes_freq.get(subword.lower(), 1))

        freq_score = math.log(max(freq, 2))  # Log smoothing

        # 2. Length Component: Cubic weighting
        # This makes a length 7 root (343) significantly more valuable than
        # a length 4 prefix (64) + length 3 suffix (27) -> (91)
        length_val = len(subword)
        length_weight = length_val ** 3

        token_score = freq_score * length_weight
        score += token_score

        if length_val == 1:
            single_char_penalty += 1

    # Normalize slightly by number of splits to prefer fewer chunks,
    # but don't over-penalize valid morphological trains (un-account-ed)
    score = score / (len(subwords) ** 0.3)

    # Penalize excessive single characters (unless it's just a suffix 's')
    if single_char_penalty > 1:
        score = score * 0.5

    return score


def split_subword_by_index(word: str,
                           word_str: str,
                           vocab_tokens: Set[str],
                           dictionary_words: Set[str],
                           max_len: int = 15,
                           min_len: int = 1,
                           subword_prefix=SUBWORD_PREFIX) -> DefaultDict[int, Set[str]]:
    """
    Generates a graph of valid subword splits for a given word based on dictionary validation.

    Keys in the return dictionary represent start indices, and values are lists of valid
    subwords starting at that index.

    Args:
        word (str): The raw word to split.
        word_str (str): The normalized word (usually lowercase and stripped of whitespaces).
        vocab_tokens (Set[str]): The current set of valid vocabulary words.
        dictionary_words (Set[str]): Set of valid dictionary words.
        max_len (int): Maximum allowed length for a subword.
        min_len (int): Minimum allowed length for a subword.
        subword_prefix (str): The subword prefix to use.

    Returns:
        DefaultDict[int, List[str]]: A mapping of start_index -> list of valid subwords.
    """
    subwords_by_index: DefaultDict[int, Set[str]] = defaultdict(set)
    word_str_equal = word == word_str
    min_len = max(1, min_len)  # Ensure minimum len is at least 1
    start = 0

    # Edge case: Word is shorter than min_len
    if len(word_str) < min_len + 1:
        for x in range(len(word)):
            subword = word[x]
            subwords_by_index[x].add(subword)
        return subwords_by_index

    # Regex handling for CamelCase or specific split patterns
    words = re.findall(CASE_SPLIT_PATTERN, word)

    boundaries = set()
    offset = 0
    pairs = {0}
    for x in words:
        boundaries.add(offset + len(x))
        offset += len(x)

    count = 0
    word_part_len = len(word_str)
    limits = {}
    mid_point = min(math.ceil(len(word) / 2), MMAP_MIN_WORD_LENGTH)
    for x in range(mid_point + 1):
        if x < 2:
            limits[x] = 3
        elif x < mid_point:
            limits[x] = 1
        else:
            limits[x] = 0

    while start <= len(word) and len(pairs) > 0:
        count += 1
        start = 0
        min_idx = word_part_len
        for x in pairs:
            if start < x <= word_part_len:
                start = x
            if x < min_idx:
                min_idx = x

        pairs.discard(start)
        limit = limits.get(start, 0)
        word_part_len = min(boundaries) if boundaries else len(word_str)

        if min_idx >= word_part_len and len(boundaries) > 1:
            boundaries.discard(word_part_len)
            word_part_len = min(boundaries) if boundaries else len(word_str)

        if start == word_part_len: continue
        for end in range(min(start + max_len, word_part_len), start, -1):
            if start == 0 and end >= len(word_str): continue

            subword_str = word_str[start:end] if end > start else word_str[start]

            if word_str_equal:
                subword = subword_str
            else:
                subword = word[start:end] if end > start else word[start]

            subword_token = subword if start < 1 and offset < 1 else f"{subword_prefix}{subword}"
            subword_in_vocab = subword_str in dictionary_words or subword_str in vocab_tokens
            token_in_vocab = subword_in_vocab if word_str_equal else subword_token in vocab_tokens
            in_vocab = subword_in_vocab or token_in_vocab

            if len(subword_str) > min_len and not in_vocab:
                continue

            subwords_by_index[start].add(subword)
            pairs.add(start + len(subword))
            pair_size = len(subwords_by_index[start])
            if pair_size > limit or len(subword) == min_len:
                break

    return subwords_by_index


def tokenize_subword_by_index(word: str,
                              vocab_tokens: Set[str],
                              dictionary: Set[str] = None,
                              max_len: int = 15,
                              min_len: int = 1,
                              subword_prefix=SUBWORD_PREFIX) -> DefaultDict[int, Set[str]]:
    """
    Generates a graph of valid subword splits for a given word based on dictionary validation.

    Keys in the return dictionary represent start indices, and values are lists of valid
    subwords starting at that index.

    Args:
        word (str): The raw word to split.
        vocab_tokens (Set[str]): The current set of valid vocabulary words.
        dictionary (Set[str]): Set of valid dictionary words.
        max_len (int): Maximum allowed length for a subword.
        min_len (int): Minimum allowed length for a subword.
        subword_prefix (str): The subword prefix to use.

    Returns:
        DefaultDict[int, List[str]]: A mapping of start_index -> list of valid subwords.
    """
    subwords_by_index: DefaultDict[int, Set[str]] = defaultdict(set)
    min_len = max(1, min_len)  # Ensure minimum len is at least 1
    start = 0

    # Edge case: Word is shorter than min_len
    if len(word) < min_len + 1:
        for x in range(len(word)):
            subwords_by_index[x].add(word[x])
        return subwords_by_index

    # Regex handling for CamelCase or specific split patterns
    words = re.findall(CASE_SPLIT_PATTERN, word)

    boundaries = set()
    offset = 0
    pairs = {0}
    for x in words:
        boundaries.add(offset + len(x))
        offset += len(x)

    count = 0
    word_part_len = len(word)
    limits = {}
    mid_point = min(math.ceil(len(word) / 2), MMAP_MIN_WORD_LENGTH)
    for x in range(mid_point + 1):
        if x < 2:
            limits[x] = 3
        elif x < mid_point:
            limits[x] = 1
        else:
            limits[x] = 0

    while start <= len(word) and len(pairs) > 0:
        count += 1
        start = 0
        min_idx = word_part_len
        for x in pairs:
            if start < x <= word_part_len:
                start = x
            if x < min_idx:
                min_idx = x

        pairs.discard(start)
        limit = limits.get(start, 0)
        word_part_len = min(boundaries) if boundaries else len(word)

        if min_idx >= word_part_len and len(boundaries) > 1:
            boundaries.discard(word_part_len)
            word_part_len = min(boundaries) if boundaries else len(word)

        if start == word_part_len: continue

        for end in range(min(start + max_len, word_part_len), start, -1):
            if start == 0 and end >= len(word): continue

            is_subword = start > 0 or offset > 0
            subword = word[start:end] if end > start else word[start]
            subword_token = f"{subword_prefix}{subword}" if word[0].isalpha() and is_subword else subword
            subword_in_vocab = subword_token in vocab_tokens
            if not subword_in_vocab and dictionary and len(subword) <= max_len and subword[0].islower():
                subword_in_vocab = subword in dictionary

            if len(subword) > min_len and not subword_in_vocab:
                continue

            subwords_by_index[start].add(subword)
            pairs.add(start + len(subword))
            pair_size = len(subwords_by_index[start])
            if pair_size > limit or len(subword) == min_len:
                break

    return subwords_by_index


def create_word_pairs(word: str, subword_by_index: DefaultDict[int, Set[str]]) -> List[List[str]]:
    """
    Traverses the subword index graph to build all valid full-word segmentations.

    Args:
        word (str): The original word (used for length validation).
        subword_by_index (DefaultDict): The graph generated by split_subword_by_index.

    Returns:
        List[List[str]]: A list of valid segmentation paths (e.g. [['sub', 'word'], ...])
    """
    results = defaultdict(list)
    try:
        for subword in subword_by_index.pop(0):
            results[len(subword)].append([subword])
    except KeyError:
        logger.error(f"No subword found for word '{word}' at start index '0'")
        return []

    for start_idx in sorted(subword_by_index.keys()):
        if start_idx in results:
            groups = results.pop(start_idx)
            for subword in subword_by_index.pop(start_idx):
                for group in groups:
                    results[start_idx + len(subword)].append(group + [subword])
        else:
            for subword in subword_by_index.pop(start_idx):
                results[start_idx + len(subword)].append([subword])

    try:
        results = results[len(word)]
    except KeyError:
        logger.error(f"No valid subword pairs found for word '{word}' with final len: '{len(word)}'")
        results = []

    return results


def perform_perfect_split(p_lookup: str,
                          p_word: str,
                          p_offset: int,
                          max_dept: int,
                          min_len: int,
                          max_len: int,
                          lookup_dictionary: Set[str],
                          vocab_tokens: Set[str],
                          subword_prefix=SUBWORD_PREFIX) -> List[List[Tuple[int, str]]]:
    """
        Attempts to find a "perfect" segmentation of a word using Iterative Deepening
        and Alternating Edge Search.

        This method explores the search space to find a combination of subwords that
        reconstruct the original word where every component is valid within the
        dictionary or vocabulary.

        Args:
            p_lookup: The normalized string to look up (e.g., lowercase).
            p_word: The raw string segment from the original text.
            p_offset: The character offset of this segment relative to the full word.
            max_dept: Maximum recursion depth for the split search.
            min_len: Minimum length of a valid subword.
            max_len: Maximum length of a valid subword.
            lookup_dictionary: Set of valid language dictionary words.
            vocab_tokens: Set of existing vocabulary tokens.

        Returns:
            List[List[Tuple[int, str]]]: A list of valid split paths found.
        """
    # 1. Initial Check: If the whole remainder is in the dictionary.
    if len(p_lookup) <= max_len and p_lookup in lookup_dictionary:
        return [[(p_offset, p_word)]]

    new_max_len = max(7, max_len, MAX_DICT_WORD_LEN)
    if p_offset > 0:
        if len(p_word) <= new_max_len and f"{subword_prefix}{p_word}" in vocab_tokens:
            return [[(p_offset, p_word)]]
    else:
        if len(p_lookup) <= new_max_len and p_word in vocab_tokens:
            return [[(p_offset, p_word)]]

    word_str_equal = p_lookup == p_word

    def _solve_at_specific_depth(curr_lookup: str,
                                 curr_word: str,
                                 curr_offset: int,
                                 cuts_remaining: int) -> List[List[Tuple[int, str]]]:

        full_len = len(curr_lookup)
        try:
            str_equal = curr_word[0].islower()
        except IndexError:
            str_equal = word_str_equal

        _max_len = MAX_DICT_WORD_LEN if str_equal else max_len

        if cuts_remaining == 0:
            in_dict = True if len(curr_lookup) < 2 else curr_lookup in lookup_dictionary
            curr_token = curr_word if (curr_offset < 1 and p_offset < 1) else f"{subword_prefix}{curr_word}"
            in_vocab = curr_token in vocab_tokens or (str_equal and curr_word in vocab_tokens)

            # 2. Leaf Node Check:
            # We allow words to exceed 'max_len' ONLY if they are in the dictionary
            # AND they are not excessively long (<= 11 chars).
            # This preserves 'establish' (9) but rejects 'autobiographical' (16).
            is_valid_len = (in_dict and len(curr_lookup) <= max_len) or (in_vocab and len(curr_word) <= _max_len)

            if p_offset > 0:
                if len(curr_lookup) < min_len:
                    is_valid_len = False
            else:
                if len(curr_lookup) < min_len and 0 < curr_offset < len(curr_lookup) - 1:
                    is_valid_len = False

            if is_valid_len:
                return [[(curr_offset, curr_word)]]
            return []

        # Determine search range
        start_i = 1 if p_offset == 0 and curr_offset == 0 else max(2, min_len)

        # We search up to 11 (max reasonable root size) to find valid roots
        end_i = full_len - 1

        if end_i < start_i:
            return []

        search_indices = []
        l, r = start_i, end_i
        while l <= r:
            if l == r:
                search_indices.append(l)
            else:
                search_indices.append(r)
                search_indices.append(l)
            l += 1
            r -= 1

        lag = len(curr_word) - len(curr_lookup)
        solutions_at_this_level = []

        for i in search_indices:
            if i > MAX_DICT_WORD_LEN: continue

            part1_lookup = curr_lookup[:i]
            part1_word_len = i + lag
            if len(part1_lookup) > _max_len:
                part1_in_dict = False
            else:
                part1_in_dict = part1_lookup in lookup_dictionary

            # 3. Recursive Step Check:
            if not part1_in_dict:
                part1_word_raw = part1_lookup if str_equal else curr_word[:part1_word_len]
                if curr_offset >0 or p_offset > 0:
                    part1_word_raw = f"{subword_prefix}{part1_word_raw}"

                part1_in_dict = part1_word_raw in vocab_tokens
                if not part1_in_dict: continue
            else:
                part1_word_raw = part1_lookup if str_equal else curr_word[:part1_word_len]

            part2_lookup = curr_lookup[i:]
            part2_word_raw = part2_lookup if str_equal else curr_word[part1_word_len:]

            recursive_results = _solve_at_specific_depth(
                curr_lookup=part2_lookup,
                curr_word=part2_word_raw,
                curr_offset=curr_offset + part1_word_len,
                cuts_remaining=cuts_remaining - 1
            )

            if recursive_results:
                current_node = (curr_offset, part1_word_raw)
                for path in recursive_results:
                    solutions_at_this_level.append([current_node] + path)

        return solutions_at_this_level

    for d in range(1, max_dept + 1):
        solutions = _solve_at_specific_depth(p_lookup, p_word, p_offset, d)
        if solutions:
            return solutions

    return []

def split_word_by_pairs(word: str,
                        spaces: int,
                        word_str_equal: bool,
                        vocab_tokens: Set[str],
                        lookup_dictionary: Set[str],
                        morphemes_freq: Counter[str],
                        max_len: int = 15,
                        min_len: int = 1,
                        start_index: int = 0,
                        subword_prefix=SUBWORD_PREFIX) -> Tuple[List[Tuple[int, str]], List[str]]:

    # --- Nested: Best Pair Split ---
    def perform_best_pair_split(process_word: str,
                                lookup_word: str,
                                current_offset: int) -> List[Tuple[int, str]]:
        best_splits = []
        try:
            subword_by_index = split_subword_by_index(
                process_word, lookup_word, vocab_tokens, lookup_dictionary, max_len, min_len, subword_prefix
            )
            candidates = create_word_pairs(process_word, subword_by_index)

            if candidates:
                best_pairs_sort = defaultdict(list)
                for pair in candidates:
                    best_pairs_sort[len(pair)].append(pair)

                candidates = best_pairs_sort[sorted(best_pairs_sort.keys())[0]]

                if len(candidates) > 1:
                    candidates_with_dummy_offset = []
                    for c in candidates:
                        dummy_tuples = [(0, w) for w in c]
                        candidates_with_dummy_offset.append((c, calculate_pair_score(dummy_tuples, morphemes_freq)))

                    candidates_with_dummy_offset.sort(key=lambda x: x[1], reverse=True)
                    best_split_strs = candidates_with_dummy_offset[0][0]
                else:
                    best_split_strs = candidates[0]

                local_idx = 0
                for token in best_split_strs:
                    best_splits.append((current_offset + local_idx, token))
                    local_idx += len(token)
        except Exception as ex:
            logger.error("Cannot find any best pairs. Performing greedy split...", exc_info=ex)

        return best_splits

    # --- Nested: Greedy Split ---
    def perform_greedy_split(process_word: str, lookup_word: str, current_offset: int) -> List[Tuple[int, str]]:
        local_idx = 0
        full_len_lookup = len(lookup_word)
        greedy_splits = []
        root_lag = len(process_word) - len(lookup_word)

        while local_idx < full_len_lookup:
            found_match = False
            remaining_len = full_len_lookup - local_idx
            search_cap = min(remaining_len, max_len)

            for chunk_len in range(search_cap, min_len - 1, -1):
                candidate = lookup_word[local_idx: local_idx + chunk_len]
                if candidate in lookup_dictionary:
                    start_idx_word = local_idx + root_lag
                    end_idx_word = start_idx_word + chunk_len
                    if local_idx == 0:
                        start_idx_word = 0
                        end_idx_word = chunk_len + root_lag

                    original_segment = process_word[start_idx_word: end_idx_word]
                    greedy_splits.append((current_offset + start_idx_word, original_segment))
                    local_idx += chunk_len
                    found_match = True
                    break

            if not found_match:
                start_idx_word = local_idx + root_lag
                if local_idx == 0: start_idx_word = 0
                unknown_char = process_word[start_idx_word: start_idx_word + 1]
                greedy_splits.append((current_offset + start_idx_word, unknown_char))
                local_idx += 1

        return greedy_splits

    results = []
    split_types = []
    is_subword = word.startswith(subword_prefix)
    current_offset = 1 if is_subword and start_index < 1 else start_index

    try:
        words = re.findall(CASE_SPLIT_PATTERN, word)
    except NameError:
        words = [word]

    if not word_str_equal and spaces > 0:
        space_token = word[:spaces]
        results.append((current_offset, space_token))
        current_offset += spaces
        word = word[spaces:]
    else:
        word = word.lstrip(subword_prefix) if current_offset > 0 else word

    if not words:
        words = [word]

    segments_to_process = []
    temp_offset = current_offset
    for w in words:
        token = f"{subword_prefix}{w}" if temp_offset > 0 else w
        if token in vocab_tokens:
            split_types.append('perfect')
            results.append((temp_offset, w))
            temp_offset += len(w)
            continue
        w_lookup = w if word_str_equal else w.lower()
        segments_to_process.append((w, w_lookup, temp_offset))
        temp_offset += len(w)

    if not segments_to_process:
        return results, split_types

    # Allow enough depth for long words (e.g. 16 chars / 3 = 5.3 -> 6 splits)
    max_dept = min(12, math.ceil(len(word) / 3))

    for seg_word, seg_lookup, seg_offset in segments_to_process:
        found_splits = None

        perfect_candidates = perform_perfect_split(
            p_lookup=seg_lookup,
            p_word=seg_word,
            p_offset=seg_offset,
            max_dept=max_dept,
            min_len=min_len,
            max_len=max_len,
            lookup_dictionary=lookup_dictionary,
            vocab_tokens=vocab_tokens,
            subword_prefix=subword_prefix,
        )

        if perfect_candidates:
            split_types.append('perfect')
            if len(perfect_candidates) > 1:
                perfect_candidates.sort(
                    key=lambda c: calculate_pair_score(c, morphemes_freq),
                    reverse=True
                )

            found_splits = perfect_candidates[0]

        if not found_splits:
            best_pairs = perform_best_pair_split(seg_word, seg_lookup, seg_offset)
            if best_pairs:
                split_types.append('best')
                found_splits = best_pairs

        if not found_splits:
            greedy_pairs = perform_greedy_split(seg_word, seg_lookup, seg_offset)
            split_types.append('greedy')
            found_splits = greedy_pairs

        if found_splits:
            results.extend(found_splits)
            # current_offset = seg_offset + len(seg_word)

    return results, split_types


def tokenize_word_by_pairs(word: str,
                           vocab_tokens: Dict[str, int]|Set[str],
                           subword_freq: Counter[str],
                           avg_str_freq: Counter[int],
                           word_pairs: LRUCache,
                           max_len: int = MAX_DICT_WORD_LEN,
                           min_len: int = 1,
                           start_index: int = 0,
                           dictionary: Set[str] = None,
                           subword_prefix=SUBWORD_PREFIX) -> Tuple[List[Tuple[int, str]], List[str]]:
    """
    Attempts to find a "perfect" segmentation of a word using Iterative Deepening
    and Alternating Edge Search.

    This method explores the search space to find a combination of subwords that
    reconstruct the original word where every component is valid within the
    dictionary or vocabulary.

    Args:
        word: The raw string segment from the original text.
        vocab_tokens: Set of existing vocabulary tokens.
        subword_freq: The Counter with subword frequencies to use for sorting subword pairs.
        avg_str_freq: The Counter with average string frequencies.
        word_pairs: The LRUCache to use for looking up the lower case word pairs.
        max_len: Maximum length of a valid subword.
        min_len: Minimum length of a valid subword.
        start_index: the starting index/offset of the subword if word is actually a subword.
        dictionary: The dictionary for which to use for lookup.
        subword_prefix: The subword prefix to use.

    Returns:
        List[List[Tuple[int, str]]]: A list of valid split paths found.
    """
    def tokenize_with_perfect_split(p_word: str,
                                    p_offset: int,
                                    max_dept: int) -> List[List[Tuple[int, str]]]:
        """
        Attempts to find a "perfect" segmentation of a word using Iterative Deepening
        and Alternating Edge Search.

        This method explores the search space to find a combination of subwords that
        reconstruct the original word where every component is valid within the
        dictionary or vocabulary.

        Args:
            p_word: The raw string segment from the original text.
            p_offset: The character offset of this segment relative to the full word.
            max_dept: Maximum recursion depth for the split search.

        Returns:
            List[List[Tuple[int, str]]]: A list of valid split paths found.
        """
        perfect_match = []
        p_lookup = word_pairs.get(p_word)
        if not p_lookup:
            p_lookup = p_word if p_word[0].islower() else p_word.lower()
            word_pairs.put(p_word, p_lookup)

        try:
            word_str_equal = p_word[0].islower()
        except IndexError:
            word_str_equal = p_word == p_lookup

        freq = subword_freq.get(p_word, 1)
        new_max_len = max(7, max_len, MAX_DICT_WORD_LEN) if word_str_equal else max_len
        p_token = f"{subword_prefix}{p_word}" if p_offset > 0 and p_word[0].isalpha() else p_word
        if len(p_word) <= new_max_len and p_token in vocab_tokens:
            return [[(p_offset, p_word)]]

        if len(p_lookup) <= max_len and dictionary and p_lookup in dictionary:
            if len(p_word) > WORD_SIZE_LIMIT or freq < SUBWORD_FREQ_LIMIT:
                perfect_match = [[(p_offset, p_word)]]
            else:
                return [[(p_offset, p_word)]]

        def _solve_at_specific_depth(curr_word: str,
                                     curr_offset: int,
                                     cuts_remaining: int,
                                     longest_substr: int) -> Tuple[int, List[List[Tuple[int, str]]]]:

            full_len = len(curr_word)

            if cuts_remaining == 0:

                is_subword_token = curr_word[0].isalpha() and (curr_offset > 0 or p_offset > 0)
                curr_token = f"{subword_prefix}{curr_word}" if is_subword_token else curr_word
                in_vocab = True if len(curr_word) < 2 else curr_token in vocab_tokens

                if not in_vocab and dictionary:
                    curr_lookup = word_pairs.get(curr_word)
                    if not curr_lookup:
                        curr_lookup = curr_word.lower()
                        word_pairs.put(curr_word, curr_lookup)
                    in_vocab = curr_lookup in dictionary

                # 2. Leaf Node Check:
                # We allow words to exceed 'max_len' ONLY if they are in the dictionary
                # AND they are not excessively long (<= 11 chars).
                # This preserves 'establish' (9) but rejects 'autobiographical' (16).
                is_valid_len = in_vocab and len(curr_word) <= max_len

                if p_offset > 0:
                    if len(curr_word) < min_len:
                        is_valid_len = False
                else:
                    if len(curr_word) < min_len and 0 < curr_offset < len(curr_word) - 1:
                        is_valid_len = False

                if is_valid_len:
                    longest_substr = max(longest_substr, len(curr_word))
                    return longest_substr, [[(curr_offset, curr_word)]]
                return longest_substr, []

            # Determine search range
            start_i = 1 if p_offset == 0 and curr_offset == 0 else max(2, min_len)

            # We search up to 11 (max reasonable root size) to find valid roots
            end_i = full_len - 1

            if end_i < start_i:
                return longest_substr, []

            search_indices = []
            l, r = start_i, end_i
            while l <= r:
                if l == r:
                    search_indices.append(l)
                else:
                    search_indices.append(r)
                    search_indices.append(l)
                l += 1
                r -= 1

            solutions_at_this_level = []

            for i in search_indices:
                if i > MAX_DICT_WORD_LEN: continue

                part1_word = curr_word[:i]

                if part1_word[0].isalpha() and (curr_offset > 0 or p_offset > 0):
                    part1_token = f"{subword_prefix}{part1_word}"
                else:
                    part1_token = part1_word

                part1_in_dict = part1_token in vocab_tokens

                if not part1_in_dict and len(part1_word) <= new_max_len:
                    part1_lookup = word_pairs.get(part1_word)
                    if not part1_lookup:
                        part1_lookup = part1_word.lower()
                        word_pairs.put(part1_word, part1_lookup)

                    part1_in_dict = part1_lookup in dictionary

                if not part1_in_dict: continue

                part2_word = curr_word[i:]

                longest_substr, recursive_results = _solve_at_specific_depth(
                    curr_word=part2_word,
                    curr_offset=curr_offset + i,
                    cuts_remaining=cuts_remaining - 1,
                    longest_substr=longest_substr,
                )

                if recursive_results:
                    current_node = (curr_offset, part1_word)
                    for path in recursive_results:
                        solutions_at_this_level.append([current_node] + path)

            return longest_substr, solutions_at_this_level

        longest_subword = 0
        for d in range(1, max_dept + 1):
            longest_subword, solutions = _solve_at_specific_depth(p_word, p_offset, d, longest_subword)

            if solutions:  # Return perfect split if found
                if perfect_match:
                    if longest_subword > WORD_SIZE_LIMIT:
                        return solutions
                    else:
                        return perfect_match
                else:
                    return solutions

            if perfect_match:  # Return the perfect match if no perfect split was found
                return perfect_match

        return []

    def perform_best_pair_split(word: str, current_offset: int) -> List[Tuple[int, str]]:
        best_splits = []
        try:
            subword_by_index = tokenize_subword_by_index(
                word, vocab_tokens, dictionary, max_len, min_len, subword_prefix
            )
            candidates = create_word_pairs(word, subword_by_index)

            if candidates:
                best_pairs_sort = defaultdict(list)
                for pair in candidates:
                    best_pairs_sort[len(pair)].append(pair)

                candidates = best_pairs_sort[sorted(best_pairs_sort.keys())[0]]

                if len(candidates) > 1:
                    candidates_with_dummy_offset = []
                    for c in candidates:
                        dummy_tuples = [(0, w) for w in c]
                        candidates_with_dummy_offset.append((c, calculate_pair_score(dummy_tuples, subword_freq)))

                    candidates_with_dummy_offset.sort(key=lambda x: x[1], reverse=True)
                    best_split_strs = candidates_with_dummy_offset[0][0]
                else:
                    best_split_strs = candidates[0]

                local_idx = 0
                for token in best_split_strs:
                    best_splits.append((current_offset + local_idx, token))
                    local_idx += len(token)
        except Exception as ex:
            logger.error("Cannot find any best pairs. Performing greedy split...", exc_info=ex)

        return best_splits

    def perform_greedy_split(word: str, current_offset: int) -> List[Tuple[int, str]]:
        local_idx = 0
        full_len_lookup = len(word)
        greedy_splits = []

        while local_idx < full_len_lookup:
            found_match = False
            remaining_len = full_len_lookup - local_idx
            search_cap = min(remaining_len, max_len)

            for chunk_len in range(search_cap, min_len - 1, -1):
                candidate = word[local_idx: local_idx + chunk_len]
                token = candidate if current_offset < 1 and local_idx < 1 else f"{subword_prefix}{candidate}"
                token_in_vocab = token in vocab_tokens

                if not token_in_vocab and dictionary and word[0].islower():
                    token_in_vocab = len(word) <= max_len and word in dictionary

                if token_in_vocab:
                    start_idx = local_idx
                    greedy_splits.append((current_offset + start_idx, candidate))
                    local_idx += chunk_len
                    found_match = True
                    break

            if not found_match:
                start_idx = local_idx
                unknown_char = word[start_idx: start_idx + 1]
                greedy_splits.append((current_offset + start_idx, unknown_char))
                local_idx += 1

        return greedy_splits

    results = []
    split_types = []
    is_subword = word.startswith(subword_prefix)
    current_offset = 1 if is_subword and start_index < 1 else start_index
    word = word.lstrip(subword_prefix) if current_offset > 0 else word
    dictionary = dictionary if dictionary else set()

    try:
        words = re.findall(CASE_SPLIT_PATTERN, word)
    except NameError:
        words = [word]

    if not words:
        words = [word]

    segments_to_process = []
    temp_offset = current_offset
    for idx, _word in enumerate(words):
        is_alphabet = len(_word) > 0 and _word[0].isalpha()
        token = f"{subword_prefix}{_word}" if temp_offset > 0 and is_alphabet else _word
        if len(_word) <= max_len and token in vocab_tokens:
            split_types.append('perfect')
            results.append((temp_offset, _word))
            temp_offset += len(_word)
            continue
        segments_to_process.append((_word, temp_offset))
        temp_offset += len(_word)

    if not segments_to_process:
        return results, split_types

    # Allow enough depth for long words (e.g. 16 chars / 3 = 5.3 -> 6 splits)
    max_dept = min(12, math.ceil(len(word) / 3))
    str_equal = (len(words) == 1 and words[0][0].islower())

    for seg_word, seg_offset in segments_to_process:
        found_splits = None
        # Determine max length base on word freq
        is_lower = str_equal or seg_word[0].islower()
        total_freq = subword_freq.get(seg_word, 1)
        freq_limit = max(MIN_WORD_FREQ, avg_str_freq.get(len(seg_word), 5000))
        max_len = get_max_word_len(total_freq, freq_limit, is_lower)

        perfect_candidates = tokenize_with_perfect_split(seg_word, seg_offset, max_dept)

        if perfect_candidates:
            split_types.append('perfect')
            if len(perfect_candidates) > 1:
                perfect_candidates.sort(
                    key=lambda c: calculate_pair_score(c, subword_freq),
                    reverse=True
                )

            found_splits = perfect_candidates[0]

        if not found_splits:
            best_pairs = perform_best_pair_split(seg_word, seg_offset)
            if best_pairs:
                split_types.append('best')
                found_splits = best_pairs

        if not found_splits:
            greedy_pairs = perform_greedy_split(seg_word, seg_offset)
            split_types.append('greedy')
            found_splits = greedy_pairs

        if found_splits:
            results.extend(found_splits)
            # current_offset = seg_offset + len(seg_word)

    return sorted(results, key=lambda pair: pair[0]), split_types


def generate_subword_splits(word: str,
                            vocab: Set[str],
                            dictionary: Set[str],
                            morphemes_freq: Counter[str],
                            avg_str_freq: Counter[int],
                            word_pairs: LRUCache,
                            max_len: int = 15,
                            min_len: int = 1,
                            start_idx: int=0,
                            subword_prefix: str=SUBWORD_PREFIX):
    """Generates all valid subword splits for a given word grouped by start indices."""
    subwords_by_index: OrderedDict[int, str] = OrderedDict()

    pairs, split_types = tokenize_word_by_pairs(
        word, vocab, morphemes_freq, avg_str_freq, word_pairs, max_len, min_len, start_idx, dictionary, subword_prefix
    )

    for pair in pairs:
        subwords_by_index[pair[0]] = pair[1]

    return subwords_by_index, split_types


@log_execution_time
def extract_subword_frequencies_old(words_freq: CounterType[str],
                                word_pair_cache: LRUCache,
                                subword_freq: CounterType[str],
                                scaled_subword_freq: CounterType[str],
                                dictionary: Set[str],
                                max_len: int = MAX_DICT_WORD_LEN,
                                is_subword: bool = False) -> None:
    """
    Extracts and updates subword frequencies from a list of words based on a
    reference dictionary and frequency heuristics.

    The function performs a greedy split of words into subwords, prioritizing
    longer matches and validating them against a dictionary or frequency thresholds.
    It then prunes subwords that are redundant prefixes of longer, more frequent subwords.

    Args:
        words_freq: A Counter object with the corpus word counts.
        word_pair_cache: The LRUCache to use for looking up the lower case word pairs.
        subword_freq: A Counter object to update with subword counts.
        scaled_subword_freq: A Counter object to update with scaled subword counts.
        dictionary: A set of valid morphemes/words for validation.
        max_len: The maximum length of a subword to consider.
        is_subword: A boolean flag indicating whether the provided words are subwords.
    """
    if not isinstance(dictionary, set):
        dictionary = set(dictionary)

    # Pre-calculate word counts to avoid re-processing duplicates
    logger.info(f"Extracting subword frequencies for {len(words_freq)} unique words "
                f"using dictionary with {len(dictionary)} entries.")

    word_splits_cache: Dict[str, List[str]] = {}

    for word, freq in words_freq.items():
        token = f"{SUBWORD_PREFIX}{word}" if is_subword else word
        if token in word_splits_cache:
            for sub in word_splits_cache[token]:
                subword_freq[sub] += 1
                scaled_subword_freq[sub] += math.sqrt(freq)
            continue

        splits = []
        start = 0
        word_len = len(word)
        end = min(start + max_len, word_len)
        word_str = word_pair_cache.get(word)
        if not word_str:
            word_str = word.lower()
            word_pair_cache.put(word, word_str)

        str_equal = word_str == word

        while start < word_len:
            sub = word_str[start:end]

            # Check if it's a valid morpheme
            in_dict = sub in dictionary
            is_valid = in_dict

            # Heuristic: if not in dict, check if surrounding context is in dict
            if not is_valid:
                valid_prefix = (start > 2 and word[:start] in dictionary)
                if valid_prefix or (end < word_len - 3 and word[end:] in dictionary):
                    is_valid = True

            if is_valid:
                if str_equal:
                    subword = f"{SUBWORD_PREFIX}{sub}" if is_subword or start > 0 else sub
                else:
                    subword = f"{SUBWORD_PREFIX}{word[start:end]}" if is_subword or start > 0 else word[start:end]

                splits.append(subword)
                subword_freq[subword] += 1
                scaled_subword_freq[subword] += math.sqrt(freq)
                # Add valid subword to lookup dictionary
                if (is_subword or start > 0) and in_dict:
                    dictionary.add(subword)

            end -= 1
            if (end - start) <= 3:
                start += 1
                end = min(start + max_len, word_len)

        word_splits_cache[token] = splits

    # Remove low frequency subwords from scaled_subword_freq
    low_freq_subwords = _filter_low_frequency_subwords(subword_freq, dictionary)
    logger.info(f"Removed {len(low_freq_subwords)} low-frequency morphemes from counter.")
    for subword in low_freq_subwords:
        del scaled_subword_freq[subword]


def _filter_low_frequency_subwords_old(subword_freq: Counter[str], dictionary: Set[str]) -> Set[str]:
    """
    Removes subwords from the frequency counter that are infrequent or
    redundant prefixes of more frequent subwords.
    """
    def identify_invalid_candidates(sorted_candidates, is_suffix_pass=False):
        to_delete = set()
        for i in range(len(sorted_candidates)):
            candidate = sorted_candidates[i]

            # Since 'candidates' is sorted, all subwords starting with 'candidate'
            # follow it immediately in the list.
            for j in range(i + 1, len(sorted_candidates)):
                next_cand = sorted_candidates[j]

                candidate_found = False
                if is_suffix_pass:
                    if not next_cand.endswith(candidate):
                        break

                    if subword_freq[candidate] <= subword_freq[next_cand]:
                        to_delete.add(candidate)
                        candidate_found = True
                else:
                    if not next_cand.startswith(candidate):
                        break

                    if subword_freq[candidate] <= subword_freq[next_cand]:
                        to_delete.add(candidate)
                        candidate_found = True

                if not candidate_found:
                    break

        return to_delete

    candidates = sorted(subword_freq.keys())
    candidates_to_delete = set()

    # 2. Delete low frequency or invalid candidates
    for i in range(len(candidates)):
        candidate = candidates[i]

        # 1. Keep if in dictionary or frequency is not too low
        if not candidate in dictionary and subword_freq[candidate] < 2:
            candidates_to_delete.add(candidate)

    # 3. Check subsequent candidates for prefix redundancy
    candidates = [word for word in candidates if word not in candidates_to_delete]
    invalid_candidates = identify_invalid_candidates(candidates)
    candidates_to_delete.update(invalid_candidates)
    print(f"Found {len(invalid_candidates)} invalid prefix candidates.")

    # 4. Check subsequent candidates for suffix redundancy
    candidates = [word for word in candidates if word not in candidates_to_delete]
    candidates = sorted(candidates, key=lambda x: x[::-1])
    data = {"suffix": candidates}
    save_word_frequencies_json("sorted_suffix_candidates.json", data, lang_code='eng')
    invalid_candidates = identify_invalid_candidates(candidates, is_suffix_pass=True)
    candidates_to_delete.update(invalid_candidates)
    print(f"invalid suffix={invalid_candidates} \nFound {len(invalid_candidates)} invalid suffix candidates.")

    for key in candidates_to_delete:
        del subword_freq[key]

    return candidates_to_delete


@log_execution_time
def extract_subword_frequencies(words_freq: Counter,
                                word_pair_cache: LRUCache,
                                morphemes_freq: Counter,
                                scaled_subword_freq: Counter,
                                dictionary: Set[str],
                                word_splits_cache: Dict[str, Dict[int, List[str]]],
                                max_len: int = MAX_DICT_WORD_LEN,
                                is_subword: bool = False) -> None:
    """
    Efficiently extracts subword frequencies and prunes redundant candidates.
    The SUBWORD_PREFIX is handled after filtering to ensure suffix matching works.
    """
    if not isinstance(dictionary, set):
        dictionary = set(dictionary)

    # Temporary counters to store frequencies without the prefix for correct filtering
    # We need to track which clean strings actually need a prefix eventually
    # (i.e., they were found in a 'subword' context: start > 0 or is_subword=True)
    prefix_freq = Counter()
    prefix_scaled_freq = Counter()

    suffix_freq = Counter()
    suffix_scaled_freq = Counter()

    logger.info(f"Extracting subword frequencies for {len(words_freq)} words "
                f"using dictionary with {len(dictionary)} entries.")

    for word, freq in words_freq.items():
        """
        word_str = word_pair_cache.get(word)
        if not word_str:
            word_str = word.lower()
            word_pair_cache.put(word, word_str)
        """

        word_len = len(word)
        in_dict = word in dictionary
        splits = defaultdict(list)
        start = 0
        end = min(start + max_len, word_len)
        # Search for subwords from current start point
        while start < word_len:  # All subwords should be at least 3 character long
            # Extract the "raw" subword (from original casing if not equal, or lower)
            subword_len = end - start
            if subword_len < 1: break

            subword = word[start:end]

            if is_subword or start > 0:
                final_key = f"{SUBWORD_PREFIX}{subword}"
                morphemes_freq[final_key] += 1
                scaled_subword_freq[final_key] += freq

                # suffix_scaled_freq[subword] += freq #  Frequency already normalized
                # if in_dict: suffix_freq[subword] += 1
            else:
                morphemes_freq[subword] += 1
                scaled_subword_freq[subword] += freq

                # prefix_scaled_freq[subword] += freq
                # if in_dict: prefix_freq[subword] += 1

            splits[start].append(subword)
            end -= 1
            if subword_len < 2:  # Process next chunk
                start += 1
                end = min(start + max_len, word_len)

        if in_dict:
            word_splits_cache[word] = splits

    """
    # --- Filtering Phase (Operates on clean strings) ---
    logger.info(f"Filtering redundant and low-frequency subwords using dictionary with {len(dictionary)} entries...")

    # Prune low frequency and redundant prefixes/suffixes
    # --- Filtering Phase (Operates on clean strings) ---
    logger.info(f"Filtering redundant and low-frequency prefix from {len(prefix_freq)} candidates...")
    valid_prefix, replaced_prefixes = _filter_subwords_logic(prefix_freq, dictionary, is_prefix=True)
    save_word_frequencies_json("replaced_prefixes.json", replaced_prefixes, lang_code='eng')

    logger.info(f"Filtering redundant and low-frequency suffix from {len(suffix_freq)} candidates...")
    valid_suffix, replaced_suffixes = _filter_subwords_logic(suffix_freq, dictionary, is_prefix=False)

    # --- Finalization Phase ---
    # Transfer filtered results to the main counters with the prefix applied
    for raw_sub in valid_prefix:
        morphemes_freq[raw_sub] += prefix_freq[raw_sub]
        scaled_subword_freq[raw_sub] += prefix_scaled_freq[raw_sub]

    # Transfer filtered results to the main counters with the suffix applied
    for raw_sub in valid_suffix:
        final_key = f"{SUBWORD_PREFIX}{raw_sub}"
        morphemes_freq[final_key] += suffix_freq[raw_sub]
        scaled_subword_freq[final_key] += suffix_scaled_freq[raw_sub]
    """


def _filter_subwords_logic(freq_counter: Counter,
                           dictionary: Set[str],
                           is_prefix: bool) -> Tuple[Set[str], DefaultDict[str, list]]:
    """
    Core pruning logic. Identifies candidates to keep.
    """
    def get_redundant(sorted_list):
        to_remove = set()
        dict_skips = 1
        for i in range(len(sorted_list)):
            cand = sorted_list[i]
            # Check subsequent items in sorted list
            for j in range(i + 1, len(sorted_list)):
                nxt = sorted_list[j]

                if len(nxt) > len(cand) + dict_skips:  # filter duplicates with a single character difference
                    break

                if is_prefix:
                    if not nxt.startswith(cand): break
                else:
                    if not nxt.endswith(cand): break

                # If the shorter candidate is less or equally frequent than the longer one,
                # it's redundant (part of a more common morpheme)
                if freq_counter[cand] <= freq_counter[nxt]:
                    if cand in dictionary:
                        dict_skips += 1
                        continue  # do not remove valid dictionary words

                    replaced_morpheme[nxt].append(cand)
                    to_remove.add(cand)
                    break

        return to_remove

    replaced_morpheme = defaultdict(list)

    # 1. Frequency thresholding
    candidates = [word for word, count in freq_counter.items() if count > 1 or word in dictionary]

    if is_prefix:
        # Standard Sort for Prefix Redundancy
        candidates.sort()
    else:
        # Reversed String Sort for Suffix Redundancy
        candidates.sort(key=lambda x: x[::-1])

    redundant_candidates = get_redundant(candidates)
    candidates = set(c for c in candidates if c not in redundant_candidates)

    return candidates, replaced_morpheme


@log_execution_time
def discover_new_morphemes_old(word_splits_cache: Dict[str, List[Tuple[int, str]]], root_min_len: int = 2):
    """
    Discover new potential prefix/suffix

    Args:
        word_splits_cache: Dictionary mapping of word and subword splits e.g. {"sunflower": [(0, 'sun'), (2, 'un'),
        (3, 'flower'), (3, 'flow'), (4, 'low'), (7, 'er')]}.
        root_min_len: The minimum length of root words to consider.
    """
    logger.info(f"Discovering new morphemes by word segmentation for {len(word_splits_cache)} words...")
    potential_prefixes = defaultdict(set)
    potential_suffixes = defaultdict(set)
    all_morphemes = {}
    new_split_count = 0
    for word in word_splits_cache:
        max_len = len(word) - 1
        all_morphemes[word] = []
        suffix_idx = 0
        prefix_idx = -1
        prefix = ""
        suffix = ""
        pairs_found = set()
        for idx, pair in enumerate(word_splits_cache[word]):
            start, root = pair
            size = len(root)

            if size < root_min_len: continue

            if prefix_idx != start:
                prefix_idx = start
                prefix = word[:start] if start >= root_min_len else ""

            suffix_start = start + size

            if suffix_idx != suffix_start:
                suffix_idx = suffix_start
                suffix = word[suffix_idx:] if len(word) - suffix_start >= root_min_len else ""

            suffix_pair_name = f"{start}-{len(root)}&{suffix_idx}-{len(suffix)}"
            prefix_pair_name = f"0-{len(prefix)}&{start}-{len(root)}"
            suffix_added = suffix_pair_name in pairs_found
            prefix_added = prefix_pair_name in pairs_found
            suffix_req_met = not suffix_added and suffix and max_len >= len(suffix) and len(root) >= root_min_len
            prefix_req_met = not prefix_added and prefix and max_len >= len(prefix) and len(root) >= root_min_len

            if prefix_req_met:
                all_morphemes[word].append([(0, prefix), (start, root)])
                pairs_found.add(prefix_pair_name)

            if suffix_req_met:
                all_morphemes[word].append([(start, root), (suffix_idx, suffix)])
                pairs_found.add(suffix_pair_name)

    logger.info(f"Created new morphemes for {new_split_count} words.")
    # Select most frequent morphemes for each word as candidates
    word_morphemes = {}
    for word in all_morphemes:
        for best_pair in all_morphemes.get(word, []):
            if len(best_pair) > 1:
                suffix = best_pair[1][1]
                if len(suffix) >= root_min_len:
                    if best_pair[0][1] == 'anatomico' and not word_morphemes:
                        word_morphemes = all_morphemes[word]
                        print(f"Using word_morphemes={word_morphemes} for word {word}")
                    potential_prefixes[best_pair[0][1]].add(suffix)

                if len(best_pair[0][1]) >= root_min_len:  # Add inflectional suffix candidate
                    potential_suffixes[suffix].add(best_pair[0][1])

                if len(best_pair) > 2 and len(suffix) >= root_min_len:  # Add derivational suffix candidate
                    potential_suffixes[best_pair[2][1]].add(suffix)

        if len(potential_prefixes) == 15:
            print(f"Current potential_prefixes: {potential_prefixes}")

        if len(potential_suffixes) == 15:
            print(f"Current potential_suffixes: {potential_suffixes}")

    return potential_prefixes, potential_suffixes


def count_normalized_stems_old(affixes_by_len, dictionary, stems_to_strip, strip_type):
    variety = 0
    normalized_stems = set()
    current_root = None
    for stem in sorted(stems_to_strip):  # Sort alphabetically
        # Count variety based on prefix grouping of *original* stems
        if current_root is None or not stem.startswith(current_root):
            variety += 1
            current_root = stem

        normalized_stem = stem
        found_strip = False
        for length, affixes in affixes_by_len.items():
            if len(stem) > length:  # Ensure stem is longer than affix
                if strip_type == "suffix":
                    affix_part = stem[-length:]
                    normalized_stem = stem[:-length]  # Ensure normalize stems are still valid words
                    if affix_part in affixes and normalized_stem in dictionary:
                        found_strip = True
                        break  # Found longest affix
                else:  # strip_type == "prefix"
                    affix_part = stem[:length]
                    normalized_stem = stem[length:]
                    if affix_part in affixes and normalized_stem in dictionary:
                        found_strip = True
                        break  # Found longest affix
        if found_strip:
            normalized_stems.add(normalized_stem)
        else:
            normalized_stems.add(stem)  # Add the original if no strip

    return variety, normalized_stems


def final_stem_variety_check_old(
        candidate_morphemes: Set[str],
        potential_morphemes_db: Dict[str, Set[str]],
        affixes_to_strip: Set[str],
        dictionary: Set[str],
        strip_type: str = "suffix"
) -> Dict[str, List[str]]:
    """
    Performs the final stem normalization and variety check on candidate morphemes.
    This is run *after* a complete list of affixes is available.

    Args:
        candidate_morphemes: The set of morphemes that passed initial filtering.
        potential_morphemes_db: The original full dict of {morpheme: Set(stems)}.
        affixes_to_strip: The *complete* set of affixes to strip (e.g., all known + all candidates).
        dictionary: language dictionary with valid meaningful words.
        strip_type: "suffix" (to strip from end) or "prefix" (to strip from start).

    Returns:
        Tuple[Set[str], Set[str]]:
        - A set of morphemes that passed the *final* normalization check.
        - A set of all stems associated with the *final* passing morphemes.
    """
    logger.info(f"Running final normalization check on {len(candidate_morphemes)} candidate morphemes...")

    final_passed_morphemes = {}

    if not candidate_morphemes:
        return final_passed_morphemes

    # Pre-group the affixes to strip by length for efficient stripping
    affixes_by_len = _group_by_length(affixes_to_strip)

    candidate_morphemes = sorted(candidate_morphemes)  # Sort alphabetically to compare next candidate
    duplicates = {'acknowl', 'acknowle'}

    for index, morpheme in enumerate(candidate_morphemes):
        stems_to_strip = {x for x in potential_morphemes_db.get(morpheme, set()) if x in dictionary}

        if len(stems_to_strip) < MIN_STEM_VARIETY: continue

        # --- 1. Normalize Stems and Calculate Variety ---
        variety, normalized_stems = count_normalized_stems(
            affixes_by_len, dictionary, stems_to_strip, strip_type
        )

        if morpheme in duplicates:
            print(f"Using variety={variety} with stem: {stems_to_strip} for morpheme={morpheme} & \nnormalized_stems={normalized_stems}")

        if len(normalized_stems) < MIN_STEM_VARIETY: continue

        try:
            # This logic is *only* for adjusting the *current* 'variety'
            # based on the next morpheme in the list.
            next_morpheme = candidate_morphemes[index + 1]
            size = len(next_morpheme) - len(morpheme)
            if size == 1 and morpheme == next_morpheme[:len(morpheme)]:  # Verify current morpheme is a substring of the next.
                next_stems = potential_morphemes_db.get(next_morpheme, set())
                next_variety, _ = count_normalized_stems(
                    affixes_by_len, dictionary, next_stems, strip_type
                )
                # 1. Verify the current and next morpheme will pass the variety test
                if next_variety >= MIN_STEM_VARIETY and variety >= MIN_STEM_VARIETY:
                    # 2. Ensure the current morpheme will still pass the variety test if duplicates are removed. e.g.
                    # morpheme1 = 'acknow' with stems: {'ing', 'ledged', 'ledges', 'ledging', 'ledge'}
                    # morpheme2 = 'acknowl' with stems: {'edged', 'edges', 'edging', 'edge'}
                    # ignore duplicates by concatenating the difference to 'acknow' stems:
                    # {'l'+'edged', 'l'+'edges', 'l'+'edging', 'l'+'edge'}, so only {'l'+'ing'} will be counted.
                    diff_str = next_morpheme[-size:]
                    variety = MIN_STEM_VARIETY - 3  # Will need at least 3 non-duplicated stems to pass
                    if strip_type == "prefix":  # Append difference to end of stem
                        for stem in next_stems:  # Ensure normalize stems are still valid words
                            normalized_stem = f"{stem}{diff_str}"
                            if normalized_stem in stems_to_strip:
                                continue  # do not count duplicate stem
                            else:
                                variety += 1
                    else:  # strip_type == "suffix". Append difference to start of stem
                        for stem in next_stems:
                            normalized_stem = f"{diff_str}{stem}"
                            if normalized_stem in stems_to_strip:
                                continue  # do not count duplicate stem
                            else:
                                variety += 1
        except IndexError:
            pass
        # --- 2. Make final decision based on *normalized* variety ---
        if variety >= MIN_STEM_VARIETY:
            final_passed_morphemes[morpheme] = sorted(stems_to_strip)  # TODO - remove sort

    logger.info(f"Final normalization check complete. {len(final_passed_morphemes)} morphemes passed.")
    return final_passed_morphemes

def _get_fast_roots(stems, affix_set, affix_lengths, dictionary, is_suffix_mode):
    """
    Collapses stems to their base roots in O(L) time.
    """
    normalized = set()
    # Priority: Dict words, then length DESC (process longer stems first)
    sorted_stems = sorted(stems, key=lambda x: (x in dictionary, len(x)), reverse=True)

    for stem in sorted_stems:
        redundant = False
        # Only check against valid affix lengths
        for L in affix_lengths:
            if len(stem) <= L: continue

            # Split stem to see if it's a root + affix
            root_cand, affix_cand = (stem[:-L], stem[-L:]) if is_suffix_mode else (stem[L:], stem[:L])

            if affix_cand in affix_set:
                # If root is in dict or already in our set, this stem is just a variant
                if root_cand in normalized or (len(root_cand) > 2 and root_cand in dictionary):
                    redundant = True
                    break

        if not redundant:
            normalized.add(stem)

    return normalized

@log_execution_time
def final_stem_variety_check_new(
        lang_code,
        candidate_morphemes,
        potential_morphemes_db,
        affixes_to_strip,
        dictionary,
        strip_type="suffix"
):
    is_suffix_mode = (strip_type == "suffix")
    affix_set = set(affixes_to_strip)
    # Cache lengths to avoid repeated set(len()) calls
    affix_lengths = sorted(set(len(a) for a in affixes_to_strip), reverse=True)

    # 1. Global Reconstruction Map (Constant time lookup)
    # Key: The combined word, Value: The high-priority morpheme that 'claimed' it
    file_name = f"{lang_code}_{strip_type}_{WORD_REGISTRY_FILE_NAME}"
    file_path = os.path.join(STORAGE_DIR, file_name)
    word_registry = load_json_file(file_path)

    # 2. Sort by Priority: Dictionary words first, then length (Longer = more specific)
    sorted_morphemes = sorted(
        candidate_morphemes,
        key=lambda x: (x in dictionary, len(x)),
        reverse=True
    )

    final_results = {}

    # 3. Process morphemes in order of authority
    for m in sorted_morphemes:
        if is_suffix_mode and len(m) < 2:
            continue  # skip short prefix

        raw_stems = potential_morphemes_db.get(m, set())
        if len(raw_stems) < MIN_STEM_VARIETY:
            continue

        # Threshold check: Dict words pass regardless, others need variety
        m_in_dict = m in dictionary
        if not m_in_dict and len(m) <= 3:
            continue

        valid_stems = set()
        claimed_words = {}

        for s in raw_stems:
            if len(m) <= MIN_SUBWORD_LEN or len(s) <= MIN_SUBWORD_LEN:
                if not m_in_dict or s not in dictionary:
                    continue  # skip invalid stems, at both must be dictionary words
                s_in_dict = True
            else:
                s_in_dict = s in dictionary

            if not m_in_dict and not s_in_dict:
                continue  # skip invalid stems, at lease one must be a valid dictionary word

            # Near-Constant Time Check
            # Universal Reconstruction: m + s for suffix, s + m for prefix
            # Handles "abnormal"+"ism" vs "abnormali"+"sm" instantly
            full_word = (m + s) if is_suffix_mode else (s + m)
            if full_word in word_registry or full_word not in dictionary:
                # If claimed by a better morpheme (like 'abnormal'), skip this stem
                continue

            # Claim this word for this morpheme
            claimed_words[full_word] = m
            valid_stems.add(s)

        if len(valid_stems) < MIN_STEM_VARIETY:
            continue

        # 4. Fast Root Normalization
        roots = _get_fast_roots(valid_stems, affix_set, affix_lengths, dictionary, is_suffix_mode)

        if m in dictionary or len(roots) >= MIN_STEM_VARIETY or len(valid_stems) >= MIN_STEM_VARIETY:
            final_results[m] = {
                "stems": sorted(list(valid_stems), key=lambda x: (x in dictionary, len(x)), reverse=True),
                "roots": sorted(list(roots), key=lambda x: (x in dictionary, len(x)), reverse=True)
            }
            # Important: Update word registry with claimed words only if morpheme pass
            for word, morpheme in claimed_words.items():
                if morpheme in roots:
                    word_registry[word] = morpheme

    save_json_file(STORAGE_DIR, file_name, word_registry)
    if is_suffix_mode:
        logger.info(f"Added {len(final_results)} new prefix morphemes.")
    else:
        logger.info(f"Added {len(final_results)} new suffix morphemes.")

    return final_results


def final_stem_variety_check_old_best(
        candidate_morphemes: Set[str],
        potential_morphemes_db: Dict[str, Set[str]],
        dictionary: Set[str],
        is_suffix_mode: bool = True,
        threshold: int = MIN_STEM_VARIETY
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Validates morphemes based on stem variety and populates the word registry.
    """
    word_registry: Dict[str, str] = {}
    trie = MorphemeLineageTrie(is_suffix_mode=is_suffix_mode)
    final_results = {}

    # Sort by length descending for greedy matching
    sorted_m = sorted(candidate_morphemes, key=len, reverse=True)

    for m in sorted_m:
        stems = potential_morphemes_db.get(m, set())
        valid_stems = set()

        # Validate stems against the dictionary
        for s in stems:
            full_word = (s + m) if is_suffix_mode else (m + s)
            if full_word in dictionary:
                valid_stems.add(s)

        # Threshold check or dictionary membership
        if len(valid_stems) >= threshold or m in dictionary:
            if not trie.is_redundant(m):
                trie.insert(m)
                final_results[m] = {"stems": sorted(valid_stems)}
                # Add confirmed pairings to registry
                for s in valid_stems:
                    full_word = (s + m) if is_suffix_mode else (m + s)
                    word_registry[full_word] = m

    # Use Trie to catch "odd" but valid morphemes in the rest of the dictionary
    word_registry = split_unclaimed_words(dictionary, word_registry, trie)

    return final_results, word_registry


class MorphemeLineageTrie:
    """
    A specialized Trie for morpheme lookups.
    Updated to support finding ALL matches for qualitative evaluation.
    """
    def __init__(self, is_suffix_mode: bool = False):
        self.trie = {}
        self.is_suffix_mode = is_suffix_mode

    def set_mode(self, is_suffix_mode: bool):
        """Dynamically toggle between prefix (False) and suffix (True) mode."""
        self.is_suffix_mode = is_suffix_mode

    def insert(self, morpheme: str):
        text = morpheme[::-1] if self.is_suffix_mode else morpheme
        node = self.trie
        for char in text:
            node = node.setdefault(char, {})
        node['_is_morpheme'] = True

    def get_all_valid_prefixes_at(self, text: str) -> List[str]:
        """
        Given a string (or substring), returns all morphemes
        starting at index 0 found in the Trie.
        """
        node = self.trie
        matches = []
        curr = ""
        for char in text:
            if char not in node:
                break
            node = node[char]
            curr += char
            if '_is_morpheme' in node:
                matches.append(curr)
        return matches

    def find_all_matches(self, word: str) -> List[str]:
        """Returns all valid morphemes found at the start (or end) of the word."""
        text = word[::-1] if self.is_suffix_mode else word
        node = self.trie
        matches = []
        curr = ""
        for char in text:
            if char not in node: break
            node = node[char]
            curr += char
            if '_is_morpheme' in node:
                m_found = curr[::-1] if self.is_suffix_mode else curr
                matches.append(m_found)

        return matches

    def find_longest_match(self, word: str) -> str:
        """Finds the longest valid morpheme match for a word segment."""
        text = word[::-1] if self.is_suffix_mode else word
        node = self.trie
        match = ""
        curr = ""
        for char in text:
            if char not in node: break
            node = node[char]
            curr += char
            if '_is_morpheme' in node:
                match = curr

        return match[::-1] if self.is_suffix_mode else match

    def is_redundant(self, morpheme: str) -> bool:
        text = morpheme[::-1] if self.is_suffix_mode else morpheme
        node = self.trie
        for char in text[:-1]:
            if char not in node: break
            node = node[char]
            if '_is_morpheme' in node: return True

        return False


@log_execution_time
def discover_new_morphemes_new_old(word_splits_cache: Dict[str, Dict[int, List[str]]],
                           morphemes_freq: CounterType[str],
                           root_min_len: int = 2):
    """
    Efficiently discovers potential prefixes and suffixes by processing word segments.

    Args:
    word_splits_cache: Dictionary mapping of word and subword splits.
    root_min_len: The minimum length of root words to consider.
    """
    logger.info(f"Discovering new morphemes for {len(word_splits_cache)} words...")

    potential_prefixes = defaultdict(set)
    potential_suffixes = defaultdict(set)

    for word, splits in word_splits_cache.items():
        new_prefixes = {}
        new_suffixes = {}
        for start, groups in splits.items():
            for prefix in groups:
                prefix_len = len(prefix)
                # 1. Check for Potential Suffixes
                freq_limit = MIN_STEM_VARIETY * 2 if prefix_len < WORD_SIZE_LIMIT else MIN_STEM_VARIETY
                freq_key = f"{SUBWORD_PREFIX}{prefix}" if start > 0 else prefix
                suffix_req_met = morphemes_freq.get(freq_key, 0) >= freq_limit and prefix_len >= root_min_len
                suffix_start = start + prefix_len
                for suffix in splits.get(suffix_start, []):
                    suffix_len = len(suffix)
                    if suffix_req_met and prefix_len > len(new_suffixes.get(suffix, "")):
                        new_suffixes[suffix] = prefix

                    if prefix_len < 2: continue

                    # 2. Check for Potential Prefixes
                    freq_limit = MIN_STEM_VARIETY * 2 if suffix_len < WORD_SIZE_LIMIT else MIN_STEM_VARIETY
                    freq_key = f"{SUBWORD_PREFIX}{suffix}" if start > 0 else suffix
                    prefix_req_met = morphemes_freq.get(freq_key, 0) >= freq_limit and suffix_len >= root_min_len

                    if prefix_req_met and start < 1 and suffix_len > len(new_prefixes.get(prefix, "")):
                        new_prefixes[prefix] = suffix

        for suffix, root in new_suffixes.items():
            potential_suffixes[suffix].add(root)

        for prefix, root in new_prefixes.items():
            potential_prefixes[prefix].add(root)

    return dict(potential_prefixes), dict(potential_suffixes)

@log_execution_time
def discover_new_morphemes(word_splits_cache: Dict[str, Dict[int, List[str]]],
                           morphemes_freq: CounterType[str],
                           root_min_len: int = 3):
    """
    Discovers potential prefixes and suffixes. Prefix accumulation uses a set to properly track stem variety.
    """
    logger.info(f"Discovering new morphemes for {len(word_splits_cache)} words...")

    potential_prefixes = defaultdict(set)
    potential_suffixes = defaultdict(set)

    for word, splits in word_splits_cache.items():
        for start, groups in splits.items():
            for segment in groups:
                seg_len = len(segment)
                if seg_len < 2: continue # Ignore single chars during discovery

                # 2. Check for Potential Prefixes
                freq_limit = MIN_STEM_VARIETY * 2 if seg_len < WORD_SIZE_LIMIT else MIN_STEM_VARIETY
                freq_key = f"{SUBWORD_PREFIX}{segment}" if start > 0 else segment
                freq_req_met = morphemes_freq.get(freq_key, 0) >= freq_limit and seg_len >= root_min_len
                if not freq_req_met: continue

                # 1. Potential Suffixes (The segment is a suffix to a stem)
                # If there is a valid 'root' before this segment
                stem = word[:start]
                if stem and len(stem) >= root_min_len:
                    potential_suffixes[segment].add(stem)

                # 2. Potential Prefixes (The segment is a prefix to a stem)
                if start == 0:
                    stem = word[seg_len:]
                    if stem and len(stem) >= root_min_len:
                        potential_prefixes[segment].add(stem)

    return dict(potential_prefixes), dict(potential_suffixes)

@log_execution_time
def initial_morpheme_filtering(potential_morphemes: Dict[str, Set[str]],
                               known_morphemes: Set[str],
                               dictionary: Set[str],
                               morphemes_freq: CounterType[str],
                               is_suffix: bool = False) -> Tuple[Set[str], Set[str], List[str]]:
    """
    Performs the first pass of filtering.
    NEW: Now checks both Stem Variety AND Morpheme Frequency.
    """
    logger.info(f"Running initial filtering on {len(potential_morphemes)} potential morphemes...")

    candidate_morphemes = set()
    passed_stems = set()
    removed = []

    for new_morpheme, stems in potential_morphemes.items():
        # 1. Determine the key to look up in the frequency counter
        freq_key = f"{SUBWORD_PREFIX}{new_morpheme}" if is_suffix else new_morpheme
        morpheme_frequency = morphemes_freq.get(freq_key, 0)

        # 2. Check both Stem Variety (unique stems) and Global Frequency
        if (len(stems) >= MIN_STEM_VARIETY and
                morpheme_frequency >= MIN_STEM_VARIETY and
                new_morpheme not in known_morphemes):

            candidate_morphemes.add(new_morpheme)
            passed_stems.update(stems)

    morpheme_type = "suffixes" if is_suffix else "prefixes"
    logger.info(f"Removed {len(removed)} duplicate {morpheme_type}...")

    return candidate_morphemes, passed_stems, removed


def evaluate_morpheme_split(word: str,
                            morpheme: str,
                            is_suffix: bool,
                            dictionary: Set[str],
                            morphemes_freq: Counter) -> float:
    """
    Scores a potential split. Higher scores favor:
    1. Stems that are complete words in the dictionary.
    2. Longer stems (root-focused).
    3. Stems with high frequency.
    """
    if is_suffix:
        stem = word[:-len(morpheme)]
        pair = [(0, stem), (len(stem), morpheme)]
    else:
        stem = word[len(morpheme):]
        pair = [(0, morpheme), (len(morpheme), stem)]

    # Base score using the existing cubic length logic
    score = calculate_pair_score(pair, morphemes_freq)

    # Bonus: Does the stem exist in our dictionary?
    if stem in dictionary:
        score *= 2.0  # Heavily favor splits that leave a valid dictionary word

    # Penalty: Very short stems (unless the word itself is very short)
    if len(stem) < MIN_SUBWORD_LEN:
        score *= 0.5

    return score


def split_unclaimed_words(dictionary: Set[str],
                          word_registry: Dict[str, List[str]],
                          trie: MorphemeLineageTrie,
                          morphemes_freq: Counter) -> Dict[str, List[str]]:
    """
    Finds the BEST split for unclaimed words by evaluating all possible
    morpheme matches from the Trie.
    """
    new_morphemes = {}
    is_suffix = trie.is_suffix_mode

    for word in dictionary:
        if word not in word_registry:
            #matches = trie.find_all_matches(word)
            splits = viterbi_split(word, trie, morphemes_freq)
            if splits:
                # Frequency-weighted selection

                #best_match = select_best_morpheme(word, matches, is_suffix, dictionary, morphemes_freq)
                word_registry[word] = splits
                new_morphemes[word] = splits

    if trie.is_suffix_mode:
        logger.info(f"Added {len(new_morphemes)} new suffix to registry using Best-Split logic.")
        save_word_frequencies_json("suffix_new_word_registry.json", new_morphemes)
    else:
        logger.info(f"Added {len(new_morphemes)} new prefix to registry using Best-Split logic.")
        save_word_frequencies_json("prefix_new_word_registry.json", new_morphemes)

    save_word_frequencies_json("morpheme_lineage_trie.json", trie.trie)

    return word_registry


def select_best_morpheme(word: str,
                         matches: List[str],
                         is_suffix: bool,
                         dictionary: Set[str],
                         morphemes_freq: Counter) -> str:
    """
    Selects the best morpheme using a hierarchical strategy:
    1. Primary: Dictionary-Validated Stem (Root Preservation).
    2. Secondary: Longest Root among valid dictionary stems.
    3. Fallback: Highest Frequency Morpheme from corpus (Statistical Significance).
    4. Tie-breaker: Longest Morpheme (Structural Integrity).
    """
    if not matches:
        return ""

    valid_stem_candidates = []

    # 1. Primary Strategy: Check for stems in dictionary
    for m in matches:
        if is_suffix:
            stem = word[:-len(m)]
        else:
            stem = word[len(m):]

        if stem in dictionary:
            valid_stem_candidates.append((m, len(stem)))

    if valid_stem_candidates:
        # Sort by stem length descending (Longest Root First)
        valid_stem_candidates.sort(key=lambda x: x[1], reverse=True)
        return valid_stem_candidates[0][0]

    # 2. Fallback Strategy: Frequency-Weighted Choice
    # Use the morphemes_freq counter to find the most common morpheme
    # We look up the 'token' version (with ## for suffixes) to match freq storage
    match_frequencies = []
    for m in matches:
        token_key = f"{SUBWORD_PREFIX}{m}" if is_suffix else m
        freq = morphemes_freq.get(token_key, 0)
        # We also consider length as a secondary tie-breaker for frequency
        match_frequencies.append((m, freq, len(m)))

    # Sort by Frequency (desc), then by Length (desc)
    match_frequencies.sort(key=lambda x: (x[1], x[2]), reverse=True)

    return match_frequencies[0][0]


def viterbi_split(word: str,
                  trie: MorphemeLineageTrie,
                  morphemes_freq: Dict[str, int],
                  base_penalty: float = 2.0,
                  short_penalty: float = 15.0) -> List[str]:
    """
    Optimal subword split using Dynamic Programming.
    Updated to use Length-Weighted Frequency scoring to prioritize longer roots.
    """
    n = len(word)
    if n == 0: return []

    # best_scores[i] is the max score found to reach index i
    best_scores = [-math.inf] * (n + 1)
    best_scores[0] = 0
    best_splits = [0] * (n + 1)

    for i in range(n):
        if best_scores[i] == -math.inf:
            continue

        remainder = word[i:]
        candidates = trie.get_all_valid_prefixes_at(remainder)

        for morpheme in candidates:
            m_len = len(morpheme)
            end_idx = i + m_len

            # Formatting key for frequency lookup
            key = f"{SUBWORD_PREFIX}{morpheme}" if i > 0 else morpheme
            freq = morphemes_freq.get(key, 0)

            if freq > 0:
                # --- Advanced Scoring Logic ---
                # We use length squared to heavily weight longer subwords.
                # Score = (length^2) * log10(frequency)
                # This ensures a 4-char morpheme beats four 1-char morphemes easily.
                length_weight = m_len ** 2
                current_score = best_scores[i] + (length_weight * math.log10(freq))

                # Apply penalties for transition
                current_score -= base_penalty

                # Harsh penalty for subwords shorter than the minimum threshold
                if m_len < MIN_SUBWORD_LEN:
                    current_score -= short_penalty

                if current_score > best_scores[end_idx]:
                    best_scores[end_idx] = current_score
                    best_splits[end_idx] = i

    if best_scores[n] == -math.inf:
        return []

    # Backtrack
    result = []
    curr = n
    while curr > 0:
        prev = best_splits[curr]
        segment = word[prev:curr]
        result.append(f"{SUBWORD_PREFIX}{segment}" if prev > 0 else segment)
        curr = prev

    return result[::-1]


def split_and_register(dictionary: Set[str],
                       trie: MorphemeLineageTrie,
                       morphemes_freq: Counter) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Splits words and populates prefix/suffix registries.
    Assumes Trie contains both prefixes and suffixes in forward-facing order
    for the Viterbi algorithm to work correctly.
    """
    prefix_registry = {}
    suffix_registry = {}

    # Ensure Trie is in forward mode for Viterbi processing
    original_mode = trie.is_suffix_mode
    trie.set_mode(False)

    for word in dictionary:
        # Higher penalty to avoid over-segmentation
        splits = viterbi_split(word, trie, morphemes_freq, base_penalty=2.0, short_penalty=12.0)

        if splits and len(splits) > 1:
            prefix_registry[word] = [splits[0]]
            suffix_registry[word] = splits[1:]

    # Restore original mode if necessary
    trie.set_mode(original_mode)
    return prefix_registry, suffix_registry

@log_execution_time
def final_stem_variety_check(
        prefix_candidates: Set[str],
        suffix_candidates: Set[str],
        potential_morphemes_db: Dict[str, Set[str]],
        dictionary: Set[str],
        morphemes_freq: Counter,
        threshold: int = MIN_STEM_VARIETY) -> Tuple[Dict[str, Any], Dict[str, List[str]], Dict[str, List[str]]]:

    final_morphemes_meta = defaultdict(set)
    trie = MorphemeLineageTrie()

    # 1. Process Prefix Candidates
    trie.set_mode(False)  # Ensure prefix mode
    for m in prefix_candidates:
        stems = potential_morphemes_db.get(m, set())
        freq = morphemes_freq.get(m, 0)

        if (freq >= threshold and len(stems) >= threshold) or m in dictionary:
            if not trie.is_redundant(m):
                trie.insert(m)
                final_morphemes_meta[m].update(stems)

    # 2. Process Suffix Candidates
    trie.set_mode(True)  # Switch to suffix mode for redundancy check/reversed insertion
    for m in suffix_candidates:
        stems = potential_morphemes_db.get(m, set())
        key = f"{SUBWORD_PREFIX}{m}"
        freq = morphemes_freq.get(key, 0)

        if (freq >= threshold and len(stems) >= threshold) or m in dictionary:
            # Suffix redundancy check works best when reversed (e.g., 'ing' vs 'ning')
            if not trie.is_redundant(m):
                # Note: If viterbi_split needs suffixes in forward order,
                # we would need a second Trie or a multi-pass approach.
                # Here we follow the logic of switching for the check.
                trie.insert(m)
                final_morphemes_meta[m].update(stems)

    # Convert meta to final format
    final_morphemes_meta = {m: {"stems": sorted(list(stems))} for m, stems in final_morphemes_meta.items()}

    # 3. Process dictionary to fill registries
    # split_and_register will internally set mode to False for Viterbi
    prefix_reg, suffix_reg = split_and_register(dictionary, trie, morphemes_freq)

    return final_morphemes_meta, prefix_reg, suffix_reg


@log_execution_time
def run_morpheme_pipeline(lang_code,
                          word_splits_cache: Dict[str, Dict[int, List[str]]],
                          morphemes_freq: CounterType[str],
                          dictionary: Set[str],
                          prefixes: Set[str],
                          suffixes: Set[str],
                          root_words: Set[str]
                          ) -> Tuple[Set[str], Set[str], Set[str]]:
    logger.info("--- Pipeline Step 1: Discovering New Morphemes ---")

    save_word_frequencies_json('word_splits_cache.json', word_splits_cache)
    potential_prefixes, potential_suffixes = discover_new_morphemes(word_splits_cache, morphemes_freq)

    save_word_frequencies_json("potential_prefixes.json", {k: list(v) for k, v in potential_prefixes.items()})
    save_word_frequencies_json("potential_suffixes.json", {k: list(v) for k, v in potential_suffixes.items()})

    logger.info("--- Pipeline Step 2 (Pass 1): Initial Filtering ---")
    # Added morphemes_freq and is_suffix flag
    candidate_prefixes, _, prefix_removed = initial_morpheme_filtering(
        potential_prefixes, prefixes, dictionary, morphemes_freq, is_suffix=False
    )
    candidate_suffixes, _, suffix_removed = initial_morpheme_filtering(
        potential_suffixes, suffixes, dictionary, morphemes_freq, is_suffix=True
    )
    deleted = {"prefix_removed": prefix_removed, "suffix_removed": suffix_removed}
    save_word_frequencies_json("duplicates_removed.json", deleted, lang_code)

    logger.info("--- Pipeline Step 3 (Pass 2): Final Normalization Check ---")
    # all_candidate_suffixes = suffixes.union(candidate_suffixes)
    # all_candidate_prefixes = prefixes.union(candidate_prefixes)
    all_candidates = deepcopy(potential_prefixes)
    all_candidates.update(potential_suffixes)

    # 1. Process prefixes
    final_passed_morphemes, prefix_reg, suffix_reg = final_stem_variety_check(
        candidate_prefixes,
        candidate_suffixes,
        potential_prefixes,
        dictionary,
        morphemes_freq
    )
    new_prefixes = [x for stems in prefix_reg.values() for x in stems]
    new_suffixes = [x for stems in suffix_reg.values() for x in stems]
    logger.info(f"Added {len(new_prefixes)} new prefixes & {len(new_suffixes)} new suffixes.")
    file_name = f"{lang_code}_prefix_{WORD_REGISTRY_FILE_NAME}"
    save_json_file(STORAGE_DIR, file_name, prefix_reg)
    file_name = f"{lang_code}_suffix_{WORD_REGISTRY_FILE_NAME}"
    save_json_file(STORAGE_DIR, file_name, suffix_reg)

    save_word_frequencies_json(f"lookup_dictionary.json", sorted(dictionary))
    save_word_frequencies_json(f"final_passed_morphemes.json", final_passed_morphemes)

    new_added_roots = {s for stems in final_passed_morphemes.values() for s in stems['stems']}.difference(dictionary)

    logger.info(f"Added {len(new_added_roots)} new roots.")
    #suffix_roots = [x for x in final_suffix_stems if morphemes_freq.get(x, 0) >= MIN_STEM_VARIETY]
    #prefix_roots = [x for x in final_prefix_stems if morphemes_freq.get(f"{SUBWORD_PREFIX}{x}", 0) >= MIN_STEM_VARIETY]

    prefixes.update(new_prefixes)
    suffixes.update(new_suffixes)
    root_words.update(new_added_roots)

    return prefixes, suffixes, root_words


class TestMorphemePipeline(unittest.TestCase):
    def setUp(self):
        # 4 examples of 'ing' (meets threshold)
        self.dictionary = {
            "working", "playing", "ending", "jumping",
            "unhappy", "unbound", "unseen", "unjust",
            "atypical"
        }
        self.potential_suffixes = {
            "ing": {"work", "play", "end", "jump"}
        }
        self.potential_prefixes = {
            "un": {"happy", "bound", "seen", "just"},
            "a": {"typical"}
        }

    def test_trie_pruning(self):
        """Confirm that nested morphemes (e.g., 'ing' inside 'eting') are pruned."""
        trie = MorphemeLineageTrie(is_suffix_mode=True)
        trie.insert("ing")
        self.assertTrue(trie.is_redundant("eting"))
        self.assertFalse(trie.is_redundant("able"))

    def test_unclaimed_splitting(self):
        """Confirm that valid morphemes are applied to words missing from registry."""
        trie = MorphemeLineageTrie(is_suffix_mode=False)
        trie.insert("un")
        # 'unjust' is in dict but we start with empty registry
        registry = split_unclaimed_words(self.dictionary, {}, trie)
        self.assertEqual(registry.get("unjust"), "un")

    def test_variety_check_integration_fixed(self):
        """Fixed test with correct threshold and example counts."""
        # Use threshold=3 to match the previous failure logic, or 4 for current setup
        results, registry = final_stem_variety_check(
            {"ing"}, self.potential_suffixes, self.dictionary, True, threshold=4
        )
        self.assertIn("ing", results)
        self.assertEqual(registry["working"], "ing")

    def test_multilingual_unicode(self):
        """Ensure the Trie handles non-ASCII characters for global support."""
        dict_cyrillic = {"переделать", "переписать"}
        pot_pref = {"пере": {"делать", "писать"}}
        results, registry = final_stem_variety_check(
            {"пере"}, pot_pref, dict_cyrillic, False, threshold=2
        )
        self.assertIn("пере", results)
        self.assertEqual(registry["переделать"], "пере")


def get_shards_to_load(
        file_name_template: str,
        total_number_of_shards: int,
        shards_to_skip: int,
        number_of_shards_to_load: int
) -> List[str]:
    """
    Generates a list of shard filenames to load, skipping a specified number.
    """
    start_shard = shards_to_skip
    end_shard = min(start_shard + number_of_shards_to_load, total_number_of_shards)

    if start_shard >= total_number_of_shards:
        return []

    shards_to_load = [
        file_name_template.format(i) for i in range(start_shard, end_shard)
    ]

    return shards_to_load


def load_c4_dataset_shards(split='train', streaming=True, shards=None) -> Optional[IterableDataset]:
    """
    Loads a specific split of the C4 dataset, with an option to load by specific shards.
    """
    try:
        # This will now log the full list of shards to dataset_processing.log
        logger.debug(f"Loading shard group with {len(shards)} files...\n{shards}")

        data_files = {split: shards} if shards is not None else None

        dataset = load_dataset(
            'allenai/c4',
            'en',
            split=split,
            data_files=data_files,
            streaming=streaming,
            trust_remote_code=True
        )

        return dataset

    except Exception as e:
        logger.error(f"Cannot load C4 dataset split '{split}'", exc_info=e)
        return None


def remove_single_spaces(tokens):
    """
    Removes single spaces to reconstruct words.
    Logic:
    - KEEP space if BOTH surrounding words are Capitalized (e.g. "Hello World").
    - REMOVE space if ANY surrounding word is lowercase (e.g. "camel Case", "this Is", "key board").
    """
    if len(tokens) < 3:
        return tokens

    selected_tokens = []
    max_idx = len(tokens) - 1

    for i, token in enumerate(tokens):
        # Check if current token is a single space, and we are not at the boundaries
        if 0 < i < max_idx and len(token) == 1 and re.match(SPACE_PATTERN, token):
            prev_token = tokens[i - 1]
            next_token = tokens[i + 1]

            if prev_token[0].isalpha() and next_token[0].isalpha():  # Check if neighbors are valid words
                continue  # Remove single space token
            else:
                # Keep space if neighbors are not both words (e.g. punctuation)
                selected_tokens.append(token)
        else:
            selected_tokens.append(token)

    return selected_tokens


def add_single_spaces(tokens):
    """
    Adds a single space between valid words.
    """
    selected_tokens = []
    max_idx = len(tokens) - 1

    for i, token in enumerate(tokens):
        selected_tokens.append(token)

        if i < max_idx:
            next_token = tokens[i + 1]

            if token[0].isalpha() and next_token[0].isalpha():  # Check if current and next are both valid words
                selected_tokens.append(' ')

    return selected_tokens


def save_words_binary(word_list, filename):
    """
    Saves a list of words to a binary file using Python's most
    efficient serialization method (pickle).

    Args:
        word_list (Any): The list/Counter of strings to save.
        filename (str): The name of the file to create.
    """
    # print(f"Saving {len(word_list)} words to {filename}...") # DEBUG
    with open(filename, 'wb') as f:
        # pickle.dump serializes the Python object directly into a file.
        # HIGHEST_PROTOCOL is the most efficient binary version available.
        pickle.dump(word_list, f, pickle.HIGHEST_PROTOCOL)


def load_words_binary(filename):
    """
    Loads a list of words from a binary pickle file.

    This function performs the "read" operation. pickle.load() is
    highly optimized (written in C) to deserialize the binary data
    back into a complete Python object (our list of words) very quickly.

    Args:
        filename (str): The name of the binary file to read.

    Returns:
        list: The deserialized list of words.
    """
    with open(filename, 'rb') as f:
        # pickle.load() reads the binary data and reconstructs the
        # Python object in one highly-optimized operation.
        word_list = pickle.load(f)

    return word_list


def normalize_text_for_nlp(text: str) -> str:
    """
    Applies a robust 3-stage normalization pipeline to clean text for NLP.

    1. NFKC normalization for compatibility chars (punctuation, ligatures, Zs spaces).
    2. Single-pass translation to delete zero-width/format chars and standardize controls.
    3. 1-to-4 Tab expansion, run last for maximum efficiency.
    """
    if not isinstance(text, str):  # Robustness check for non-string input
        return ""

    # --- 1. NFKC Normalization (Heavy lifting) ---
    # This is the key first step. It normalizes all 17 "Zs" (Space Separator)
    # characters (like U+3000, U+2009, U+00A0) into a standard U+0020 space.
    # It also handles ligatures (e.g., 'ﬁ' -> 'fi') and other compatibility chars.
    text = unicodedata.normalize('NFKC', text)

    # --- 2. Combined Translation (1-to-1 and 1-to-0) ---
    # Uses the pre-computed map for maximum efficiency.
    # This pass deletes a wide range of 'Cf' (Format) and zero-width
    # characters. It also standardizes all remaining C0/C1 control
    # characters and Unicode line breaks to a single space.
    text = text.translate(_ROBUST_TRANSLATION_MAP)

    # --- 3. Tab Expansion (1-to-4) ---
    # This must be run *after* normalization and translation.
    # We can't use the translation map as it's a 1-to-many replacement.
    text = text.replace('\t', '    ')

    return text


def _download_shard_task(args):
    """
    Task 1: Downloads a file from a URL to a temp path.
    Returns: ("DOWNLOAD_COMPLETE", (url, temp_file_path)) or ("DOWNLOAD_FAILED", url)
    """
    url, temp_dir, queue = args
    socket.setdefaulttimeout(60.0)
    # Create temp file
    fd, temp_filename = tempfile.mkstemp(suffix=".json.gz", dir=temp_dir)

    # Setup logger for this worker
    _logger = setup_logging(file_name=f"download_{os.getpid()}.log")

    try:
        _logger.info(f"Worker {os.getpid()} downloading: {url}")
        os.close(fd)

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(temp_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        _logger.info(f"Worker {os.getpid()} finished download: {temp_filename}")
        queue.put(("DOWNLOAD_COMPLETE", (url, temp_filename)))

    except Exception as e:
        _logger.error(f"Download failed for {url}: {e}")
        queue.put(("DOWNLOAD_FAILED", url))
        if 'temp_filename' in locals() and os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass


def _process_chunk_worker(args):
    """
    Task 2: Processes a specific chunk of lines from a local file.
    Returns: ("CHUNK_COMPLETE", (source_file_path, result_bin_path))
    """
    file_path, start_line, num_lines, text_column, unicode_pattern, temp_dir, queue = args
    chunk_id = f"{os.getpid()}_{start_line}"
    try:
        chunk_counter = Counter()
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:  # Efficiently read lines from gzip
            # Skip to start line using islice
            lines_gen = itertools.islice(f, start_line, start_line + num_lines)
            lines_buffer = list(lines_gen)

        for line in lines_buffer:
            if not line: continue
            try:
                data = json.loads(line)
                text = data.get(text_column, "")
                if text:
                    text = normalize_text_for_nlp(text)
                    tokens = remove_single_spaces(re.findall(unicode_pattern, text))
                    chunk_counter.update(tokens)
            except json.JSONDecodeError:
                continue

        # Save result
        fd, res_filename = tempfile.mkstemp(suffix=f"_chunk_{chunk_id}.bin", dir=temp_dir)
        os.close(fd)
        save_words_binary(chunk_counter, res_filename)

        # Return the source file path so the main thread knows which file this chunk belongs to
        queue.put(("CHUNK_COMPLETE", (file_path, res_filename)))

    except Exception as e:
        print(f"Chunk worker error on {file_path} lines {start_line}: {e}")
        queue.put(("CHUNK_FAILED", file_path))


def build_word_frequencies_from_c4(
        file_name_template: str,
        total_shards: int,
        shards_to_skip: int,
        total_shards_to_load: int,
        text_column: str,
        num_workers: int
) -> CounterType[str]:
    """
    Orchestrates the multi-process download and frequency counting using a
    queue for real-time aggregation and a retry mechanism.
    """
    logger.info(f"Building word frequencies from {total_shards_to_load} shards...")
    start_time = time.time()
    temp_dir = tempfile.mkdtemp(prefix="c4_pipeline_")
    logger.info(f"Created pipeline temp dir: {temp_dir}")

    word_frequency = Counter()
    manager = Manager()
    queue = manager.Queue()

    pending_urls = get_shards_to_load(file_name_template, total_shards, shards_to_skip, total_shards_to_load)

    # Track progress
    active_downloads = 0
    active_chunks = 0

    # file_chunk_tracker: Maps downloaded_file_path -> number_of_chunks_remaining
    # Used to determine when to delete the downloaded file.
    file_chunk_tracker = {}

    # We consider "completed" when the file is downloaded AND processed AND deleted
    completed_shards_count = 0
    total_shards_expected = len(pending_urls)

    retry_counts = Counter()
    MAX_RETRIES = 2

    try:
        # --- Standard Pool (No nested context/daemon issues) ---
        pool = Pool(processes=num_workers)
        logger.info(f"Pipeline started. Pool size: {num_workers}")

        while completed_shards_count < total_shards_expected:

            # 1. Submit Downloads if slots available
            while pending_urls and active_downloads < MAX_CONCURRENT_DOWNLOADS:
                url = pending_urls.pop(0)
                args = (url, temp_dir, queue)
                pool.apply_async(_download_shard_task, args=(args,))
                active_downloads += 1
                logger.info(f"Submitted DOWNLOAD task. Active Downloads: {active_downloads}")

            # 2. Event Loop
            try:
                message = queue.get()
                msg_type, payload = message

                if msg_type == "DOWNLOAD_COMPLETE":
                    url, file_path = payload
                    active_downloads -= 1

                    # Calculate chunks IN MAIN THREAD (as requested)
                    try:
                        line_count = 0
                        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                            for _ in f: line_count += 1

                        if line_count == 0:
                            logger.warning(f"File empty: {file_path}")
                            os.remove(file_path)
                            completed_shards_count += 1
                            continue

                        num_chunks = math.ceil(line_count / LINES_PER_CHUNK)
                        logger.info(f"Shard ready: {file_path}. Submitting {line_count} lines, {num_chunks} chunks...")

                        # Initialize tracker for this file
                        file_chunk_tracker[file_path] = num_chunks

                        # Submit chunk tasks
                        for i in range(num_chunks):
                            start_line = i * LINES_PER_CHUNK
                            chunk_args = (file_path, start_line, LINES_PER_CHUNK, text_column, UNICODE_PATTERN, temp_dir, queue)
                            pool.apply_async(_process_chunk_worker, args=(chunk_args,))
                            active_chunks += 1

                    except Exception as e:
                        logger.error(f"Error preparing chunks for {file_path}: {e}")
                        if os.path.exists(file_path): os.remove(file_path)
                        completed_shards_count += 1

                elif msg_type == "DOWNLOAD_FAILED":
                    url = payload
                    active_downloads -= 1
                    retry_counts[url] += 1
                    if retry_counts[url] <= MAX_RETRIES:
                        logger.warning(f"Retrying download: {url}")
                        pending_urls.insert(0, url)
                    else:
                        logger.error(f"Permanently failed download: {url}")
                        completed_shards_count += 1

                elif msg_type == "CHUNK_COMPLETE":
                    source_file, res_filename = payload
                    active_chunks -= 1

                    # Aggregate
                    try:
                        temp_file_path = os.path.join(temp_dir, res_filename)
                        if os.path.exists(temp_file_path):
                            batch_counts = load_words_binary(temp_file_path)
                            word_frequency.update(batch_counts)
                            os.remove(temp_file_path)
                    except Exception as e:
                        logger.error(f"Failed to load chunk result {res_filename}: {e}")

                    # Update tracker for source file
                    if source_file in file_chunk_tracker:
                        file_chunk_tracker[source_file] -= 1

                        # cleanup check
                        if file_chunk_tracker[source_file] <= 0:
                            logger.info(f"Finished processing all chunks for: {source_file}")
                            del file_chunk_tracker[source_file]
                            try:
                                os.remove(source_file)
                            except OSError:
                                pass
                            completed_shards_count += 1
                            logger.info(
                                f"Progress: {completed_shards_count}/{total_shards_expected} shards fully processed.")

                elif msg_type == "CHUNK_FAILED":
                    source_file = payload
                    active_chunks -= 1
                    # Even if failed, we decrement the counter so we don't hang forever
                    if source_file in file_chunk_tracker:
                        file_chunk_tracker[source_file] -= 1
                        if file_chunk_tracker[source_file] <= 0:
                            del file_chunk_tracker[source_file]
                            try:
                                if os.path.exists(source_file): os.remove(source_file)
                            except:
                                pass
                            completed_shards_count += 1

            except Exception as e:
                logger.error(f"Event loop error: {e}", exc_info=True)
                break

        logger.info("All tasks processed. Closing pool.")
        pool.close()
        pool.join()
        manager.shutdown()

    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
        pool.terminate()
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    logger.info(f"Build word frequencies completed in {time.time() - start_time:.2f}s")

    return word_frequency


def add_high_frequency_words_to_dictionary(dictionary: Set[str],
                                           lang_code: str,
                                           word_str_freq: CounterType[str],
                                           avg_word_freq: Counter[int]) -> List[str]:
    logger.info("Adding high frequency words to dictionary...")
    lang_characters = set()
    for word, frequency in word_str_freq.most_common(2000):
        if len(word) < 2: continue
        lang_characters.update(word)

    data = {f"{lang_code}_characters": list(lang_characters)}
    save_word_frequencies_json("character_freq.json", data, lang_code)

    words_to_add = []
    for word, freq in word_str_freq.items():
        length = len(word)
        if length < 2 or length > MAX_DICT_WORD_LEN: continue
        valid_word = True

        unique_count = 0
        last_char = ""
        for char in word:
            if char not in lang_characters:
                valid_word = False
                break

            if last_char != char:
                unique_count += 1
                last_char = char

        if not valid_word: continue
        if (unique_count / length) < 0.5: continue

        word_limit = max(MIN_WORD_FREQ, avg_word_freq.get(length, 1000))
        if word[0].isalpha() and freq > word_limit and word not in dictionary:
            words_to_add.append(word)

    if words_to_add:
        dictionary.update(words_to_add)
        logger.info(f"Added {len(words_to_add)} new high-frequency words to '{lang_code}' dictionary.")

    return words_to_add


@log_execution_time
def build_average_frequencies(word_frequencies):
    average_frequency = {}
    count = Counter()

    for word, freq in word_frequencies.items():
        word_len = len(word) - len(SUBWORD_PREFIX) if word.startswith(SUBWORD_PREFIX) else len(word)
        if word_len < 1: continue
        num_of_values = count.get(word_len, 0) + 1
        running_average = average_frequency.get(word_len, 0)
        average_frequency[word_len] = add_to_running_average(freq, running_average, num_of_values)
        count[word_len] += 1

    return Counter(average_frequency)


def pre_process_word_frequencies(word_frequencies: CounterType[str]) -> Tuple[CounterType[str], LRUCache]:
    """
    Pre-process word frequency count to:
    1. Create the lower-case `word_str_freq` Counter.
    2. Create the `word_pairs` lookup dictionary.
    3. Find subwords in long compound words and add them to the main `word_frequencies` Counter.
    """
    logger.info(f"Pre-processing {len(word_frequencies)} words...")

    final_word_str_freq = Counter()
    word_pairs_cache = LRUCache()
    remove_count = 0
    long_count = 0

    for word, frequency in list(word_frequencies.items()):
        if frequency < MIN_TOTAL_FREQUENCY:
            del word_frequencies[word]
            remove_count += 1
            continue

        # Regex handling for CamelCase or specific split patterns
        words = re.findall(CASE_SPLIT_PATTERN, word)
        needs_update = True if len(words) > 1 else False
        if needs_update:
            del word_frequencies[word]

        for idx, _word in enumerate(words):
            if len(_word) > MAX_WORD_LEN:
                long_count += 1
                continue
            # Step 2: Create lower-case counter and word_pairs
            word_str = word.lower()
            word_pairs_cache.put(word, word_str)
            final_word_str_freq[word_str] += frequency
            if idx > 0:
                if needs_update:
                    word_frequencies[f"{SUBWORD_PREFIX}{_word}"] += frequency
            else:
                if needs_update:
                    word_frequencies[_word] += frequency

    logger.info(f"Pre-processing complete. Removed {remove_count} low frequency & {long_count} long words (greater that 50 characters).")
    return final_word_str_freq, word_pairs_cache


def sort_by_frequency(word: str, subword_freq: Counter):
    """Sort by frequency"""
    return subword_freq[word]


def align_scale_and_cap(corpus_frequencies: Counter,
                        max_freq_cap: float,
                        scale_range: tuple=None) -> CounterType[str]:
    """
    1. Scales the entire corpus_frequencies Counter so the most common word equals max_freq_cap.
    2. Ensures rare words from the dictionary are added to the scaled Counter using the minimum observed frequency.
    3. Aligns dictionary_words with this new scaled data and calculates final Log-Zipf scores.

    Returns:
        tuple: (aligned_scores_dict, scaled_counter_object)
    """

    # --- Step 0: Pre-check ---
    if not corpus_frequencies:
        return Counter()

    # --- Step 1: Scale the Counter (Cap Enforcement) ---
    scale_range = scale_range if scale_range else (100, max_freq_cap)

    # --- Step 2: Logarithmic Scaling (Zipf Adjustment) ---
    log_values = {w: np.log10(val) for w, val in corpus_frequencies.items()}

    vals = list(log_values.values())
    min_log = min(vals)
    max_log = max(vals)

    # Handle edge case: all words have the same frequency
    if max_log - min_log == 0:
        return corpus_frequencies

    # --- Step 3: Final Min-Max Scaling to Target Range ---
    aligned_scores = Counter()
    min_scale, max_scale = scale_range

    for word in sorted(log_values.keys(), key=lambda x: log_values[x], reverse=True):
        normalized_val = (log_values[word] - min_log) / (max_log - min_log)
        scaled_val = normalized_val * (max_scale - min_scale) + min_scale
        aligned_scores[word] += math.floor(scaled_val)

    return aligned_scores


def add_dictionary_words_to_corpus(corpus_frequencies, dictionary_words):
    # Identify the lowest frequency in the counter to determine default fill value.
    min_corpus_count = min(corpus_frequencies.values())
    fill_value = max(min_corpus_count * 5, 100)
    # --- Step 3: Ensure all dictionary words are in the scaled_counter. ---
    for word in dictionary_words:
        corpus_frequencies[word] += fill_value


def extract_prefix_suffix_roots(lang_code, word_frequency, word_pairs, avg_str_freq, vocab_words, dictionary):

    global PREFIXES, SUFFIXES, ROOT_WORDS

    logger.info("--- Starting Morpheme Discovery Phase ---")
    # --- Step 1: Pre-process word frequencies (Single-Threaded) ---
    logger.info("--- 1. Pre-processing word frequencies ---")

    max_word_len = MMAP_MIN_WORD_LENGTH * 2 - 1
    scaled_subword_freq = Counter()
    morphemes_freq = Counter()
    long_words = []
    subwords = set()
    full_words = set()
    spaces = 0
    for word, freq in list(word_frequency.items()):
        is_subword = True if word.startswith(SUBWORD_PREFIX) else False
        word_len = len(word) - len(SUBWORD_PREFIX) if is_subword else len(word)
        start = len(SUBWORD_PREFIX) if is_subword else 0

        if word_len < 2: continue

        if is_subword:
            subwords.add(word[start:])
        else:
            full_words.add(word)

        word_str = word_pairs.get(word)
        if not word_str:
            word_str = word.lower()
            word_pairs.put(word, word_str)

    # Process subwords
    start_time = time.time()
    word_splits_cache: Dict[str, Dict[int, List[str]]] = {}
    sub_words_freq = Counter({word: word_frequency[word] for word in subwords})
    extract_subword_frequencies(
        sub_words_freq, word_pairs, morphemes_freq, scaled_subword_freq, dictionary, word_splits_cache, is_subword=True
    )
    logger.info(f"Extracted frequencies for {len(sub_words_freq)} subwords in {time.time() - start_time} seconds.")

    start_time = time.time()
    full_words_freq = Counter({word: word_frequency[word] for word in full_words})
    extract_subword_frequencies(
        full_words_freq, word_pairs, morphemes_freq, scaled_subword_freq, dictionary, word_splits_cache
    )

    # Remove low frequency subwords
    morphemes_freq = Counter({k: v for k, v in morphemes_freq.items() if v > 1 or k in dictionary})
    scaled_subword_freq = Counter({key: scaled_subword_freq[key] for key in morphemes_freq})
    logger.info(f"Extracted frequencies for {len(full_words_freq)} words in {time.time() - start_time} seconds.")

    words_by_freq = sorted(scaled_subword_freq.keys(), key=lambda x: scaled_subword_freq[x], reverse=True)
    scaled_subword_freq = Counter({x: max(math.floor(scaled_subword_freq[x]), 1) for x in words_by_freq})

    # Update scaled_morphemes_freq
    scaled_morphemes_freq = deepcopy(morphemes_freq)
    for word in scaled_morphemes_freq:
        scaled_morphemes_freq[word] += np.cbrt(scaled_subword_freq[word])

    # Save final morphemes frequencies
    save_word_frequencies_json(MORPHEME_FREQ_FILE_NAME, morphemes_freq, lang_code=lang_code)
    save_word_frequencies_json("scaled_subword_freq.json", scaled_subword_freq, lang_code=lang_code)
    save_word_frequencies_json("scaled_morphemes_freq.json", scaled_morphemes_freq, lang_code=lang_code)
    save_word_frequencies_json('scaled_word_frequency.json', word_frequency, lang_code=lang_code)

    # --- MORPHEME PIPELINE (TWO-PASS) ---
    (
        PREFIXES,
        SUFFIXES,
        ROOT_WORDS
    ) = run_morpheme_pipeline(
        lang_code,
        word_splits_cache,
        morphemes_freq,
        dictionary,
        PREFIXES,
        SUFFIXES,
        ROOT_WORDS,
    )

    # 7. Save Updated Morpheme Lists
    save_morphemes(
        lang_code,
        MORPHEME_FILE,
        PREFIXES,
        SUFFIXES,
        ROOT_WORDS  # Save the single consolidated set
    )

    return morphemes_freq, scaled_subword_freq, scaled_morphemes_freq


def add_subword_groups(subwords_by_index, subword_stem_info: Dict[str, Set[str]]):
    groups = []
    # Get the string of the first subword to check for leading spaces
    first_subword_str = subwords_by_index[0]
    str_size = len(first_subword_str.lstrip())
    first_token_idx = 0 if str_size > 0 else 1
    subword_stem = ""

    for start_idx in sorted(subwords_by_index.keys()):
        if start_idx > first_token_idx:
            subword = f"{SUBWORD_PREFIX}{subwords_by_index[start_idx]}"
            subword_stems = subword_stem_info.get(subword, set())
            subword_stems.add(subword_stem)
            subword_stem_info[subword] = subword_stems
            subword_stem = f"{subword_stem}{subwords_by_index[start_idx]}"  # Update stem with new subword.
        else:
            subword = subwords_by_index[start_idx]
            subword_stems = subword_stem_info.get(subword, set())
            subword_stems.add(subword_stem)
            subword_stem_info[subword] = subword_stems
            subword_stem = subword

        groups.append(subword)

    return groups


def get_max_word_len(freq, average_freq, is_lower, word_len_counter=None):
    if is_lower and freq > max(average_freq * 0.7, HIGH_WORD_FREQ):
        if freq > max(average_freq * 3, HIGH_WORD_FREQ):
            max_len = 15
        else:
            max_len = 12
    elif freq > max(average_freq * 0.3, MIN_WORD_FREQ) and is_lower:
        max_len = 7
    else:
        max_len = 6

    if word_len_counter is not None:
        word_len_counter[max_len] += 1

    return max_len


def process_low_frequency_subwords(subword_stem_variety: Dict[str, Set[str]],
                                   morphemes_freq: Counter[str],
                                   avg_str_freq: Counter[int],
                                   vocab_words: Set[str],
                                   dictionary_words: Set[str]
                                   ) -> Tuple[Set[str], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Split low frequency subwords. Temporarily removes the low-freq word from the vocab.
    """
    failed_splits = {}
    subword_replacements = {}
    word_pairs = LRUCache()
    new_tokens = set()
    full_word = 0

    logger.info(f"Processing {len(subword_stem_variety)} low frequency subwords for replacements...")
    logger.info(f"Current vocab size before processing: {len(vocab_words)}")

    for subword in sorted(subword_stem_variety.keys(), key=len, reverse=True):
        total_freq = morphemes_freq.get(subword, 1)
        stems = subword_stem_variety[subword]
        stems = [x for x in stems if x]
        subword_len = len(subword) - len(SUBWORD_PREFIX) if stems else len(subword)
        idx = len(SUBWORD_PREFIX) if stems else 0
        # TODO - freq_limit_met = total_freq < max(FREQUENCY_LIMIT * 5, MAX_FREQUENCY_SPLIT)
        freq_limit_met = total_freq < 110 and subword_len > MIN_SUBWORD_LEN
        #min_freq = total_freq * 1.1  # Minimum of 20 % efficiency gain
        offset = 1 if stems else 0  # set offset to 1 for subwords with prefix for vocab lookup matching

        if subword_len > MIN_SUBWORD_LEN + 1 and len(stems) < 10 and freq_limit_met:

            word = subword.lstrip(SUBWORD_PREFIX) if stems else subword
            word_str = word if word[idx].islower() else word.lower()
            word_pairs.put(word, word_str)
            subword_len = len(word_str)
            variety_req_met = len(stems) > 6 or total_freq > 12

            if subword_len <= WORD_SIZE_LIMIT + 1 and word[idx].islower() and variety_req_met:
                continue

            if len(set(word)) < 3 or (len(word_str) < MMAP_MIN_WORD_LENGTH and subword in new_tokens):
                continue

            max_len = len(word_str) - 1 if len(word_str) < MMAP_MIN_WORD_LENGTH + 1 else MMAP_MIN_WORD_LENGTH

            # Hide the subword from vocab to force a split ---
            vocab_words.discard(subword)
            in_vocab = word in vocab_words
            if in_vocab:
                dictionary_words.discard(word)
                vocab_words.discard(word)

            # Use dictionary_words and vocab_words to find splits
            subwords_by_index, split_types = generate_subword_splits(
                word, vocab_words, dictionary_words, morphemes_freq, avg_str_freq, word_pairs,
                max_len=max_len, start_idx=offset
            )
            if in_vocab:  # Add back work to dictionary after generating subwords
                dictionary_words.add(word)
                if stems:
                    vocab_words.add(word)  # Add back word to vocab

            subword_groups = []
            new_total_freq = total_freq
            max_token = 3 # max token must be greater than 3
            longest_subword = ""
            min_len_limit = min(MIN_SUBWORD_LEN, math.floor(subword_len / 2))
            if not stems:  # Original word without subword prefix
                full_word += 1
                token = subwords_by_index.pop(0)
                subword_groups.append(token)
                if len(token) > max_token:
                    max_token = len(token)
                    longest_subword = token

                if len(token) >= min_len_limit:
                    freq = morphemes_freq.get(token, 1)
                    new_total_freq = max(new_total_freq, freq)

            for start_idx in sorted(subwords_by_index.keys()):
                token_len = len(subwords_by_index[start_idx])
                token = f"{SUBWORD_PREFIX}{subwords_by_index[start_idx]}"
                if token_len > max_token:
                    max_token = token_len
                    longest_subword = token

                if token_len >= min_len_limit:
                    freq = morphemes_freq.get(token, 1)
                    new_total_freq = max(new_total_freq, freq)

                new_tokens.add(token)
                subword_groups.append(token)

            max_splits = math.ceil(subword_len / MIN_SUBWORD_LEN)
            total_freq_req_met = new_total_freq > total_freq or longest_subword in vocab_words
            min_len_met = 1 < len(subword_groups) <= max_splits
            # Validate the split
            if min_len_met and total_freq_req_met:
                vocab_words.update(subword_groups)  # Add subwords replacements to vocab
                subword_replacements[subword] = subword_groups
            else:
                vocab_words.add(subword)  # Add back original token if failed
                failure_info = {
                    "total_freq_req_met": total_freq_req_met, "min_len_met": min_len_met,
                    "group_size": len(subword_groups)
                }
                subword_groups.append(failure_info)
                failed_splits[subword] = subword_groups

    return vocab_words, subword_replacements, failed_splits


def group_subwords_by_length(avg_str_freq,
                             subword_frequency,
                             dictionary_words,
                             vocab_words,
                             word_pair_cache,
                             words_to_split,
                             morphemes_freq):
    """
    Groups subwords by index for words that need to be split.
    Uses generate_subword_splits to handle long compound words.
    """
    greedy_splits = {}
    perfect_splits = {}
    word_len_counter = Counter()
    dictionary_words.update(vocab_words)
    subword_stem_variety: Dict[str, Set[str]] = {}

    for word, freq in words_to_split.items():
        word_str = word_pair_cache.get(word)
        start = len(SUBWORD_PREFIX) if word.startswith(SUBWORD_PREFIX) else 0
        if not word_str:
            word_str = word.lower()
            word_pair_cache.put(word, word_str)

        is_lower = word[start].islower()
        freq_limit = max(MIN_WORD_FREQ, avg_str_freq.get(len(word_str), 1))
        max_len = get_max_word_len(freq, freq_limit, is_lower, word_len_counter)

        subwords_by_index, split_types = generate_subword_splits(
            word, vocab_words, dictionary_words, avg_str_freq, subword_frequency, word_pair_cache, max_len=max_len
        )
        groups = add_subword_groups(subwords_by_index, subword_stem_variety)
        vocab_words.update(groups)

        if 'perfect' not in split_types:
            greedy_splits[word] = groups
        else:
            perfect_splits[word] = groups

    logger.info(f"Max word length counts: {word_len_counter}")
    logger.info(f"Found Greedy Split: {len(greedy_splits)}")
    logger.info(f"Found Perfect Split: {len(perfect_splits)}")
    save_word_frequencies_json(GREEDY_FILE, greedy_splits)
    save_word_frequencies_json(PERFECT_FILE, perfect_splits)
    save_word_frequencies_json('subword_frequency.json', subword_frequency)

    # Replace low frequency subwords
    updated_vocab, subword_replacements, failed_splits = process_low_frequency_subwords(
        subword_stem_variety, morphemes_freq, avg_str_freq, vocab_words, dictionary_words
    )
    subword_replacement_file = os.path.join(STORAGE_DIR, 'subword_replacements.json')
    failed_replacement_file = os.path.join(STORAGE_DIR, 'failed_replacement.json')
    save_word_frequencies_json(subword_replacement_file, subword_replacements)
    save_word_frequencies_json(failed_replacement_file, failed_splits)
    logger.info(f"{len(subword_replacements)} subwords replaced.")
    logger.info(f"{len(failed_splits)} replacements failed.")


def load_dictionary(lang_code):
    start = time.time()
    dictionaries = {}
    dictionary_words = set()
    try:
        config_file = os.path.join(STORAGE_DIR, CONFIG_FILE_NAME)
        if os.path.exists(config_file):
            # First try loading dictionary from file.
            with open(config_file, "r", encoding="utf-8") as f:
                dictionaries = json.load(f).get(DICTIONARIES, {})
        try:
            # Convert loaded dictionary lists back to sets
            dictionary_words = set(dictionaries[lang_code])
            logger.info(f"Loaded '{lang_code}' dictionary words with size: {len(dictionary_words)} "
                        f"in {time.time() - start} seconds.")
        except KeyError:
            dictionary_words = load_default_dictionary(lang_code)
            logger.info(f"Created new '{lang_code}' dictionary words with size: {len(dictionary_words)} "
                        f"in {time.time() - start} seconds.")
    except KeyError:
        msg = f"Unsupported language code: '{lang_code}'. Please add support to load_dictionary & SUPPORTED_LANGUAGES."
        logger.error(msg)

    return dictionary_words


def load_default_dictionary(lang_code):
    default_dictionary = set()
    try:
        lookup_name = SUPPORTED_LANGUAGES[lang_code]
        lookup_function = globals()[lookup_name]
        default_dictionary = lookup_function()
    except (KeyError, AttributeError):
        logger.error(f"Unsupported language code: '{lang_code}'. Please add support to load_default_dictionary.")

    return default_dictionary


def _load_dictionaries():
    dictionaries = {}
    for lang_code in SUPPORTED_LANGUAGES:
        dictionaries[lang_code] = load_dictionary(lang_code)

    return dictionaries


def build_wordpiece_vocab(
    lang_code: str,
    word_freq: Counter,
    str_freq: Counter,
    word_pairs: Dict[str, Tuple[str, int]],
    morphemes_freq: Counter,
    avg_str_freq: Counter[int] = None,
    vocab_config: Tuple[Dict, Dict] = None,
    morpheme_config: Tuple[Set, Set, Set] = None,
):
    """
    Builds a WordPiece vocabulary using an efficient two-pass system.
    Supports injection of vocab and morpheme configs for testing.
    """
    if morpheme_config:
        PREFIXES, SUFFIXES, ROOT_WORDS = morpheme_config
    else:
        PREFIXES, SUFFIXES, ROOT_WORDS = load_morphemes(lang_code, MORPHEME_FILE)

    avg_str_freq = avg_str_freq or build_average_frequencies(str_freq)
    vocab, config = vocab_config if vocab_config else _load_tokenizer_vocab(STORAGE_DIR)
    dictionary_words = config[DICTIONARIES].get(lang_code, set())
    vocab_words = set(vocab.keys())

    dictionary_words.update(PREFIXES)
    dictionary_words.update(SUFFIXES)
    dictionary_words.update(ROOT_WORDS)

    p_size = len(PREFIXES)
    r_size = len(ROOT_WORDS)
    s_size = len(SUFFIXES)

    logger.info(
        f"Added {p_size} prefixes, {s_size} suffixes & {r_size} roots to lookup dictionary."
    )

    subword_frequency = Counter()
    words_to_split = Counter()
    symbol_str = 0
    min_str = 0

    for word in sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True):
        freq = word_freq[word]
        word_info = word_pairs.get(word) or get_initial_str_spaces(word)
        word_pairs[word] = word_info
        word_str, spaces = word_info
        vocab_words.update(word)
        subword_frequency.update(word)
        subword_frequency[word] += freq

        if 0 < len(word) < MMAP_MIN_WORD_LENGTH and not word[0].isalpha() and word not in vocab_words:
            vocab_words.add(word)
            symbol_str += 1
            continue

        min_criteria_met = freq > FREQUENCY_LIMIT or word_str in dictionary_words

        if len(word_str) <= MIN_SUBWORD_LEN and min_criteria_met and word not in vocab_words:
            word_str_equal = is_word_str_equal(spaces, word, word_str)
            pairs, _ = split_word_by_pairs(
                word,
                spaces,
                word_str_equal,
                vocab_words,
                dictionary_words,
                morphemes_freq,
                freq,
            )
            if len(pairs) == 1:
                vocab_words.add(pairs[0][1])
                dictionary_words.add(pairs[0][1])
                subword_frequency[pairs[0][1]] += freq
                min_str += 1
                continue
            elif len(pairs) > 1:
                if pairs[0][1] not in vocab_words:
                    vocab_words.add(pairs[0][1])
                    dictionary_words.add(pairs[0][1])
                    subword_frequency[pairs[0][1]] += freq
                    min_str += 1
                for pair in pairs[1:]:
                    subword = f"{SUBWORD_PREFIX}{pair[1]}"
                    if subword not in vocab_words:
                        vocab_words.add(subword)
                        dictionary_words.add(subword)
                        subword_frequency[subword] += freq
                        min_str += 1
                continue
            else:
                logger.error(f"Cannot split word: '{word}'")
                continue

        if len(word_str) <= MIN_SUBWORD_LEN or len(word) > MAX_WORD_LEN:
            continue

        words_to_split[word] += freq

    logger.info(
        f"Added {symbol_str} characters & {min_str} short high frequent words to vocab."
    )
    logger.info(f"Initial vocab size: {len(vocab_words)}.")
    logger.info(f"Found {len(words_to_split)} words to be split.")

    logger.info(
        f"Starting Pass 2: Building and processing subword groups using {len(vocab_words)} vocab words..."
    )
    # TODO - optimize (takes too long)
    group_subwords_by_length(
        avg_str_freq,
        subword_frequency,
        dictionary_words,
        vocab_words,
        word_pairs,
        words_to_split,
        morphemes_freq,
    )
    logger.info(f"Pass 2 complete. Found {len(subword_frequency)} unique subwords.")
    del str_freq
    del avg_str_freq

    sort_function = functools.partial(sort_by_frequency, subword_freq=subword_frequency)

    vocab_words = sorted(vocab_words, key=sort_function, reverse=True)

    logger.info("Vocabulary build complete.")

    return vocab_words, subword_frequency

def add_words_to_vocab(vocab, vocab_words):
    count = 0
    for word in vocab_words:
        if word not in vocab:
            vocab[word] = len(vocab)
            count += 1

    logger.info(f"Added {count} new tokens to vocabulary.")
    save_word_frequencies_json(VOCAB_FILE_NAME, vocab)


class WordPieceTokenizer:
    """
    A WordPiece Tokenizer that tokenizes text based on a pre-trained vocabulary.

    Args:
        load_directory: The full directory path name from which to load the vocabulary.
        unk_token: The string representation of the unknown token.
        subword_prefix: The prefix to denote a subword piece (e.g., "##").
    """
    def __init__(self, load_directory: str="", unk_token: str = UNK, subword_prefix: str = SUBWORD_PREFIX):
        self.vocab_dir = load_directory or STORAGE_DIR
        self.special_tokens = {v: idx for idx, v in enumerate(SPECIAL_TOKENS)}
        vocab, config = self.load(self.vocab_dir)
        self.vocab = vocab if len(vocab) > len(self.special_tokens) else deepcopy(self.special_tokens)
        self.dictionary_words = set()
        self.word_pairs = LRUCache(DEFAULT_TOP_K_WORDS)
        self.pre_tokenized_vocab = LRUCache(DEFAULT_TOP_K_WORDS)
        self._load_morphemes_frequencies()
        self.unk_token = unk_token
        self.subword_prefix = subword_prefix
        self.unk_token_id = self.vocab[self.unk_token]
        self.pre_tokenizer_re = re.compile(UNICODE_PATTERN)
        for key, value in config.items():
            if key == DICTIONARIES:
                for lang in value:
                    self.dictionary_words.update(value[lang])
            elif key == "pre_tokenized_vocab":
                self.pre_tokenized_vocab.update(value)
            else:
                setattr(self, key, value)

        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        self.special_tokens_inv = {v: k for k, v in self.special_tokens.items()}
        self.tokenizer_updated = False

        # TODO - delete
        self.pre_tokenized_vocab.clear()

        if self.unk_token not in self.vocab:
            raise ValueError(f"Unknown token '{self.unk_token}' not found in the vocabulary.")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _load_morphemes_frequencies(self):
        self.morphemes_freq = Counter()
        for language in SUPPORTED_LANGUAGES:
            self.morphemes_freq += load_word_frequencies_json(MORPHEME_FREQ_FILE_NAME, lang_code=language)

        return self.morphemes_freq

    def _add_new_vocab_words(self, words: list[str]):
        """
        Add new words to the vocabulary.

        Args:
            words: A list of new vocab words sorted by frequency (most frequent first).
        """
        if not words: return

        new_word_count = 0
        for word in words:
            if word not in self.vocab:
                new_word_count += 1
                idx = len(self.vocab)
                self.vocab[word] = idx

                # Update reverse lookup vocab
                self.vocab_inv[idx] = word

        if new_word_count > 0:
            logger.info(f"Added {new_word_count} new vocab words. Vocab size: {len(self.vocab)}")
            self.save(self.vocab_dir)
        else:
            logger.info("No new unique words to add to vocab.")

    def _get_updated_config(self):
        tokenizer_config = _load_tokenizer_config(self.vocab_dir)
        tokenizer_config.update({
            "unk_token": self.unk_token,
            "subword_prefix": self.subword_prefix,
            "pre_tokenized_vocab": dict(self.pre_tokenized_vocab),
            "special_tokens": self.special_tokens,
        })

        return tokenizer_config

    def _reset_vocab(self):
        """Reset Vocabulary."""
        self.dictionaries = {}
        self.dictionary_words = {}
        self.pre_tokenized_vocab = {}
        self.vocab = {k: v for k, v in self.special_tokens.items()}
        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        shutil.rmtree(self.vocab_dir)

    def _convert_subwords_to_tokens(self, subword_pairs: List[str]):
        tokens = [self.vocab[token] for token in subword_pairs]
        return tokens

    def _convert_tokens_to_subwords(self, tokens: List[int]):
        new_tokens = [self.vocab_inv[token] for token in tokens]
        return new_tokens

    def _tokenize_word(self, word: str, use_vocab_cache=True) -> List[str]:
        """Tokenizes a single pre-tokenized word into subwords using a greedy approach."""
        if len(word) == 0: return []

        if word in self.vocab:
            return [word]

        if use_vocab_cache:
            tokens = self.pre_tokenized_vocab.get(word)
            if tokens:
                return tokens  #self._convert_tokens_to_subwords(tokens)

        pairs, split_types = tokenize_word_by_pairs(
            word, self.vocab, self.morphemes_freq, self.word_pairs, subword_prefix=self.subword_prefix
        )

        tokens = [pairs[0][1]]
        for pair in pairs[1:]:
            tokens.append(f"{self.subword_prefix}{pair[1]}")

        if use_vocab_cache:
            self.pre_tokenized_vocab.put(word, tokens)  #self._convert_subwords_to_tokens(tokens)
            self.tokenizer_updated = True

        return tokens

    def tokenize(self, text: str) -> List[str]:
        """Converts a string into a sequence of tokens."""
        text = normalize_text_for_nlp(text)
        tokens = []
        for token in remove_single_spaces(self.pre_tokenizer_re.findall(text)):
            tokens.extend(self._tokenize_word(token))

        return tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Converts a string into a sequence of token IDs."""
        tokens = self.tokenize(text)
        token_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        if add_special_tokens:
            bos_id = self.vocab.get(BOS)
            eos_id = self.vocab.get(EOS)
            if bos_id is not None:
                token_ids.insert(0, bos_id)
            if eos_id is not None:
                token_ids.append(eos_id)

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Converts a sequence of token IDs back into a string."""
        if not token_ids: return ""

        tokens = []
        for tok in token_ids:
            if tok in self.special_tokens_inv: continue  # Ignore special tokens
            token = self.vocab_inv.get(tok, self.unk_token)
            if token.startswith(self.subword_prefix):
                tokens[-1] = f"{tokens[-1]}{token.lstrip(self.subword_prefix)}"
            else:
                tokens.append(token)

        tokens = add_single_spaces(tokens)

        return "".join(tokens)

    def split_word_by_length(self,
                             word: str,
                             word_pairs: Dict[str, Tuple[str, int]],
                             vocab_words: Set[str],
                             min_len=1,
                             max_len=MAX_DICT_WORD_LEN):
        """Generates all valid subword splits for a given word grouped by start indices."""
        subwords_by_length: DefaultDict[int, List[Tuple[int, str]]] = defaultdict(list)
        word_info = word_pairs.get(word) or get_initial_str_spaces(word)
        word_str, spaces = word_info

        word_str_equal = word[spaces:] == word_str
        if not word_str_equal and spaces > 0:  # Add spaces as a separate token for Upper/Sentence case words
            space_str = word[:spaces]
            subwords_by_length[len(space_str)].append((0, space_str))

        for start in range(len(word)):
            for end in range(start+min_len, min(start+max_len+1, len(word_str)+1)):
                if start == 0 and end >= len(word_str): continue

                subword_str = word_str[start:end]

                if len(subword_str) > min_len and subword_str not in vocab_words:
                    continue

                if word_str_equal:
                    if spaces == 1 and start < spaces:
                        start_idx = start
                    else:
                        start_idx = start + spaces
                else:
                    start_idx = start + spaces
                    if start < spaces == 1 and len(subword_str) > 3:  # Only use meaningful subwords
                        subword_full_str = word[start:end + spaces]
                        if subword_full_str in vocab_words:  # Use subword with space if already in vocab
                            subwords_by_length[len(subword_full_str)].append((start, subword_full_str))
                            continue

                subword = subword_str if word_str_equal and start > spaces else word[start_idx:end + spaces]
                subwords_by_length[len(subword)].append((start_idx, subword))
                vocab_words.add(subword)

        return subwords_by_length


    def group_subword_by_index(self,
                               word: str,
                               word_pairs: Dict[str, str],
                               subwords_by_length: dict[int, List[Tuple[int, int]]],
                               lookup_vocab: dict[str, int],
                               min_len=1,
                               max_len=7) -> DefaultDict[int, List[int]]:
        """Generates all valid subword splits for a given word grouped by start indices."""
        subword_splits: DefaultDict[int, List[int]] = defaultdict(list)
        word_info = word_pairs.get(word) or get_initial_str_spaces(word)
        word_str, spaces = word_info

        min_len = max(1, min_len)  # Ensure minimum len is at least 1
        if len(word_str) < min_len+1:
            start = 0
            end = start + spaces + 1
            size = len(word) - spaces
            subword = word[start:end]
            if subword not in lookup_vocab:
                lookup_vocab[subword] = len(lookup_vocab)
            subword_splits[0].append(lookup_vocab[subword])
            idx = start + spaces
            for x in range(1, size - 1, 1):
                subword = word[idx + x]
                if subword not in lookup_vocab:
                    lookup_vocab[subword] = len(lookup_vocab)
                subword_splits[x].append(lookup_vocab[subword])

            return subword_splits

        # Filter subwords splits to only keep the best subword pairs
        covered_indices = Counter()
        for length in sorted(subwords_by_length.keys(), reverse=True):
            if length > max_len: continue
            for start, subword_key in subwords_by_length[length]:
                end = start + length - 1
                if length > 1 and (covered_indices.get(start, 0) > 3 or covered_indices.get(end, 0) > 3): continue
                covered_indices.update(list(range(start, end+1)))
                subword_splits[start].append(subword_key)

        return subword_splits

    @log_execution_time
    def build_wordpiece_vocab(self,
                              lang_code: str,
                              str_freq: Counter,
                              word_pair_cache: LRUCache,
                              subword_frequency: Counter,
                              morphemes_freq: Counter,
                              scaled_morphemes_freq: Counter,
                              avg_str_freq: Counter[int] = None,
                              vocab_config: Tuple[Dict, Dict] = None,
                              morpheme_config: Tuple[Set, Set, Set] = None) -> Dict[str, int]:
        """
        Builds a WordPiece vocabulary using an efficient two-pass system.

        Args:
            lang_code: ISO 639-3 code of the selected language.
            str_freq: The Counter object with normalized word frequency.
            word_pair_cache: The LRUCache to use for looking up the lower case word pairs.
            subword_frequency: The Counter with subword frequencies.
            morphemes_freq: The Counter with prefix, suffix & root frequencies.
            scaled_morphemes_freq: The Counter with prefix, suffix & root scaled & aligned with frequencies.
            avg_str_freq: [Optional] Average word frequency by length counter.
            vocab_config: [Optional] Tuple(vocab, config) where 'vocab' is a dictionary with
                the current vocab tokens. 'config' is a dictionary with current vocab configurations.
            morpheme_config: [Optional] Tuple(prefix, suffix, roots) with the loaded prefix, suffix & roots.

        Returns:
            A Dict mapping tokens to integer IDs.
        """
        if morpheme_config:
            PREFIXES, SUFFIXES, ROOT_WORDS = morpheme_config
        else:
            PREFIXES, SUFFIXES, ROOT_WORDS = load_morphemes(lang_code, MORPHEME_FILE)

        avg_str_freq = avg_str_freq or build_average_frequencies(str_freq)
        vocab, config = vocab_config if vocab_config else self.load(STORAGE_DIR)
        dictionary_words = set(config[DICTIONARIES].get(lang_code, []))
        vocab_words = set(vocab.keys())

        dictionary_words.update(PREFIXES)
        dictionary_words.update(SUFFIXES)
        dictionary_words.update(ROOT_WORDS)

        p_size = len(PREFIXES)
        r_size = len(ROOT_WORDS)
        s_size = len(SUFFIXES)

        logger.info(
            f"Added {p_size} prefixes, {s_size} suffixes & {r_size} roots to lookup dictionary."
        )

        words_to_split = Counter()
        min_str = 0

        for word, freq in subword_frequency.items():

            if word.startswith(SUBWORD_PREFIX): continue  # Skip subwords

            word_str = word_pair_cache.get(word)
            if not word_str:
                word_str = word.lower()
                word_pair_cache.put(word, word_str)

            word_len = len(word)
            if word_len > 1:
                vocab_words.add(word[0])
                for char in word[1:]:
                    vocab_words.add(char)
            elif word_len == 1:
                vocab_words.add(word)

            if 0 < word_len < MMAP_MIN_WORD_LENGTH and not word[0].isalpha() and word not in vocab_words:
                vocab_words.add(word)
                continue

            min_criteria_met = freq > MIN_WORD_FREQUENCY or word_str in dictionary_words

            if word_len <= MIN_SUBWORD_LEN and min_criteria_met and word not in vocab_words:
                pairs, _ = tokenize_word_by_pairs(
                    word,
                    vocab_words,
                    scaled_morphemes_freq,
                    avg_str_freq,
                    word_pair_cache,
                    dictionary=dictionary_words,
                )
                if len(pairs) == 1:
                    vocab_words.add(pairs[0][1])
                    dictionary_words.add(pairs[0][1])
                    min_str += 1
                    continue
                elif len(pairs) > 1:
                    if pairs[0][1] not in vocab_words:
                        vocab_words.add(pairs[0][1])
                        dictionary_words.add(pairs[0][1])
                        min_str += 1
                    for pair in pairs[1:]:
                        subword = f"{SUBWORD_PREFIX}{pair[1]}"
                        if subword not in vocab_words:
                            vocab_words.add(subword)
                            dictionary_words.add(subword)
                            min_str += 1
                    continue
                else:
                    logger.error(f"Cannot split word: '{word}'")
                    continue

            if word_len <= MIN_SUBWORD_LEN or word_len > MAX_WORD_LEN:
                continue

            words_to_split[word] += freq

        logger.info(f"Added {min_str} short high frequent words to vocab.")
        logger.info(f"Initial vocab size: {len(vocab_words)}.")
        logger.info(f"Found {len(words_to_split)} words to be split.")
        # save_word_frequencies_json(f"initial_vocab.json", {"vocab_words": list(vocab_words)})

        logger.info(
            f"Starting Pass 2: Building and processing subword groups using {len(vocab_words)} vocab words..."
        )
        group_subwords_by_length(
            avg_str_freq,
            scaled_morphemes_freq,
            dictionary_words,
            vocab_words,
            word_pair_cache,
            words_to_split,
            morphemes_freq
        )
        logger.info(f"Pass 2 complete. Found {len(subword_frequency)} unique subwords.")
        del str_freq
        del avg_str_freq

        sort_function = functools.partial(sort_by_frequency, subword_freq=subword_frequency)
        vocab_words = sorted(vocab_words, key=sort_function, reverse=True)
        self._add_new_vocab_words(vocab_words)

        logger.info("Vocabulary build complete.")

        return self.vocab


    def save(self, save_directory: str):
        """Saves the tokenizer's vocabulary and configuration to a directory."""
        tokenizer_config = self._get_updated_config()
        _save_tokenizer_vocab(save_directory, self.vocab)
        _save_tokenizer_config(save_directory, tokenizer_config)

    @classmethod
    def load(cls, load_directory: str):
        """Loads a tokenizer from a directory."""
        vocab = _load_tokenizer_vocab(load_directory)
        config = _load_tokenizer_config(load_directory)

        return vocab, config

# --- Unit Tests ---
class TestFrequencyAlignment(unittest.TestCase):

    def setUp(self):
        # Setup a dummy corpus with a maximum count of 1000 and a minimum of 10
        self.raw_corpus = Counter({
            "common": 1000,
            "mid": 500,
            "rare": 10
        })
        # The dictionary includes a word that is missing from the corpus
        self.vocab = {"common", "mid", "rare", "missing_word"}
        self.max_cap = 100.0

    def test_counter_scaling_cap(self):
        """Test if the corpus is correctly scaled to the new max_freq (100.0) and maintains proportionality."""
        scaled_counter = align_scale_and_cap(self.vocab, self.raw_corpus, self.max_cap)

        # Max word ("common": 1000) should now be 100.0
        self.assertAlmostEqual(scaled_counter["common"], 100.0)

        # Mid word ("mid": 500) should now be 50.0
        self.assertAlmostEqual(scaled_counter["mid"], 50.0)

    def test_missing_word_counter_update(self):
        """
        Crucial Test: Verify the missing word is added to the returned scaled_counter
        with the correct minimum scaled frequency.
        """
        scaled_counter = align_scale_and_cap(self.vocab, self.raw_corpus, self.max_cap)

        # 1. Check if the missing word exists in the scaled counter
        self.assertIn("missing_word", scaled_counter)

        # 2. Check the minimum expected scaled frequency
        # Original min was 10. Scaling factor is 100/1000 = 0.1.
        # New minimum frequency is 10 * 0.1 = 1.0.
        expected_min_freq = 1.0

        # 3. Check that the missing word's frequency equals the new minimum
        self.assertAlmostEqual(scaled_counter["missing_word"], expected_min_freq)

        # 4. Check that the final Log-Zipf score is the same for the truly rare word and the missing word (both are '0.0')
        self.assertAlmostEqual(scaled_counter["missing_word"], 0.0)
        self.assertAlmostEqual(scaled_counter["rare"], 0.0)
        self.assertAlmostEqual(scaled_counter["missing_word"], scaled_counter["rare"])

    def test_zipf_score_bounds(self):
        """Test if final scores are bounded between 0 and 1."""
        scores = align_scale_and_cap(self.vocab, self.raw_corpus, self.max_cap)
        self.assertAlmostEqual(max(scores.values()), 1.0)
        self.assertAlmostEqual(min(scores.values()), 0.0)


class TestWordSplitting(unittest.TestCase):

    def setUp(self):
        # Mock Data
        # SWIPES
        self.dictionary = {
            "auto", "bio", "graph", "ical", "biographical", "autobiographical",
            "theme", "themes", "t", "hemes", "s", "the", "mes", "wipe", "wipes",
            "swipe", "swipes"
        }
        self.vocab = set()

        # Frequencies: "Theme" + "s" should score higher than "T" + "hemes"
        self.freq = Counter({
            "theme": 71,
            "<|##|>theme": 28,
            "<|##|>the": 1768,
            "the": 2791,
            "s": 2125,
            "<|##|>s": 4810,
            "<|##|>themes": 38,
            "themes": 7,
            "t": 850,
            "<|##|>hemes": 20,
            "hemes": 1,
            "<|##|>swipe": 10,
            "swipe": 13,
            "wipe": 12,
            "<|##|>wipe": 26,
            "<|##|>wipes": 10,
            "w": 2437,
            "auto": 1827,
            "<|##|>auto": 106,
            "bio": 2224,
            "<|##|>bio": 449,
            "<|##|>graph": 504,
            "graph": 168,
            "<|##|>ical": 971,
            "ical": 28,
            "<|##|>mes": 966,
            "mes": 767
        })

    def test_perform_perfect_split_max_len_constraint(self):
        """
        Test that perform_perfect_split respects max_len.
        Input: 'AUTOBIOGRAPHICAL' (16 chars)
        max_len: 5
        Expected: Should NOT attempt to split at index 15, 14, etc.
        Should find splits like ['auto', 'bio', 'graph', 'ical'] where chunks <= 5.
        """
        word = "AUTOBIOGRAPHICAL"
        word_str = "autobiographical"

        # We need a depth of at least 3 splits (4 parts) to fit 16 chars into max_len 5
        # 16 / 4 = 4 chars avg.
        results = perform_perfect_split(
            p_lookup=word_str,
            p_word=word,
            p_offset=0,
            max_dept=5,
            min_len=2,
            max_len=5,
            lookup_dictionary=self.dictionary,
            vocab_tokens=self.vocab
        )

        self.assertTrue(len(results) > 0, "Should find a solution")

        for path in results:
            for _, token in path:
                self.assertLessEqual(len(token), 5, f"Token '{token}' exceeds max_len 5")

    def test_perform_perfect_split_returns_all_candidates(self):
        """
        Test that perform_perfect_split returns multiple valid pairings at the same depth.
        Word: "THEMES"
        Dict contains: "theme", "s", "t", "hemes"
        Depth 1 (1 cut) possibilities: ["theme", "s"] and ["t", "hemes"]
        """
        word = "THEMES"
        # Ensure dictionary has necessary parts

        results = perform_perfect_split(
            p_lookup="themes",
            p_word="Themes",
            p_offset=0,
            max_dept=2,
            min_len=2,
            max_len=5,
            lookup_dictionary=self.dictionary,
            vocab_tokens=self.vocab
        )

        # We expect at least two paths:
        # 1. (0, 'Theme'), (5, 's')
        # 2. (0, 'T'), (1, 'hemes')

        self.assertTrue(len(results) >= 2, "Should return multiple valid split paths")

        # flatten to strings for easy checking
        str_results = [[t[1] for t in path] for path in results]
        self.assertIn(['Theme', 's'], str_results)
        self.assertIn(['T', 'hemes'], str_results)

    def test_split_word_by_pairs_sorting(self):
        """
        Test that split_word_by_pairs uses the frequency counter to pick the best perfect split.
        Given "Themes", it should prefer ["Theme", "s"] over ["T", "hemes"] because
        "Theme" and "s" have higher frequencies in self.freq.
        """
        # Mock functions that split_word_by_pairs depends on if they aren't in the same file
        # assuming the code provided above is fully integrated.

        results, types = split_word_by_pairs(
            word="Themes",
            spaces=0,
            word_str_equal=False,
            vocab_tokens=self.vocab,
            lookup_dictionary=self.dictionary,
            morphemes_freq=self.freq,
            max_len=5,
            min_len=1
        )

        self.assertIn('perfect', types)
        # We expect the winner to be Theme + s based on frequency
        self.assertEqual(results[0][1], "Theme")
        self.assertEqual(results[1][1], "s")

        # Test SWIPES
        results, types = split_word_by_pairs(
            word="SWIPES",
            spaces=0,
            word_str_equal=False,
            vocab_tokens=self.vocab,
            lookup_dictionary=self.dictionary,
            morphemes_freq=self.freq,
            max_len=5,
            min_len=1
        )

        self.assertIn('perfect', types)
        # We expect the winner to be Theme + s based on frequency
        self.assertEqual(results[0][1], "SWIPE")
        self.assertEqual(results[1][1], "S")


class TestStringReconstruction(unittest.TestCase):
    def reconstruct(self, text):
        """Helper to run the full cycle"""
        tokens = re.findall(UNICODE_PATTERN, text)
        cleaned = remove_single_spaces(tokens)
        reconstructed_tokens = add_single_spaces(cleaned)
        return "".join(reconstructed_tokens)

    def test_joined_camel_case(self):
        """
        This test fails with the old code.
        Input 'CamelCase' should remain 'CamelCase'.
        Old regex split it into 'Camel', 'Case' -> became 'Camel Case'.
        """
        original = "CamelCase"
        self.assertEqual(self.reconstruct(original), original)

    def test_camel_case_split(self):
        original = "camel Case"
        self.assertEqual(self.reconstruct(original), original)

    def test_pascal_case_split(self):
        original = "this Is"
        self.assertEqual(self.reconstruct(original), original)

    def test_standard_lowercase_split(self):
        original = "key board"
        self.assertEqual(self.reconstruct(original), original)

    def test_proper_nouns_kept(self):
        original = "Hello World"
        self.assertEqual(self.reconstruct(original), original)

    def test_mixed_complex_sentence(self):
        original = "Hello World this Is camel Case test New iPhone sales"
        self.assertEqual(self.reconstruct(original), original)

    def test_punctuation_safety(self):
        original = "Hello, World."
        self.assertEqual(self.reconstruct(original), original)


class TestLowFrequencySplitting(unittest.TestCase):
    def setUp(self):
        self.vocab = {"<|##|>oddity", "<|##|>odd", "<|##|>ity", "odd", "ity"}
        self.dict = {"oddity", "odd", "ity"}
        self.freq = Counter({
            "<|##|>odd": 500, "<|##|>ity": 500, "<|##|>oddity": 50, "<|##|>o":64229, "<|##|>ov":13453,
            "<|##|>er":16940, "<|##|>over": 1200
        })

        # Subword variety inputs
        self.stem_variety = {"<|##|>oddity": {"radi"}}

    def test_split_logic_respects_removal(self):
        # Case: <|##|>oddity should split into <|##|>odd + <|##|>ity
        # because it is low freq (50) and components are high freq.
        # It relies on temporarily hiding <|##|>oddity from vocab.

        new_vocab, replacements, failed_splits = process_low_frequency_subwords(
            self.stem_variety, self.freq, self.vocab.copy(), self.dict
        )

        self.assertIn("<|##|>oddity", replacements)
        self.assertEqual(replacements["<|##|>oddity"], ["<|##|>odd", "<|##|>ity"])

    def test_short_subword_ignored(self):
        # Too short to split
        variety = {"<|##|>over": {"do"}}
        new_vocab, replacements, failed_splits = process_low_frequency_subwords(
            variety, self.freq, {"<|##|>over", "<|##|>o", "<|##|>ov", "<|##|>er"}, {"do", "over"}
        )
        self.assertEqual(len(replacements), 0)
        self.assertEqual(len(failed_splits), 0)


class TestWordPieceTokenizerComprehensive(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for vocab storage
        self.test_dir = tempfile.mkdtemp()

        # Create a mock vocabulary
        self.mock_vocab = {
            "<|pad|>": 0,
            "<|unk|>": 1,
            "<|bos|>": 2,
            "<|eos|>": 3,
            "<|mask|>": 4,
            "the": 5,
            "quick": 6,
            "brown": 7,
            "fox": 8,
            "jump": 9,
            "##le": 10,
            "##s": 11,
            "un": 12,
            "##happy": 13,
            "apple": 14,
            "banana": 15,
            "App": 16,
            "##ed": 17,
            " ": 18,
            "  ": 19,
            "   ": 20,
            "    ": 21,
        }

        self.mock_config = {
            "dictionaries": {
                "eng": ["the", "quick", "brown", "fox", "jump", "jumped", "unhappy", "apple", "banana"]
            },
            "token_frequencies": {
                "the": 1000,
                "quick": 500
            }
        }

        # Write mock files
        with open(os.path.join(self.test_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(self.mock_vocab, f)

        with open(os.path.join(self.test_dir, 'tokenizer_config.json'), 'w', encoding='utf-8') as f:
            json.dump(self.mock_config, f)

        # Initialize tokenizer pointing to temp dir
        self.tokenizer = WordPieceTokenizer(load_directory=self.test_dir, subword_prefix='##')

        # Manually populate morphemes_freq for splitting logic testing
        self.tokenizer.morphemes_freq = Counter({
            "un": 500,
            "##happy": 400,
            "jump": 600,
            "##ed": 1000
        })

    def tearDown(self):
        # Cleanup temp directory
        shutil.rmtree(self.test_dir)

    # --- Initialization Tests ---
    def test_vocab_loading(self):
        """Test if vocabulary loads correctly from JSON."""
        self.assertEqual(self.tokenizer.vocab_size, 22)
        self.assertEqual(self.tokenizer.vocab["jump"], 9)
        self.assertEqual(self.tokenizer.vocab["<|unk|>"], 1)

    def test_missing_token_fallback(self):
        """Ensure UNK token is present and identified."""
        self.assertTrue(self.tokenizer.unk_token in self.tokenizer.vocab)
        self.assertEqual(self.tokenizer.unk_token_id, 1)

    # --- Encoding Tests ---
    def test_encode_simple_words(self):
        """Test encoding of words that exist directly in vocab."""
        text = "the quick brown fox"
        ids = self.tokenizer.encode(text)
        expected_ids = [5, 6, 7, 8]
        self.assertEqual(ids, expected_ids)

    def test_encode_subword_splitting(self):
        """Test if 'jumped' splits into 'jump' + '##ed'."""
        # 'jumped' is not in vocab ID map, but 'jump' and '##ed' are.
        # It is in the dictionary config, so it should attempt to split.
        text = "jumped"
        ids = self.tokenizer.encode(text)
        expected_ids = [9, 17]  # jump, ##ed
        self.assertEqual(ids, expected_ids)

    def test_encode_unknown_word(self):
        """Test handling of words completely outside vocab and dictionary."""
        text = "xyzzzz"
        ids = self.tokenizer.encode(text)
        # Should return UNK token ID(s) depending on character fallback
        # Since 'x', 'y', 'z' aren't in mock vocab, it should likely hit UNK
        self.assertTrue(all(x == self.tokenizer.unk_token_id for x in ids))

    def test_encode_special_tokens(self):
        """Test adding BOS and EOS tokens."""
        text = "apple"
        ids = self.tokenizer.encode(text, add_special_tokens=True)
        expected_ids = [2, 14, 3]  # <|bos|>, apple, <|eos|>
        self.assertEqual(ids, expected_ids)

    # --- Decoding Tests ---
    def test_decode_simple(self):
        """Test decoding IDs back to string."""
        ids = [5, 6, 14]  # the, quick, apple
        text = self.tokenizer.decode(ids)
        self.assertEqual(text, "the quick apple")

    def test_decode_merge_subwords(self):
        """Test that '##' prefixes are removed and words merged."""
        ids = [12, 13]  # un, ##happy
        text = self.tokenizer.decode(ids)
        self.assertEqual(text, "unhappy")

    def test_decode_with_special_tokens(self):
        """Test that special tokens are ignored in decode output."""
        ids = [2, 14, 3]  # <|bos|>, apple, <|eos|>
        text = self.tokenizer.decode(ids)
        self.assertEqual(text, "apple")

    # --- Logic & Helper Tests ---
    def test_calculate_pair_score(self):
        """
        Test the scoring logic.
        Longer roots should have significantly higher scores due to cubic weighting.
        """
        # Case A: Short prefix + short suffix
        # un (2 chars), do (2 chars)
        pair_a = [(0, "un"), (2, "do")]

        # Case B: Long root
        # establish (9 chars)
        pair_b = [(0, "establish")]

        # Mock frequencies
        freqs = Counter({"un": 100, "do": 100, "establish": 50})

        score_a = calculate_pair_score(pair_a, freqs)
        score_b = calculate_pair_score(pair_b, freqs)

        # Even though 'un' and 'do' are more frequent, 'establish' is much longer
        # Cubic weight: 2^3 = 8 vs 9^3 = 729. establish should win.
        self.assertGreater(score_b, score_a)

    def test_generate_subword_splits(self):
        """Test if the generator finds valid splits for a compound word."""
        # Setup specific environment for this test function
        vocab = {"un", "##happy", "##s"}
        dictionary = {"unhappy", "unhappiness", "happy", "un"}
        morphemes = Counter({"un": 50, "##happy": 50, "##s": 100})

        word = "unhappy"
        word_str = "unhappy"
        max_len = 5
        spaces = 0

        subwords_by_index, split_types = generate_subword_splits(
            word, word_str, spaces, vocab, dictionary, morphemes, max_len, subword_prefix='##'
        )

        # We expect index 0 to contain "un"
        self.assertIn(0, subwords_by_index)
        self.assertEqual(subwords_by_index[0], "un")

        # We expect the next index (len("un")=2) to contain "happy"
        self.assertIn(2, subwords_by_index)
        self.assertEqual(subwords_by_index[2], "happy")

    # --- Edge Cases ---
    def test_empty_string(self):
        self.assertEqual(self.tokenizer.encode(""), [])
        self.assertEqual(self.tokenizer.decode([]), "")

    def test_whitespaces(self):
        # Should handle extra whitespaces based on regex
        ids = self.tokenizer.encode("   the    apple   ")
        self.assertEqual(ids, [20, 5, 21, 14, 20])

    def test_case_sensitivity_handling(self):
        # "Apple" should match "App", "##le" if "Apple" isn't in vocab
        ids = self.tokenizer.encode("Apple")
        self.assertEqual(ids, [16, 10])

    # --- Save/Load Consistency ---
    def test_save_and_reload(self):
        """Test that saving the tokenizer and reloading it preserves data."""
        save_dir = os.path.join(self.test_dir, "saved_model")
        self.tokenizer.save(save_dir)

        # Check files exist
        self.assertTrue(os.path.exists(os.path.join(save_dir, 'vocab.json')))
        self.assertTrue(os.path.exists(os.path.join(save_dir, 'tokenizer_config.json')))

        # Reload
        new_tokenizer = WordPieceTokenizer(load_directory=save_dir)
        self.assertEqual(new_tokenizer.vocab_size, self.tokenizer.vocab_size)
        self.assertEqual(new_tokenizer.encode("jumped"), self.tokenizer.encode("jumped"))


def try_word_piece_tokenizer():
    global PREFIXES, SUFFIXES, ROOT_WORDS
    # Ensure multiprocessing works correctly when run as a script
    multiprocessing.set_start_method('spawn', force=True)
    # 1. Define your dataset parameters
    file_name_template = 'https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/en/c4-train.{0:05d}-of-01024.json.gz'
    total_shards_in_dataset = 1024
    num_shards = 25  # Load 25 shards total
    shards_to_skip = 10  # Skip the first 10 (start at 10)
    shard_group_size = 1  # How many shards each worker process handles at a time
    text_column = "text"
    lang_code = 'eng'
    num_workers = min(cpu_count(), num_shards)
    # Load morphemes using the updated function
    PREFIXES, SUFFIXES, ROOT_WORDS = load_morphemes(lang_code, MORPHEME_FILE)
    logger.info(f"Processing {num_shards} shards starting from {shards_to_skip}")
    logger.info(f"Using {num_workers} worker processes.")
    # 3. Process the dataset
    try:
        tokenizer = WordPieceTokenizer()
        # Load config to get the dictionary
        vocab = tokenizer.vocab
        config = _load_tokenizer_config(STORAGE_DIR)
        dictionary = config[DICTIONARIES].get(lang_code, set())

        # --- Check for existing word frequencies ---
        logger.info(f"Loading existing frequencies...")
        word_frequency = load_word_frequencies_json(WORD_FREQ_FILE_NAME, lang_code=lang_code)
        if not word_frequency:
            word_frequency = build_word_frequencies_from_c4(
                file_name_template=file_name_template,
                total_shards=total_shards_in_dataset,
                shards_to_skip=shards_to_skip,
                total_shards_to_load=num_shards,
                text_column=text_column,
                num_workers=num_workers
            )
            if word_frequency:
                save_word_frequencies_json(WORD_FREQ_FILE_NAME, word_frequency, lang_code=lang_code)

        _start = time.time()
        # This call *mutates* word_frequency (adds subwords) and returns the *new* maps
        word_str_freq, word_pair_cache = pre_process_word_frequencies(word_frequency)
        logger.info(f"Word-pair pre-processing complete in {time.time() - _start:.2f} s.")

        # Scale and align dictionary words with the frequency counts
        add_dictionary_words_to_corpus(word_frequency, dictionary)

        word_frequency = align_scale_and_cap(word_frequency, max_freq_cap=CORPUS_FREQ_SCALE)
        word_str_freq = align_scale_and_cap(word_str_freq, max_freq_cap=CORPUS_FREQ_SCALE)

        # We now have word_str_freq, which is the {lowercase: freq} map we need.
        avg_word_freq = build_average_frequencies(word_str_freq)
        save_word_frequencies_json("word_str_freq.json", word_str_freq, lang_code=lang_code)

        # Add high frequency words
        new_words = add_high_frequency_words_to_dictionary(dictionary, lang_code, word_str_freq, avg_word_freq)

        # --- Save the updated config (with new dictionary words) ---
        if new_words:
            file_name = "new_dictionary_words.json"
            file_path = os.path.join(STORAGE_DIR, file_name)
            dictionary_words_json = load_json_file(file_path)
            new_words.extend(dictionary_words_json.get(lang_code, []))
            dictionary_words_json[lang_code] = sorted(new_words)
            config[DICTIONARIES][lang_code] = dictionary
            save_json_file(STORAGE_DIR, file_name, dictionary_words_json)
            _save_tokenizer_config(STORAGE_DIR, config)

        # Extract & update prefix, suffix and roots from frequencies
        vocab_words = set(vocab.keys())
        morphemes_freq, subword_frequency, scaled_morphemes_freq = extract_prefix_suffix_roots(
            lang_code, word_frequency, word_pair_cache, avg_word_freq, vocab_words, dictionary
        )

        # Build tokenizer vocab
        _start = time.time()
        vocab = tokenizer.build_wordpiece_vocab(
            lang_code, word_str_freq, word_pair_cache, subword_frequency, morphemes_freq, scaled_morphemes_freq, avg_word_freq,
            vocab_config=(vocab, config),
            morpheme_config=(PREFIXES, SUFFIXES, ROOT_WORDS)
        )

        logger.info(f"Build wordpiece vocabulary completed in {time.time() - _start:.2f} s.")
        logger.info(f"New vocabulary size: {len(vocab)}")
        # add_words_to_vocab(vocab, vocab_words)
        # TODO: Create pre-tokenize cache of top 100,000 most frequent words not in vocab
    except KeyboardInterrupt:
        logger.info("Manually stopped processing.")
    except Exception as e:
        logger.error(f"An error occurred in the main processing loop: {e}", exc_info=True)


@log_execution_time
def evaluate_tokenizer(
        corpus: Counter,
        tokenizer_func: Callable[[str], list],
        vocab_set: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """
    Benchmarks a tokenizer function against a word corpus with added vocabulary validation.

    Understanding the Metrics:
     * Weighted Tokens per Word:
       This is often the most critical metric for efficiency. It tells you how much your sequence length will expand.
       * Lower is usually better for transformer models, as it allows you to fit more actual text into
       the model's fixed context window (e.g., 512 or 8192 tokens).
     * Fertility Rate:
       This measures the "granularity" of the tokenizer.
       * A rate close to 1.0 suggests the tokenizer splits nearly every character.
       * A rate closer to 0.2 - 0.3 implies the tokenizer is successfully matching whole words or large subwords.
     * Speed (Words/Sec):
       This measures the Python-level overhead + the tokenizer logic.
     * Vocab Utilization:
       It calculates what percentage of your provided vocabulary was actually triggered by the corpus.
       This helps identify if the vocabulary is bloated with unused tokens.

    Args:
        corpus (Counter): A Counter object where keys are words and values are frequencies.
        tokenizer_func (Callable): A function that accepts a string and returns a list of tokens.
        vocab_set (Set[str], optional): A set of valid tokens to check for OOV rates.

    Returns:
        Dict: A dictionary containing comprehensive performance metrics.
    """

    # 1. Setup Tracking Variables
    unique_words = list(corpus.keys())
    total_corpus_occurrences = sum(corpus.values())

    generated_unique_tokens = set()
    token_counts_per_word = []
    total_chars = 0

    # Vocab tracking
    oov_tokens = []
    oov_token_occurrences = 0
    total_tokens_generated = 0

    # 2. Execution Phase (Timing the unique word processing)
    start_time = time.perf_counter()

    raw_output = []
    for word in unique_words:
        tokens = tokenizer_func(word)
        raw_output.append(tokens)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # 3. Analysis Phase
    for i, word in enumerate(unique_words):
        tokens = raw_output[i]
        freq = corpus[word]

        # Basic Stats
        num_tokens = len(tokens)
        token_counts_per_word.append(num_tokens)
        generated_unique_tokens.update(tokens)
        total_chars += len(word)

        # Weighted totals for later calculations
        word_total_tokens = num_tokens * freq
        total_tokens_generated += word_total_tokens

        # OOV Analysis (if vocab provided)
        if vocab_set:
            for token in tokens:
                if token not in vocab_set:
                    # We weight OOV by how often this word appears in corpus
                    oov_token_occurrences += freq
                    oov_tokens.append(token)

    print(f"oov_tokens={oov_tokens}\nFound {len(oov_tokens)} oov tokens.")
    # 4. Calculation of Weighted Metrics
    projected_total_chars = sum(len(word) * corpus[word] for word in unique_words)

    # Calculate vocab efficiency metrics if vocab is provided
    vocab_metrics = {}
    if vocab_set:
        # Utilization: What % of the provided vocab did we actually use?
        vocab_utilization = len(generated_unique_tokens.intersection(vocab_set)) / len(vocab_set) if len(
            vocab_set) > 0 else 0

        # OOV Ratio: What % of total generated tokens were not in the provided vocab?
        oov_ratio = oov_token_occurrences / total_tokens_generated if total_tokens_generated > 0 else 0

        vocab_metrics = {
            "provided_vocab_size": len(vocab_set),
            "vocab_utilization_pct": round(vocab_utilization * 100, 2),
            "oov_token_ratio_pct": round(oov_ratio * 100, 4),
            "total_oov_occurrences": oov_token_occurrences,
            "oov_tokens": oov_tokens
        }

    # 5. Compile Report
    report = {
        "speed": {
            "total_time_sec": round(elapsed_time, 4),
            "unique_words_per_sec": int(len(unique_words) / elapsed_time) if elapsed_time > 0 else 0,
            "chars_per_sec": int(total_chars / elapsed_time) if elapsed_time > 0 else 0,
        },
        "efficiency": {
            "unique_tokens_generated": len(generated_unique_tokens),
            "avg_tokens_per_word": round(statistics.mean(token_counts_per_word), 2),
            "weighted_tokens_per_word": round(total_tokens_generated / total_corpus_occurrences, 2),
            "fertility_rate": round(total_tokens_generated / projected_total_chars,
                                    2) if projected_total_chars > 0 else 0
        },
        "vocabulary_analysis": vocab_metrics,
        "corpus_stats": {
            "unique_words_input": len(unique_words),
            "total_occurrences_input": total_corpus_occurrences,
        }
    }

    return report


def print_report(name: str, report: Dict[str, Any]):
    """Helper to print the dictionary nicely."""
    print(f"\n{'=' * 10} REPORT: {name.upper()} {'=' * 10}")

    speed = report['speed']
    eff = report['efficiency']
    vocab = report.get('vocabulary_analysis', {})

    print(f"Processing Speed:     {speed['unique_words_per_sec']:,} words/sec (Unique)")
    print(f"Weighted Tks/Word:    {eff['weighted_tokens_per_word']} (Lower is better)")
    print(f"Fertility Rate:       {eff['fertility_rate']} tokens/char")

    if vocab:
        print("-" * 20)
        print(f"OOV Rate:             {vocab['oov_token_ratio_pct']}%")
        print(f"Vocab Utilization:    {vocab['vocab_utilization_pct']}% of provided dict")
    print("=" * 40)


def benchmark_comparison(corpus: Counter, custom_tokenizer_func: Callable, vocab_set: Optional[Set] = None):
    """
    Compares the custom tokenizer against a fast Hugging Face tokenizer.
    """
    print(f"\nStarting Benchmark Comparison using corpus with {len(corpus)} word frequencies "
          f"& vocab with {len(vocab_set)} tokens...")

    # 1. Evaluate Custom
    custom_report = evaluate_tokenizer(corpus, custom_tokenizer_func, vocab_set)
    print_report("Custom Tokenizer", custom_report)

    # 2. Evaluate Hugging Face (if installed)
    if not TRANSFORMERS_AVAILABLE:
        print("\n[!] 'transformers' library not found. Skipping HF benchmark.")
        print("    Run `pip install transformers` to enable comparison.")
        return

    # Using GPT-2 Fast tokenizer as a standard baseline for speed/efficiency
    model_name = "gpt2"
    print(f"\nLoading Hugging Face '{model_name}' (Fast) for comparison...")

    try:
        # use_fast=True ensures we get the Rust-backed tokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Wrapper to match signature (text -> list of strings)
        def hf_wrapper(text):
            return hf_tokenizer.tokenize(text)

        hf_report = evaluate_tokenizer(corpus, hf_wrapper)
        print_report(f"HF {model_name} (Fast)", hf_report)

        # 3. Comparative Summary
        custom_speed = custom_report['speed']['unique_words_per_sec']
        hf_speed = hf_report['speed']['unique_words_per_sec']
        speed_diff = custom_speed / hf_speed if hf_speed > 0 else 0

        print("\n" + "*" * 40)
        print("COMPARISON SUMMARY")
        print("*" * 40)
        if custom_speed > hf_speed:
            print(f"Speed: Custom is {round(speed_diff, 2)}x FASTER than HF.")
        else:
            print(f"Speed: Custom is {round(hf_speed / custom_speed, 2)}x SLOWER than HF.")

        c_eff = custom_report['efficiency']['weighted_tokens_per_word']
        h_eff = hf_report['efficiency']['weighted_tokens_per_word']
        print(f"Compression: Custom ({c_eff}) vs HF ({h_eff}) [Weighted Tokens/Word]")

    except Exception as e:
        print(f"Error loading HF tokenizer: {e}")


if __name__ == "__main__":
    # 3. Run evaluation
    """
    tokenizer = WordPieceTokenizer()
    # TODO - sorted_word_frequencies.json
    word_frequency = load_word_frequencies_json(WORD_FREQ_FILE_NAME, lang_code='eng')
    stats = evaluate_tokenizer(word_frequency, tokenizer.tokenize)
    if tokenizer.tokenizer_updated:
        tokenizer.save(tokenizer.vocab_dir)
    benchmark_comparison(word_frequency, tokenizer.tokenize, tokenizer.vocab)
    """

    try_word_piece_tokenizer()
