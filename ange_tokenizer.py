"""
ange_tokenizer.py — Semantic Morpheme Tokenizer
================================================

A language-agnostic, morphologically-aware tokenizer that segments words into
linguistically valid subword units using statistical scoring, embedding-based
semantic similarity, and a graph-based dynamic-programming split enumerator.

Architecture overview
---------------------
SemanticTokenizer
  ├─ SemanticTokenLexicon  — scoring via cosine-similarity over live embeddings
  ├─ CandidateCache        — per-word k-best split pool with slot-based training
  └─ SimpleTransformerXL   — lightweight embedding backend (numpy / PyTorch)

Key algorithms
--------------
* Graph-DP split enumeration  — O(n × V) per word, where V = vocab tokens that
  span from position i.  Enumerate ALL valid tokenizations up to a budget cap.
* calculate_split_score       — Cubic-length × log-frequency × validity-multiplier
  scoring that strongly prefers real morphemes over arbitrary byte-fragments.
* infer_language_bias         — Statistical analysis of dict_words to determine
  whether a language is primarily suffixing or prefixing, used to tune the
  affix-stripping order inside extract_productive_roots.
* Training cycle              — Stochastic slot selection (85/10/5 %) combined
  with pair-affinity updates nudge the lexicon toward linguistically valid splits.

Supported inputs
----------------
Any ISO 639-3 language and any Unicode script.  Latin-script neoclassical roots
"""

import os
import sys
import json
import time
import shutil
import torch
import regex as re
import numpy as np
import unicodedata
import statistics
from functools import wraps
from typing import (
    List, Set, Dict, Any, Tuple, Optional, Iterable, Counter as CounterType, Callable, FrozenSet
)
from collections import Counter, defaultdict, OrderedDict, deque

from context_aware_io import save_json_file, load_json_file
from latin_greek_root_extractor import _build_corpus_profile, get_combining_vowels, extract_productive_roots
from morph_segmenter_ica import MorphSegmenterICA, get_language_family
from morpheme_extractor import extract_productive_affixes
from morpho_rules import CompiledRules
from copy import deepcopy
import logging
import math

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
CASE_SPLIT_PATTERN = re.compile(
    r'\p{Lu}\p{Ll}+|\p{Ll}+|\p{Lu}+(?=\p{Lu}\p{Ll}{2,})|\p{Lu}+|\p{Alphabetic}+',
    re.UNICODE,
)
# Pre-tokenizer: preserve runs of 2+ whitespace chars as a single token,
# collapse single whitespace into separate tokens, match everything else.
UNICODE_PATTERN = re.compile(
    r'\p{White_Space}{2,}'   # 2+ consecutive whitespace (preserves formatting)
    r'|\p{Alphabetic}+'             # word tokens
    r'|\p{White_Space}'             # single whitespace
    r'|\p{M}|\p{N}|\p{P}|\p{S}'     # marks, numbers, punctuation, symbols
    r'|\p{C}|.',                    # control chars, anything else
    re.UNICODE,
)
WHITE_SPACE_CHARS = {" ", "\n", "\t", "\r", "\f", "\v"}

# --- Load Initial Morphemes (Global Scope) ---
PREFIXES, SUFFIXES, ROOT_WORDS, DERIVED_WORDS = set(), set(), set(), set()

# --- Pre-computed constants for text normalization ---
DELETION_CHARS = (
    '\u00ad\u180e\u200b\u200c\u200d\u200e\u200f'
    '\u202a\u202b\u202c\u202d\u202e'
    '\u2060\u2061\u2062\u2063\u2064'
    '\u206A\u206B\u206C\u206D\u206E\u206F'
    '\ufeff\ufff9\ufffa\ufffb'
    '\ufe00\ufe01\ufe02\ufe03\ufe04\ufe05\ufe06\ufe07'
    '\ufe08\ufe09\ufe0a\ufe0b\ufe0c\ufe0d\ufe0e\ufe0f'
)

C0_CONTROLS_TO_SPACE = (
    '\u0000\u0001\u0002\u0003\u0004\u0005\u0006\u0007\u0008'
    '\u000A\u000B\u000C\u000D\u000E\u000F'
    '\u0010\u0011\u0012\u0013\u0014\u0015\u0016\u0017'
    '\u0018\u0019\u001A\u001B\u001C\u001D\u001E\u001F'
)
C1_CONTROLS_TO_SPACE = (
    '\u007F'
    '\u0080\u0081\u0082\u0083\u0084\u0085\u0086\u0087\u0088\u0089\u008A\u008B\u008C\u008D\u008E\u008F'
    '\u0090\u0091\u0092\u0093\u0094\u0095\u0096\u0097\u0098\u0099\u009A\u009B\u009C\u009D\u009E\u009F'
)
UNICODE_BREAKS_TO_SPACE = '\u2028\u2029'

REPLACEMENT_CHARS = C0_CONTROLS_TO_SPACE + C1_CONTROLS_TO_SPACE + UNICODE_BREAKS_TO_SPACE

_ROBUST_TRANSLATION_MAP = str.maketrans(
    REPLACEMENT_CHARS,
    ' ' * len(REPLACEMENT_CHARS),
    DELETION_CHARS,
)

DICT_WORDS = "words.json"
WORD_FREQ_FILE_NAME = 'word_frequencies.json'
MORPHEME_FREQ_FILE_NAME = 'morpheme_frequencies.json'
VOCAB_FILE_NAME = 'vocab.json'

try:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT_DIR = os.getcwd()

LOG_DIR = os.path.join(ROOT_DIR, "logs")
TEST_DIR = os.path.join(ROOT_DIR, "tests")
TEST_FILES_DIR = os.path.join(TEST_DIR, "files")
STORAGE_DIR = os.path.join(ROOT_DIR, "vocab_files")
VOCAB_FILE = os.path.join(STORAGE_DIR, VOCAB_FILE_NAME)
MORPHEME_FILE = os.path.join(STORAGE_DIR, 'morphemes.json')
FILTERED_WORD_FREQ_FILE = os.path.join(STORAGE_DIR, 'dict_word_frequencies.json')

# Slot selection probabilities during training
SLOT_1_PROB = 0.85
SLOT_2_PROB = 0.10
SLOT_3_PROB = 0.05
SLOT_REST_PROB = 0.01

CONSECUTIVE_LOWER_THRESHOLD = 3
MAX_VOCAB_PER_LANGUAGE = 200_000  # ---- Maximum vocab budget per language ----

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.WARNING,
    stream=sys.stdout,
)
logger = logging.getLogger("SemanticTokenizer")


# ============================================================================
# Scoring helpers
# ============================================================================

def calculate_split_score(
        token_ids: Tuple[int, ...],
        id_to_token: Dict[int, str],
        morphemes_freq: CounterType[str],
) -> float:
    """
    Calculates a default score for a candidate split (tuple of token IDs).

    Uses Cubic Length Weighting (length^3) * log-smoothed frequency,
    **plus** a validity multiplier that sharply distinguishes real
    morphemes (high frequency in the dictionary extraction) from
    arbitrary fragments.

    The key insight: a token like "skysc" will have very low morpheme_freq
    (it never appears as a real prefix/suffix/root), while "sky" or "scrap"
    have high frequencies.  The validity multiplier amplifies this gap so
    that fewer-but-invalid tokens can never outscore more-but-valid tokens.
    """
    if not token_ids:
        return 0.0

    subwords = [id_to_token.get(tid, "?") for tid in token_ids]
    word_str_equal = _is_lower_case_words(subwords)

    score = 0.0
    single_char_penalty = 0
    num_valid = 0
    last_idx = len(subwords) - 1

    for idx, subword in enumerate(subwords):
        freq = morphemes_freq.get(subword, 0)
        if not word_str_equal:
            freq = max(freq, morphemes_freq.get(subword.lower(), 0))

        freq_score = math.log(max(freq, 2))
        length_weight = len(subword) ** 3

        # --- Validity multiplier ---
        # High-frequency morphemes (freq >= 5) get a strong bonus.
        # Low-frequency or unseen fragments get heavily penalized.
        # This ensures "skysc" (freq ~0-1) scores far below "sky" (freq >> 5).
        if freq >= 5:
            validity = 2.0  # known productive morpheme
            num_valid += 1
        elif freq >= 2:
            validity = 1.0  # plausible but rare
        else:
            validity = 0.15  # arbitrary fragment — heavy penalty

        score += freq_score * length_weight * validity

        if len(subword) == 1 and idx != last_idx:
            single_char_penalty += 1

    # Token-count normalization: slight penalty for more tokens, but not
    # so aggressive that 2 invalid tokens beat 3 valid ones.
    score = score / (len(subwords) ** 0.3)

    # Bonus when ALL parts are valid morphemes
    if num_valid == len(subwords):
        score *= 1.5

    if single_char_penalty > 1:
        score *= 0.75

    return score


def get_common_affixes_by_percentile(
        prod_counter: CounterType,
        percentile: float = 0.90,
        percentile_by_len: dict[int, float] | None = None,
        max_len: int = 10,
        min_freq=80,
) -> Set[str]:
    percentile_by_len = percentile_by_len or {i: percentile for i in range(1, max_len + 1)}
    by_length: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    for affix, prod in prod_counter.items():
        if len(affix) > max_len:
            continue
        by_length[len(affix)].append((affix, prod))

    selected: Set[str] = set()
    for length, items in by_length.items():
        if not items:
            continue
        prods = sorted(p for _, p in items)
        n = len(prods)
        percent = percentile_by_len.get(length, percentile)
        threshold_idx = max(0, math.ceil(percent * n) - 1)
        minimum_freq_idx = max(0, math.ceil(percentile * n) - 1)
        if n <= 3:
            threshold = max(min_freq, prods[0])
            minimum_freq = max(min_freq, prods[0])
        else:
            threshold = max(min_freq, prods[threshold_idx])
            minimum_freq = max(min_freq, prods[minimum_freq_idx])
        for affix, prod in items:
            if prod >= threshold:
                selected.add(affix)
            elif length > 3 and prod >= minimum_freq:
                for i in range(1, len(affix) - 1):
                    if affix[:i] in selected and affix[i:] in selected:
                        selected.add(affix)
                        break
    return selected


def _is_lower_case_words(subwords: List[str]):
    try:
        return all(sub[0].islower() for sub in subwords)
    except IndexError:
        return True


def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds.")
        return result
    return wrapper


# ============================================================================
# Dictionary / corpus helpers
# ============================================================================

def get_skyscraper_training_sentences() -> List[List[str]]:
    raw = [
        "the sky scraper towers over the city sky line",
        "a tall sky scraper was built down town",
        "sky scrapers define the modern city sky line",
        "the sky scraper reached into the sky above",
        "workers built the sky scraper floor by floor",
        "the glass sky scraper reflected the blue sky",
        "every sky scraper starts with a strong found ation",
        "sky scrapers are symbols of urban develop ment",
        "the sky scraper stood tall against the sky",
        "from the sky scraper you can see the whole city",
        "the tallest sky scraper in the world is a tower",
        "sky line views from the sky scraper are amaz ing",
        "the sky scraper was designed by a famous architect",
        "new sky scrapers are being built every year",
        "the sky was clear above the sky scraper",
        "a sky scraper is also called a high rise building",
        "sky scrapers need deep found ations in the ground",
        "the sky scraper cast a long shadow over the street",
        "engineers designed the sky scraper to with stand wind",
        "the sky scraper had over one hundred floors",
        "the back ground of the photo showed sky scrapers",
        "under ground parking is common beneath sky scrapers",
        "high light of the tour was visiting the sky scraper",
        "the water melon vendor stood near the sky scraper",
        "fire men rushed to the sky scraper during the alarm",
        "note book in hand she sketched the sky scraper",
        "moon light illuminated the sky scraper at night",
        "rain coat needed as clouds gathered around the sky scraper",
        "un account ed funds were found in the sky scraper office",
        "the account ability report covered sky scraper safety",
        "under standing sky scraper engineering requires study",
        "extra ordinary views from the sky scraper observation deck",
        "inter connected sky scrapers formed a complex",
    ]
    return [s.split() for s in raw]


def load_morphemes(filepath: str) -> Tuple[Set[str], Set[str], Set[str]]:
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return (
                set(data.get("prefixes", [])),
                set(data.get("suffixes", [])),
                set(data.get("root_words", [])),
            )
        except json.JSONDecodeError:
            return set(), set(), set()
    return set(), set(), set()


def save_morphemes(filepath: str, prefixes: Set[str], suffixes: Set[str], root_words: Set[str]):
    data = {
        "prefixes": sorted(list(prefixes)),
        "suffixes": sorted(list(suffixes)),
        "root_words": sorted(list(root_words)),
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_word_frequencies_json(file_name: str, lang_code: str = '') -> Counter:
    if lang_code:
        file_name = f'{lang_code}_{file_name}'
    filepath = os.path.join(STORAGE_DIR, file_name)
    if not os.path.exists(filepath):
        return Counter()
    try:
        with open(filepath, 'r', encoding="utf-8") as f:
            return Counter(json.load(f))
    except Exception:
        return Counter()


def save_word_frequencies_json(file_name: str, word_counter, lang_code: str = ''):
    if lang_code:
        file_name = f'{lang_code}_{file_name}'
    filepath = os.path.join(STORAGE_DIR, file_name)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = dict(word_counter) if isinstance(word_counter, Counter) else word_counter
    with open(filepath, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ============================================================================
# Pair scoring
# ============================================================================

def calculate_pair_score(
    subword_pairs_tuples: List[Tuple[int, str]],
    morphemes_freq: Counter,
    dictionary: Set[str],
) -> float:
    subwords = [x[1].lower() for x in subword_pairs_tuples]
    word = "".join(subwords)
    if not word:
        return 0.0

    score = 0.0
    all_parts_valid = True
    single_char_penalty = 0
    last_idx = len(word) - 1

    for idx, subword in subword_pairs_tuples:
        word_str = subword.lower()
        length_val = len(word_str)
        freq = morphemes_freq.get(word_str, 1)

        freq_score = math.log(max(freq, 2))
        length_weight = length_val ** 2.2

        if word_str in dictionary:
            score += (length_val * 500) + (300 * freq_score)
        elif word_str in PREFIXES or word_str in SUFFIXES:
            score += (length_val * 350) + (200 * freq_score)
        elif word_str in DERIVED_WORDS:
            score += (length_val * 200) + (150 * freq_score)
        else:
            all_parts_valid = False
            score -= 1500 if length_val < 5 else 500

        score += (freq_score * length_weight)

        if length_val == 1 and idx != last_idx:
            single_char_penalty += 1

    if single_char_penalty > 1:
        score *= 0.2 if single_char_penalty > 1 else 0.4

    lengths = [len(s) for s in subwords]
    if len(lengths) > 1:
        balance = min(lengths) / max(lengths)
        score *= (1 + (balance * 0.2))

    if all_parts_valid:
        score *= 3

    return score / (len(subwords) ** 0.5)


# ============================================================================
# Space handling for pre/post tokenization
# ============================================================================

def remove_single_spaces(tokens):
    """
    Removes single spaces to reconstruct words.
    KEEP space if BOTH surrounding words are Capitalized (e.g. "Hello World").
    REMOVE space if ANY surrounding word is lowercase.
    Multi-char whitespace tokens (indentation, etc.) are always kept.
    """
    if len(tokens) < 3:
        return tokens

    selected_tokens = []
    max_idx = len(tokens) - 1

    for i, token in enumerate(tokens):
        if 0 < i < max_idx and len(token) == 1 and re.match(SPACE_PATTERN, token):
            prev_token = tokens[i - 1]
            next_token = tokens[i + 1]
            if prev_token[0].isalpha() and next_token[0].isalpha():
                continue  # Remove single space between word tokens
            else:
                selected_tokens.append(token)
        else:
            selected_tokens.append(token)

    return selected_tokens


def add_single_spaces(tokens):
    """Adds a single space between valid word tokens."""
    selected_tokens = []
    max_idx = len(tokens) - 1

    for i, token in enumerate(tokens):
        selected_tokens.append(token)
        if i < max_idx:
            next_token = tokens[i + 1]
            if token[0].isalpha() and next_token[0].isalpha():
                selected_tokens.append(' ')

    return selected_tokens


def normalize_text_for_nlp(text: str) -> str:
    """Robust 3-stage normalization: NFKC → translation → tab expansion."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKC', text)
    text = text.translate(_ROBUST_TRANSLATION_MAP)
    text = text.replace('\t', '    ')
    return text


# ============================================================================
# Lexicon: Semantic scoring via embeddings
# ============================================================================

class SemanticTokenLexicon:
    """
    Scoring via Semantic Search over real-time model embeddings.

    score = base_score + semantic_affinity_bonus
      base_score  = 1.0 / log(num_tokens + 1)
      affinity    = mean cosine-sim of adjacent pairs, scaled to [0, 0.05]
    """

    def __init__(self, model):
        self.model = model
        self._refresh_embeddings()
        self.pair_affinity: Dict[Tuple[str, str], float] = {}
        self.pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    def _refresh_embeddings(self):
        """Extract embedding weights as a numpy array and normalize."""
        emb = self.model.get_embedding_weights()  # always returns np.ndarray
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        self.normalized_embeddings = emb / (norm + 1e-8)
        self.vocab_size = emb.shape[0]
        self.embedding_dim = emb.shape[1]

    def refresh_from_model(self):
        self._refresh_embeddings()
        self.pair_affinity.clear()

    def get_pair_affinity(self, token1: str, token2: str) -> float:
        pair = (token1, token2)
        if pair not in self.pair_affinity:
            sim = self.model.similarity(token1, token2)
            self.pair_affinity[pair] = sim
        return self.pair_affinity[pair]

    def score_tokenization(self, tokens: List[str]) -> float:
        n = len(tokens)
        if n == 0:
            return 0.0

        base_score = 1.0 / np.log(max(2, n + 1))

        affinity_bonus = 0.0
        if n >= 2:
            total = sum(self.get_pair_affinity(tokens[i], tokens[i + 1]) for i in range(n - 1))
            mean_aff = total / (n - 1)
            affinity_bonus = ((mean_aff + 1.0) / 2.0) * 0.05

        return float(base_score + affinity_bonus)

    def update_pair_scores(self, tokens: List[str], observed_score: float, lr: float = 0.1):
        if len(tokens) < 2:
            return
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            cur = self.get_pair_affinity(tokens[i], tokens[i + 1])
            self.pair_affinity[pair] = (1 - lr) * cur + lr * observed_score
            self.pair_counts[pair] += 1

    def get_stats(self) -> Dict:
        if not self.pair_affinity:
            return {"total_pairs_computed": 0, "total_pairs_updated": 0, "mean_affinity": 0.0}
        aff = list(self.pair_affinity.values())
        updated = sum(1 for c in self.pair_counts.values() if c > 0)
        return {
            "total_pairs_computed": len(self.pair_affinity),
            "total_pairs_updated": updated,
            "mean_affinity": float(np.mean(aff)),
            "max_affinity": float(np.max(aff)),
            "min_affinity": float(np.min(aff)),
            "std_affinity": float(np.std(aff)),
        }


# ============================================================================
# Candidate cache
# ============================================================================

class CandidateEntry:
    __slots__ = ('tokens', 'score', 'prev_score', 'consecutive_lower', 'enabled')

    def __init__(self, tokens: Tuple[str, ...], score: float):
        self.tokens = tokens
        self.score = score
        self.prev_score = score
        self.consecutive_lower = 0
        self.enabled = True

    def update_score(self, new_score: float):
        if new_score < self.score:
            self.consecutive_lower += 1
        else:
            self.consecutive_lower = 0
        self.prev_score = self.score
        self.score = new_score
        if self.consecutive_lower >= CONSECUTIVE_LOWER_THRESHOLD:
            self.enabled = False

    def __repr__(self):
        state = "ON" if self.enabled else "OFF"
        return f"CandidateEntry({list(self.tokens)}, score={self.score:.4f}, {state})"


class CandidateCache:
    def __init__(self):
        self._candidates: Dict[str, List[CandidateEntry]] = {}
        self._rest_deque: Dict[str, deque] = {}
        self.hits = 0
        self.misses = 0

    def put_candidates(self, word: str, entries: List[CandidateEntry]):
        self._candidates[word] = entries
        self._rebuild_deque(word)

    def _rebuild_deque(self, word: str):
        entries = self._candidates.get(word, [])
        rest = deque()
        for i in range(3, len(entries)):
            if entries[i].enabled:
                rest.append(i)
        self._rest_deque[word] = rest

    def select_candidate_for_training(self, word: str) -> Optional[CandidateEntry]:
        entries = self._candidates.get(word)
        if not entries:
            self.misses += 1
            return None
        self.hits += 1

        enabled_top3 = [i for i in range(min(3, len(entries))) if entries[i].enabled]
        if not enabled_top3:
            dq = self._rest_deque.get(word, deque())
            if dq:
                idx = dq.popleft()
                dq.append(idx)
                return entries[idx]
            return None

        r = np.random.random()
        if r <= SLOT_1_PROB and len(enabled_top3) >= 1:
            return entries[enabled_top3[0]]
        if r <= SLOT_1_PROB + SLOT_2_PROB and len(enabled_top3) >= 2:
            return entries[enabled_top3[1]]
        if r <= SLOT_1_PROB + SLOT_2_PROB + SLOT_3_PROB and len(enabled_top3) >= 3:
            return entries[enabled_top3[2]]

        dq = self._rest_deque.get(word, deque())
        if dq:
            idx = dq.popleft()
            dq.append(idx)
            if entries[idx].enabled:
                return entries[idx]
        return entries[enabled_top3[0]]

    def get_best(self, word: str) -> Optional[CandidateEntry]:
        entries = self._candidates.get(word)
        if entries and entries[0].enabled:
            self.hits += 1
            return entries[0]
        self.misses += 1
        return None

    def get_all_tokens(self, word: str) -> Optional[List[Tuple[str, ...]]]:
        entries = self._candidates.get(word)
        if entries is not None:
            self.hits += 1
            return [e.tokens for e in entries]
        self.misses += 1
        return None

    def get_entries(self, word: str) -> Optional[List[CandidateEntry]]:
        return self._candidates.get(word)

    def has_candidates(self, word: str) -> bool:
        return word in self._candidates

    def re_rank(self, word: str):
        entries = self._candidates.get(word)
        if entries:
            entries.sort(key=lambda e: e.score, reverse=True)
            self._rebuild_deque(word)

    def get_stats(self) -> Dict:
        total = self.hits + self.misses
        total_candidates = sum(len(v) for v in self._candidates.values())
        total_enabled = sum(sum(1 for e in v if e.enabled) for v in self._candidates.values())
        return {
            "cached_words": len(self._candidates),
            "total_candidates": total_candidates,
            "enabled_candidates": total_enabled,
            "disabled_candidates": total_candidates - total_enabled,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
        }

# ============================================================================
# Language typology: suffixing vs. prefixing bias
# ============================================================================

#: Empirically determined ratio thresholds used by ``infer_language_bias``.
#: A language is considered *strongly suffixing* when the suffix-to-prefix
#: productivity ratio exceeds ``_STRONG_SUFFIXING_RATIO`` and *weakly suffixing*
#: when it exceeds ``_WEAK_SUFFIXING_RATIO``.
_STRONG_SUFFIXING_RATIO: float = 1.4
_WEAK_SUFFIXING_RATIO: float   = 1.05


def infer_language_bias(
    dict_words,
    *,
    min_affix_len: int = 2,
    max_affix_len: int = 8,
    min_type_freq: int = 5,
    sample_size: Optional[int] = None,
    _strong_ratio: float = _STRONG_SUFFIXING_RATIO,
    _weak_ratio:   float = _WEAK_SUFFIXING_RATIO,
) -> Dict[str, Any]:
    """Determine whether a language is primarily suffixing or prefixing.

    Uses a **paradigm-based** statistical analysis of *dict_words* — the most
    corpus-size-independent measure available — to classify the language on the
    suffixing ↔ prefixing typological axis (Dryer 2013; WALS Chapter 26).

    Algorithm — paradigm-based stem recovery
    -----------------------------------------
    For every word *W* in *dict_words* and every candidate affix length *L* in
    [*min_affix_len*, *max_affix_len*]:

    * **Suffix test** — strip the last *L* characters.  If the remaining stem
      ``W[:-L]`` is itself present in *dict_words*, record *W* as a
      *suffix-derived word* and increment the type-frequency counter for the
      suffix ``W[-L:]``.
    * **Prefix test** — strip the first *L* characters.  If the remaining stem
      ``W[L:]`` is present in *dict_words*, record *W* as a *prefix-derived word*
      and increment the type-frequency counter for the prefix ``W[:L]``.

    Affixes seen in fewer than *min_type_freq* distinct words are discarded as
    hapax fragments.  Productivity is then scored as::

        productivity(affix) = type_freq(affix) × len(affix)

    Two complementary ratios are computed and combined geometrically:

    * **prod_ratio** — ``suffix_productivity / prefix_productivity`` (reflects
      morpheme richness).
    * **word_ratio** — ``suffix_derived_words / prefix_derived_words`` (reflects
      how many distinct word-types exhibit each kind of affixation).

    The geometric mean of the two ratios is the final discriminant.  Values
    above *_strong_ratio* (default 1.4) indicate *strongly suffixing*; values
    below ``1 / _strong_ratio`` indicate *strongly prefixing*.

    Why paradigm-based?
    -------------------
    Raw substring frequency conflates root-initial/final sequences with true
    affixes.  By requiring the stripped stem to be a *dictionary word*, the
    method counts only **attested morphological relationships** — the same
    criterion used in linguistic typology fieldwork.

    Parameters
    ----------
    dict_words : Iterable[str]
        Lowercased dictionary words for the target language.  Needs ≥ 50 types
        for reliable results; fewer triggers an ``"insufficient_data"`` result.
    min_affix_len : int, optional
        Shortest candidate affix (default ``2``).  Set to ``1`` for highly
        agglutinative languages with single-character morphemes (e.g. Georgian).
    max_affix_len : int, optional
        Longest candidate affix (default ``8``).
    min_type_freq : int, optional
        Minimum number of distinct word-types that must exhibit an affix for it
        to count as productive (default ``5``).
    sample_size : int or None, optional
        If given, uses a deterministic sample of this many words.  Useful for
        very large dictionaries (> 200 k words).  ``None`` uses the full set.
    _strong_ratio : float, optional
        Ratio threshold for *strongly* suffixing / prefixing (default ``1.4``).
    _weak_ratio : float, optional
        Ratio threshold for *weakly* suffixing / prefixing (default ``1.05``).

    Returns
    -------
    Dict[str, Any]
        Keys:

        ``"bias"`` : str
            One of ``"strongly_suffixing"``, ``"weakly_suffixing"``,
            ``"neutral"``, ``"weakly_prefixing"``, ``"strongly_prefixing"``,
            or ``"insufficient_data"``.
        ``"is_suffixing"`` : bool
            ``True`` when bias is ``"strongly_suffixing"`` or
            ``"weakly_suffixing"``; defaults to ``True`` when insufficient data
            (safe default — most world languages are primarily suffixing).
        ``"suffix_productivity"`` : float
            Sum of ``type_freq × length`` over all qualifying suffixes.
        ``"prefix_productivity"`` : float
            Sum of ``type_freq × length`` over all qualifying prefixes.
        ``"ratio"`` : float
            Geometric mean of *prod_ratio* and *word_ratio*.
        ``"suffix_candidates"`` : int
            Qualifying suffix types after the *min_type_freq* filter.
        ``"prefix_candidates"`` : int
            Qualifying prefix types after the *min_type_freq* filter.
        ``"suffix_derived_words"`` : int
            Distinct word-types identified as suffix-derived.
        ``"prefix_derived_words"`` : int
            Distinct word-types identified as prefix-derived.
        ``"per_length_ratios"`` : Dict[int, float]
            Per-length ``suffix_prod / prefix_prod`` ratios.
        ``"words_analysed"`` : int
            Word-types actually analysed (after optional sampling).
        ``"confidence"`` : float
            Heuristic confidence in [0, 1].

    Raises
    ------
    TypeError
        If *dict_words* is not iterable.
    ValueError
        If *min_affix_len* > *max_affix_len* or either is ≤ 0.

    Examples
    --------
    English (strongly suffixing):

    >>> eng = {w + s for w in ["walk","talk","play","work","run"]
    ...        for s in ["ed","ing","er","ly","ness","tion","ment","able"]}
    >>> eng |= {"walk","talk","play","work","run"}  # add base forms
    >>> result = infer_language_bias(eng)
    >>> result["bias"]
    'strongly_suffixing'
    >>> result["is_suffixing"]
    True

    Swahili-like (strongly prefixing):

    >>> sw = {p + b for b in ["toto","jiji","kono","lango","shule","nyumba"]
    ...       for p in ["m","wa","ki","vi","mu","mi","u","i","a","ha"]}
    >>> sw |= {"toto","jiji","kono","lango","shule","nyumba"}
    >>> result = infer_language_bias(sw)
    >>> result["bias"]
    'strongly_prefixing'

    References
    ----------
    Dryer, M. S. (2013). Prefixing vs. Suffixing in Inflectional Morphology.
    In M. S. Dryer & M. Haspelmath (eds.), *The World Atlas of Language
    Structures Online*. Max Planck Institute for Evolutionary Anthropology.
    https://wals.info/chapter/26

    Bickel, B. & Nichols, J. (2013). Exponence of Selected Inflectional
    Formatives. In *WALS Online* (ch. 21a). MPI-EVA.
    """
    # ── Input validation ────────────────────────────────────────────────────
    if not hasattr(dict_words, '__iter__'):
        raise TypeError("dict_words must be an iterable of strings")
    if min_affix_len < 1 or max_affix_len < 1:
        raise ValueError("min_affix_len and max_affix_len must be positive integers")
    if min_affix_len > max_affix_len:
        raise ValueError(
            f"min_affix_len ({min_affix_len}) must be <= max_affix_len ({max_affix_len})"
        )

    # ── Build word set and optionally sample ────────────────────────────────
    if not isinstance(dict_words, (set, frozenset)):
        word_list: List[str] = list(dict_words)
    else:
        word_list = list(dict_words)

    if sample_size is not None and sample_size < len(word_list):
        seed = 0
        for w in word_list[:8]:
            seed ^= hash(w)
        rng = __import__('random').Random(seed & 0xFFFFFFFF)
        rng.shuffle(word_list)
        word_list = word_list[:sample_size]

    n_words = len(word_list)

    # ── Insufficient data guard ─────────────────────────────────────────────
    _MINIMUM_WORDS = 50
    if n_words < _MINIMUM_WORDS:
        return {
            "bias":                 "insufficient_data",
            "is_suffixing":         True,  # safe default: most world languages suffix
            "suffix_productivity":  0.0,
            "prefix_productivity":  0.0,
            "ratio":                0.0,
            "suffix_candidates":    0,
            "prefix_candidates":    0,
            "suffix_derived_words": 0,
            "prefix_derived_words": 0,
            "per_length_ratios":    {},
            "words_analysed":       n_words,
            "confidence":           0.0,
        }

    # ── Paradigm-based stem recovery ────────────────────────────────────────
    # A word W is "suffix-derived" if W[:-L] ∈ word_set for some valid L.
    # A word W is "prefix-derived" if W[L:]  ∈ word_set for some valid L.
    # Requiring the stem to be a dictionary word is the key to robustness:
    # it ensures we count attested morphological relationships, not arbitrary
    # substrings.
    word_set: Set[str] = set(word_list)

    suffix_type_freq: Counter  = Counter()   # affix_str → # word-types bearing it
    prefix_type_freq: Counter  = Counter()

    suffix_derived_words: Set[str] = set()
    prefix_derived_words: Set[str] = set()

    # Per-length productivity accumulators for diagnostic output
    suffix_prod_by_len: Dict[int, float] = defaultdict(float)
    prefix_prod_by_len: Dict[int, float] = defaultdict(float)

    # Minimum stem length: require the remaining stem has >= min_affix_len chars
    # so single-character "stems" don't inflate counts.
    _MIN_STEM = max(min_affix_len, 2)

    for word in word_list:
        wlen = len(word)
        # Word must be long enough to have both an affix AND a valid stem
        if wlen <= min_affix_len + _MIN_STEM - 1:
            continue
        hi = min(wlen - _MIN_STEM, max_affix_len)

        for length in range(min_affix_len, hi + 1):
            # ── Suffix test ──
            stem_before = word[:-length]
            if stem_before in word_set:
                suffix = word[-length:]
                suffix_type_freq[suffix] += 1
                suffix_derived_words.add(word)

            # ── Prefix test ──
            stem_after = word[length:]
            if stem_after in word_set:
                prefix = word[:length]
                prefix_type_freq[prefix] += 1
                prefix_derived_words.add(word)

    # ── Filter hapax affixes and compute productivity ───────────────────────
    sfx_prod_total: float = 0.0
    pfx_prod_total: float = 0.0
    n_sfx_candidates: int = 0
    n_pfx_candidates: int = 0

    for affix, freq in suffix_type_freq.items():
        if freq >= min_type_freq:
            prod = freq * len(affix)
            sfx_prod_total += prod
            suffix_prod_by_len[len(affix)] += prod
            n_sfx_candidates += 1

    for affix, freq in prefix_type_freq.items():
        if freq >= min_type_freq:
            prod = freq * len(affix)
            pfx_prod_total += prod
            prefix_prod_by_len[len(affix)] += prod
            n_pfx_candidates += 1

    # ── Two complementary ratios ────────────────────────────────────────────
    # prod_ratio:  morpheme richness (weighted by length)
    # word_ratio:  proportion of words exhibiting each affix type
    n_sfx_words = len(suffix_derived_words)
    n_pfx_words = len(prefix_derived_words)

    def _safe_ratio(a: float, b: float) -> float:
        if b == 0.0:
            return float('inf') if a > 0 else 1.0
        return a / b

    prod_ratio = _safe_ratio(sfx_prod_total, pfx_prod_total)
    word_ratio = _safe_ratio(float(n_sfx_words), float(n_pfx_words))

    # Geometric mean — avoids one extreme ratio dominating
    if prod_ratio == float('inf') and word_ratio == float('inf'):
        combined = float('inf')
    elif prod_ratio == float('inf'):
        combined = word_ratio * 2.0
    elif word_ratio == float('inf'):
        combined = prod_ratio * 2.0
    else:
        combined = math.sqrt(prod_ratio * word_ratio)

    # ── Per-length ratios (diagnostic) ─────────────────────────────────────
    per_length_ratios: Dict[int, float] = {}
    all_lengths = sorted(set(suffix_prod_by_len) | set(prefix_prod_by_len))
    for L in all_lengths:
        per_length_ratios[L] = _safe_ratio(
            suffix_prod_by_len.get(L, 0.0),
            prefix_prod_by_len.get(L, 0.0),
        )

    # ── Classification ──────────────────────────────────────────────────────
    if combined > _strong_ratio:
        bias = "strongly_suffixing"
    elif combined > _weak_ratio:
        bias = "weakly_suffixing"
    elif combined < (1.0 / _strong_ratio):
        bias = "strongly_prefixing"
    elif combined < (1.0 / _weak_ratio):
        bias = "weakly_prefixing"
    else:
        bias = "neutral"

    is_suffixing: bool = bias in ("strongly_suffixing", "weakly_suffixing")

    # ── Confidence heuristic ────────────────────────────────────────────────
    total_candidates = n_sfx_candidates + n_pfx_candidates
    cand_conf  = min(1.0, total_candidates / 30)
    words_conf = min(1.0, n_words / 2000)
    confidence = (cand_conf + words_conf) / 2.0

    logger.debug(
        "infer_language_bias: n=%d, sfx_words=%d, pfx_words=%d, "
        "prod_ratio=%.3f, word_ratio=%.3f, combined=%.3f, bias=%s",
        n_words, n_sfx_words, n_pfx_words, prod_ratio, word_ratio, combined, bias,
    )

    return {
        "bias":                 bias,
        "is_suffixing":         is_suffixing,
        "suffix_productivity":  sfx_prod_total,
        "prefix_productivity":  pfx_prod_total,
        "ratio":                combined if combined != float('inf') else 999.0,
        "suffix_candidates":    n_sfx_candidates,
        "prefix_candidates":    n_pfx_candidates,
        "suffix_derived_words": n_sfx_words,
        "prefix_derived_words": n_pfx_words,
        "per_length_ratios":    per_length_ratios,
        "words_analysed":       n_words,
        "confidence":           confidence,
    }

# ============================================================================
# SemanticTokenizer
# ============================================================================

class SemanticTokenizer:
    """
    Semantic Search Tokenizer with Trainable Embeddings.

    Accepts a model object that provides:
      - vocab_size (int)
      - id_to_token(i) -> str
      - has_token(tok) -> bool
      - similarity(tok1, tok2) -> float
      - get_embedding_weights() -> np.ndarray of shape (vocab_size, dim)
    """

    def __init__(
        self,
        model,
        load_directory: str = "",
        unk_token: str = UNK,
        subword_prefix: str = SUBWORD_PREFIX,
        vocab: Dict[str, int] = None,
    ):
        self.model = model
        self.lexicon = SemanticTokenLexicon(model)
        self.vocab_dir = load_directory or STORAGE_DIR
        self.special_tokens = {v: idx for idx, v in enumerate(SPECIAL_TOKENS)}
        vocab = vocab or self.load_vocab(self.vocab_dir)
        self.vocab = vocab if len(vocab) > len(self.special_tokens) else deepcopy(self.special_tokens)
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.special_tokens_inv = {v: k for k, v in self.special_tokens.items()}
        self.unk_token = unk_token
        self.subword_prefix = subword_prefix
        self.unk_token_id = self.vocab[self.unk_token]
        self.pre_tokenizer_re = re.compile(UNICODE_PATTERN)

        # Build vocab from model's token set
        for i in range(model.vocab_size):
            tok = model.id_to_token(i)
            if tok is not None:
                self.vocab[tok] = i
                self.id_to_token[i] = tok

        self.cache = CandidateCache()
        self.inference_mode = False

        # Extract morpheme frequencies
        self.morpheme_freq: Counter = Counter()

        if not self.morpheme_freq:
            # Fallback: lightweight extraction from model tokens only
            self._extract_candidate_substrings_from_tokens()

    # ------------------------------------------------------------------
    # Vocab persistence
    # ------------------------------------------------------------------

    def save_vocab(self, save_directory: str):
        save_json_file(save_directory, VOCAB_FILE_NAME, self.vocab)

    @classmethod
    def load_vocab(cls, load_directory: str):
        return load_json_file(os.path.join(load_directory, VOCAB_FILE_NAME))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------
    # Internal morpheme helpers
    # ------------------------------------------------------------------

    def _extract_candidate_substrings_from_tokens(self):
        """Quick morpheme freq approximation using vocab ids since sorted by frequency."""
        variance = 5
        freq = self.vocab_size * variance

        for tok in sorted(self.vocab.keys(), key=lambda x: self.vocab[x]):
            if tok and tok not in SPECIAL_TOKENS:
                self.morpheme_freq[tok] += freq
                freq -= variance

    @staticmethod
    def _extract_morpheme_frequencies(dictionary: Set[str],
                                      min_len: int = 1,
                                      max_len: int = 10) -> Tuple[Counter, Counter, Counter]:
        """
        Extract substring frequencies from a dictionary of words.
        Returns (morpheme_freq, prefix_freq, suffix_freq).
        """
        morpheme_freq, prefix_freq, suffix_freq = Counter(), Counter(), Counter()

        for word in dictionary:
            wlen = len(word)
            if wlen <= max_len:
                morpheme_freq[word] += 1
            for length in range(min_len, min(wlen, max_len + 1)):
                prefix = word[:length]
                suffix = word[-length:]
                prefix_freq[prefix] += 1
                suffix_freq[suffix] += 1

                for start in range(1, wlen - length):
                    morpheme_freq[word[start:start + length]] += 1

        # Update morpheme_freq with prefix/suffix counts
        morpheme_freq.update(prefix_freq)
        morpheme_freq.update(suffix_freq)

        return morpheme_freq, prefix_freq, suffix_freq

    def run_morpheme_pipeline(
        self,
        lang_code: str,
        dict_words: Set[str],
        min_morpheme_len: int = 1,
        desired_root_len: int = 6,
        min_root_freq: int = 4,
        max_bound_root_len: int=10,
        max_free_root_len: int=15,
        prod_freq_len_multi: int=2,
        script: str = "latin",
        combining_forms: Optional[Set[str]] = None,
        orth_rules: Optional[CompiledRules] = None,
    ) -> Tuple[Set[str], Set[str], Set[str], CounterType, CounterType, List[Tuple[str, float]]]:
        """
        Build the core tokenizer vocabulary by combining all morpheme classes.

        Orchestrates the full morpheme extraction pipeline:

        1.  **extract_productive_roots** — discovers free roots, prefix roots,
            suffix roots, and combining forms from *dict_words*
            via statistical analysis.  Internally calls
            ``extract_latin__greek_roots`` when no *combining_forms* are
            supplied so that neoclassical roots are always represented.

        2.  **Merge** — unions all six morpheme classes (prefixes, suffixes,
            free roots, combining forms, prefix roots, suffix roots) into a
            single ranked vocabulary list.

        3.  **Score** — ranks every token by
            ``calculate_improved_stem_score`` so that the caller (``build_vocab``)
            can fill the vocab budget in productivity order without any
            additional sorting.

        The method is designed to be called **once** per language and its
        results cached; all internal helpers use O(|dict_words|) data
        structures so duplicate work across repeated calls is avoided.

        Parameters
        ----------
        lang_code :
            ISO 639-3 language code (e.g. ``"eng"``).  Forwarded to
            ``extract_productive_roots`` for rule caching.
        dict_words :
            Set of lowercased dictionary words for the target language.
        min_morpheme_len :
            Minimum token length admitted into the scored output list.
        desired_root_len :
            Maximum bound-root length passed to ``extract_productive_roots``.
        min_root_freq :
            Minimum morpheme frequency; forwarded to
            ``extract_productive_roots`` as *min_root_freq*.
        script :
            Writing-system name (``"latin"``, ``"cyrillic"``, ``"greek"``).
            Controls which combining-vowel set is used during root extraction.
        combining_forms :
            Optional pre-computed set of combining forms.  When ``None``,
            ``extract_productive_roots`` discovers them automatically via
            ``extract_latin__greek_roots``.
        orth_rules :
            Optional pre-compiled ``CompiledRules`` object.  When ``None``,
            ``extract_productive_roots`` discovers rules automatically via
            ``discover_orth_rules``.

        Returns
        -------
        Tuple of seven elements:
            prefixes       — productive prefix morphemes (Set[str])
            common_prefixes — subset that appear in *dict_words* (Set[str])
            suffixes       — productive suffix morphemes (Set[str])
            common_suffixes — subset that appear in *dict_words* (Set[str])
            roots          — free roots (Set[str])
            bound_roots    — prefix + suffix bound roots (Set[str])
            scores         — list of (token, score) pairs sorted descending
                             by ``calculate_improved_stem_score``, covering
                             *all* six morpheme classes (List[Tuple[str, float]])

        Notes
        -----
        *   All six sets may overlap: a token can be simultaneously a free
            root AND a combining form.  Deduplication is handled by
            ``build_vocab`` when constructing the final vocab dict.
        *   The ``scores`` list uses ``calculate_improved_stem_score`` from
            ``morpho_rules`` with a combining-form type multiplier (1.5×) for
            neoclassical roots and an affix multiplier (1.05×) for bound roots,
            ensuring productive morphemes outrank rare dictionary fragments.
        *   Performance: on a 500 K-word English dictionary this method runs
            in roughly 30–60 s (dominated by ``extract_latin__greek_roots``).
            Pass pre-computed *combining_forms* and *orth_rules* to skip the
            expensive discovery steps on repeated calls.
        """
        from morpho_rules import calculate_improved_stem_score
        from morpho_rules import load_json_file

        # ── Step 1: resolve affix lists for this lang_code ───────────────
        # Priority:
        #   1. Pre-built JSON files  ({lang_code}_prefixes.json / _suffixes.json)
        #      — shipped with the language pack or saved from a previous run.
        #   2. On-the-fly discovery via morpheme_extractor.run_morpheme_pipeline
        #      — used automatically when no pre-built files exist for this
        #      lang_code, so the tokenizer is truly zero-configuration for any
        #      new language or domain-specific corpus.
        base_dir = (os.path.dirname(os.path.abspath(__file__))
                    if "__file__" in dir() else os.getcwd())
        prefixes_path = os.path.join(base_dir, f"{lang_code}_prefixes.json")
        suffixes_path = os.path.join(base_dir, f"{lang_code}_suffixes.json")

        # Use provided dict_words as canonical set (already lowercased by caller)
        if not isinstance(dict_words, set):
            dict_words = set(dict_words)

        # Fast path: load pre-built affix lists from disk.
        prefixes: Set[str] = set(load_json_file(prefixes_path))
        suffixes: Set[str] = set(load_json_file(suffixes_path))
        if not (prefixes and suffixes):
            # Slow path: discover affixes from dict_words on the fly.
            # morpheme_extractor.run_morpheme_pipeline uses Information-Theory
            # metrics (Boundary Entropy, Variation of Information, sibling
            # competition) to extract productive prefixes and suffixes without
            # any language-specific rules — works for any ISO 639-3 code.
            from morpheme_extractor import run_morpheme_pipeline as _me_pipeline
            (
                prefixes,
                suffixes,
                _me_pfx_freq,
                _me_sfx_freq,
            ) = _me_pipeline(dict_words, min_len=min_morpheme_len, max_len=max(max_bound_root_len, desired_root_len))

            # Persist the discovered affixes so subsequent calls skip discovery.
            if dict_words:  # don't write empty files
                save_json_file(base_dir, f"{lang_code}_prefixes.json", sorted(prefixes))
                save_json_file(base_dir, f"{lang_code}_suffixes.json", sorted(suffixes))

        # ── Step 1b: infer language bias ──────────────────────────────────────
        # Use statistical analysis of dict_words to determine whether this
        # language is primarily suffixing or prefixing, then wire the result
        # into extract_productive_roots so the correct side is tried first.
        bias_result = infer_language_bias(dict_words)
        is_suffixing: bool = bias_result["is_suffixing"]
        logger.info(
            "run_morpheme_pipeline [%s]: language bias=%s (ratio=%.2f, confidence=%.2f)",
            lang_code, bias_result["bias"], bias_result["ratio"], bias_result["confidence"],
        )
        # ── Step 2: discover productive morphemes ────────────────────────
        (
            free_roots,
            prefix_roots,
            suffix_roots,
            combining_forms,
            morpheme_freq,
            prefix_freq,
            suffix_freq,
        ) = extract_productive_roots(
            lang_code=lang_code,
            dict_words=dict_words,
            suffixes=suffixes,
            prefixes=prefixes,
            script=script,
            min_root_freq=min_root_freq,
            desired_root_len=desired_root_len,
            max_free_root_len=max_free_root_len,
            max_bound_root_len=max_bound_root_len,
            prod_freq_len_multi=prod_freq_len_multi,
            is_suffixing=is_suffixing,
            combining_forms=combining_forms,
            orth_rules=orth_rules,
        )
        # ── Step 3: promote neoclassical compound bound roots ────────────
        # Some bound roots discovered by extract_productive_roots are genuine
        # neoclassical combining forms that Pattern A/B/C in
        # extract_latin__greek_roots missed — typically because the root is
        # long (≥ 7 chars) and only appears in a handful of compound words
        # (e.g. "metallurg" appears only in metallurgical/metallurgist/…).
        #
        # Detection: a bound root R is a neoclassical compound constituent when
        # it appears as a non-initial segment in ≥ 2 compound words that ALSO
        # contain a known combining form as their LEFT part.  This is the same
        # "2+ distinct left anchors" criterion used by decompose_compound_forms.
        #
        # Promoted roots are added to `combining_forms` so they receive the
        # 1.5× type multiplier in scoring — correctly ranking them ahead of
        # ordinary bound affixes.
        #
        # Combined complexity: O(|words| × avg_word_len) — ~11× faster than
        # the triple-nested loop on a 500 K English dictionary.
        _long_bound = {r for r in prefix_roots | suffix_roots if len(r) >= 6 and r not in combining_forms}
        if _long_bound and combining_forms:
            _cf_source = combining_forms | prefixes | prefix_roots | suffixes | suffix_roots
            _min_cf, _max_cf = 3, desired_root_len

            # ── Build the two forward prefix tries ───────────────────────
            # Each trie node is a plain dict; terminals carry the stored string.
            # Using a local sentinel avoids any dict-key collision with chars.
            _END = None  # None cannot appear as a character key

            def _build_trie(strings, min_len=0, max_len=999):
                root: Dict = {}
                for s in strings:
                    slen = len(s)
                    if not (min_len <= slen <= max_len):
                        continue
                    node = root
                    for ch in s:
                        node = node.setdefault(ch, {})
                    node[_END] = s
                return root

            # cf_trie: walk word left→right to find all valid cf split points.
            # Only insert cf strings in the [_min_cf, _max_cf] length window
            # because shorter/longer slices are never valid split positions.
            _cf_trie = _build_trie(_cf_source, min_len=_min_cf, max_len=_max_cf)

            # br_trie: walk the right fragment left→right to find a bound root.
            _br_trie = _build_trie(_long_bound)

            # ── Scan dict_words with the two tries ───────────────────────
            _anchor_counts: Dict[str, int] = defaultdict(int)
            for w in dict_words:
                wlen = len(w)
                if wlen < _min_cf + 3:
                    continue

                # Walk cf_trie: one O(min(|w|, _max_cf)) pass finds all valid
                # left-split positions where w[:i+1] is a known combining form.
                _node = _cf_trie
                _cap = min(wlen - 3, _max_cf)
                # Per-word dedup: each bound root is counted at most once per
                # word regardless of how many cf splits expose it, preventing
                # double-counting when e.g. both "electro" and "electrom" are
                # valid left splits for the same compound word.
                _counted_this_word: set = set()
                for _i, _ch in enumerate(w):
                    if _i >= _cap:
                        break
                    if _ch not in _node:
                        break
                    _node = _node[_ch]
                    if _i + 1 >= _min_cf and _END in _node:
                        # Valid cf split found.  Walk br_trie on the right
                        # fragment collecting ALL bound roots that are prefixes
                        # of it (not just the shortest).  This is critical for
                        # roots like "metallurg" whose right fragment also
                        # starts with shorter bound root "metallurgi" — stopping
                        # at the first terminal would under-count the longer root.
                        _r_node = _br_trie
                        for _rch in w[_i + 1:]:
                            if _rch not in _r_node:
                                break
                            _r_node = _r_node[_rch]
                            if _END in _r_node:
                                _br = _r_node[_END]
                                if _br not in _counted_this_word:
                                    _anchor_counts[_br] += 1
                                    _counted_this_word.add(_br)
                        # Continue walking cf_trie — longer valid cf splits
                        # may expose different (non-duplicate) bound roots.

            # Promote roots seen after ≥ 2 distinct combining-form anchors
            _promoted = {br for br, cnt in _anchor_counts.items() if cnt >= 2}
            combining_forms = combining_forms | _promoted

        # ── Step 4: union of all token types → score and rank ───────────
        # Collect all unique tokens with their type membership flags.
        # Each token is scored exactly once; type flags are OR-merged when
        # a token belongs to multiple classes (e.g. free root + combining form).
        # Score every token once using the improved stem scoring formula.
        prod_prefixes = extract_productive_affixes(
            prefixes, prefix_freq, is_suffix=False, dict_words=dict_words, min_len=2, top_n=250
        )
        prod_suffixes = extract_productive_affixes(
            suffixes, suffix_freq, is_suffix=True, dict_words=dict_words, min_len=1, top_n=250
        )
        common_prefixes = {x for x, s in prod_prefixes}
        common_suffixes = {x for x, s in prod_suffixes}
        scored_tokens: List[Tuple[str, float]] = []
        for tok in combining_forms | prefixes | prefix_roots | suffixes | suffix_roots:
            score = calculate_improved_stem_score(
                stem=tok,
                frequency=morpheme_freq.get(tok, 1),
                stem_position_length=0.5,
                is_prefixes=tok in common_prefixes,
                is_suffixes=tok in common_suffixes,
                is_dict_words=tok in dict_words,
                is_combining_forms=tok in combining_forms,
            )
            scored_tokens.append((tok, score))

        # Sort descending by score so build_vocab can greedily fill the budget.
        scored_tokens.sort(key=lambda x: x[1], reverse=True)

        return set(prefixes), set(suffixes), combining_forms, prefix_freq, suffix_freq, scored_tokens

    @log_execution_time
    def build_vocab(
        self,
        corpus_freq: Counter,
        dict_words: Set[str],
        lang_code: str,  # ISO 639-3 code e.g. 'eng' for English
        max_vocab: Optional[int] = None,
        min_morpheme_freq: int = 4,
        max_morpheme_len: int = 10,
        min_morpheme_len: int = 1,
        verbose: bool = True,
    ) -> Dict[str, int]:
        """
        Build a case-sensitive vocab from *corpus_word_freq* (a Counter of
        word→frequency) and *dict_words* (a set of the top 250 k valid
        dictionary words for the target language).

        The algorithm is language-agnostic and works for any ISO 639-3 code:

        1. **Infer language bias** — call :func:`infer_language_bias` on
           *dict_words* to determine whether the language is suffixing or
           prefixing.  This controls the affix-stripping order in step 2.
        2. **Extract morphemes** — scan *dict_words* for prefix / suffix /
           root substrings and count their frequency across the dictionary.
        2. **Rank by productivity** — score each morpheme candidate by
           ``freq × len(morpheme)`` so that frequent, long roots outrank
           rare short fragments.
        3. **Select vocab tokens** — greedily fill the vocab budget
           (default 230 k per language) starting with:
             a. Special tokens
             b. Single characters found in the corpus (case-sensitive)
             c. Top-ranked morpheme candidates (roots > affixes)
             d. High-frequency corpus words not yet covered
        4. **Case-sensitivity** — the vocab stores tokens exactly as they
           appear; "apple", "Apple", and "APPLE" are separate entries.
        5. **Whitespace preservation** — the pre-tokenizer regex already
           captures runs of 2+ whitespace characters as single tokens so
           code indentation is preserved.

        Returns the final vocab dict ``{token: id}``.
        """
        budget = max_vocab or MAX_VOCAB_PER_LANGUAGE
        t0 = time.time()

        # --- 1. Extract morphemes from dict_words ---------------------------
        if verbose:
            print(f"[build_vocab:{lang_code}] Extracting morphemes from {len(dict_words)} dict words …")

        # Classify into prefix / suffix / root for downstream use
        prefixes, suffixes, combining_forms, prefix_freq, suffix_freq, scores = self.run_morpheme_pipeline(
            lang_code, dict_words, min_morpheme_len, max_morpheme_len, min_morpheme_freq
        )
        # --- 2. Build the vocab ---------------------------------------------
        new_vocab: Dict[str, int] = {}
        next_id = 0

        # (a) special tokens
        for tok in SPECIAL_TOKENS:
            next_id = self.add_to_vocab(new_vocab, tok, next_id)

        # (b) all single characters that appear in the corpus (case-sensitive)
        char_counts: CounterType = Counter(c for word in corpus_freq for c in word)
        characters = sorted(char_counts.keys(), key=lambda x: char_counts[x], reverse=True)
        for ch in characters:  # Add subword characters first
            if ch.isalpha() and ch not in new_vocab and next_id < budget:
                next_id = self.add_to_vocab(new_vocab, f"{SUBWORD_PREFIX}{ch}", next_id)

        for ch in characters:  # Add all characters
            if ch not in new_vocab and next_id < budget:
                next_id = self.add_to_vocab(new_vocab, ch, next_id)

        # (c) whitespace tokens (space, tab-as-4-spaces, newline, common indents)
        for ws_tok in [' ', '\n', '    ', '        ', '  ']:
            if ws_tok not in new_vocab and next_id < budget:
                next_id = self.add_to_vocab(new_vocab, ws_tok, next_id)

        # (d) top-ranked morphemes by productivity score
        # scored already sorted in descending by morpheme pipeline
        p_count, s_count = 0, 0
        for morph, _score in scores:
            if next_id >= budget:
                logger.info(f"Final token score: {_score}.")
                break

            if suffix_freq.get(morph, 0) > 1:  # Add subword token
                next_id = self.add_to_vocab(new_vocab, f"{self.subword_prefix}{morph}", next_id)
                s_count += 1

            if prefix_freq.get(morph, 0) > 1:  # Add prefix token
                next_id = self.add_to_vocab(new_vocab, morph, next_id)
                p_count += 1

        logger.info(f"Added {p_count} prefixes and {s_count} subwords to vocab. Initial Vocab Size: {len(new_vocab)}")

        # (e) high-frequency corpus words not yet covered (case-sensitive)
        corpus_profile = _build_corpus_profile(corpus_freq)
        lang_script = corpus_profile.primary_script.lower()
        lang_family = get_language_family(lang_code)
        vowels = corpus_profile.corpus_vowels
        cv_set = get_combining_vowels(lang_script)
        segmenter = MorphSegmenterICA(
            lang_code, lang_family, dict_words, prefixes, suffixes, combining_forms, vowels, cv_set
        )
        logger.info(f"Extracting vocab token from corpus with {len(corpus_freq)} words...")
        start_time = time.perf_counter()
        desired_len = 6
        short_count = 0
        word = ""
        word_registry = {}
        for word, _freq in corpus_freq.most_common():
            if next_id >= budget:
                logger.info(f"Final token frequency: {_freq}.")
                break

            if len(word) < 1: continue  # skip any empty string

            if len(word) < desired_len:
                next_id = self._extract_vocab_tokens(
                    new_vocab, next_id, segmenter, word, word_registry, dict_words, desired_len
                )
                short_count += 1
                continue

            next_id = self._extract_vocab_tokens(
                new_vocab, next_id, segmenter, word, word_registry, dict_words, desired_len
            )

        logger.info(f"Finished extracting vocab token from corpus in {time.perf_counter() - start_time} seconds.")
        logger.info(f"Added {short_count} short words to vocab. Final word added: '{word}'")
        save_json_file(STORAGE_DIR, "word_registry.json", word_registry)

        # --- 3. Install the vocab -------------------------------------------
        self.vocab = new_vocab
        self.id_to_token = {v: k for k, v in new_vocab.items()}

        elapsed = time.time() - t0
        if verbose:
            print(
                f"[build_vocab:{lang_code}] Done in {elapsed:.2f}s — "
                f"vocab size: {len(new_vocab)}, "
                f"prefixes: {len(prefixes)}, "
                f"suffixes: {len(suffixes)}, "
                f"combining_forms: {len(combining_forms)}"
            )

        self.save_vocab(self.vocab_dir)

        return new_vocab

    def _extract_vocab_tokens(self, new_vocab, next_id, segmenter, word, word_registry, dict_words, desired_len=6):
        tokens = []
        start_idx = 0
        for idx, w in enumerate(self.split_word_with_case_correction(word, dict_words)):
            if w[0].isupper():
                split, breakdown = segmenter.segment(w.lower(), max_len=5)
                for i, sub in enumerate(split):
                    if idx > 0 or i > 0:  # Extract token in original word case.
                        tok = f"{self.subword_prefix}{word[start_idx:start_idx + len(sub)]}"
                    else:
                        tok = word[start_idx:start_idx + len(sub)]

                    next_id = self.add_to_vocab(new_vocab, tok, next_id)
                    tokens.append(tok)
                    start_idx += len(sub)
            else:
                if idx == 0 and len(w) < desired_len:  # Skip segmentation of short words
                    next_id = self.add_to_vocab(new_vocab, w, next_id)
                    tokens.append(w)
                    start_idx += len(w)
                else:
                    split, breakdown = segmenter.segment(w)
                    for i, sub in enumerate(split):
                        tok = f"{self.subword_prefix}{sub}" if idx > 0 or i > 0 else sub
                        next_id = self.add_to_vocab(new_vocab, tok, next_id)
                        tokens.append(tok)
                        start_idx += len(sub)

        word_registry[word] = tokens
        return next_id

    def split_word_with_case_correction(self, word, dict_words):
        parts = CASE_SPLIT_PATTERN.findall(word)
        if len(parts) > 1 and parts[-1][0].isupper() and parts[-1][-1].islower() and parts[-2][-1].isupper():
            # Split correction for split like QRcode: ["Q", "Rcode"] -> ["QR", "code"]
            str_lower = parts[-1].lower()
            next_str = str_lower[1:]
            if str_lower not in dict_words and next_str in dict_words:
                parts[-2] = parts[-2] + parts[-1][0]
                parts[-1] = next_str

        return parts

    def add_to_vocab(self, vocab, word, next_id):
        if word not in vocab:
            vocab[word] = next_id
            next_id += 1

        return next_id

    # ------------------------------------------------------------------
    # Vocab mutation
    # ------------------------------------------------------------------

    def add_words_to_vocab(self, words: List[str]):
        if not words:
            return
        new_word_count = 0
        for word in words:
            if word not in self.vocab:
                new_word_count += 1
                idx = len(self.vocab)
                self.vocab[word] = idx
                self.id_to_token[idx] = word
        if new_word_count > 0:
            logger.info(f"Added {new_word_count} new vocab words. Vocab size: {len(self.vocab)}")
            self.save_vocab(self.vocab_dir)

    def _reset_vocab(self):
        self.vocab = {k: v for k, v in self.special_tokens.items()}
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        if os.path.exists(self.vocab_dir):
            shutil.rmtree(self.vocab_dir)

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _tokenize_word(self, word: str, use_cache: bool = True) -> List[str]:
        """
        Tokenize a single word into vocab token strings.

        In **inference mode**, returns the best (highest-scored) cached split.
        In **training mode**, uses slot-based selection probabilities
        (like ``select_candidate_for_training``) to sample from the
        candidate pool, which introduces exploration.

        Always returns ``List[str]`` — a list of token strings.
        """
        if not word:
            return []

        if self.inference_mode:
            # --- Inference: deterministic best split -----------------------
            best = self.cache.get_best(word) if use_cache else None
            if best is not None:
                return list(best.tokens)
            # Cache miss → enumerate, then return best
            self.find_all_valid_splits_dp(word, use_cache=use_cache)
            best = self.cache.get_best(word)
            if best is not None:
                return list(best.tokens)
            # Ultimate fallback: character split
            return [ch if ch in self.vocab else self.unk_token for ch in word]

        else:
            # --- Training: stochastic slot-based selection ------------------
            # Make sure candidates exist
            if not self.cache.has_candidates(word):
                self.find_all_valid_splits_dp(word, use_cache=use_cache)

            selected: Optional[CandidateEntry] = self.cache.select_candidate_for_training(word)
            if selected is not None:
                return list(selected.tokens)

            # Fallback if no candidate was selected
            best = self.cache.get_best(word)
            if best is not None:
                return list(best.tokens)
            return [ch if ch in self.vocab else self.unk_token for ch in word]

    def tokenize(self, text: str) -> List[str]:
        """Converts a string into a sequence of tokens."""
        text = normalize_text_for_nlp(text)
        tokens: List[str] = []
        for token in remove_single_spaces(self.pre_tokenizer_re.findall(text)):
            tokens.extend(self._tokenize_word(token))
        return tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
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
        if not token_ids:
            return ""
        tokens = []
        for tok in token_ids:
            if tok in self.special_tokens_inv:
                continue
            token = self.id_to_token.get(tok, self.unk_token)
            if token.startswith(self.subword_prefix):
                tokens[-1] = f"{tokens[-1]}{token.lstrip(self.subword_prefix)}"
            else:
                tokens.append(token)
        tokens = add_single_spaces(tokens)
        return "".join(tokens)

    # ------------------------------------------------------------------
    # Graph-based DP split enumeration
    # ------------------------------------------------------------------

    def _build_token_graph(self, word: str, max_subword_len: int = 15) -> Dict[int, List[Tuple[int, str]]]:
        n = len(word)
        graph: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        for i in range(n):
            end_limit = min(i + max_subword_len, n)
            for j in range(i + 1, end_limit + 1):
                substr = word[i:j]
                if substr in self.vocab:
                    graph[i].append((j, substr))
        return graph

    def find_all_valid_splits_dp(
        self,
        word: str,
        min_root_len: int=4,
        max_subword_len=15,
        use_cache: bool = True,
    ) -> List[Tuple[str, ...]]:
        """
        Enumerate valid tokenizations via graph-based DP.
        Returns list of token-string tuples sorted by calculate_split_score.
        """
        cached = self.cache.get_all_tokens(word) if use_cache else None
        if cached is not None:
            return cached

        n = len(word)
        if n == 0:
            return []

        graph = self._build_token_graph(word, max_subword_len)
        max_candidates = len(graph)

        dp: List[Optional[List[List[str]]]] = [None] * (n + 1)
        dp[0] = [[]]

        for i in range(n):
            if dp[i] is None:
                continue
            paths = dp[i]
            if len(paths) > max_candidates:
                paths.sort(key=lambda x: len(x))  # keep candidates with longest root
                best_size = len(paths[0])
                best_size = best_size + 1 if best_size < 3 else best_size
                best_count = sum(1 for x in paths if len(x) <= best_size)
                paths = paths[:min(best_count, max_candidates)]
                dp[i] = paths
            for next_pos, token_str in graph.get(i, []):
                if dp[next_pos] is None:
                    dp[next_pos] = []
                for path in paths:
                    dp[next_pos].append(path + [token_str])

        if dp[n] is None:
            # TODO - should be using subword characters. e.g. "#s"
            valid_chars = tuple(c if c in self.vocab else UNK for c in word)
            entries = [CandidateEntry(valid_chars, 0.0)]
            self.cache.put_candidates(word, entries)
            return [valid_chars]

        max_splits = math.ceil(len(word) / min_root_len) + 1
        candidates = [s for s in dp[n] if len(s) <= max_splits]  # Select top candidate splits

        def default_score(token_tuple: Tuple[str, ...]) -> float:
            ids = tuple(self.vocab.get(t, 0) for t in token_tuple)
            return calculate_split_score(ids, self.id_to_token, self.morpheme_freq)

        candidates.sort(key=default_score, reverse=True)

        if len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]

        # Create CandidateEntry objects with blended scores
        entries: List[CandidateEntry] = []
        max_ds = max((default_score(c) for c in candidates), default=1.0)
        if max_ds <= 0:
            max_ds = 1.0

        for tokens in candidates:
            ds = default_score(tokens)
            ss = self.lexicon.score_tokenization(list(tokens))
            norm_ds = ds / max_ds
            blended = ss * (0.3 + 0.7 * norm_ds)
            entries.append(CandidateEntry(tokens, blended))

        entries.sort(key=lambda e: e.score, reverse=True)
        self.cache.put_candidates(word, entries)
        return [e.tokens for e in entries]

    def score_tokenization(self, tokens) -> float:
        if isinstance(tokens, tuple):
            return self.lexicon.score_tokenization(list(tokens))
        return self.lexicon.score_tokenization(tokens)

    def find_best_split(
        self,
        word: str,
        return_top_k: int = 5,
        use_cache: bool = True,
    ) -> Tuple[List[str], float, List[Tuple[List[str], float]]]:
        if not word:
            return [""], 0.0, [([""], 0.0)]

        # FAST PATH: inference
        if self.inference_mode and use_cache:
            best = self.cache.get_best(word)
            if best is not None:
                strs = list(best.tokens)
                return strs, best.score, [(strs, best.score)]

        self.find_all_valid_splits_dp(word, use_cache=use_cache)

        entries = self.cache.get_entries(word)
        if entries and len(entries) > 0:
            best_entry = entries[0]
            best_strs = list(best_entry.tokens)
            best_score = best_entry.score
            top_k = [(list(e.tokens), e.score) for e in entries[:return_top_k]]
            return best_strs, best_score, top_k

        char_split = list(word)
        return char_split, 0.0, [(char_split, 0.0)]

    # ------------------------------------------------------------------
    # Training cycle
    # ------------------------------------------------------------------

    def training_cycle(
        self,
        words: List[str],
        epochs: int = 1,
        learning_rate: float = 0.1,
        verbose: bool = True,
    ):
        for word in words:
            if word:
                self.find_all_valid_splits_dp(word)

        for epoch in range(epochs):
            self.lexicon.refresh_from_model()
            total_score = 0.0
            total_updates = 0

            for word in words:
                if not word:
                    continue
                entries = self.cache.get_entries(word)
                if not entries:
                    continue

                selected = self.cache.select_candidate_for_training(word)
                if selected is None:
                    continue

                new_score = self.lexicon.score_tokenization(list(selected.tokens))
                selected.update_score(new_score)
                self.lexicon.update_pair_scores(list(selected.tokens), new_score, learning_rate)
                total_updates += 1
                total_score += new_score

                for entry in entries[:3]:
                    if entry is selected or not entry.enabled:
                        continue
                    es = self.lexicon.score_tokenization(list(entry.tokens))
                    entry.update_score(es)
                    self.lexicon.update_pair_scores(list(entry.tokens), es, learning_rate * 0.5)
                    total_updates += 1
                    total_score += es

                self.cache.re_rank(word)

            if verbose:
                avg = total_score / total_updates if total_updates else 0
                stats = self.lexicon.get_stats()
                cstats = self.cache.get_stats()
                print(
                    f"Epoch {epoch + 1}: Avg score: {avg:.4f}, "
                    f"Pairs: {stats['total_pairs_computed']}, "
                    f"Updates: {total_updates}, "
                    f"Cache hit: {cstats['hit_rate']:.1%}"
                )

    def set_inference_mode(self, enabled: bool = True):
        self.inference_mode = enabled

    def get_stats(self) -> Dict:
        """Return aggregated runtime statistics for the tokenizer.

        Returns
        -------
        Dict with keys:

        ``"lexicon"`` : Dict
            Statistics from :class:`SemanticTokenLexicon`: pair affinity
            counts, mean/max/min/std affinity, and hit-rate.
        ``"cache"`` : Dict
            Statistics from :class:`CandidateCache`: cached word count,
            total/enabled/disabled candidates, and cache hit-rate.
        ``"vocab_size"`` : int
            Current vocabulary size (including special tokens).
        """
        return {
            "lexicon":    self.lexicon.get_stats(),
            "cache":      self.cache.get_stats(),
            "vocab_size": self.vocab_size,
        }

# ============================================================================
# Model & embedding primitives
# ============================================================================

class ModelArgs:
    vocab_size, pad_token_id, bos_token_id, eos_token_id = 1000, -1, -1, -1
    word_embed_dim, embed_dim, ffn_dim, n_layer, n_heads, max_seq_len = 8192, 512, 2048, 8, 8, 64
    dropout, weight_decay, clm_lr, base_lr, min_lr = 0.3, 1e-2, 8e-5, 1e-3, 1e-6
    num_sensitivity_levels, num_languages, num_tasks = 3, 7, 12
    embed_group_size, embed_device = 10, None
    device, dtype = None, None
    weights: Dict[str, float] = {}
    db_path: str = "vector_db"
    expert_dim, expert_ffn_dim = 128, 256
    adaptive_softmax_cutoffs: Optional[List[int]] = None

    def __init__(self, **kwargs):
        self.dtype = kwargs.get("dtype", torch.float32)
        self.device = kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.embed_device = self.device
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_kwargs(self):
        kwargs = {k: v for k, v in vars(ModelArgs).items() if not k.startswith('__') and not callable(v)}
        kwargs.update(vars(self))
        return kwargs


class SimpleTransformerXL:
    """
    A lightweight model wrapper that provides:
      - vocab_size (int)
      - id_to_token(i) -> str | None
      - has_token(tok) -> bool
      - similarity(tok1, tok2) -> float
      - get_embedding_weights() -> np.ndarray (V, D)
      - train_model(sentences, tokenize_func, …)

    When torch is available, this wraps real nn.Embedding.
    Otherwise falls back to pure numpy for testing.
    """

    def __init__(self, vocab_size: int, d_model: int = 128, vocab_tokens: Optional[List[str]] = None):
        self._vocab_size = vocab_size
        self.d_model = d_model

        # Build token ↔ id maps
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        if vocab_tokens:
            for i, tok in enumerate(vocab_tokens):
                self.vocab[tok] = i
                self.inv_vocab[i] = tok

        # Embedding storage (numpy for portability)
        self._embeddings = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.1

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def id_to_token(self, idx: int) -> Optional[str]:
        return self.inv_vocab.get(idx)

    def has_token(self, tok: str) -> bool:
        return tok in self.vocab

    def get_embedding_weights(self) -> np.ndarray:
        """Return embedding matrix as plain numpy array."""
        return self._embeddings.copy()

    def similarity(self, tok1: str, tok2: str) -> float:
        """Cosine similarity between two tokens."""
        id1 = self.vocab.get(tok1)
        id2 = self.vocab.get(tok2)
        if id1 is None or id2 is None:
            return 0.0
        e1 = self._embeddings[id1]
        e2 = self._embeddings[id2]
        n1 = np.linalg.norm(e1)
        n2 = np.linalg.norm(e2)
        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0
        return float(np.dot(e1, e2) / (n1 * n2))

    def train_model(self, sentences: List[List[str]], tokenize_func=None, epochs: int = 1, learning_rate: float = 0.1):
        """
        Simple co-occurrence-based embedding training.
        Tokens that appear adjacent in sentences get their embeddings
        nudged closer together.
        """
        for epoch in range(epochs):
            total_updates = 0
            for sent in sentences:
                for i in range(len(sent) - 1):
                    t1, t2 = sent[i], sent[i + 1]
                    id1 = self.vocab.get(t1)
                    id2 = self.vocab.get(t2)
                    if id1 is None or id2 is None:
                        continue
                    # Nudge embeddings towards each other
                    diff = self._embeddings[id2] - self._embeddings[id1]
                    self._embeddings[id1] += learning_rate * diff * 0.01
                    self._embeddings[id2] -= learning_rate * diff * 0.01
                    total_updates += 1

# ============================================================================
# Factory
# ============================================================================

def create_tokenizer(
    vocab_tokens: Optional[List[str]] = None,
    embedding_dim: int = 128,
    verbose: bool = True,
) -> Tuple[SemanticTokenizer, SimpleTransformerXL]:
    """Create a SemanticTokenizer with a SimpleTransformerXL model."""
    if vocab_tokens is None:
        chars = list("abcdefghijklmnopqrstuvwxyz0123456789?")
        morphemes = [
            # Prefixes
            "un", "re", "in", "im", "ir", "il", "dis", "en", "em", "non",
            "over", "mis", "sub", "pre", "inter", "fore", "de", "trans",
            "super", "semi", "anti", "mid", "under", "out", "extra", "counter",
            "auto", "bio", "geo", "micro", "nano", "psych", "socio", "thermo",
            "ultra", "multi", "poly",
            # Suffixes
            "ed", "ing", "er", "est", "ly", "ness", "ment", "tion", "sion",
            "ity", "ous", "ive", "ful", "less", "able", "ible", "ize", "ise",
            "al", "ial", "ical", "ary", "ory", "ery", "ry",
            "ance", "ence", "ant", "ent", "ism", "ist",
            # Common roots
            "sky", "scraper", "scrap", "tower", "build", "ground", "house",
            "light", "water", "fire", "rain", "wind", "moon", "sun", "star",
            "back", "fore", "under", "over", "after", "every", "some", "any",
            "day", "night", "time", "line", "side", "way", "man", "men",
            "account", "stand", "hold", "come", "go", "take", "make", "give",
            "know", "think", "find", "tell", "work", "play", "run", "walk",
            "high", "low", "long", "short", "big", "small", "good", "bad",
            "new", "old", "first", "last", "great", "little",
            # Compound parts
            "dog", "cat", "fish", "bird", "book", "note", "key", "board",
            "door", "head", "hand", "foot", "eye", "ear", "ring", "cap",
            "guard", "shield", "brush", "melon", "apple", "berry", "cake",
            "coat", "sand", "ice", "berg", "mate", "quick", "slow",
            "pan", "pine", "green", "red", "blue", "black", "white",
            # Full common words
            "the", "and", "that", "with", "have", "this", "from", "they",
            "which", "words", "language", "computer", "system", "information",
            "together", "between", "before", "better", "everything", "someone",
            "somewhere", "successful", "different", "important", "community",
            "government", "development", "education", "running", "walking",
            "learning", "thinking", "playing", "design", "report", "energy",
            "health", "social", "public", "family", "school", "student",
            "power", "level", "point", "area", "place", "group", "number",
            "interest", "value", "order", "action", "change", "effect",
            "reason", "course", "form", "plan", "book", "city", "name",
            "road", "body", "side", "door", "head", "mind", "kind", "part",
            "word", "line", "home", "case", "fact",
            # BPE struggle word parts
            "ther", "apist", "pen", "insula", "mis", "tle", "toe",
            "inf", "amous", "interpret", "not", "withstanding",
            "ordinary", "iosyncr", "asy", "rupt", "ible",
            "abo", "lish", "amen", "anta", "gon", "ass", "inate",
            "bar", "ometer", "ast", "rophe", "circ", "ums", "nce",
            "fi", "cial", "art", "ificial", "represent", "ative",
            "lege", "nda", "car", "pet", "gran", "ate",
            "stra", "wb", "erry", "amb", "urger",
            "cakes", "neath", "noon", "where", "conscious",
            "productive", "connected", "aut", "obi", "ography",
            "iod", "iversity", "eco", "op", "olitics", "organ",
            "otechnology", "otherapy", "iop", "olitical", "mod",
            "ynamic", "raviolet",
            "bel", "ieve", "yond", "set", "ow", "bet", "ray",
            "tw", "ixt", "itched", "wi", "lder", "gan",
            "mans", "laughter", "elly",
            "kn", "ee", "skirts", "raper",
            "ooth", "ysc", "sk", "sc", "ra", "per",
            # Longer fragments
            "background", "foreground", "underground", "afternoon",
            "everywhere", "anybody", "infrastructure", "rastructure",
            "disciplinary", "ciplinary", "standing", "ability",
            "ynthesis", "photos", "dis",
            "photograph", "synthesis", "understand", "accountable",
            "discipline", "photo", "count", "able",
            "dream", "jump", "talk", "weak", "sad",
            "teach", "acher", "fort", "ef", "ide",
            "hap", "pi", "wal", "ked",
            "small", "result", "example", "question", "history",
            "future", "science", "research", "culture", "control",
            "program", "market", "policy", "process", "problem",
            "success",
            # Training sentence tokens
            "towers", "tall", "built", "down", "town",
            "scrapers", "modern", "define", "reached", "above",
            "workers", "floor", "glass", "reflected", "starts",
            "strong", "found", "ation", "ations", "symbols", "urban",
            "develop", "stood", "against", "whole", "see", "world",
            "tallest", "views", "amaz", "designed", "famous",
            "architect", "being", "year", "clear", "also", "called",
            "rise", "building", "need", "deep", "cast", "shadow",
            "street", "engineers", "hundred",
            "showed", "parking", "common", "beneath",
            "tour", "visiting", "vendor", "near",
            "rushed", "during", "alarm", "sketched",
            "illuminated", "needed", "clouds", "gathered", "around",
            "funds", "were", "office", "covered", "safety",
            "engineering", "requires", "study", "observation", "deck",
            "formed", "complex",
            # skysc + raper fragments (must be in vocab for test)
            "skysc",
        ]
        vocab_tokens = sorted(set(chars + morphemes))

    if verbose:
        print(f"Building model with {len(vocab_tokens)} vocab tokens...")

    model = SimpleTransformerXL(
        vocab_size=len(vocab_tokens),
        d_model=embedding_dim,
        vocab_tokens=vocab_tokens,
    )

    tok = SemanticTokenizer(model=model)

    return tok, model

# ============================================================================
# BENCHMARKING
# ============================================================================
class TokenizerBenchmark:
    def __init__(self):
        self.results = {}

    def benchmark_semantic(self, tokenizer: SemanticTokenizer, words: List[str]):
        r = {"name": "Semantic Tokenizer V8", "tokens_per_word": [], "inference_time": [], "compression_ratio": []}
        tokenizer.training_cycle(words, epochs=1, learning_rate=0.15, verbose=True)
        tokenizer.set_inference_mode(True)
        for w in words:
            tokenizer.find_best_split(w)
        for w in words:
            t0 = time.perf_counter()
            best, sc, _ = tokenizer.find_best_split(w)
            dt = time.perf_counter() - t0
            r["tokens_per_word"].append(len(best))
            r["inference_time"].append(dt * 1000)
            r["compression_ratio"].append(len(w) / max(1, len(best)))
        self.results["semantic"] = r
        return r

    def print_comparison(self):
        print("\n" + "=" * 80)
        print("TOKENIZER BENCHMARK")
        print("=" * 80)
        for _, r in self.results.items():
            print(f"\n{r['name']}:")
            print(f"  Average tokens per word: {np.mean(r['tokens_per_word']):.2f}")
            print(f"  Average inference time:  {np.mean(r['inference_time']):.4f} ms")
            print(f"  Average compression:     {np.mean(r['compression_ratio']):.2f}x")

# ============================================================================
# DEMO
# ============================================================================
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
    c_size = len(corpus)
    v_size = len(vocab_set) if vocab_set else "N/A"
    print(f"Running full tokenizer benchmark using corpus with {c_size} words and vocab with {v_size} tokens...")

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
    # Try importing transformers for the benchmark section
    try:
        from transformers import AutoTokenizer
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        TRANSFORMERS_AVAILABLE = False

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
        logger.exception(f"Error loading HF tokenizer: {e}")


def run_demo():
    print("=" * 80)
    print("SEMANTIC TOKENIZER V8 - SEMANTIC SEARCH")
    print("=" * 80)
    lang_code = "eng"

    tok, model = create_tokenizer()

    # Demo build vocab
    corpus_freq = Counter(load_json_file(os.path.join(STORAGE_DIR, f"{lang_code}_{WORD_FREQ_FILE_NAME}")))
    dict_words = set(load_json_file(os.path.join(STORAGE_DIR, f"{lang_code}_{DICT_WORDS}")))
    new_vocab = tok.build_vocab(corpus_freq, dict_words, lang_code=lang_code)
    print(f"Created new vocab with size: {len(new_vocab)}")
    save_json_file(STORAGE_DIR, f"{lang_code}_{VOCAB_FILE_NAME}", new_vocab)

    # Full Benchmark Comparison
    tok.inference_mode = True
    corpus = Counter(dict(corpus_freq.most_common(500_000)))
    benchmark_comparison(corpus, tok.tokenize, set(new_vocab.keys()))

    test_words = ["understanding", "accountability", "skyscraper",
                  "background", "afternoon", "everything"]

    print("\n" + "=" * 80)
    print("BEFORE TRAINING")
    print("=" * 80)
    before = {}
    for word in test_words:
        best, score, top_k = tok.find_best_split(word, return_top_k=3)
        before[word] = (best, score, len(best))
        print(f"\n  '{word}'")
        print(f"    Best: {best} ({len(best)} tokens, score: {score:.4f})")
        for s, sc in top_k[:3]:
            print(f"      {len(s)} tokens: {s} -> {sc:.4f}")

    print("\n" + "=" * 80)
    print("TRAINING MODEL ON SENTENCES")
    print("=" * 80)
    sentences = get_skyscraper_training_sentences()
    model.train_model(sentences, epochs=10)
    print(f"  Trained on {len(sentences)} sentences, {model.vocab_size} tokens in vocab")

    print("\n" + "=" * 80)
    print("TRAINING TOKENIZER (3 epochs)")
    print("=" * 80)
    tok.training_cycle(test_words, epochs=3, learning_rate=0.15)

    print("\n" + "=" * 80)
    print("AFTER TRAINING")
    print("=" * 80)
    for word in test_words:
        if word not in before:
            continue
        best, score, top_k = tok.find_best_split(word, return_top_k=3)
        bsplit, bscore, btokens = before[word]
        status = "\u2713" if score >= bscore * 0.99 else "\u2717"
        print(f"\n  '{word}'")
        print(f"    Before: {bsplit} ({btokens} tokens, score: {bscore:.4f})")
        print(f"    After:  {best} ({len(best)} tokens, score: {score:.4f}) {status}")
        print(f"  Top 3 candidates:")
        for split, cand_score in top_k[:3]:
            tag = ""
            if split == ['sk', 'ysc', 'raper']:
                tag = " <-- BAD"
            print(f"    {len(split)} tokens: {split} -> {cand_score:.4f}{tag}")

    # Pair affinities
    print("\n" + "=" * 80)
    print("PAIR AFFINITIES (after training)")
    print("=" * 80)
    for t1, t2 in [("sky", "scraper"), ("sky", "sc"), ("sk", "ysc"), ("ysc", "raper"),
                    ("under", "standing"), ("account", "ability"), ("back", "ground")]:
        if model.has_token(t1) and model.has_token(t2):
            aff = tok.lexicon.get_pair_affinity(t1, t2)
            print(f"  ('{t1}', '{t2}'): {aff:.4f}")

    print("\n" + "=" * 80)
    print("BENCHMARKING")
    print("=" * 80)
    bench = TokenizerBenchmark()
    tok2, _ = create_tokenizer(verbose=False)
    bench.benchmark_semantic(tok2, test_words)
    bench.print_comparison()


if __name__ == "__main__":
    run_demo()