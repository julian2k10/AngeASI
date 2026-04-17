import gc
import os
import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple, Optional, NamedTuple, Any, Callable
from collections import OrderedDict
from enum import Enum
import logging
import functools
import itertools
import random
from ddgs import DDGS
import regex as re
import numpy as np
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --- Configuration ---
torch.set_float32_matmul_precision('high')
SEED = 0  # For reproducibility
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# --- Logging Configuration ---
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
date_fmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(format=LOG_FORMAT, datefmt=date_fmt, level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("HierarchicalMoE-Transformer")

REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
REGISTRY_FILE = "skills_registry.json"


# ============================================================================
# LAYER 3: TASKCLASS ENUM (5-8 Fundamental Behavioral Categories)
# ============================================================================

class TaskClass(Enum):
    """
    Core task categories with distinct computational behavior.
    These map to fundamental differences in:
    - Padding direction (prepend vs append)
    - Reset token semantics (BOS vs EOS)
    - Loss function (language modeling vs classification)
    - Output format (sequence vs classification logits)
    """
    GENERATIVE = "generative"  # CLM-like (left-to-right)
    REVERSE_GENERATIVE = "reverse_gen"  # RCLM-like (right-to-left)
    DISCRIMINATIVE = "discriminative"  # Classification
    SEQUENCE_LABELING = "seq_label"  # Token-level classification
    UNCLASSIFIED = "unclassified"  # Unknown/learning

    def is_generative(self) -> bool:
        """Returns True for any generative task."""
        return self in (TaskClass.GENERATIVE, TaskClass.REVERSE_GENERATIVE)

    def is_reverse(self) -> bool:
        """Returns True for right-to-left tasks."""
        return self == TaskClass.REVERSE_GENERATIVE

    def is_discriminative(self) -> bool:
        """Returns True for classification/discriminative tasks."""
        return self in (TaskClass.DISCRIMINATIVE, TaskClass.SEQUENCE_LABELING)

    @property
    def ordinal(self) -> int:
        """Stable integer ordinal (0-4) suitable for embedding lookup."""
        return list(TaskClass).index(self)

    @staticmethod
    def from_ordinal(idx: int) -> 'TaskClass':
        """Reverse of ordinal — used when decoding task_class_ids tensors."""
        return list(TaskClass)[idx]

    @staticmethod
    def count() -> int:
        """Number of distinct TaskClass values — size of the embedding table."""
        return len(TaskClass)


# ── TaskClass-keyed loss-scale weights (replaces Task-name-keyed SCALE_FACTORS) ──
# Generative tasks are harder → higher weight to prevent loss starvation.
TASKCLASS_SCALE_FACTORS: Dict[str, float] = {
    TaskClass.REVERSE_GENERATIVE.value: 2.0,
    TaskClass.GENERATIVE.value: 1.5,
    TaskClass.SEQUENCE_LABELING.value: 1.0,
    TaskClass.DISCRIMINATIVE.value: 1.2,
    TaskClass.UNCLASSIFIED.value: 1.0,
}


# --- Backward Compatibility: Keep Existing Task Enum ---
class Task(Enum):
    """Legacy enum—kept for backward compatibility with existing code."""
    RCLM = 0
    CLM = 1
    SUMMARIZATION_ABSTRACTIVE = 2
    SUMMARIZATION_EXTRACTIVE = 3
    QUESTION_ANSWERING_GENERATIVE = 4
    QUESTION_ANSWERING_EXTRACTIVE = 5
    SENSITIVITY_CLASSIFICATION = 6
    LANGUAGE_IDENTIFICATION = 7
    TASK_IDENTIFICATION = 8
    CODE_GENERATION = 9
    TRANSLATION = 10
    UNCLASSIFIED_SKILL = 11
    FIM = 12  # Fill-in-the-Middle: decoder-only, unified bidirectional objective

    @property
    def task_class(self) -> TaskClass:
        """Map legacy Task to modern TaskClass."""
        mapping = {
            Task.RCLM: TaskClass.REVERSE_GENERATIVE,
            Task.CLM: TaskClass.GENERATIVE,
            Task.SUMMARIZATION_ABSTRACTIVE: TaskClass.GENERATIVE,
            Task.SUMMARIZATION_EXTRACTIVE: TaskClass.DISCRIMINATIVE,
            Task.QUESTION_ANSWERING_GENERATIVE: TaskClass.GENERATIVE,
            Task.QUESTION_ANSWERING_EXTRACTIVE: TaskClass.DISCRIMINATIVE,
            Task.SENSITIVITY_CLASSIFICATION: TaskClass.DISCRIMINATIVE,
            Task.LANGUAGE_IDENTIFICATION: TaskClass.SEQUENCE_LABELING,
            Task.TASK_IDENTIFICATION: TaskClass.DISCRIMINATIVE,
            Task.CODE_GENERATION: TaskClass.GENERATIVE,
            Task.TRANSLATION: TaskClass.GENERATIVE,
            Task.FIM: TaskClass.GENERATIVE,
            Task.UNCLASSIFIED_SKILL: TaskClass.UNCLASSIFIED,
        }
        return mapping.get(self, TaskClass.UNCLASSIFIED)


# ============================================================================
# LAYER 2: SKILL REGISTRY (Dynamic JSON-Backed Registry)
# ============================================================================

@dataclass
class Skill:
    """Metadata for a learnable skill."""

    name: str
    skill_id: str
    task_class: TaskClass
    description: str
    languages: Optional[List[str]] = None
    confidence: float = 0.5
    difficulty: str = "medium"
    domain: str = "general"
    tags: List[str] = field(default_factory=list)
    parent_skills: List[str] = field(default_factory=list)
    example_prompts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize skill to dictionary."""
        data = asdict(self)
        data['task_class'] = self.task_class.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'Skill':
        """Deserialize skill from dictionary."""
        data = data.copy()
        if isinstance(data['task_class'], str):
            # Support both value-based lookup ("generative") and
            # name-based lookup ("GENERATIVE") for backward compatibility.
            raw = data['task_class']
            try:
                data['task_class'] = TaskClass(raw)  # by value e.g. "generative"
            except ValueError:
                data['task_class'] = TaskClass[raw.upper()]  # by name  e.g. "GENERATIVE"
        return cls(**data)

    def __hash__(self):
        return hash(self.skill_id)

    def __eq__(self, other):
        if isinstance(other, Skill):
            return self.skill_id == other.skill_id
        return False


class ISO639_3:
    """Utility class for ISO 639-3 language code operations."""

    @staticmethod
    def is_valid(code: str) -> bool:
        """Check if a code is a valid ISO 639-3 code (3 lowercase letters)."""
        return isinstance(code, str) and len(code) == 3 and code.islower() and code.isalpha()

    @staticmethod
    def get_language_name(iso639_3_code: str) -> Optional[str]:
        """Get the English name of a language from its ISO 639-3 code."""
        language_names = {
            "eng": "English", "deu": "German", "fra": "French", "spa": "Spanish",
            "ita": "Italian", "por": "Portuguese", "rus": "Russian", "hin": "Hindi",
            "cmn": "Mandarin Chinese", "jpn": "Japanese", "kor": "Korean", "ara": "Arabic",
            "nld": "Dutch", "swe": "Swedish", "pol": "Polish", "zho": "Chinese",
        }
        return language_names.get(iso639_3_code)

    @staticmethod
    def get_family(iso639_3_code: str) -> Optional[str]:
        """Get the language family for an ISO 639-3 code."""
        return ISO639_3_LANGUAGE_FAMILIES.get(iso639_3_code)


class SkillRegistry:
    """Dynamic registry of learnable skills (Layer 2)."""

    def __init__(self, registry_path: Optional[str] = None):
        self.skills: Dict[str, Skill] = {}
        self.skill_aliases: Dict[str, str] = {}
        self.load_from_file(registry_path)

    def register_skill(self, skill: Skill) -> None:
        """Register a new skill."""
        if skill.skill_id in self.skills:
            logger.warning(f"Overwriting existing skill: {skill.skill_id}")
        self.skills[skill.skill_id] = skill
        logger.debug(f"Registered skill: {skill.name} ({skill.skill_id})")

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Retrieve skill by ID (resolves aliases)."""
        resolved_id = self.skill_aliases.get(skill_id, skill_id)
        return self.skills.get(resolved_id)

    def get_skills_by_task_class(self, task_class: TaskClass) -> List[Skill]:
        """Get all skills of a specific task class."""
        return [s for s in self.skills.values() if s.task_class == task_class]

    def get_skills_by_language(self, language: str) -> List[Skill]:
        """Get all skills applicable to a language (using ISO 639-3 code)."""
        # Validate ISO 639-3 code
        if not ISO639_3.is_valid(language):
            logger.warning(f"Invalid ISO 639-3 code: {language}")
            return []

        return [s for s in self.skills.values()
                if s.languages is None or language in s.languages]

    def get_skills_by_language_family(self, family: str) -> List[Skill]:
        """Get all skills for languages in a specific family."""
        # Get all ISO 639-3 codes for this family
        family_codes = [
            code for code, fam in ISO639_3_LANGUAGE_FAMILIES.items()
            if fam == family
        ]

        # Get skills using these codes
        skills = []
        for code in family_codes:
            skills.extend(self.get_skills_by_language(code))

        return list(set(skills))  # Remove duplicates

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        all_skills = list(self.skills.values())
        return {
            "total_skills": len(all_skills),
            "by_task_class": {
                tc.value: len(self.get_skills_by_task_class(tc))
                for tc in TaskClass
            },
            "avg_confidence": sum(s.confidence for s in all_skills) / len(all_skills) if all_skills else 0,
        }

    def save_to_file(self, path: str) -> None:
        """Persist registry to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {skill_id: skill.to_dict()
                for skill_id, skill in self.skills.items()}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.skills)} skills to {path}")

    def load_from_file(self, path: str) -> None:
        """Load registry from JSON.

        Handles two on-disk layouts:
        - **Nested** (skills_registry.json production format):
          ``{category: {skill_id: skill_dict, ...}, ...}``
        - **Flat** (written by ``save_to_file``):
          ``{skill_id: skill_dict, ...}``
        """
        if path is None or not os.path.exists(path):
            if path is not None:
                logger.warning(f"Registry file not found: {path}")
            return

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for key, value in data.items():
            if key.startswith('_'):  # skip metadata entries
                continue
            if isinstance(value, dict) and 'skill_id' in value:
                # --- Flat format: key is skill_id, value is skill_dict ---
                try:
                    skill = Skill.from_dict(value)
                    self.register_skill(skill)
                except Exception as e:
                    logger.error(f"Failed to load skill '{key}': {e}")
            elif isinstance(value, dict):
                # --- Nested format: key is category, value is {skill_id: skill_dict} ---
                for skill_id, skill_data in value.items():
                    if skill_id.startswith('_'):
                        continue
                    try:
                        skill = Skill.from_dict(skill_data)
                        self.register_skill(skill)
                    except Exception as e:
                        logger.error(f"Failed to load skill '{skill_id}' in category '{key}': {e}")

        logger.info(f"Loaded {len(self.skills)} skills from {path}")


# ============================================================================
# EXISTING CODE WITH TASKCLASS INTEGRATION
# ============================================================================

# ISO 639-3 Language Mapping
LANGUAGE_MAPPING = {
    "Indo-European": {
        "eng": "English",  # ISO 639-3: eng
        "deu": "German",  # ISO 639-3: deu
        "fra": "French",  # ISO 639-3: fra
        "spa": "Spanish",  # ISO 639-3: spa
        "hin": "Hindi",  # ISO 639-3: hin
        "rus": "Russian",  # ISO 639-3: rus
        "por": "Portuguese",  # ISO 639-3: por
        "ita": "Italian",  # ISO 639-3: ita
        "nld": "Dutch",  # ISO 639-3: nld
        "swe": "Swedish",  # ISO 639-3: swe
        "pol": "Polish",  # ISO 639-3: pol
    }
}
LANGUAGE_FAMILIES = OrderedDict({
    "Indo-European": list(LANGUAGE_MAPPING["Indo-European"].keys()),
})

# ISO 639-3 Language Family Mapping (extended)
ISO639_3_LANGUAGE_FAMILIES = {
    # Indo-European
    "eng": "Indo-European", "deu": "Indo-European", "fra": "Indo-European",
    "spa": "Indo-European", "ita": "Indo-European", "por": "Indo-European",
    "rus": "Indo-European", "pol": "Indo-European", "nld": "Indo-European",
    "swe": "Indo-European", "hin": "Indo-European",
    # Sino-Tibetan
    "cmn": "Sino-Tibetan", "zho": "Sino-Tibetan", "jpn": "Japonic",
    # Koreanic
    "kor": "Koreanic",
    # Afro-Asiatic
    "ara": "Afro-Asiatic",
}

# --- Special Tokens ---
BOS_TOKEN = "bos_token"
EOS_TOKEN = "eos_token"
SEP_TOKEN = "sep_token"
PAD_TOKEN = "pad_token"
UNK_TOKEN = "unk_token"
MASK_TOKEN = "mask_token"
BOS = "<|bos|>"
EOS = "<|eos|>"
SEP = "<|sep|>"  # marks the start of the target span (for sequence to sequence generation target)
PAD = "<|pad|>"
UNK = "<|unk|>"
MASK = "<|mask|>"
# FIM (Fill-in-the-Middle) sentinel tokens
PRE  = "<|pre|>"   # marks the start of the prefix context
SUF  = "<|suf|>"   # marks the start of the suffix context
MID  = "<|mid|>"   # marks the start of the middle span (generation target)
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS, SEP, MASK, PRE, SUF, MID]
SEPARATE_WHITE_SPACE = r'\p{White_Space}+|\p{Lu}\p{Ll}+|\p{Alphabetic}+|\p{M}|\p{N}|\p{P}|\p{S}|\p{C}|.'
JOINT_SPACES = r'\p{Zs}\p{Lu}\p{Ll}+|\p{Zs}\p{Lu}+|\p{Zs}\p{Ll}+|\p{White_Space}+|\p{Lu}\p{Ll}+|\p{Alphabetic}+|\p{M}|\p{N}|\p{P}|\p{S}|\p{C}|.'
WORD_PATTERN = r'\p{Zs}\p{Lu}\p{Ll}+|\p{Zs}\p{Lu}+|\p{Zs}\p{Ll}+|\p{White_Space}+|\p{Lu}\p{Ll}+|\p{Alphabetic}+|\p{M}|\p{N}|\p{P}|\p{S}|\p{C}|.'

# ── Seed vocabulary ──────────────────────────────────────────────────────────
# Build the initial vocab by tokenising seed sentences with the *same*
# WORD_PATTERN that DummyTokenizer.encode() uses.  This guarantees that
# space-prefixed tokens like " cat", " jumps", etc. are included, so they
# never fall back to <|unk|> during basic English tokenisation.
# expand_module_vocab() in test helpers follows the same pattern to extend
# this vocab with task-specific text at test-time.
_SEED_SENTENCES = [
    "the cat sat on a mat.",
    "The quick brown fox jumps over the lazy dog.",
    "the cat jumps over the dog",
    "A dog sat on the brown mat",
    "A brown fox chases after the small mouse",
    "The quick dog jumps over the lazy cat",
    "the dog is lazy", "a quick brown cat", "the fox is quick",
]
_seed_words: set = set()
for _s in _SEED_SENTENCES:
    _seed_words.update(re.findall(WORD_PATTERN, _s))
VOCAB: List[str] = SPECIAL_TOKENS + sorted(_seed_words - set(SPECIAL_TOKENS))

TOKEN_TO_ID: Dict[str, int] = {token: i for i, token in enumerate(VOCAB)}
ID_TO_TOKEN: Dict[int, str] = {i: token for token, i in TOKEN_TO_ID.items()}
PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID = TOKEN_TO_ID[PAD], TOKEN_TO_ID[BOS], TOKEN_TO_ID[EOS]
MASK_TOKEN_ID = TOKEN_TO_ID[MASK]
SEP_TOKEN_ID  = TOKEN_TO_ID[SEP]
PRE_TOKEN_ID  = TOKEN_TO_ID[PRE]
SUF_TOKEN_ID  = TOKEN_TO_ID[SUF]
MID_TOKEN_ID  = TOKEN_TO_ID[MID]
VOCAB_SIZE = len(VOCAB)

# --- Task Type Groups ---
GENERATIVE_TASKS = {Task.CLM, Task.TRANSLATION, Task.CODE_GENERATION, Task.SUMMARIZATION_ABSTRACTIVE,
                    Task.QUESTION_ANSWERING_GENERATIVE, Task.FIM}

_LANG_DB_FILE = os.path.join(os.getcwd(), 'data', 'lang_family_data.json')


def _load_lang_db() -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Load lang_family_data.json and build the two lookup structures.

    Returns
    -------
    code_to_family : dict[str, str]
        Maps every ISO 639-3 code to its LINGUISTIC_PRIORITIES family key.
    family_to_codes : dict[str, list[str]]
        Maps every family key to its language codes, ordered by population
        descending (most-spoken first).
    """
    if not os.path.exists(_LANG_DB_FILE):
        logger.warning(
            "_load_lang_db: %s not found — LANG_CODE_TO_FAMILY will be empty. "
            "Run _rebuild_lang_family_data() to generate it.",
            _LANG_DB_FILE,
        )
        return {}, {}

    with open(_LANG_DB_FILE, "r", encoding="utf-8") as fh:
        entries = json.load(fh)

    code_to_family: Dict[str, str] = {}
    family_to_codes: Dict[str, List[str]] = {}

    # Entries are pre-sorted by (family asc, population desc) in the JSON file,
    # so insertion order already gives us the correct population ranking per family.
    for entry in entries:
        code = entry["language_code"]
        family = entry["family"]
        code_to_family[code] = family
        family_to_codes.setdefault(family, []).append(code)

    logger.info(
        "_load_lang_db: loaded %d language codes across %d families from %s",
        len(code_to_family), len(family_to_codes), _LANG_DB_FILE,
    )
    return code_to_family, family_to_codes


# Module-level lookup structures — populated once at import time.
LANG_CODE_TO_FAMILY: Dict[str, str]
_FAMILY_TO_LANG_CODES: Dict[str, List[str]]
LANG_CODE_TO_FAMILY, _FAMILY_TO_LANG_CODES = _load_lang_db()


def get_language_family(lang_code: str) -> str:
    """Return the LINGUISTIC_PRIORITIES family key for an ISO 639-3 code.

    Performs a case-insensitive O(1) lookup in ``LANG_CODE_TO_FAMILY``.

    Falls back to ``"Germanic"`` when the code is not found and emits a
    ``logger.warning`` so gaps in the database are discoverable without
    raising exceptions.

    Parameters
    ----------
    lang_code : str
        A 3-letter ISO 639-3 code, e.g. ``"eng"``, ``"fra"``, ``"jpn"``.
        ISO 639-1 two-letter codes are **not** supported.

    Returns
    -------
    str
        A family name guaranteed to be a key in ``LINGUISTIC_PRIORITIES``,
        e.g. ``"Germanic"``, ``"Italic"``, ``"Japonic"``.

    Examples
    --------
    >>> get_language_family("eng")
    'Germanic'
    >>> get_language_family("jpn")
    'Japonic'
    >>> get_language_family("tam")
    'Dravidian'
    >>> get_language_family("UNKNOWN")
    'Germanic'
    """
    family = LANG_CODE_TO_FAMILY.get(lang_code.lower())
    if family is None:
        logger.warning(
            "get_language_family: unrecognised lang_code %r — "
            "falling back to 'Germanic'. If this code is valid, "
            "regenerate lang_family_data.json via _rebuild_lang_family_data().",
            lang_code,
        )
        return "Germanic"
    return family


def get_family_languages(family: str) -> List[str]:
    """Return all ISO 639-3 codes that belong to *family*, sorted by
    population descending (most-spoken language first).

    Parameters
    ----------
    family : str
        A key from ``LINGUISTIC_PRIORITIES``, e.g. ``"Germanic"``,
        ``"Dravidian"``, ``"Turkic"``.  Case-sensitive.

    Returns
    -------
    list[str]
        ISO 639-3 codes for all languages in the family, ordered from
        highest to lowest speaker population.  Returns an empty list if
        the family is not recognised.

    Examples
    --------
    >>> get_family_languages("Germanic")[:3]
    ['eng', ...]          # English first (~3 B speakers)
    >>> get_family_languages("Japonic")
    ['jpn', ...]
    >>> get_family_languages("Unknown")
    []
    """
    codes = _FAMILY_TO_LANG_CODES.get(family)
    if codes is None:
        logger.warning(
            "get_family_languages: unrecognised family %r. "
            "Valid families are the keys of LINGUISTIC_PRIORITIES.",
            family,
        )
        return []
    return list(codes)


class DummyTokenizer:
    """
    Tokenizer backed by the module-level VOCAB.

    All token-id lookups go through the ``vocab`` and ``id_to_token``
    *instance properties*, which re-read the module globals on every access.
    This means that after a call to ``expand_module_vocab()`` (or any other
    code that rebinds ``TOKEN_TO_ID`` / ``ID_TO_TOKEN`` on the module) every
    existing ``DummyTokenizer`` instance will immediately see the new
    vocabulary — no re-instantiation required.

    Instance attributes ``pad_token_id``, ``bos_token_id``, ``eos_token_id``,
    ``mask_token_id``, and ``vocab_size`` are similarly live properties so
    that any vocab expansion is reflected automatically.
    """

    # ------------------------------------------------------------------
    # Live properties — read the current module globals on every access.
    # This is the canonical fix for the stale-global / @classmethod bug:
    # encode() and decode() now call self.vocab / self.id_to_token instead
    # of the bare names TOKEN_TO_ID / ID_TO_TOKEN which could be stale
    # if the module attribute was rebound after import.
    # ------------------------------------------------------------------

    @staticmethod
    def _mod():
        """Return the ange_moe_asi module object, however it was imported."""
        import sys
        # The module registers itself under its own __name__.  Try the bare
        # name first (direct run / plain import), then the package-qualified
        # name (pytest via 'asi.ange_moe_asi').  This guarantees we always
        # patch the same object that expand_module_vocab targets.
        return (
            sys.modules.get('ange_moe_asi')
            or sys.modules.get('asi.ange_moe_asi')
        )

    @property
    def vocab(self) -> Dict[str, int]:
        """Live token→id mapping; always reflects the current module-level TOKEN_TO_ID."""
        return self._mod().TOKEN_TO_ID

    @property
    def id_to_token(self) -> Dict[int, str]:
        """Live id→token mapping; always reflects the current module-level ID_TO_TOKEN."""
        return self._mod().ID_TO_TOKEN

    @property
    def pad_token_id(self) -> int:
        return self._live('pad_token_id', 'PAD_TOKEN_ID')

    @property
    def bos_token_id(self) -> int:
        return self._live('bos_token_id', 'BOS_TOKEN_ID')

    @property
    def eos_token_id(self) -> int:
        return self._live('eos_token_id', 'EOS_TOKEN_ID')

    @property
    def mask_token_id(self) -> int:
        return self._live('mask_token_id', 'MASK_TOKEN_ID')

    @property
    def sep_token_id(self) -> int:
        return self._live('sep_token_id', 'SEP_TOKEN_ID')

    @property
    def pre_token_id(self) -> int:
        """FIM Prefix sentinel token ID."""
        return self._live('pre_token_id', 'PRE_TOKEN_ID')

    @property
    def suf_token_id(self) -> int:
        """FIM Suffix sentinel token ID."""
        return self._live('suf_token_id', 'SUF_TOKEN_ID')

    @property
    def mid_token_id(self) -> int:
        """FIM Middle sentinel token ID."""
        return self._live('mid_token_id', 'MID_TOKEN_ID')

    @property
    def vocab_size(self) -> int:
        return self._live('vocab_size', 'VOCAB_SIZE')

    def __init__(self, **kwargs):
        # Caller overrides (e.g. DummyTokenizer(vocab_size=1000)) are stored
        # in a private dict so they shadow the live-property defaults without
        # conflicting with the property descriptor on the class.
        # We bypass __setattr__ entirely to avoid triggering any descriptor.
        object.__setattr__(self, '_overrides', dict(kwargs))

    def _live(self, key: str, module_attr: str):
        """Return caller-override if set; otherwise read the live module global."""
        overrides = object.__getattribute__(self, '_overrides')
        if key in overrides:
            return overrides[key]
        return getattr(self._mod(), module_attr)

    # ------------------------------------------------------------------
    # encode / decode — instance methods using self.vocab / self.id_to_token
    # ------------------------------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = False,
               return_tensors: Optional[str] = None):
        """
        Tokenise *text* using WORD_PATTERN and look up ids from ``self.vocab``
        (which is always the current, possibly expanded vocabulary).

        Special tokens (BOS, EOS, PAD, SEP, MASK, PRE, SUF, MID, …) that appear
        verbatim in the text are recognised atomically BEFORE WORD_PATTERN is
        applied, so that ``<|pre|>`` is always a single token ID rather than
        five character pieces.  This is essential for FIM prompts.
        """
        t2i = self.vocab          # live dict — never stale
        unk_id = t2i[UNK]
        bos_id = t2i[BOS]
        eos_id = t2i[EOS]

        # Build an ordered list of (token_string, token_id) for every special
        # token that is currently in the vocabulary, sorted longest-first so that
        # longer matches (e.g. '<|pre|>') take priority over any substrings.
        _specials = sorted(
            ((tok, tid) for tok, tid in t2i.items() if tok.startswith('<|') and tok.endswith('|>')),
            key=lambda x: len(x[0]), reverse=True,
        )
        if _specials:
            # Split text into alternating (special, non-special) pieces and
            # tokenise each non-special piece with WORD_PATTERN.
            import re as _re
            _special_strs = [_re.escape(s) for s, _ in _specials]
            _split_pat = '(' + '|'.join(_special_strs) + ')'
            pieces = _re.split(_split_pat, text)
            token_ids_body: List[int] = []
            for piece in pieces:
                if piece == '':
                    continue
                if piece in t2i:          # it's a special token
                    token_ids_body.append(t2i[piece])
                else:                     # regular text — apply WORD_PATTERN
                    token_ids_body.extend(
                        t2i.get(tok, unk_id) for tok in re.findall(WORD_PATTERN, piece)
                    )
        else:
            token_ids_body = [t2i.get(tok, unk_id) for tok in re.findall(WORD_PATTERN, text)]

        if add_special_tokens:
            token_ids = [bos_id] + token_ids_body + [eos_id]
        else:
            token_ids = token_ids_body

        if return_tensors == 'pt':
            return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)  # [1, seq_len]

        # Round-trip check (emit warning when text cannot be perfectly reconstructed).
        try:
            decoded_text = self.decode(token_ids)
            assert decoded_text == text
        except AssertionError:
            logger.warning(
                f"Cannot perfectly tokenize text: '{text}' \nDecoded text: '{decoded_text}'"
            )

        return token_ids

    def decode(self, token_ids) -> str:
        """
        Convert a list of ids back to a string using ``self.id_to_token``
        (which is always the current, possibly expanded vocabulary).
        Skips BOS, EOS, PAD, and the three FIM sentinel tokens
        (PRE, SUF, MID) so that generated text is clean of structural markers.
        """
        i2t = self.id_to_token    # live dict — never stale
        t2i = self.vocab
        # All structural/control tokens are excluded from decoded output
        special_ids = {t2i.get(tok, -1) for tok in (BOS, EOS, PAD, PRE, SUF, MID)}
        special_ids.discard(-1)
        return "".join(i2t.get(_id, UNK) for _id in token_ids if _id not in special_ids)

    def __call__(self, text: str, **kwargs):
        return self.encode(text, **kwargs)

def pad_tokens(
        token_ids: List[List[int]],
        pad_token_id: int,
        max_len: int,
        task: Optional[Task] = None,
        task_class: Optional[TaskClass] = None
) -> List[List[int]]:
    """
    Pads and truncates a list of token sequences.
    Now supports both legacy Task enum and new TaskClass.
    """
    # Resolve to task_class if needed
    if task_class is None:
        if task is None:
            task_class = TaskClass.GENERATIVE  # default
        else:
            task_class = task.task_class
    padded_ids = []
    for tokens in token_ids:
        truncated = tokens[:max_len]
        num_to_pad = max_len - len(truncated)
        if task_class.is_reverse():  # REVERSE_GENERATIVE
            padded_ids.append(([pad_token_id] * num_to_pad) + truncated)
        else:
            padded_ids.append(truncated + ([pad_token_id] * num_to_pad))
    return padded_ids


def create_stateful_batches(
        token_ids: List[List[int]],
        pad_token_id: int,
        max_len: int,
        batch_size: int,
        task: Optional[Task] = None,
        task_class: Optional[TaskClass] = None
) -> List[torch.Tensor]:
    """
    Flattens tokens, organizes them into parallel streams, and chunks them into stateful batches.
    Now supports both legacy Task enum and new TaskClass.
    """
    # Resolve to task_class if needed
    if task_class is None:
        if task is None:
            task_class = TaskClass.GENERATIVE  # default
        else:
            task_class = task.task_class

    # Step 1: Flatten all sentences into a single stream of tokens.
    all_tokens = [token for sentence in token_ids for token in sentence]
    if not all_tokens:
        logger.error("List Empty. Not enough data to create a full batch.")
        return []

    # Step 2: Trim the sequence so it can be evenly divided into 'batch_size' streams.
    num_streams = batch_size
    tokens_per_stream = math.ceil(len(all_tokens) / num_streams)
    num_batches = math.ceil(tokens_per_stream / max_len)
    total_tokens = num_batches * max_len * batch_size
    tokens_per_stream = total_tokens // num_streams
    tokens_needed = total_tokens - len(all_tokens)

    _pct_pad = 100.0 * max(0, tokens_needed) / max(total_tokens, 1)
    logger.debug(
        "[create_stateful_batches] task_class=%s  raw_tokens=%d  streams=%d  "
        "max_len=%d  total_slots=%d  padding_added=%d (%.1f%%)  num_batches=%d",
        task_class.value, len(all_tokens), num_streams, max_len,
        total_tokens, max(0, tokens_needed), _pct_pad, num_batches,
    )
    if _pct_pad > 50:
        logger.warning(
            "[create_stateful_batches] %.1f%% of the batch is padding — "
            "the dataset may be too small for batch_size=%d.  "
            "Consider reducing batch_size or adding more data.",
            _pct_pad, batch_size,
        )

    # tokens_needed slots are filled with pad_token_id so each stream
    # ends cleanly on a max_len boundary without corrupting content order.
    padding = [pad_token_id] * max(0, tokens_needed)
    if task_class.is_reverse():  # REVERSE_GENERATIVE
        all_tokens = padding + all_tokens
    else:
        all_tokens.extend(padding)

    streams = torch.tensor(all_tokens, dtype=torch.long).view(num_streams, tokens_per_stream).contiguous()
    num_batches = tokens_per_stream // max_len
    batched_data = []
    for i in range(num_batches):
        batch = streams[:, i * max_len: (i + 1) * max_len]
        batched_data.append(batch)

    return batched_data


def _generate_sub_sequence_id_parallel(input_ids: torch.Tensor, pad_token_id: int, bos_token_id: int) -> torch.Tensor:
    """
    Generates unique sub-sequence IDs for parallel streams in a batch.
    Each BOS token starts a new sub-sequence within its stream (row).
    """
    is_pad = (input_ids == pad_token_id)
    is_bos = (input_ids == bos_token_id)

    # cumsum(is_bos) creates segment IDs within each row
    sub_sequence_id_per_row = is_bos.long().cumsum(dim=1)

    # Create a large offset for each row to make IDs globally unique across the batch
    bsz = input_ids.shape[0]
    row_offset = torch.arange(bsz, device=input_ids.device).unsqueeze(1) * (input_ids.shape[1] + 1)

    sub_sequence_id = sub_sequence_id_per_row + row_offset
    return sub_sequence_id.masked_fill(is_pad, 0)

# diagonal
DIAGONAL = 0
GRADIENT = 0

def clm_collate_fn(
        batch: torch.Tensor,
        task: Optional[Task] = None,
        task_class: Optional[TaskClass] = None,
        tokenizer: Optional[Any] = None,
) -> Dict[str, torch.Tensor]:
    """
    Creates shifted labels and attention masks.
    Now supports both legacy Task enum and new TaskClass.
    Defaults to TaskClass.GENERATIVE when neither is provided.

    Args:
        batch:     Input token-id tensor [batch, seq_len] (or [seq_len]).
        task:      Legacy Task enum (resolved to task_class when provided).
        task_class: Modern TaskClass enum; takes precedence over task.
        tokenizer: Optional tokenizer instance.  When supplied, PAD / BOS /
                   EOS token ids are read from ``tokenizer.pad_token_id`` etc.
                   so that a dynamically expanded vocabulary is handled
                   correctly.  Falls back to module-level globals when None.
    """
    global DIAGONAL, GRADIENT
    # Resolve to task_class if needed
    if task_class is None:
        if task is None:
            task_class = TaskClass.GENERATIVE  # default: causal LM
        else:
            task_class = task.task_class

    # Resolve special-token ids — prefer tokenizer, fall back to module globals.
    if tokenizer is not None:
        _pad_id = tokenizer.pad_token_id
        _bos_id = tokenizer.bos_token_id
        _eos_id = tokenizer.eos_token_id
    else:
        _pad_id = PAD_TOKEN_ID
        _bos_id = BOS_TOKEN_ID
        _eos_id = EOS_TOKEN_ID

    if batch.dim() == 1:
        batch = batch.unsqueeze(0)

    input_ids = batch
    device = input_ids.device
    is_pad = (input_ids == _pad_id)
    _, seq_len = input_ids.shape

    reset_token_id = _eos_id if task_class.is_reverse() else _bos_id

    sub_sequence_id = _generate_sub_sequence_id_parallel(input_ids, _pad_id, reset_token_id)
    boundary_mask = (sub_sequence_id.unsqueeze(2) == sub_sequence_id.unsqueeze(1)) & (sub_sequence_id.unsqueeze(2) > 0)
    padding_mask = ~is_pad.unsqueeze(1)

    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100
    labels.masked_fill_(is_pad, -100)

    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    final_mask_bool = causal_mask & boundary_mask & padding_mask
    attention_mask = ~final_mask_bool

    output_dict = {'input_ids': input_ids, 'attention_mask': attention_mask}

    if task_class.is_reverse():
        output_dict['reverse_clm_labels'] = labels
    else:
        output_dict['clm_labels'] = labels

    is_reset = (input_ids == reset_token_id)
    is_continuation_row = (input_ids[:, 0] != reset_token_id) & (input_ids[:, 0] != _pad_id)
    reset_cumsum = is_reset.long().cumsum(dim=1)
    mask_before_first_reset = (reset_cumsum == 0)
    is_continuation = is_continuation_row.unsqueeze(1) & mask_before_first_reset

    # ── Diagnostic: flag batches where every label is masked or the mask blocks
    #    all self-attention — these indicate a data-pipeline bug.
    #
    # Suppressed for:
    #   • Single-token sequences (seq_len == 1) — cached generation steps where
    #     the token has no BOS/EOS context are expected to have all-masked labels.
    #   • Continuation-only batches — packed stream rows with no reset token are
    #     legitimate mid-stream continuations, not data bugs.
    # ──────────────────────────────────────────────────────────────────────────
    # _reset_id: BOS for CLM/FIM, EOS for RCLM — must match clm_collate_fn reset logic.
    _reset_id  = _eos_id if task_class.is_reverse() else _bos_id
    _has_reset = (input_ids == _reset_id).any().item()
    _is_cached_gen_step = (seq_len == 1)
    _skip_diag = _is_cached_gen_step or not _has_reset

    if not _skip_diag and GRADIENT < 20:
        _lbl_key = 'reverse_clm_labels' if task_class.is_reverse() else 'clm_labels'
        _lbl = output_dict[_lbl_key]
        _n_active_labels = (_lbl != -100).sum().item()
        if _n_active_labels == 0:
            GRADIENT += 1
            logger.warning(
                "[clm_collate_fn] ALL labels are -100 for task_class=%s — "
                "no gradient signal in this batch.  "
                "Check that the dataset appended target tokens after the prompt.",
                task_class.value,
            )
    # Check diagonal of attention mask: non-PAD token[i] must always attend to itself.
    # PAD tokens are legitimately blocked from self-attention (sub_sequence_id=0
    # means boundary_mask[b,i,i] is False for pad positions), so we only flag
    # non-padding positions whose diagonal is True.
    # Skip for cached generation steps where single-token batches have no reset context.
    if not _skip_diag and attention_mask.shape[-2] == attention_mask.shape[-1]:
        diag_masked = attention_mask[:, range(seq_len), range(seq_len)]
        # Exclude PAD positions — they are correctly masked everywhere including diagonal.
        non_pad_diag_masked = diag_masked & ~is_pad
        if non_pad_diag_masked.any():
            affected = non_pad_diag_masked.nonzero(as_tuple=False).tolist()
            if DIAGONAL < 5:
                logger.warning(
                    "[clm_collate_fn] %d non-PAD token(s) are masked from attending to THEMSELVES "
                    "(diagonal of attention_mask is True). This is almost always a bug. "
                    "Affected positions: %s",
                    non_pad_diag_masked.sum().item(),
                    affected,
                )
                if len(affected) > 2:
                    DIAGONAL += 1
                else:
                    DIAGONAL += 0.25

    output_dict.update({
        'sub_sequence_id': sub_sequence_id,
        'is_continuation': is_continuation,
        'is_pad': is_pad
    })
    return output_dict


# ============================================================================
# SKILL ROUTER INTEGRATION (Layer 1 Output → Layer 2 Lookup → Layer 3 Config)
# ============================================================================

class SkillExecutor:
    """Execute skills using routing + task class information."""

    def __init__(
            self,
            skill_registry: SkillRegistry,
            model: Optional['SecureEncoderDecoderMoE'] = None,
            tokenizer: Optional[Any] = None,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.registry = skill_registry
        self.model = model
        # Tokenizer resolution priority:
        #   1. Explicitly passed tokenizer
        #   2. The model's own tokenizer (always set since __init__ change)
        #   3. Fallback DummyTokenizer (when no model is provided)
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif model is not None and hasattr(model, 'tokenizer'):
            self.tokenizer = model.tokenizer
        else:
            self.tokenizer = DummyTokenizer()
        self.device = device

    def execute(
            self,
            skill_id: str,
            input_text: str,
            max_length: int = 512,
            batch_size: int = 1,
            return_raw: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a skill on input text.
        Demonstrates the three-layer integration:
        - Layer 1: Routing (conceptual - this is where router output would come from)
        - Layer 2: Skill Registry lookup
        - Layer 3: TaskClass resolution for data pipeline
        """
        # Layer 2: Look up skill in registry
        skill = self.registry.get_skill(skill_id)
        if skill is None:
            logger.warning(f"Skill '{skill_id}' not found; using fallback")
            skill = self.registry.get_skill("unclassified")

        # Layer 3: Resolve TaskClass to configure data pipeline
        task_class = skill.task_class
        logger.info(f"Executing skill '{skill.name}' with task class '{task_class.value}'")

        # Prepare input — encode returns [1, seq_len] when return_tensors='pt'
        input_ids = self.tokenizer.encode(
            input_text,
            add_special_tokens=True,
            return_tensors='pt'
        )
        if input_ids.shape[1] == 0:
            return {'error': 'Empty input'}

        # Convert to flat list-of-lists expected by create_stateful_batches
        token_list: List[List[int]] = input_ids.tolist()  # [[tok0, tok1, ...]]
        actual_len = input_ids.shape[1]
        effective_max_len = min(max_length, max(actual_len, 16))

        # Layer 3: Create batches based on TaskClass
        if task_class.is_generative():
            batches = create_stateful_batches(
                token_list,
                pad_token_id=self.tokenizer.pad_token_id,
                max_len=effective_max_len,
                batch_size=batch_size,
                task_class=task_class,
            )
        else:
            # For discriminative tasks, use simpler batching
            batches = [input_ids]

        # Prepare output
        result = {
            'skill_id': skill_id,
            'skill_name': skill.name,
            'task_class': task_class.value,
            'confidence': skill.confidence,
            'batches_processed': len(batches),
        }

        return result


class AccessLevel(Enum):
    LEVEL_0_PUBLIC = 0
    LEVEL_1_INTERNAL = 1
    LEVEL_2_CONFIDENTIAL = 2


class User(NamedTuple):
    id: str
    access_level: AccessLevel


# --- Helper Function to create token-perfect data ---
def create_token_level_pairs(text, snippet_len, tokenizer):
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(text_ids) <= snippet_len: return text, tokenizer.decode(text_ids)
    start = random.randint(0, len(text_ids) - snippet_len)
    snippet_ids = text_ids[start:start + snippet_len]
    return text, tokenizer.decode(snippet_ids)


# --- Stateful Supervised Data Pipeline ---
class StatefulCollator:
    """A stateful collator that manages context for parallel streams."""

    def __init__(self, user: User, task: Task, device: torch.device,
                 tokenizer: Optional[Any] = None):
        self.user = user
        self.task = task
        self.device = device
        self.stream_lengths: Dict[int, int] = {}
        # Store the tokenizer so all token-id lookups go through it instead of
        # module-level globals, which may be stale if the vocab was expanded
        # after this instance was created.
        self.tokenizer = tokenizer if tokenizer is not None else DummyTokenizer()
        self.reset_token_id = (
            self.tokenizer.eos_token_id if self.task == Task.RCLM
            else self.tokenizer.bos_token_id
        )

    def reset(self):
        """Resets the state of the collator for a new epoch."""
        self.stream_lengths = {}

    def __call__(self, batch: list) -> Dict[str, torch.Tensor]:
        is_dict_input = isinstance(batch[0], dict)
        if is_dict_input:
            # Batch is a list containing a single dictionary from StatefulSupervisedDataset
            batch_data = batch[0]
            input_tensor = batch_data['input_ids'].to(self.device)
        else:
            # Batch is a list containing a single tensor from TextDataset
            batch_data = {}
            input_tensor = batch[0].to(self.device)

        # --- Generate base batch data ---
        bsz, seq_len = input_tensor.shape
        # Pass tokenizer so clm_collate_fn derives special-token ids from it.
        final_batch = clm_collate_fn(input_tensor, self.task, tokenizer=self.tokenizer)

        # Update batch with any pre-computed labels
        for k, v in batch_data.items():
            if k != 'input_ids': final_batch[k] = v.to(self.device)

        # --- Handle stateful lengths for parallel streams ---
        past_lengths_list = [self.stream_lengths.get(i, 0) for i in range(bsz)]
        final_batch['past_lengths'] = torch.tensor(past_lengths_list, device=self.device).unsqueeze(1)

        _pad_id = self.tokenizer.pad_token_id  # live — always current
        for i in range(bsz):
            row_tokens = input_tensor[i]
            is_reset_indices = (row_tokens == self.reset_token_id).nonzero(as_tuple=True)[0]
            if is_reset_indices.numel() > 0:
                # If a reset token exists, the new stream length is the number of
                # non-pad tokens in the final segment (from the last reset token).
                last_reset_idx = is_reset_indices[-1].item()
                final_segment = row_tokens[last_reset_idx:]
                new_len = (final_segment != _pad_id).sum().item()
                self.stream_lengths[i] = new_len
            else:
                # If no reset token, the entire row is a continuation. Add the number of
                # non-pad tokens to the previous length.
                non_pad_count = (row_tokens != _pad_id).sum().item()
                self.stream_lengths[i] = self.stream_lengths.get(i, 0) + non_pad_count

        # --- Combine and finalize batch ---
        # Legacy task_ids kept for backward compatibility; also emit task_class_ids
        # for the new TaskClass-based routing path (smaller embedding, 5 vs 12 values).
        final_batch['task_ids'] = torch.full_like(final_batch['input_ids'], self.task.value)
        final_batch['task_class_ids'] = torch.full_like(
            final_batch['input_ids'], self.task.task_class.ordinal
        )
        final_batch['access_levels'] = torch.full_like(final_batch['input_ids'], self.user.access_level.value)
        return final_batch


class StatefulSupervisedDataset(Dataset):
    def __init__(self, data_pairs: List[Tuple], task: Task, tokenizer: DummyTokenizer, batch_size: int, max_len: int,
                 label_map: Optional[Dict[str, int]] = None):
        self.batches = []
        token_sequences = []
        # Store labels as lists of lists first to maintain sequence boundaries, then flatten later.
        label_sequences = {
            'clm_labels': [], 'access_levels': [], 'src_lang_ids': [],
            'start_positions': [], 'end_positions': [], 'summary_ext_labels': []
        }

        is_generative = task in GENERATIVE_TASKS

        # ── Diagnostic counters (printed once per dataset construction) ────────
        _n_valid_spans   = 0   # QA_EXT / SUM_EXT: spans actually found in text
        _n_missing_spans = 0   # spans searched but not found (label → -100)
        _first_seq_logged = False   # log the first sequence for each task

        for text, label_data in data_pairs:
            # For generative tasks (seq2seq), combine prompt and target.
            # Format: [BOS] prompt target [EOS]
            # Efficient loss masking: prompt tokens are set to -100 so the model
            # only trains on the target sequence (no gradient for the source).
            #
            #   input_ids:  BOS  s0 … sN  t0  t1 … tM  EOS
            #   labels:    -100 -100 … -100  t1  t2 … tM  EOS  -100
            #
            # CLM next-token shift: label[i] predicts input_ids[i+1].
            #   Positions 0 .. prompt_len-2 → -100  (masked source prefix)
            #   Position  prompt_len-1      → t0    (last source token predicts first target)
            #   Positions prompt_len ..     → t1..tM, EOS
            #   Last position              → -100   (EOS has nothing after it)
            if is_generative:
                prompt_ids = tokenizer.encode(text, add_special_tokens=True)
                if prompt_ids and prompt_ids[-1] == tokenizer.eos_token_id:
                    prompt_ids = prompt_ids[:-1]  # Remove EOS from prompt

                target_ids = tokenizer.encode(label_data, add_special_tokens=True)
                if target_ids and target_ids[0] == tokenizer.bos_token_id:
                    target_ids = target_ids[1:]  # Remove BOS from target

                token_ids = prompt_ids + target_ids
                # Loss masking: suppress prompt, train on target only.
                # label[i] = input_ids[i+1] (CLM shift), so:
                #   position prompt_len-1 predicts target_ids[0] (boundary)
                #   positions prompt_len+ predict target_ids[1..] and EOS
                if len(prompt_ids) > 0:
                    labels = ([-100] * (len(prompt_ids) - 1)) + target_ids + [-100]
                else:
                    labels = target_ids + [-100]
                labels = labels[:len(token_ids)]
                while len(labels) < len(token_ids):
                    labels.append(-100)
                label_sequences['clm_labels'].append(labels)

                if not _first_seq_logged:
                    _first_seq_logged = True
                    n_masked  = labels.count(-100)
                    n_active  = len(labels) - n_masked
                    logger.info(
                        "[Dataset:%s] First seq — total_tokens=%d  prompt_tokens=%d  "
                        "target_tokens=%d  active_labels=%d  masked_labels=%d  "
                        "prompt[:5]=%s  target[:5]=%s",
                        task.name, len(token_ids), len(prompt_ids), len(target_ids),
                        n_active, n_masked,
                        tokenizer.decode(prompt_ids[:5]),
                        tokenizer.decode(target_ids[:5]),
                    )
            else:
                token_ids = tokenizer.encode(text, add_special_tokens=True)

                if not _first_seq_logged:
                    _first_seq_logged = True
                    logger.debug(
                        "[Dataset:%s] First seq — total_tokens=%d  text[:60]=%r",
                        task.name, len(token_ids), text[:60],
                    )

            token_sequences.append(token_ids)
            n = len(token_ids)

            # Handle labels for other tasks
            if task == Task.SENSITIVITY_CLASSIFICATION:
                label_sequences['access_levels'].append([label_data.value] * n)
            elif task == Task.LANGUAGE_IDENTIFICATION:
                label_sequences['src_lang_ids'].append([label_map[label_data]] * n)
            elif task == Task.QUESTION_ANSWERING_EXTRACTIVE:
                snippet_ids = tokenizer.encode(label_data, add_special_tokens=False)
                start_idx = -1
                # A simple search for the snippet in the tokenized context
                for i in range(len(token_ids) - len(snippet_ids) + 1):
                    if token_ids[i:i + len(snippet_ids)] == snippet_ids:
                        start_idx = i
                        break
                start_pos = start_idx if start_idx != -1 else -100
                end_pos = (start_idx + len(snippet_ids) - 1) if start_idx != -1 else -100
                # Store span labels only at position 0 of the sequence (the BOS /
                # first token).  Every other position is -100.  This ensures the
                # label survives stream-packing and the per-subsequence sampling in
                # calculate_loss (is_first_in_subsequence) selects exactly one label
                # per sample, giving a valid scalar index into the logit tensor.
                span_labels_start = [-100] * n
                span_labels_end   = [-100] * n
                if n > 0:
                    span_labels_start[0] = start_pos
                    span_labels_end[0]   = end_pos
                label_sequences['start_positions'].append(span_labels_start)
                label_sequences['end_positions'].append(span_labels_end)
                if start_idx != -1:
                    _n_valid_spans += 1
                else:
                    _n_missing_spans += 1
                    logger.debug(
                        "[Dataset:%s] Span NOT FOUND — snippet_ids=%s not in token_ids[:20]=%s",
                        task.name, snippet_ids[:6], token_ids[:20],
                    )
            elif task == Task.SUMMARIZATION_EXTRACTIVE:
                bin_labels = [0] * n
                snippet_ids = tokenizer.encode(label_data, add_special_tokens=False)
                start_idx = -1
                for i in range(len(token_ids) - len(snippet_ids) + 1):
                    if token_ids[i:i + len(snippet_ids)] == snippet_ids:
                        start_idx = i
                        break
                if start_idx != -1:
                    for i in range(len(snippet_ids)):
                        bin_labels[start_idx + i] = 1
                    _n_valid_spans += 1
                else:
                    _n_missing_spans += 1
                label_sequences['summary_ext_labels'].append(bin_labels)

        # ── Post-construction diagnostics ──────────────────────────────────────
        if task in (Task.QUESTION_ANSWERING_EXTRACTIVE, Task.SUMMARIZATION_EXTRACTIVE):
            total_spans = _n_valid_spans + _n_missing_spans
            logger.debug(
                "[Dataset:%s] Span search complete — valid=%d / %d  missing=%d  "
                "(missing spans produce -100 labels and contribute NO gradient)",
                task.name, _n_valid_spans, total_spans, _n_missing_spans,
            )
            if _n_valid_spans == 0:
                logger.warning(
                    "[Dataset:%s] *** ALL SPANS MISSING — loss will never converge. "
                    "Check that snippet text exists verbatim in the encoded context. ***",
                    task.name,
                )

        logger.info(
            "[Dataset:%s] Built %d sequences  batch_size=%d  max_len=%d",
            task.name, len(token_sequences), batch_size, max_len,
        )

        # First, create the token batches. This determines the final, padded size of our stream.
        self.token_batches = create_stateful_batches(token_sequences, tokenizer.pad_token_id, max_len, batch_size, task)
        if not self.token_batches:
            return  # Exit if no data was produced

        # Calculate the total number of elements across all generated token batches.
        total_tokens_in_batches = sum(batch.numel() for batch in self.token_batches)

        # Now, flatten all the label sequences into single streams.
        all_flat_labels = {}
        for key, list_of_lists in label_sequences.items():
            if list_of_lists:
                all_flat_labels[key] = [item for sublist in list_of_lists for item in sublist]

        # Process each label stream to match the token stream's structure and size.
        self.label_batches = {key: [] for key in all_flat_labels}
        for key, stream in all_flat_labels.items():
            padding_needed = total_tokens_in_batches - len(stream)
            if padding_needed < 0:
                # This case should ideally not happen with the corrected logic.
                # Truncate if necessary, though it might indicate a logic error.
                padded_stream = stream[:total_tokens_in_batches]
                logger.warning(f"Label stream for '{key}' was truncated. This may indicate an issue.")
            else:
                padded_stream = stream + [-100] * padding_needed

            num_total_batches = len(self.token_batches)
            tokens_per_stream = total_tokens_in_batches // batch_size

            # Ensure the total size is divisible by batch_size before viewing
            if total_tokens_in_batches % batch_size != 0:
                raise ValueError("Total token count in batches must be divisible by batch_size.")

            label_tensor_streams = torch.tensor(padded_stream, dtype=torch.long).view(batch_size, tokens_per_stream)

            for i in range(num_total_batches):
                self.label_batches[key].append(label_tensor_streams[:, i * max_len: (i + 1) * max_len])

        # Combine token and label batches into the final format.
        for i in range(len(self.token_batches)):
            batch_data = {'input_ids': self.token_batches[i]}
            for key, batches in self.label_batches.items():
                if i < len(batches):
                    batch_data[key] = batches[i]
            self.batches.append(batch_data)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]




def create_fim_sequence(
        token_ids: List[int],
        tokenizer: Any,
        rng: random.Random,
        fim_rate: float = 0.5,
) -> Tuple[List[int], List[int]]:
    """
    Convert a single flat token sequence into Fill-in-the-Middle (FIM) format.

    With probability ``fim_rate`` the sequence is rearranged into PSM order:

        <PRE> [prefix] <SUF> [suffix] <MID> [middle] <EOS>

    The returned labels mask everything before ``<MID>`` with ``-100`` so that
    loss is only computed on the **middle span** — the bridging prediction target.
    The model therefore learns to generate the middle given both boundary contexts.

    With probability ``1 - fim_rate`` the original token order is kept (standard CLM).

    Args:
        token_ids: Raw encoded tokens (should include BOS at index 0 and EOS at end).
        tokenizer: Tokenizer instance providing pre/suf/mid/eos token IDs.
        rng:       Seeded Random instance for reproducibility.
        fim_rate:  Fraction of sequences to convert to FIM format (default 0.5).

    Returns:
        (input_ids, labels) where ``labels[i] = -100`` for non-target positions.
    """
    pre_id = tokenizer.pre_token_id
    suf_id = tokenizer.suf_token_id
    mid_id = tokenizer.mid_token_id
    eos_id = tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id

    # Strip BOS/EOS to work with the raw content tokens
    content = token_ids
    if content and content[0] == bos_id:
        content = content[1:]
    if content and content[-1] == eos_id:
        content = content[:-1]

    # Fall back to standard CLM for very short sequences
    if len(content) < 3 or rng.random() >= fim_rate:
        full = [bos_id] + content + [eos_id]
        # CLM labels: shift by 1 (standard next-token prediction)
        if len(full) > 1:
            labels = full[1:] + [-100]
        else:
            labels = [-100]
        return full, labels

    # Choose two random split points to create prefix / middle / suffix
    # Ensure each part has at least 1 token
    lo = rng.randint(1, len(content) - 1)
    hi = rng.randint(lo + 1, len(content))  # hi <= len(content)
    prefix = content[:lo]
    middle = content[lo:hi]
    suffix = content[hi:]

    # PSM layout: <PRE> prefix <SUF> suffix <MID> middle <EOS>
    # The model sees full bidirectional context (prefix + suffix) and must
    # generate the bridging middle span.
    sentinel_pre  = [pre_id]
    sentinel_suf  = [suf_id]
    sentinel_mid  = [mid_id]
    sentinel_end  = [eos_id]

    input_ids = (sentinel_pre + prefix +
                 sentinel_suf + suffix +
                 sentinel_mid + middle +
                 sentinel_end)

    # Loss masking: only supervise the middle span and the final EOS.
    # Everything up to and including <MID> is masked with -100.
    context_len = len(sentinel_pre) + len(prefix) + len(sentinel_suf) + len(suffix) + len(sentinel_mid)
    target_ids  = middle + sentinel_end   # tokens the model must predict

    # CLM shift: label[i] = input_ids[i+1]
    #   context positions (0..context_len-2) → -100
    #   position context_len-1 (<MID> sentinel) → middle[0]  (boundary signal)
    #   positions context_len .. end-1          → middle[1..], EOS
    #   last position                           → -100
    labels = ([-100] * (context_len - 1)) + target_ids + [-100]
    labels = labels[:len(input_ids)]
    while len(labels) < len(input_ids):
        labels.append(-100)

    return input_ids, labels


class FimDataset(Dataset):
    """
    A drop-in replacement / companion to ``TextDataset`` that applies
    Fill-in-the-Middle (FIM) transformations to raw text data.

    Each sentence is randomly converted (with probability ``fim_rate``) into
    PSM format: ``<PRE> prefix <SUF> suffix <MID> middle <EOS>``.
    The loss is masked so the model only predicts the **middle span**.

    Sequences are packed into stateful batches exactly as ``TextDataset`` does,
    preserving the Transformer-XL style context window across steps.

    The ``clm_labels`` key contains the masked label tensor — identical to the
    label format used by ``TextDataset`` / ``clm_collate_fn`` so the same
    ``calculate_loss`` path handles it without modification.
    """

    def __init__(
            self,
            raw_sentences: List[str],
            max_len: int,
            batch_size: int,
            tokenizer: Optional[Any] = None,
            fim_rate: float = 0.5,
            seed: int = 0,
    ):
        """
        Args:
            raw_sentences: Plain text strings to tokenise and pack.
            max_len:       Maximum token sequence length per batch step.
            batch_size:    Number of parallel streams (stateful batching).
            tokenizer:     Tokenizer with encode()/decode() and FIM token IDs.
                           Falls back to DummyTokenizer when None.
            fim_rate:      Fraction of sequences rearranged into PSM format.
                           0.0 = pure CLM; 1.0 = all FIM.  Default 0.5.
            seed:          RNG seed for reproducible FIM splits.
        """
        self._rng = random.Random(seed)
        _tok = tokenizer if tokenizer is not None else DummyTokenizer()

        all_input_ids:  List[int] = []
        all_label_ids:  List[int] = []

        for sentence in raw_sentences:
            raw = _tok.encode(sentence, add_special_tokens=True)
            ids, lbls = create_fim_sequence(raw, _tok, self._rng, fim_rate)
            all_input_ids.extend(ids)
            all_label_ids.extend(lbls)

        if not all_input_ids:
            logger.error("FimDataset: no tokens produced from raw_sentences.")
            self.data: List[Dict[str, torch.Tensor]] = []
            return

        # Pack into stateful stream batches (same as create_stateful_batches)
        # We manually tile because we need aligned label batches.
        num_streams = batch_size
        total = len(all_input_ids)
        tokens_per_stream = math.ceil(total / num_streams)
        num_batches = math.ceil(tokens_per_stream / max_len)
        total_slots = num_batches * max_len * num_streams
        pad_needed = total_slots - total
        pad_id = _tok.pad_token_id

        all_input_ids.extend([pad_id] * max(0, pad_needed))
        all_label_ids.extend([-100]  * max(0, pad_needed))

        inp_t = torch.tensor(all_input_ids[:total_slots], dtype=torch.long)
        lbl_t = torch.tensor(all_label_ids[:total_slots], dtype=torch.long)

        tokens_per_stream = total_slots // num_streams
        inp_streams = inp_t.view(num_streams, tokens_per_stream)
        lbl_streams = lbl_t.view(num_streams, tokens_per_stream)

        self.data = []
        for i in range(num_batches):
            inp_batch = inp_streams[:, i * max_len: (i + 1) * max_len]
            lbl_batch = lbl_streams[:, i * max_len: (i + 1) * max_len]
            self.data.append({'input_ids': inp_batch, 'clm_labels': lbl_batch})

        _pct_fim = 100.0 * sum(
            1 for l in all_label_ids[:total_slots] if l != -100
        ) / max(total_slots, 1)
        logger.info(
            "FimDataset: %d sentences → %d batches  "
            "fim_rate=%.0f%%  active_label_density=%.1f%%",
            len(raw_sentences), len(self.data), 100.0 * fim_rate, _pct_fim,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]

class SimpleTranslationDataset(Dataset):
    """Custom Dataset for handling translation pairs"""

    def __init__(self, pairs, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        for src_text, src_lang, tgt_text, tgt_lang in pairs:
            src_ids = self.tokenizer.encode(src_text, add_special_tokens=True)
            tgt_ids = self.tokenizer.encode(tgt_text, add_special_tokens=True)
            combined_ids = src_ids + tgt_ids[1:]  # Skip target's BOS token
            self.data.append({
                'ids': combined_ids,
                'src_len': len(src_ids),
                'src_lang': src_lang,
                'tgt_lang': tgt_lang
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TextDataset(Dataset):
    def __init__(self, raw_sentences: list[str], max_len: int, batch_size: int,
                 tokenizer: Optional[Any] = None, compacted: bool = False, task: Task = Task.CLM):
        """
        Args:
            raw_sentences: List of raw text strings to tokenise.
            max_len:       Maximum token sequence length.
            batch_size:    Number of parallel streams for stateful batching.
            tokenizer:     Tokenizer instance with an encode() method and
                           ``pad_token_id`` attribute.  When None a fresh
                           DummyTokenizer is created whose ``pad_token_id``
                           property always reads the current module-level
                           PAD_TOKEN_ID, so vocab expansions made via
                           expand_module_vocab() are reflected automatically.
            compacted:     If True, pack sequences into stateful batches.
            task:          Task enum that controls padding direction.
        """
        self.max_len = max_len
        self.task = task
        _tok = tokenizer if tokenizer is not None else DummyTokenizer()
        raw_token_ids = [_tok.encode(s, add_special_tokens=True) for s in raw_sentences]
        if self.task == Task.RCLM:
            token_ids = [tokens[::-1] for tokens in raw_token_ids]
        else:
            token_ids = raw_token_ids

        # Use the tokenizer's own pad_token_id — live property, never stale.
        _pad_id = _tok.pad_token_id
        if compacted:
            self.data = create_stateful_batches(token_ids, _pad_id, max_len, batch_size, task=self.task)
        else:
            self.data = pad_tokens(token_ids, _pad_id, max_len, task=self.task)

        self._num_tokens = sum(d.numel() if isinstance(d, torch.Tensor) else len(d) for d in self.data)
        if self._num_tokens < 1:
            logger.error("There are no tokens in the dataset")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        return self.data[idx]


class WebSearchTool:
    def search(self, query: str, max_results: int = 3) -> str:
        logger.info(f"Executing web search for: '{query}'")
        try:
            with DDGS() as ddgs:
                results = list(itertools.islice(ddgs.text(query, region='us-en', safesearch='off'), max_results))
                if not results: return "Web search results: No relevant information found."
                formatted_results = "\n".join([f"Title: {r['title']}\nSnippet: {r['body']}" for r in results])
                return f"Web search results:\n{formatted_results}"
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return "Web search results: Could not fetch information."


class EfficientKMeans:
    def __init__(self, n_clusters: int, max_iter: int = 100, tol: float = 1e-4, device: torch.device = 'cpu'):
        self.n_clusters, self.max_iter, self.tol, self.device = n_clusters, max_iter, tol, device
        self.centroids: Optional[torch.Tensor] = None
        self.labels_: Optional[torch.Tensor] = None

    def fit(self, X: torch.Tensor):
        if X.shape[0] < self.n_clusters:
            logger.warning(f"Number of samples ({X.shape[0]}) < n_clusters ({self.n_clusters}). Using fewer clusters.")
            self.n_clusters = max(1, X.shape[0])
        if self.n_clusters == 0:
            return self

        n_samples, _ = X.shape
        X_device = X.to(self.device)
        indices = torch.randperm(n_samples, device=self.device)[:self.n_clusters]
        self.centroids = X_device[indices].clone()

        for _ in range(self.max_iter):
            distances = torch.cdist(X_device, self.centroids)
            labels = torch.argmin(distances, dim=1)
            counts = torch.bincount(labels, minlength=self.n_clusters).float().unsqueeze(1)
            new_centroids = torch.zeros_like(self.centroids)
            new_centroids.scatter_add_(0, labels.unsqueeze(1).expand_as(X_device), X_device)
            new_centroids /= counts.clamp(min=1)

            empty_clusters = (counts.squeeze() == 0)
            if empty_clusters.any():
                num_empty = empty_clusters.sum().item()
                rand_indices = torch.randperm(n_samples, device=self.device)[:num_empty]
                new_centroids[empty_clusters] = X_device[rand_indices]

            if torch.norm(new_centroids - self.centroids) < self.tol: break
            self.centroids = new_centroids

        self.labels_ = torch.argmin(torch.cdist(X_device, self.centroids), dim=1)
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.centroids is None: raise RuntimeError("KMeans model has not been fitted yet.")
        return torch.argmin(torch.cdist(X.to(self.device), self.centroids), dim=1)


class MemoryStore:
    def __init__(self, embed_dim: int, db_path: str = "vector_db",
                 device: torch.device = torch.device('cpu'),
                 initial_capacity: int = 100):
        self.device, self.db_path, self.embed_dim = device, db_path, embed_dim
        self.vectors_path, self.meta_path = f"{db_path}_vectors.mmap", f"{db_path}_meta.json"
        self.kmeans = EfficientKMeans(n_clusters=max(1, initial_capacity // 25), device=self.device)
        self.index_built = False
        self._load_or_create_db(initial_capacity)
        logger.info(f"Vector DB initialized. Entries: {len(self.metadata)}. Capacity: {self.vectors.shape[0]}.")

    def _load_or_create_db(self, capacity: int):
        if os.path.exists(self.meta_path) and os.path.exists(self.vectors_path):
            with open(self.meta_path, 'r') as f:
                self.metadata = json.load(f)
            map_capacity = max(capacity, len(self.metadata))
            self.vectors = np.memmap(self.vectors_path, dtype='float32', mode='r+',
                                     shape=(map_capacity, self.embed_dim))
        else:
            self.metadata = []
            self.vectors = np.memmap(self.vectors_path, dtype='float32', mode='w+', shape=(capacity, self.embed_dim))

        if len(self.metadata) > self.kmeans.n_clusters: self._build_index()

    def _resize_db(self):
        current_capacity = self.vectors.shape[0]
        new_capacity = current_capacity * 2
        logger.info(f"Resizing vector DB from {current_capacity} to {new_capacity}...")
        if hasattr(self.vectors, '_mmap'): self.vectors._mmap.close()
        old_vectors = np.copy(self.vectors)
        self.vectors = np.memmap(self.vectors.filename, dtype='float32', mode='w+',
                                 shape=(new_capacity, self.embed_dim))
        self.vectors[:current_capacity] = old_vectors
        self.vectors.flush()

    def _build_index(self):
        if not self.metadata: return
        logger.info("Building semantic index...")
        entry_count = len(self.metadata)
        active_vectors = torch.from_numpy(self.vectors[:entry_count]).to(self.device)
        self.kmeans.fit(active_vectors)
        self.index_built = True
        logger.info("Semantic index built.")

    def add_correction(self, prompt: str, corrected_output: str, encoder_fn: Callable[[str], np.ndarray]):
        if len(self.metadata) >= self.vectors.shape[0]: self._resize_db()
        vector = encoder_fn(prompt)
        entry_idx = len(self.metadata)
        self.vectors[entry_idx] = vector
        self.metadata.append({'prompt': prompt, 'output': corrected_output})
        self.vectors.flush()
        with open(self.meta_path, 'w') as f: json.dump(self.metadata, f)
        self.index_built = False
        logger.info(f"Added new memory: '{prompt[:50]}...'")

    def retrieve_examples(self, prompt: str, encoder_fn: Callable[[str], np.ndarray], k: int = 1) -> List[
        Tuple[str, str]]:
        if not self.metadata: return []
        if not self.index_built: self._build_index()
        query_vector = torch.from_numpy(encoder_fn(prompt)).to(self.device).unsqueeze(0)
        cluster_id = self.kmeans.predict(query_vector).item()
        cluster_indices_np = np.where(self.kmeans.labels_.cpu().numpy() == cluster_id)[0]
        if len(cluster_indices_np) == 0: return []

        candidate_vectors = torch.from_numpy(self.vectors[cluster_indices_np]).to(self.device)
        similarities = F.cosine_similarity(query_vector, candidate_vectors)
        num_to_get = min(k, len(cluster_indices_np))
        top_k_indices_in_cluster = torch.topk(similarities, k=num_to_get).indices
        original_indices = cluster_indices_np[top_k_indices_in_cluster.cpu().numpy()]
        return [(self.metadata[i]['prompt'], self.metadata[i]['output']) for i in original_indices]

    def close(self):
        if hasattr(self, 'vectors') and self.vectors is not None and hasattr(self.vectors, '_mmap'):
            self.vectors.flush()
            self.vectors._mmap.close()
            self.vectors = None
            logger.info("Memory store file handle released.")


# ======================================================================================
# --- Model & Security Primitives ---
# ======================================================================================
class ModelArgs:
    # Best word_embed_dim: 81920, 65536, 32768
    vocab_size, pad_token_id, bos_token_id, sep_token_id, eos_token_id = 1000, -1, -1, -1, -1
    word_embed_dim, embed_dim, ffn_dim, n_layer, n_heads, max_seq_len = 8192, 512, 2048, 8, 8, 64
    dropout, weight_decay, clm_lr, base_lr, min_lr = 0.3, 1e-2, 8e-5, 1e-3, 1e-6
    embed_group_size, embed_device = 10, torch.device("cpu")
    num_sensitivity_levels, num_languages, num_task_classes = 3, 7, 5  # 5 = len(TaskClass)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype, device = torch.float32, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights: Dict[str, float] = {}
    db_path: str = "vector_db"
    # --- MoE Configuration ---
    expert_dim, expert_ffn_dim = 128, 256

    # List of cutoffs, e.g., [10000, 20000]. If None, uses standard Linear layer.
    # cutoffs[0] must be >= HEAD_FRACTION * _TOKEN_PARTITION_SIZE = 0.40 * 25000 = 10000
    # to ensure the head covers the top 15-20% of tokens (80-90% of usage).
    adaptive_softmax_cutoffs: Optional[List[int]] = [10000, 20000]

    def __init__(self, **kwargs): [setattr(self, k, v) for k, v in kwargs.items()]

    def get_kwargs(self):
        kwargs = {k: v for k, v in vars(ModelArgs).items() if not k.startswith('__') and not callable(v)}
        kwargs.update(vars(self))
        return kwargs


# ── Legacy SCALE_FACTORS kept for backward compat with calculate_loss ──────────
# Primary weighting is now done via TASKCLASS_SCALE_FACTORS (defined above).
# These per-task values are used inside calculate_loss to resolve individual
# logit keys that belong to the same TaskClass but need different treatment.
SCALE_FACTORS = {
    Task.RCLM.name: 2.0, Task.CLM.name: 1.5,
    Task.LANGUAGE_IDENTIFICATION.name: 1.0, Task.SENSITIVITY_CLASSIFICATION.name: 1.2,
    Task.TASK_IDENTIFICATION.name: 1.0, Task.QUESTION_ANSWERING_EXTRACTIVE.name: 1.5,
    Task.QUESTION_ANSWERING_GENERATIVE.name: 2.0, Task.SUMMARIZATION_EXTRACTIVE.name: 1.0,
    Task.SUMMARIZATION_ABSTRACTIVE.name: 2.5, Task.CODE_GENERATION.name: 2.0,
    Task.TRANSLATION.name: 2.0, Task.UNCLASSIFIED_SKILL.name: 1.0,
}

# ── Skill (Task) → output logit key mapping ─────────────────────────────────
# Each fine-grained Task maps to named logit tensors in the outputs dict.
# TaskClass-level routing (encoder/decoder path) is handled by SecureEncoderDecoderMoE.
TASK_TO_LOGIT_NAMES = {
    Task.RCLM: "reverse_clm_logits", Task.CLM: "clm_logits",
    Task.SUMMARIZATION_EXTRACTIVE: "summary_ext_logits",
    Task.SUMMARIZATION_ABSTRACTIVE: "summary_abs_logits",
    Task.TRANSLATION: "translation_logits",
    Task.CODE_GENERATION: "code_logits",
    Task.QUESTION_ANSWERING_GENERATIVE: "qa_gen_logits",
    Task.QUESTION_ANSWERING_EXTRACTIVE: ("qa_ext_start_logits", "qa_ext_end_logits"),
    Task.LANGUAGE_IDENTIFICATION: {"encoder": "src_lang_logits", "decoder": "tgt_lang_logits"},
    Task.SENSITIVITY_CLASSIFICATION: {"encoder": "src_sensitivity_logits", "decoder": "tgt_sensitivity_logits"},
    Task.TASK_IDENTIFICATION: "task_id_logits",
    Task.UNCLASSIFIED_SKILL: "unclassified_skill_logits",
    Task.FIM: "fim_logits",             # decoder-only; shares the lm_head weight
}

# Build fast O(1) lookup sets/dicts from the table above
LOGIT_NAMES: set = set()
LOGIT_NAMES_TO_TASK: Dict[str, Task] = {}
for _task, _logit in TASK_TO_LOGIT_NAMES.items():
    if isinstance(_logit, str):
        LOGIT_NAMES.add(_logit);
        LOGIT_NAMES_TO_TASK[_logit] = _task
    elif isinstance(_logit, tuple):
        LOGIT_NAMES.update(_logit)
        for _n in _logit: LOGIT_NAMES_TO_TASK[_n] = _task
    else:  # dict {"encoder": ..., "decoder": ...}
        LOGIT_NAMES.update(_logit.values())
        for _n in _logit.values(): LOGIT_NAMES_TO_TASK[_n] = _task

# ── TaskClass → logit names (used for fast routing in _apply_task_heads) ─────
# Maps each TaskClass to the set of logit keys it can produce.
TASKCLASS_TO_LOGIT_NAMES: Dict[TaskClass, set] = {tc: set() for tc in TaskClass}
for _task, _logit in TASK_TO_LOGIT_NAMES.items():
    _tc = _task.task_class
    if isinstance(_logit, str):
        TASKCLASS_TO_LOGIT_NAMES[_tc].add(_logit)
    elif isinstance(_logit, tuple):
        TASKCLASS_TO_LOGIT_NAMES[_tc].update(_logit)
    else:
        TASKCLASS_TO_LOGIT_NAMES[_tc].update(_logit.values())

# Token partition slice size (Architecture Layer 5b: 25k-token slices per expert)
_TOKEN_PARTITION_SIZE = 25_000


class AdaptiveSoftmaxHead(nn.Module):
    """
    Efficient adaptive softmax output head that (during inference) restricts
    the softmax to only the token-partition slices claimed by *active* experts
    plus any tokens that share the same semantic cluster as those active tokens
    (i.e. tokens whose cluster centroid is closest to at least one active token's
    centroid).  This follows the architecture plan's Layer 6 spec:

        V = V_lang1 ∪ V_lang2 ∪ … ∪ V_langN  (active experts only)
        + any token in the same semantic space as those active tokens.

    During training the standard AdaptiveLogSoftmaxWithLoss forward is used,
    which already avoids a full vocab projection by construction.
    """

    # Head size as a fraction of partition_size (40% of 25 000 = 10 000).
    # The head must cover at least the top 15-20% of tokens that account for
    # 80-90% of usage, so 40% gives a comfortable safety margin.
    _HEAD_FRACTION: float = 0.40

    def __init__(
            self,
            d_model: int,
            vocab_size: int,
            cutoffs: List[int],
            div_value: float = 4.0,
            partition_size: int = _TOKEN_PARTITION_SIZE,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.partition_size = partition_size

        # ── Auto-scale cutoffs to be valid for this vocab_size ───────────────
        # PyTorch requires every cutoff to be in [1, vocab_size-1] and strictly
        # ascending.  When running with a small test vocabulary (e.g. 29 tokens)
        # the production cutoffs [10000, 20000] would be out of range, so we
        # scale them proportionally and deduplicate.
        scaled: List[int] = []
        for c in sorted(cutoffs):
            clamped = max(1, min(c, vocab_size - 1))
            if not scaled or clamped > scaled[-1]:
                scaled.append(clamped)
        # If scaling collapsed everything to a single bucket (tiny vocab),
        # drop all cutoffs and fall back to a single flat head over the full vocab.
        if len(scaled) < 2 and vocab_size <= len(scaled) + 2:
            scaled = []
        self.cutoffs = scaled

        # ── Head-size validation (production-scale vocabs only) ──────────────
        # The AdaptiveLogSoftmaxWithLoss "head" covers tokens [0, cutoffs[0]).
        # For the model to represent the bulk of the token-frequency distribution
        # without bottlenecking, the head must span at least the first 40% of
        # each active expert's token partition (the first ~10 000 tokens out of
        # the 25 000-token _TOKEN_PARTITION_SIZE).  Research on sub-word
        # vocabularies shows the top 15-20% of tokens account for 80-90% of all
        # token occurrences, so 40% is a conservative safe threshold.
        #
        # The validation only applies when vocab_size >= partition_size (i.e.
        # a production-scale vocabulary).  Small test vocabs are proportionally
        # scaled and exempt from the absolute minimum.
        #
        # Concretely: min_head_size = HEAD_FRACTION × partition_size
        #   = 0.40 × 25 000 = 10 000  (production default)
        if self.cutoffs and vocab_size >= partition_size:
            head_size = self.cutoffs[0]
            min_head_size = max(1, int(round(self._HEAD_FRACTION * partition_size)))
            if head_size < min_head_size:
                raise ValueError(
                    f"AdaptiveSoftmaxHead: head size (cutoffs[0]={head_size}) is too small "
                    f"to cover the bulk of the token distribution.  "
                    f"With partition_size={partition_size} and HEAD_FRACTION={self._HEAD_FRACTION:.0%}, "
                    f"cutoffs[0] must be at least {min_head_size} "
                    f"(≈ the first {self._HEAD_FRACTION:.0%} of each {partition_size}-token partition, "
                    f"which covers the top 15-20%% of tokens responsible for 80-90%% of usage).  "
                    f"Increase cutoffs[0] to ≥{min_head_size} or reduce partition_size."
                )
            logger.info(
                "AdaptiveSoftmaxHead: head_size=%d  partition_size=%d  "
                "head_coverage=%.1f%%  (min_required=%.1f%%  ≈top-15-20%% usage)",
                head_size, partition_size,
                100.0 * head_size / max(partition_size, 1),
                100.0 * self._HEAD_FRACTION,
            )
        elif self.cutoffs:
            logger.debug(
                "AdaptiveSoftmaxHead: small vocab (%d < partition_size=%d), "
                "cutoffs scaled to %s — head-fraction check skipped.",
                vocab_size, partition_size, self.cutoffs,
            )

        # AdaptiveLogSoftmaxWithLoss combines the projection and the loss
        # calculation for efficiency — no full vocab softmax needed during training.
        self.asm = nn.AdaptiveLogSoftmaxWithLoss(
            in_features=d_model,
            n_classes=vocab_size,
            cutoffs=self.cutoffs,
            div_value=div_value,
            head_bias=True,
        )
        # ------------------------------------------------------------------
        # Semantic-neighbour index (built lazily on first inference call).
        # Maps every token-id to an integer cluster label so that we can
        # quickly expand the active token set to include all tokens that share
        # the same semantic cluster as any active token.
        # ------------------------------------------------------------------
        self._token_cluster_labels: Optional[torch.Tensor] = None  # [vocab_size]

    # ------------------------------------------------------------------
    # Helpers: partition / semantic expansion
    # ------------------------------------------------------------------

    def _build_cluster_index(self, device: torch.device) -> None:
        """
        Build a lightweight cluster index from the head-projection weight
        matrix (shape [vocab_size, d_model] or [head_size, d_model]).

        We cluster the *output* embedding rows with a simple k-means variant
        so that semantically similar tokens share the same cluster label.
        Only done once and cached as a non-parameter buffer.
        """
        # Use the head-cluster projection weights as token representations.
        # nn.AdaptiveLogSoftmaxWithLoss stores its head projection as `head`.
        try:
            weight = self.asm.head.weight.detach()  # [head_out, d_model]
        except AttributeError:
            weight = next(self.asm.parameters()).detach()  # fallback

        n_clusters = max(1, weight.size(0) // 8)
        n_tokens = self.vocab_size

        # Use random initialisation for speed; we only need coarse grouping.
        torch.manual_seed(0)
        indices = torch.randperm(min(weight.size(0), n_tokens))[:n_clusters]
        centroids = weight[indices].clone().float()  # [K, d_model]

        # --- Mini k-means (5 iterations is enough for coarse clustering) ---
        head_w = weight.float()  # [head_out, d_model]
        for _ in range(5):
            # Distances: [head_out, K]
            dists = torch.cdist(head_w, centroids)
            labels_head = dists.argmin(dim=1)  # [head_out]
            # Update centroids
            for k in range(n_clusters):
                members = head_w[labels_head == k]
                if members.numel() > 0:
                    centroids[k] = members.mean(dim=0)

        # Map all vocab_size positions to a cluster.
        # Positions beyond head_out get label = closest centroid of the head.
        full_labels = torch.zeros(n_tokens, dtype=torch.long)
        head_size = min(head_w.size(0), n_tokens)
        full_labels[:head_size] = labels_head[:head_size]
        if n_tokens > head_size:
            # Assign remaining tail tokens to the most populated cluster
            mode_label = labels_head.mode().values
            full_labels[head_size:] = mode_label

        self._token_cluster_labels = full_labels.to(device)
        logger.debug(
            "AdaptiveSoftmaxHead: built cluster index (%d clusters, %d tokens).",
            n_clusters, n_tokens,
        )

    def _get_active_token_indices(
            self,
            active_expert_ids: Optional[List[int]],
            device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Return a 1-D LongTensor of token indices to compute logits for.

        The set is:
          1. The combined partition slices of *active* experts.
          2. All tokens that share the same semantic cluster as any token
             in set (1).

        Returns None when the full vocabulary should be used, which lets the
        caller take the fast asm.log_prob path.  This includes:
          - active_expert_ids is None (no hint provided).
          - The number of active experts is large enough that their combined
            partitions trivially cover the entire vocabulary
            (len(active_expert_ids) * partition_size >= vocab_size), so there
            is nothing to be gained from the sparse path.
        """
        if active_expert_ids is None:
            return None  # caller will use full vocab

        # Fast path: if the active partitions are guaranteed to cover the full
        # vocabulary, skip the loop and semantic expansion entirely.
        if len(active_expert_ids) * self.partition_size >= self.vocab_size:
            return None

        # --- Step 1: union of partition slices -------------------------
        token_sets: List[torch.Tensor] = []
        for expert_id in active_expert_ids:
            start = expert_id * self.partition_size
            end = min(start + self.partition_size, self.vocab_size)
            if start >= self.vocab_size:
                continue
            token_sets.append(torch.arange(start, end, device=device))

        if not token_sets:
            return None  # no valid partitions → full vocab

        active_tokens = torch.cat(token_sets).unique()

        # --- Step 2: semantic expansion --------------------------------
        if self._token_cluster_labels is None:
            self._build_cluster_index(device)

        labels = self._token_cluster_labels.to(device)  # [vocab_size]
        # Clusters represented by any active token
        active_clusters = labels[active_tokens.clamp(0, len(labels) - 1)].unique()

        # All tokens whose cluster overlaps with active clusters
        cluster_mask = torch.isin(labels, active_clusters)
        semantic_tokens = cluster_mask.nonzero(as_tuple=True)[0]

        # Union of partition tokens + semantic neighbours
        combined = torch.cat([active_tokens, semantic_tokens]).unique().sort().values
        return combined  # [M] where M ≤ vocab_size

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
            self,
            hidden: torch.Tensor,
            target: Optional[torch.Tensor] = None,
            active_expert_ids: Optional[List[int]] = None,
    ) -> Any:
        """
        Args:
            hidden:           [batch, seq_len, d_model] or [N, d_model]
            target:           [batch, seq_len] or [N]  — training labels
            active_expert_ids: list of integer expert IDs whose 25k-token
                              partition slices should be included in the
                              inference softmax.  None → full vocab.
        """
        original_shape = hidden.shape
        if len(original_shape) == 3:
            hidden = hidden.view(-1, original_shape[-1])  # [N, D]

        # ==============================================================
        # TRAINING MODE — use efficient ASM loss (no full softmax)
        # ==============================================================
        if target is not None:
            if len(target.shape) == 2:
                target = target.view(-1)

            valid_mask = target != -100
            if not valid_mask.any():
                return None

            valid_hidden = hidden[valid_mask]
            valid_target = target[valid_mask]

            # ASM internally only projects to head-size + tail clusters,
            # never over the full vocab — already efficient by construction.
            return self.asm(valid_hidden, valid_target)

        # ==============================================================
        # INFERENCE MODE — restrict softmax to active-expert token slice
        # ==============================================================
        device = hidden.device

        active_indices = self._get_active_token_indices(active_expert_ids, device)

        if active_indices is None:
            # Full-vocab path — no expert hint, or fast path detected full coverage.
            log_probs_full = self.asm.log_prob(hidden)  # [N, vocab_size]
            if len(original_shape) == 3:
                return log_probs_full.view(
                    original_shape[0], original_shape[1], self.vocab_size
                )
            return log_probs_full

        # ------------------------------------------------------------------
        # Sparse path — asm.log_prob is NEVER called here.
        # _get_active_token_indices guarantees M < vocab_size when it returns
        # a non-None tensor (full-coverage is caught above as None → asm.log_prob).
        #
        # Project hidden onto only the M active token rows, log_softmax over
        # that slice, scatter into a full-vocab -inf tensor so downstream
        # argmax / beam-search produces valid ids.  Cost O(N·M·D) vs O(N·V·D).
        # The active slice forms a valid normalised distribution (logsumexp ≈ 0).
        # ------------------------------------------------------------------
        N = hidden.size(0)
        M = active_indices.numel()

        log_probs_full = torch.full(
            (N, self.vocab_size), float("-inf"), device=device, dtype=hidden.dtype
        )

        head_weight = self.asm.head.weight           # [head_out, d_model]
        head_bias   = getattr(self.asm.head, 'bias', None)
        cutoff0     = self.cutoffs[0]
        head_mask   = active_indices < cutoff0
        head_ids    = active_indices[head_mask]
        raw_scores  = torch.full((N, M), float("-inf"), device=device, dtype=hidden.dtype)

        # ── Head-range tokens ────────────────────────────────────────────
        if head_ids.numel() > 0:
            head_rows   = head_weight[head_ids]
            head_logits = hidden @ head_rows.T
            if head_bias is not None:
                head_logits = head_logits + head_bias[head_ids]
            head_pos_in_M = head_mask.nonzero(as_tuple=True)[0]
            raw_scores[:, head_pos_in_M] = head_logits

        # ── Tail-cluster tokens ──────────────────────────────────────────
        prev_cutoff = cutoff0
        for cluster_idx, tail_module in enumerate(self.asm.tail):
            next_cutoff = self.cutoffs[cluster_idx + 1] if (cluster_idx + 1) < len(self.cutoffs) else self.vocab_size
            tail_mask_c = (active_indices >= prev_cutoff) & (active_indices < next_cutoff)
            tail_ids_g  = active_indices[tail_mask_c]
            if tail_ids_g.numel() == 0:
                prev_cutoff = next_cutoff
                continue
            local_ids   = tail_ids_g - prev_cutoff
            reduced     = tail_module[0](hidden)
            out_weight  = tail_module[1].weight
            out_bias_t  = tail_module[1].bias
            tail_logits = reduced @ out_weight[local_ids].T
            if out_bias_t is not None:
                tail_logits = tail_logits + out_bias_t[local_ids]
            tail_pos_in_M = tail_mask_c.nonzero(as_tuple=True)[0]
            raw_scores[:, tail_pos_in_M] = tail_logits
            prev_cutoff = next_cutoff

        # Log-softmax over the active slice only — never touches inactive tokens.
        log_probs_full[:, active_indices] = F.log_softmax(raw_scores, dim=-1)

        if len(original_shape) == 3:
            return log_probs_full.view(
                original_shape[0], original_shape[1], self.vocab_size
            )
        return log_probs_full


class RelativePositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.embedding_table = nn.Parameter(torch.empty(2 * max_len - 1, d_model))
        nn.init.normal_(self.embedding_table, mean=0, std=0.02)
        self.max_len = max_len

    def forward(self, seq_len_q: int, seq_len_k: int, device: torch.device, pos_offset: int = 0) -> torch.Tensor:
        q_pos = torch.arange(pos_offset, pos_offset + seq_len_q, dtype=torch.long, device=device).unsqueeze(1)
        k_pos = torch.arange(seq_len_k, dtype=torch.long, device=device).unsqueeze(0)
        relative_pos = k_pos - q_pos
        max_rel_dist = self.max_len - 1
        clamped_relative_pos = torch.clamp(relative_pos, -max_rel_dist, max_rel_dist)
        return self.embedding_table[clamped_relative_pos + max_rel_dist]


class ReformerAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        if args.embed_dim % args.n_heads != 0:
            raise ValueError(f"embed_dim ({args.embed_dim}) must be divisible by n_heads ({args.n_heads})")

        self.embed_dim = args.embed_dim
        self.num_heads = args.n_heads
        self.head_dim = args.embed_dim // self.num_heads

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.r_w_bias = nn.Parameter(torch.randn(self.num_heads, self.head_dim))
        self.r_r_bias = nn.Parameter(torch.randn(self.num_heads, self.head_dim))
        self.relative_pos = RelativePositionalEmbedding(self.head_dim, max_len=args.max_seq_len)

    def forward(self, x: torch.Tensor,
                mems: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                content_stream: Optional[torch.Tensor] = None,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                update_kv_cache: bool = True
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        bsz, qlen, _ = x.shape
        is_cross_attention = content_stream is not None

        if is_cross_attention:
            kv_input = content_stream
        else:
            kv_input = torch.cat([mems, x], dim=1) if mems is not None else x

        k_seq_len_initial = kv_input.size(1)
        q = self.q_proj(x).view(bsz, qlen, self.num_heads, self.head_dim)
        k_new = self.k_proj(kv_input).view(bsz, k_seq_len_initial, self.num_heads, self.head_dim)
        v_new = self.v_proj(kv_input).view(bsz, k_seq_len_initial, self.num_heads, self.head_dim)
        k = k_new.permute(0, 2, 1, 3)
        v = v_new.permute(0, 2, 1, 3)
        if past_kv is not None:
            try:
                k_for_attn = torch.cat([past_kv[0], k], dim=2)
                v_for_attn = torch.cat([past_kv[1], v], dim=2)
            except RuntimeError:
                logger.error(f"Cannot cat past_kv[0].shape={past_kv[0].shape} -> k.shape={k.shape}")
                k_for_attn, v_for_attn = past_kv
        else:
            k_for_attn, v_for_attn = k, v

        if update_kv_cache:
            present_kv = (k_for_attn.detach(), v_for_attn.detach())
        else:
            present_kv = past_kv

        query_pos_offset = past_kv[0].size(2) if past_kv is not None else 0
        klen = k_for_attn.size(2)
        q = q.permute(0, 2, 1, 3)
        ac = torch.einsum('bhid,bhjd->bhij', q + self.r_w_bias.unsqueeze(0).unsqueeze(2), k_for_attn)
        pos_emb = self.relative_pos(qlen, klen, q.device, pos_offset=query_pos_offset)
        bd = torch.einsum("bhid,ijd->bhij", q + self.r_r_bias.unsqueeze(0).unsqueeze(2), pos_emb)
        attn_scores = (ac + bd) * (self.head_dim ** -0.5)
        mask_fill_value = torch.finfo(attn_scores.dtype).min
        if attn_mask is not None:
            final_mask = attn_mask.unsqueeze(1)
        else:
            final_mask = torch.zeros_like(attn_scores, dtype=torch.bool)
            if klen > 1:  # Create a standard causal mask for autoregressive decoding
                causal_mask = torch.triu(torch.ones(qlen, klen, device=x.device, dtype=torch.bool),
                                         diagonal=klen - qlen + 1)
                final_mask = final_mask | causal_mask

        attn_scores = attn_scores.masked_fill(final_mask, mask_fill_value)
        # Use F.softmax directly — it is internally numerically stable (subtracts
        # the per-row max implicitly).  The previous manual max-subtraction caused
        # a KV-cache inconsistency: the global max of attn_scores differs between a
        # full L-token pass and a 1-token cached pass attending to L cached keys,
        # producing different shifted values and therefore different softmax outputs.
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=x.dtype)
        attn_output = torch.einsum('bhij,bhjd->bhid', attn_probs, v_for_attn)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(bsz, qlen, self.embed_dim)
        return self.out_proj(attn_output), present_kv


class SwiFFN(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.embed_dim, args.ffn_dim, bias=False)
        self.w2 = nn.Linear(args.ffn_dim, args.embed_dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.silu(self.w1(x))))


class SlicedEmbedding(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.embedding_dim = args.word_embed_dim
        self.segment_size = args.embed_group_size
        self.num_segments = math.ceil(args.vocab_size / args.embed_group_size)
        self.segments = nn.ModuleList()
        for i in range(self.num_segments):
            start_idx = i * self.segment_size
            end_idx = min((i + 1) * self.segment_size, args.vocab_size)
            num_embeddings = end_idx - start_idx
            padding_idx = None
            if args.pad_token_id is not None and start_idx <= args.pad_token_id < end_idx:
                padding_idx = args.pad_token_id - start_idx

            # TODO - offload embedding of rarely used tokens
            self.segments.append(
                nn.Embedding(num_embeddings, self.embedding_dim, padding_idx, device=args.embed_device)
            )
        logger.info(f"Initialized SlicedEmbedding with {self.num_segments} segments of size {self.segment_size}")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        flat_ids = input_ids.flatten()
        output_flat = input_ids.new_empty(
            (B * L, self.embedding_dim), dtype=self.segments[0].weight.dtype
        )
        # Determine which segments are active in this batch
        segment_indices = torch.div(flat_ids, self.segment_size, rounding_mode='floor')

        # Process each active segment only once for maximum efficiency
        for seg_idx in torch.unique(segment_indices):
            # Find all tokens in the batch that belong to this segment
            mask = (segment_indices == seg_idx)

            # Get the original token IDs and their positions in the flattened tensor
            original_ids_in_segment = flat_ids[mask]

            # Convert global token IDs to local indices within the segment's embedding table
            local_indices = original_ids_in_segment % self.segment_size

            # Retrieve embeddings from the corresponding segment
            embeddings = self.segments[seg_idx](local_indices)

            # Place the retrieved embeddings into the correct positions in the output tensor
            output_flat[mask] = embeddings

        return output_flat.view(B, L, self.embedding_dim)


class ExpertFFN(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty((args.expert_ffn_dim, args.expert_dim)))
        self.w2 = nn.Parameter(torch.empty((args.expert_dim, args.expert_ffn_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.w1.data, self.w2.data


class VocabularyControlledExperts(nn.Module):
    """
    An FFN layer that routes tokens to a dedicated expert based on their vocabulary ID.
    This architecture creates a specialized feed-forward network for each token in the
    vocabulary, enabling a massive increase in the model's capacity. Computation
    remains efficient as only the experts for tokens present in a batch are used.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        # Create a dedicated expert FFN for each token in the vocabulary.
        self.compress = nn.Linear(args.embed_dim, args.expert_dim, bias=False)
        self.expand = nn.Linear(args.expert_dim, args.embed_dim, bias=False)
        self.experts = nn.ModuleList([ExpertFFN(args) for _ in range(self.vocab_size)])
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Processes tokens with their dedicated vocabulary-based experts. This optimized
        forward pass avoids Python loops by vectorizing the expert computation. It uses
        a sort-and-dispatch method to group tokens by their expert, then applies all
        active expert FFNs in a single batched matrix multiplication, significantly
        improving GPU utilization.
        """
        # Step 1: Compress the input embeddings and flatten the batch for token-level processing.
        x = self.compress(x)
        B, L, D_expert = x.shape
        x_flat = x.reshape(-1, D_expert)
        ids_flat = input_ids.reshape(-1)

        # Step 2: Sort tokens by their expert ID (vocabulary ID) to create contiguous groups.
        # This allows us to process all tokens for a given expert in a single slice.
        sorted_ids, sorted_indices = torch.sort(ids_flat)
        sorted_x = x_flat[sorted_indices]

        # Step 3: Identify the active experts and the number of tokens for each.
        unique_ids, counts = torch.unique_consecutive(sorted_ids, return_counts=True)

        # --- Vectorized Expert Processing ---
        # Step 4a: Gather weights of all active experts.
        w1_weights, w2_weights = [], []
        for i in unique_ids:
            w1, w2 = self.experts[i]()
            w1_weights.append(w1)
            w2_weights.append(w2)

        # Step 4b: Create a mapping from each token to its corresponding expert's weights.
        expert_map = torch.repeat_interleave(torch.arange(len(unique_ids), device=x.device), counts)
        w1_weights = torch.stack(w1_weights)[expert_map]
        w2_weights = torch.stack(w2_weights)[expert_map]

        # Step 4c: Apply the expert FFNs in a batched fashion with activation and dropout.
        # Unsqueeze adds a dimension for matrix-vector product using bmm.
        hidden = self.dropout(F.silu(torch.bmm(w1_weights, sorted_x.unsqueeze(-1))))

        # Apply the second parameter weight.
        expert_output_bmm = torch.bmm(w2_weights, hidden)

        # Squeeze the last dimension to get the final expert outputs in sorted order.
        output_sorted = expert_output_bmm.squeeze(-1)

        # Step 5: Scatter the results back to their original token positions.
        inverse_permutation = torch.argsort(sorted_indices)
        output_flat = output_sorted[inverse_permutation]

        # Step 6: Reshape the output and expand it back to the model's embedding dimension.
        final_out = self.expand(output_flat.view(B, L, D_expert))
        return final_out


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, general_expert: SwiFFN):
        super().__init__()
        self.ffn_norm = nn.RMSNorm(args.embed_dim)
        self.ffn = VocabularyControlledExperts(args)
        self.general_expert = general_expert
        self.attn_norm = nn.RMSNorm(args.embed_dim)
        self.attn = ReformerAttention(args)
        # Initialize cross-attention layer only if needed, can be conditional
        self.cross_attn_norm = nn.RMSNorm(args.embed_dim)
        self.cross_attn = ReformerAttention(args)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self,
                input_ids: torch.Tensor,
                h_content: torch.Tensor,
                h_query: Optional[torch.Tensor] = None,
                content_mask: Optional[torch.Tensor] = None,
                query_mask: Optional[torch.Tensor] = None,
                encoder_output: Optional[torch.Tensor] = None,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # --- Feed-Forward Network for both streams ---
        ffn_input = self.ffn_norm(h_content)
        general_knowledge = self.dropout(self.general_expert(ffn_input))
        ffn_out = self.dropout(self.ffn(ffn_input, input_ids=input_ids))
        h_content = h_content + general_knowledge + ffn_out

        if h_query is not None:
            ffn_out = self.ffn(self.ffn_norm(h_query), input_ids=input_ids)
            h_query = h_query + self.dropout(ffn_out)

        # --- Self-Attention for Content and Query Streams ---
        # 1. Content Stream: Standard self-attention (Q, K, V from h_content)
        #    This stream uses the KV cache during generation.
        attn_out, present_kv = self.attn(
            self.attn_norm(h_content),
            attn_mask=content_mask,
            past_kv=past_kv
        )
        h_content = h_content + self.dropout(attn_out)

        # 2. Query Stream: Q from h_query, K/V from h_content
        #    This stream is only used during training and does not use a KV cache.
        if h_query is not None:
            attn_out, _ = self.attn(
                self.attn_norm(h_query),
                attn_mask=query_mask,
                content_stream=self.attn_norm(h_content)
            )
            h_query = h_query + self.dropout(attn_out)

        # --- Cross-Attention (only for decoder blocks) ---
        if encoder_output is not None:
            # Both streams cross-attend to the same encoder output
            attn_out, _ = self.cross_attn(self.cross_attn_norm(h_content), content_stream=encoder_output)
            h_content = h_content + self.dropout(attn_out)
            if h_query is not None:
                attn_out, _ = self.cross_attn(self.cross_attn_norm(h_query), content_stream=encoder_output)
                h_query = h_query + self.dropout(attn_out)

        return h_content, h_query, present_kv


class TaskHeadManager(nn.Module):
    """
    Manages output heads keyed on TaskClass rather than individual Task IDs.

    Architecture (Layer 6 — Vocabulary Consolidation):
      - GENERATIVE / REVERSE_GENERATIVE  → shared generative LM head (or per-skill
                                           specialised heads for translation, code, etc.)
      - DISCRIMINATIVE                   → span-extraction or scalar heads
      - SEQUENCE_LABELING                → per-position classification heads
      - UNCLASSIFIED                     → generative fallback head

    Using TaskClass as the primary axis shrinks the embedding table from
    len(Task)=12 entries to len(TaskClass)=5, reducing memory and lookup cost.
    Fine-grained per-skill heads are still kept inside `task_heads` but are
    dispatched via TaskClass routing in forward().
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.activation = nn.SiLU()
        self.dtype = args.dtype

        # TaskClass sets that need non-linear pre-activation before the head
        self.non_linear_task_classes = {
            TaskClass.DISCRIMINATIVE,
            TaskClass.SEQUENCE_LABELING,
        }
        # Fine-grained Task set that additionally need non-linear activation
        self.non_linear_tasks = {
            Task.QUESTION_ANSWERING_EXTRACTIVE, Task.SUMMARIZATION_EXTRACTIVE,
            Task.LANGUAGE_IDENTIFICATION, Task.SENSITIVITY_CLASSIFICATION,
            Task.TASK_IDENTIFICATION,
        }

        def create_generative_head():
            if args.adaptive_softmax_cutoffs is not None:
                return AdaptiveSoftmaxHead(args.embed_dim, args.vocab_size, args.adaptive_softmax_cutoffs)
            return nn.Linear(args.embed_dim, args.vocab_size, bias=False, dtype=self.dtype)

        # Shared generative head for CLM / RCLM / UNCLASSIFIED
        self.lm_head = create_generative_head()

        # Per-skill specialised heads — keyed on Task.name for ModuleDict compat
        self.task_heads = nn.ModuleDict({
            # Generative skills that share the common LM head
            Task.CLM.name: self.lm_head,
            Task.RCLM.name: self.lm_head,
            Task.UNCLASSIFIED_SKILL.name: self.lm_head,
            # Generative skills with dedicated heads (may learn different distributions)
            Task.SUMMARIZATION_ABSTRACTIVE.name: create_generative_head(),
            Task.TRANSLATION.name: create_generative_head(),
            Task.CODE_GENERATION.name: create_generative_head(),
            Task.QUESTION_ANSWERING_GENERATIVE.name: create_generative_head(),
            # FIM shares the lm_head: same decoder path, same vocabulary distribution.
            # Training signal from FIM spans steers the SAME weights as CLM, which is
            # how a unified bidirectional representation is achieved efficiently.
            Task.FIM.name: self.lm_head,
            # Discriminative / sequence-labeling heads
            Task.QUESTION_ANSWERING_EXTRACTIVE.name: nn.Linear(args.embed_dim, 2, dtype=self.dtype),
            Task.SUMMARIZATION_EXTRACTIVE.name: nn.Linear(args.embed_dim, 1, dtype=self.dtype),
            Task.LANGUAGE_IDENTIFICATION.name: nn.Linear(args.embed_dim, args.num_languages, dtype=self.dtype),
            Task.SENSITIVITY_CLASSIFICATION.name: nn.Linear(args.embed_dim, args.num_sensitivity_levels,
                                                            dtype=self.dtype),
            Task.TASK_IDENTIFICATION.name: nn.Linear(args.embed_dim, len(TaskClass), dtype=self.dtype),
        })
        logger.info(
            "Initialized TaskHeadManager (TaskClass-routed). "
            f"Adaptive Softmax: {args.adaptive_softmax_cutoffs is not None}"
        )

    def forward(
            self,
            task: Task,
            hidden_state: torch.Tensor,
            forward_pass: str,
            labels: Optional[torch.Tensor] = None,
            active_expert_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Route a hidden state through the appropriate output head.

        Routing is TaskClass-first:
          1. Determine TaskClass → select head family
          2. Apply non-linear activation when required by TaskClass
          3. Forward through the skill-specific head
          4. Map output to the named logit key(s) for calculate_loss

        Args:
            task:              Fine-grained Task enum (skill identifier).
            hidden_state:      [B, L, D] or [N, D] encoder/decoder output.
            forward_pass:      "encoder" or "decoder" (for dual-stream heads).
            labels:            Training labels — triggers efficient ASM loss path.
            active_expert_ids: Expert partition IDs for inference-time vocab slicing.
        """
        task_class = task.task_class  # O(1) property lookup

        # Non-linear pre-activation: required for discriminative / sequence tasks
        needs_activation = (
                task_class in self.non_linear_task_classes
                or task in self.non_linear_tasks
        )
        state = self.activation(hidden_state) if needs_activation else hidden_state
        state = state.to(self.dtype)

        head = self.task_heads[task.name]

        # Dispatch: AdaptiveSoftmaxHead has an extra active_expert_ids arg
        if isinstance(head, AdaptiveSoftmaxHead):
            logits_or_loss = head(state, target=labels, active_expert_ids=active_expert_ids)
        else:
            logits_or_loss = head(state)

        # Map output to the correct logit key name(s)
        logit_name_map = TASK_TO_LOGIT_NAMES[task]
        if task == Task.QUESTION_ANSWERING_EXTRACTIVE:
            start_logits, end_logits = logits_or_loss.split(1, dim=-1)
            return {logit_name_map[0]: start_logits.squeeze(-1),
                    logit_name_map[1]: end_logits.squeeze(-1)}
        if isinstance(logit_name_map, dict):
            return {logit_name_map[forward_pass]: logits_or_loss}
        return {logit_name_map: logits_or_loss}


class SecureEncoderDecoderMoE(nn.Module):
    """
    A combined-objective RCLM/CLM model with Transformer-XL style memory. It operates in one of three modes:
    (Encoder-Only, Decoder-Only, Encoder-Decoder) based on the task. RCLM objective enhances bidirectional
    understanding, while CLM supports autoregressive tasks.
    """

    def __init__(self, args: ModelArgs, tokenizer: Optional[Any] = None):
        """
        Args:
            args:      Model configuration (ModelArgs).
            tokenizer: Tokenizer instance providing encode/decode/vocab.
                       Defaults to DummyTokenizer() when not provided.
                       Always stored as self.tokenizer so all methods that
                       need encoding/decoding (generate, similarity, etc.)
                       can use it without creating temporary instances.
        """
        super().__init__()
        self.args, self.device, self.vocab_size = args, args.device, args.vocab_size
        self._setup_language_and_task_data(args)
        self.tokenizer = tokenizer if tokenizer is not None else DummyTokenizer()
        # Snapshot the token↔id mappings as they exist right now (at construction
        # time) so that similarity() and most_similar() always look up ids against
        # the same vocabulary that _embeddings was built from.  If the module-level
        # TOKEN_TO_ID is later expanded by expand_module_vocab(), _vocab_snapshot
        # stays fixed and _embeddings_cache is automatically invalidated by
        # reset_weights() — callers must call reset_weights() (or simply re-create
        # the model) after a vocab expansion if they want updated embeddings.
        raw_vocab: Dict[str, int] = dict(self.tokenizer.vocab)  # snapshot copy
        # Guard: only keep entries whose id is within the model's vocab_size.
        self._vocab_snapshot: Dict[str, int] = {
            t: i for t, i in raw_vocab.items() if i < args.vocab_size
        }
        # Routing sets: map fine-grained Tasks to encoder/decoder path.
        # TaskClass-level routing (REVERSE_GENERATIVE→encoder, GENERATIVE→decoder,
        # etc.) is handled in the routing helper methods; these sets cover the
        # task-specific overrides within each TaskClass.
        self.encoder_only_tasks = {Task.RCLM, Task.QUESTION_ANSWERING_EXTRACTIVE, Task.SUMMARIZATION_EXTRACTIVE,
                                   Task.TASK_IDENTIFICATION}
        self.decoder_only_tasks = {Task.CLM, Task.CODE_GENERATION, Task.UNCLASSIFIED_SKILL, Task.FIM}
        self.encoder_decoder_tasks = {Task.TRANSLATION, Task.SUMMARIZATION_ABSTRACTIVE, Task.LANGUAGE_IDENTIFICATION,
                                      Task.SENSITIVITY_CLASSIFICATION, Task.QUESTION_ANSWERING_GENERATIVE}
        self.tasks_with_decoder = self.decoder_only_tasks.union(self.encoder_decoder_tasks)
        self.word_embedding = SlicedEmbedding(args)
        self.access_level_embedding = nn.Embedding(args.num_sensitivity_levels, args.embed_dim)
        self.language_embedding = nn.Embedding(args.num_languages, args.embed_dim)
        # task_class_embedding: len(TaskClass)=5 entries vs old 12-entry task_ids table.
        self.task_class_embedding = nn.Embedding(args.num_task_classes, args.embed_dim)
        self.position_in_sequence_embedding = nn.Embedding(args.max_seq_len, args.embed_dim)
        self.query_embedding = nn.Parameter(torch.randn(1, 1, args.embed_dim))
        # Create a compression network for embedding
        self.embed_ffn = nn.Linear(args.word_embed_dim, args.embed_dim, bias=False)
        self.general_expert = SwiFFN(args)
        assert args.n_layer >= 2, "n_layer must be at least 2 for encoder-only and decoder-only layers."
        num_shared_layers = args.n_layer - 2
        self.shared_layers = nn.ModuleList(
            [TransformerBlock(args, self.general_expert) for _ in range(num_shared_layers)])
        self.encoder_only_layer = TransformerBlock(args, self.general_expert)
        self.decoder_only_layer = TransformerBlock(args, self.general_expert)
        self.memory_store = MemoryStore(embed_dim=args.embed_dim, device=self.device, db_path=args.db_path)
        self.web_search_tool = WebSearchTool()
        self.out_norm = nn.RMSNorm(args.embed_dim)
        self.task_manager = TaskHeadManager(args)
        self.reset_weights()
        logger.info(f"Initialized {self.__class__.__name__} with args={self.args.get_kwargs()}")
        logger.info(f"{self.__class__.__name__} has {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M parameters.")

    def reset_weights(self):
        self._invalidate_embedding_cache()
        self.apply(self._init_weights)

    def _setup_language_and_task_data(self, args: ModelArgs):
        all_langs = sorted(list(set(itertools.chain(*LANGUAGE_FAMILIES.values()))))
        self.lang_to_global_id = {lang: i for i, lang in enumerate(all_langs)}
        args.num_sensitivity_levels, args.num_task_classes, args.num_languages = len(
            AccessLevel), TaskClass.count(), len(self.lang_to_global_id)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad(): module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.RMSNorm):
            if hasattr(module, 'weight'): module.weight.data.fill_(1.0)

    def _calculate_positions(self, input_ids: torch.Tensor, past_lengths: torch.Tensor,
                             reset_token_id: int) -> torch.Tensor:
        """
        Calculates globally correct token positions for parallel streams using vectorized operations.
        Positions reset to 0 at each reset token within a stream.
        """
        # --- Step 1: Create basic masks ---
        is_pad = (input_ids == self.args.pad_token_id)
        is_reset_marker = (input_ids == reset_token_id)

        # --- Step 2: Calculate positions within each segment (a segment starts with a reset token) ---
        # A reset happens at a reset token or at the very start of the sequence for counting purposes.
        is_reset = is_reset_marker.clone()
        is_reset[:, 0] = True

        # Per-token increments are 1 for non-padded tokens, 0 for padded.
        increments = (~is_pad).long()

        # Calculate a running sum of tokens. This forms the basis of positions.
        naive_cumsum = increments.cumsum(dim=1)

        # To get segment-local positions, we must subtract the total count of tokens
        # from all previous segments. We use a trick with cummax to propagate the
        # value of the sum at the start of a segment to all tokens within it.
        # The value to subtract is the sum *before* the reset token, hence the shift.
        shifted_cumsum = torch.cat([torch.zeros_like(naive_cumsum[:, :1]), naive_cumsum[:, :-1]], dim=1)
        correction_values = torch.where(is_reset, shifted_cumsum, 0)
        correction_matrix = correction_values.cummax(dim=1).values

        # Positions are 0-indexed, so we subtract 1 from the 1-indexed cumsum.
        positions = naive_cumsum - correction_matrix - 1

        # --- Step 3: Add past_lengths offset for continuation sequences ---
        # A row is a continuation if its first token is not a reset token.
        is_continuation_row = (input_ids[:, 0] != reset_token_id).unsqueeze(1)

        # The offset only applies to the very first segment of a continuation row.
        segment_id = is_reset.long().cumsum(dim=1)
        is_first_segment = (segment_id == 1)

        # Create a mask where both conditions are true.
        offset_mask = is_continuation_row & is_first_segment

        # Add the initial offset from past_lengths.
        positions += past_lengths * offset_mask.long()

        # --- Step 4: Final cleanup for padding and clamping ---
        # Padded tokens must have a position of 0.
        positions.masked_fill_(is_pad, 0)

        # Clamp positions to the maximum allowed sequence length.
        return torch.clamp(positions, 0, self.args.max_seq_len - 1)

    def encode(self, batch, encoder_past_kv, h_content, past_seq_len, tasks_in_batch):
        output_logits = {}
        new_past_kv = []
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask')
        attention_mask = self._update_kv_cache_masking(batch, attention_mask, past_seq_len)
        h_query = None
        for idx, layer in enumerate(self.shared_layers + [self.encoder_only_layer]):
            layer_past_kv = encoder_past_kv[idx] if encoder_past_kv and len(encoder_past_kv) > idx else None
            h_content, _, present_kv = layer(
                input_ids, h_content, h_query, attention_mask, past_kv=layer_past_kv
            )
            new_past_kv.append(present_kv)
        encoder_output = self.out_norm(h_content)
        return encoder_output, output_logits, new_past_kv

    def decode(self, batch, decoder_past_kv, h_content, past_seq_len, tasks_in_batch, encoder_output,
               is_encoded_decoder_task):
        output_logits = {}
        new_past_kv = []
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask')
        attention_mask = self._update_kv_cache_masking(batch, attention_mask, past_seq_len)
        h_query = None
        for idx, layer in enumerate(self.shared_layers + [self.decoder_only_layer]):
            layer_past_kv = decoder_past_kv[idx] if decoder_past_kv and len(decoder_past_kv) > idx else None
            is_shared_layer = idx < len(self.shared_layers)
            cross_attn_source = encoder_output if is_encoded_decoder_task and not is_shared_layer else None
            h_content, _, present_kv = layer(
                input_ids, h_content, h_query, attention_mask, encoder_output=cross_attn_source, past_kv=layer_past_kv
            )
            new_past_kv.append(present_kv)
        decoder_output = self.out_norm(h_content)
        return decoder_output, output_logits, new_past_kv

    def forward(
            self,
            batch: Dict[str, torch.Tensor],
            encoder_past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            decoder_past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            task_class: Optional[TaskClass] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        """
        Forward pass following the Updated Architecture Design Plan (Layers 1-6).

        Args:
            batch:           Dict containing 'input_ids', optional 'task_ids',
                             'access_levels', 'language_ids', 'past_lengths', etc.
            encoder_past_kv: KV cache from previous encoder pass (Transformer-XL style).
            decoder_past_kv: KV cache from previous decoder pass.
            task_class:      Optional TaskClass override.  When provided this takes
                             precedence over task_ids for routing decisions (Layer 1/3
                             integration).  Allows callers that already resolved a
                             TaskClass (e.g. via SkillRegistry + SkillExecutor) to
                             skip the task_id → TaskClass translation step.

        Returns:
            (outputs_dict, new_encoder_past_kv, new_decoder_past_kv)
        """
        input_ids = batch['input_ids']
        task_ids = batch.get('task_ids')

        # ------------------------------------------------------------------
        # Layer 1 → Layer 3: Determine TaskClass for routing
        # ------------------------------------------------------------------
        # Priority: explicit task_class arg > task_ids in batch > default (GENERATIVE)
        if task_class is not None:
            # Caller has already resolved the TaskClass (e.g. from SkillRegistry).
            # Derive a compatible legacy Task set so existing head logic still works.
            resolved_task_class = task_class
            tasks_in_batch = self._task_class_to_tasks(task_class)
        elif task_ids is not None:
            tasks_in_batch = {Task(v.item()) for v in torch.unique(task_ids[:, 0])}
            # Derive the dominant TaskClass from the batch's task set
            resolved_task_class = self._resolve_batch_task_class(tasks_in_batch)
        else:
            tasks_in_batch = set()
            resolved_task_class = TaskClass.GENERATIVE

        # ------------------------------------------------------------------
        # Determine reset token based on TaskClass (Layer 1d / Layer 3b)
        # ------------------------------------------------------------------
        is_r_clm_task = resolved_task_class.is_reverse() or any(
            t == Task.RCLM for t in tasks_in_batch
        )
        reset_token_id_for_positions = (
            self.args.eos_token_id if is_r_clm_task else self.args.bos_token_id
        )

        # --- Create Initial Embeddings ---
        input_embeds = self.word_embedding(input_ids)
        input_embeds = self.embed_ffn(input_embeds)

        # --- Calculate Positions (Layer 1d: Token Partition Router) ---
        past_lengths = batch.get(
            'past_lengths',
            torch.zeros((input_ids.shape[0], 1), device=self.device, dtype=torch.long),
        )
        position_in_sequence = self._calculate_positions(
            input_ids, past_lengths, reset_token_id_for_positions
        )
        pos_embeds = self.position_in_sequence_embedding(position_in_sequence)
        h = input_embeds + pos_embeds

        past_seq_len = 0
        if encoder_past_kv and encoder_past_kv[0] is not None:
            past_seq_len = encoder_past_kv[0][0].size(2)
        elif decoder_past_kv and decoder_past_kv[0] is not None:
            past_seq_len = decoder_past_kv[0][0].size(2)

        # --- Conditioning embeddings ---
        if batch.get('access_levels') is not None:
            h += self.access_level_embedding(batch['access_levels'])
        # Layer 1b: TaskClass conditioning — embed the TaskClass ordinal (0-4) for
        # each token position. This is more compact than the legacy task_ids (0-11).
        task_class_ids = batch.get('task_class_ids')
        if task_class_ids is None and task_ids is not None:
            # Backward-compat: derive task_class_ids from legacy task_ids on-the-fly
            task_class_ids = torch.zeros_like(task_ids)
            for unique_val in torch.unique(task_ids):
                tc = Task(unique_val.item()).task_class
                task_class_ids[task_ids == unique_val] = tc.ordinal
        if task_class_ids is not None:
            h += self.task_class_embedding(task_class_ids)
        if batch.get('language_ids') is not None:
            h += self.language_embedding(batch['language_ids'])

        # ------------------------------------------------------------------
        # Layer 1a-1d: Routing — determine encoder/decoder path via TaskClass
        # ------------------------------------------------------------------
        is_encoder_task = self._is_encoder_task(resolved_task_class, tasks_in_batch)
        is_decoder_task = self._is_decoder_task(resolved_task_class, tasks_in_batch)
        is_encoded_decoder_task = self._is_encoded_decoder_task(resolved_task_class, tasks_in_batch)

        new_encoder_past_kv = []
        new_decoder_past_kv = []

        encoder_output = None
        decoder_output = None
        outputs: Dict[str, Any] = {}

        # Layer 5: Parallel execution — Encoder Processing
        if is_encoder_task or is_encoded_decoder_task:
            encoder_output, output_logits, new_encoder_past_kv = self.encode(
                batch, encoder_past_kv, h, past_seq_len, tasks_in_batch
            )
            outputs.update(output_logits)

        # Layer 5: Parallel execution — Decoder Processing
        if is_decoder_task or is_encoded_decoder_task:
            decoder_output, output_logits, new_decoder_past_kv = self.decode(
                batch, decoder_past_kv, h, past_seq_len, tasks_in_batch,
                encoder_output, is_encoded_decoder_task,
            )
            outputs.update(output_logits)

        # ------------------------------------------------------------------
        # Layer 6: Vocabulary consolidation — Task-specific output heads
        # ------------------------------------------------------------------
        if task_ids is not None:
            task_id_per_sequence = task_ids[:, 0]
            self._apply_task_heads(
                encoder_output, decoder_output,
                task_id_per_sequence, batch, outputs,
                active_expert_ids=self._get_active_expert_ids(tasks_in_batch),
            )

        # Expose encoder/decoder hidden states for downstream use (e.g. generate)
        if encoder_output is not None:
            outputs['encoder_output'] = encoder_output
        if decoder_output is not None:
            outputs['decoder_output'] = decoder_output

        return outputs, new_encoder_past_kv, new_decoder_past_kv

    # ------------------------------------------------------------------
    # Layer 1 routing helpers: TaskClass → encoder/decoder path
    # ------------------------------------------------------------------

    def _task_class_to_tasks(self, tc: TaskClass) -> set:
        """Map a TaskClass to a representative set of legacy Tasks for head dispatch."""
        if tc == TaskClass.REVERSE_GENERATIVE:
            return {Task.RCLM}
        if tc == TaskClass.GENERATIVE:
            return {Task.CLM}
        if tc == TaskClass.DISCRIMINATIVE:
            return {Task.SENSITIVITY_CLASSIFICATION}
        if tc == TaskClass.SEQUENCE_LABELING:
            return {Task.LANGUAGE_IDENTIFICATION}
        return {Task.UNCLASSIFIED_SKILL}

    def _resolve_batch_task_class(self, tasks_in_batch: set) -> TaskClass:
        """Derive the dominant TaskClass from a mixed-task batch."""
        if not tasks_in_batch:
            return TaskClass.GENERATIVE
        # Use the first task's class; mixed batches are handled per-row in _apply_task_heads
        return next(iter(tasks_in_batch)).task_class

    def _is_encoder_task(self, tc: TaskClass, tasks: set) -> bool:
        """True when the resolved task class requires encoder-only processing."""
        return (
                tc.is_reverse()
                or any(t in self.encoder_only_tasks for t in tasks)
        )

    def _is_decoder_task(self, tc: TaskClass, tasks: set) -> bool:
        """True when the resolved task class requires decoder-only processing."""
        return (
                (tc == TaskClass.GENERATIVE and not tasks)  # pure generative hint
                or any(t in self.decoder_only_tasks for t in tasks)
        )

    def _is_encoded_decoder_task(self, tc: TaskClass, tasks: set) -> bool:
        """True when the resolved task class requires encoder+decoder (seq2seq)."""
        return any(t in self.encoder_decoder_tasks for t in tasks)

    def _get_active_expert_ids(self, tasks_in_batch: set) -> Optional[List[int]]:
        """
        Return the list of expert IDs (integer indices) active in this forward
        pass.  Used by AdaptiveSoftmaxHead to restrict the inference softmax to
        the 25k-token partitions of those experts.

        Expert IDs are derived from the task.value so that each task occupies a
        deterministic, non-overlapping slice of the vocabulary partition space.
        Returns None when all tasks are active (→ full vocab, no restriction).
        """
        if not tasks_in_batch:
            return None
        return [t.value for t in tasks_in_batch]

    def _update_kv_cache_masking(self, batch: Dict[str, torch.Tensor], content_mask: Optional[torch.Tensor],
                                 past_seq_len: int) -> Optional[torch.Tensor]:
        """
        Adjusts the content mask for stateful (KV-cached) forward passes.
        It prepends a mask for the past keys, allowing attention only if a sequence
        is a continuation from the previous batch.
        """
        # TODO - Instead of a dict (which requires item-by-item lookups),
        #  use a torch tensor mapping for faster vectorized operations
        if past_seq_len > 0 and content_mask is not None:
            input_ids = batch['input_ids']
            # is_continuation has shape (bsz, qlen)
            is_continuation = batch.get('is_continuation', torch.zeros_like(input_ids, dtype=torch.bool))

            # can_see_past_mask has shape (bsz, qlen, past_seq_len)
            can_see_past_mask = is_continuation.unsqueeze(2).expand(-1, -1, past_seq_len)

            # Attention masks use True for positions to MASK OUT.
            cross_batch_boundary_mask = ~can_see_past_mask

            # Prepend the cross-batch mask to the intra-batch mask.
            # New shape: (bsz, qlen, past_seq_len + qlen)
            return torch.cat([cross_batch_boundary_mask, content_mask], dim=-1)

        return content_mask

    def _apply_task_heads(
            self,
            encoder_output,
            decoder_output,
            task_ids_per_sequence,
            batch,
            outputs,
            active_expert_ids: Optional[List[int]] = None,
    ):
        unique_tasks = torch.unique(task_ids_per_sequence[task_ids_per_sequence != -100])
        for task_id_val in unique_tasks:
            task = Task(task_id_val.item())
            mask = (task_ids_per_sequence == task_id_val)
            if not mask.any(): continue
            is_pooled = task in {Task.SENSITIVITY_CLASSIFICATION, Task.LANGUAGE_IDENTIFICATION,
                                 Task.TASK_IDENTIFICATION}

            # Retrieve labels for Adaptive Softmax efficient training path.
            # Labels are ONLY passed to the head during training — during eval/
            # inference the head must always return a log-prob tensor regardless
            # of whether clm_labels happen to be present in the batch.
            labels = None
            if self.training and self.args.adaptive_softmax_cutoffs is not None:
                if task == Task.CLM:
                    labels = batch.get('clm_labels')
                elif task == Task.RCLM:
                    labels = batch.get('reverse_clm_labels')
                elif task == Task.FIM:
                    # FIM shares the lm_head and clm_labels (PSM loss on middle span)
                    labels = batch.get('clm_labels')

                if labels is not None:
                    labels = labels[mask]

            def apply_head(output_tensor, forward_pass_name):
                if output_tensor is None: return
                state_to_use = output_tensor[mask].mean(dim=1) if is_pooled else output_tensor[mask]

                task_logits = self.task_manager(
                    task, state_to_use, forward_pass_name,
                    labels=labels,
                    active_expert_ids=active_expert_ids,
                )

                for name, value in task_logits.items():
                    if value is None: continue

                    if isinstance(value, tuple) and hasattr(value, 'loss'):
                        outputs[name] = value
                    else:
                        if name not in outputs:
                            final_shape = list(value.shape)
                            final_shape[0] = output_tensor.size(0)
                            outputs[name] = torch.full(
                                torch.Size(final_shape), -100.,
                                device=self.args.device, dtype=self.args.dtype,
                            )
                        outputs[name][mask] = value.to(self.args.dtype)

            if task in self.encoder_decoder_tasks:
                apply_head(encoder_output, "encoder")
                apply_head(decoder_output, "decoder")
            elif task in self.decoder_only_tasks:
                apply_head(decoder_output, "decoder")
            elif task in self.encoder_only_tasks:
                apply_head(encoder_output, "encoder")

    @torch.no_grad()
    def _get_sentence_embedding(self, text: str) -> np.ndarray:
        self.eval()
        tokens = self.tokenizer.encode(text, add_special_tokens=True, return_tensors='pt').to(self.device)
        # Use an encoder-only task to ensure encoder_output is generated
        batch = clm_collate_fn(tokens, task=Task.SUMMARIZATION_EXTRACTIVE)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        # Pass both legacy task_ids (for head dispatch) and new task_class_ids
        _tc = Task.SUMMARIZATION_EXTRACTIVE.task_class
        batch['task_ids'] = torch.full_like(batch['input_ids'], Task.SUMMARIZATION_EXTRACTIVE.value)
        batch['task_class_ids'] = torch.full_like(batch['input_ids'], _tc.ordinal)
        outputs, _, _ = self.forward(batch, task_class=_tc)
        hidden_state = outputs.get('encoder_output')
        if hidden_state is None:
            logger.error("Could not get encoder_output for sentence embedding.")
            return np.zeros(self.args.embed_dim, dtype=np.float32)
        is_pad = batch['is_pad']
        mask = ~is_pad.unsqueeze(-1).expand_as(hidden_state)
        pooled = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled.squeeze(0).cpu().numpy()

    def learn_from_correction(self, prompt: str, corrected_output: str):
        self.memory_store.add_correction(prompt, corrected_output, encoder_fn=self._get_sentence_embedding)

    # ------------------------------------------------------------------
    # Token-similarity interface
    # ------------------------------------------------------------------

    @property
    def vocab(self) -> Dict[str, int]:
        """
        Token-to-id mapping snapshotted at model construction time.

        Using the snapshot (rather than the live tokenizer.vocab) guarantees that
        ids returned here are always valid indices into _embeddings, even after
        expand_module_vocab() has been called and TOKEN_TO_ID has been rebound.
        """
        return self._vocab_snapshot

    @property
    def _embeddings(self) -> np.ndarray:
        """
        Projected token embeddings as a float32 numpy array [vocab_size, embed_dim].

        Mirrors the forward embedding pipeline:
          1. Concatenate all SlicedEmbedding segment weights.
          2. Project through embed_ffn (word_embed_dim -> embed_dim).

        The result is cached as _embeddings_cache and cleared by
        _invalidate_embedding_cache() whenever weights are updated.
        """
        if getattr(self, "_embeddings_cache", None) is not None:
            return self._embeddings_cache

        with torch.no_grad():
            # Step 1: reconstruct full vocabulary weight matrix [vocab_size, word_embed_dim]
            seg_weights: List[torch.Tensor] = [
                seg.weight.to(self.device, torch.float32)
                for seg in self.word_embedding.segments
            ]
            full_emb = torch.cat(seg_weights, dim=0)[: self.vocab_size]  # guard against padding

            # Step 2: project through embed_ffn  [vocab_size, embed_dim]
            proj = self.embed_ffn.weight.to(self.device, torch.float32)  # [embed_dim, word_embed_dim]
            projected = full_emb @ proj.T

        self._embeddings_cache: np.ndarray = projected.cpu().numpy()
        return self._embeddings_cache

    def _invalidate_embedding_cache(self) -> None:
        """Discard the cached embedding matrix (called after weight resets)."""
        self._embeddings_cache = None

    def similarity(self, tok1: str, tok2: str) -> float:
        """
        Cosine similarity between two tokens in the model's embedding space.

        Embeddings are projected through embed_ffn so they live in the same
        embed_dim-dimensional space used by all transformer layers, giving a
        semantically meaningful similarity score.

        Args:
            tok1: First token string (must be in the model vocabulary).
            tok2: Second token string (must be in the model vocabulary).

        Returns:
            Cosine similarity in [-1, 1], or 0.0 if either token is
            out-of-vocabulary or has a zero-norm embedding.

        Example:
            >>> model.similarity("cat", "dog")
            0.83
        """
        vocab = self.vocab
        id1 = vocab.get(tok1)
        id2 = vocab.get(tok2)
        if id1 is None or id2 is None:
            logger.debug(
                "similarity: OOV token(s) — "
                f"tok1={tok1!r} ({'found' if id1 is not None else 'OOV'}), "
                f"tok2={tok2!r} ({'found' if id2 is not None else 'OOV'})"
            )
            return 0.0

        emb = self._embeddings  # [vocab_size, embed_dim] numpy float32
        e1: np.ndarray = emb[id1]
        e2: np.ndarray = emb[id2]
        n1 = float(np.linalg.norm(e1))
        n2 = float(np.linalg.norm(e2))
        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0
        return float(np.dot(e1, e2) / (n1 * n2))

    def most_similar(
            self,
            token: str,
            top_k: int = 10,
            exclude_special: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Return the top_k tokens most similar to token by cosine similarity.

        Computed in a single batched matrix operation (O(V * D)), efficient
        even for large vocabularies.

        Args:
            token:           Query token string.
            top_k:           Number of results (default 10).
            exclude_special: Exclude PAD/BOS/EOS/UNK/MASK from results.

        Returns:
            List of (token_str, cosine_similarity) pairs, sorted descending.
            Returns [] if token is OOV.

        Example:
            >>> model.most_similar("cat", top_k=3)
            [("dog", 0.91), ("mat", 0.87), ("fox", 0.84)]
        """
        vocab = self.vocab
        id_query = vocab.get(token)
        if id_query is None:
            logger.debug(f"most_similar: {token!r} not in vocabulary.")
            return []

        emb = self._embeddings  # [V, D] numpy float32
        query_vec = emb[id_query]  # [D]
        query_norm = float(np.linalg.norm(query_vec))
        if query_norm < 1e-10:
            return []

        # Batched cosine similarity: O(V * D) single matmul
        norms = np.linalg.norm(emb, axis=1)  # [V]
        safe_norms = np.where(norms < 1e-10, 1.0, norms)
        dots = emb @ query_vec  # [V]
        sims: np.ndarray = dots / (safe_norms * query_norm)

        # Mask query token, zero-norm tokens, and optionally special tokens.
        # Using -2.0 as a sentinel below the [-1, 1] cosine range.
        sims[id_query] = -2.0
        # Zero-norm tokens produce meaningless cosine scores — exclude them.
        sims[norms < 1e-10] = -2.0
        if exclude_special:
            for t in [PAD, UNK, BOS, EOS, MASK]:
                sid = vocab.get(t)
                if sid is not None:
                    sims[sid] = -2.0

        # Count how many candidates remain after masking (score > -2)
        valid_mask = sims > -1.5  # -2.0 is our sentinel for excluded slots
        n_valid = int(valid_mask.sum())
        k = min(top_k, n_valid)
        if k == 0:
            return []

        # Top-k via argpartition over valid candidates only (O(V) average).
        # We copy() the index arrays so the subsequent [::-1] reverse-slice
        # produces a contiguous writable array (required on Windows).
        valid_indices = np.where(valid_mask)[0]
        valid_sims = sims[valid_indices]
        n = len(valid_sims)
        if n == 0:
            return []
        if k >= n:
            top_local = np.argsort(valid_sims)[::-1].copy()
        else:
            part = np.argpartition(valid_sims, -k)[-k:]
            top_local = part[np.argsort(valid_sims[part])[::-1]].copy()
        top_idx = valid_indices[top_local]

        id_to_token = {v: t for t, v in vocab.items()}  # inverted snapshot
        return [(id_to_token.get(int(i), f"<id:{i}>"), float(sims[i]))
                for i in top_idx]

    @torch.no_grad()
    def generate(self, prompt: str, user: User, task: Task, max_new_tokens=100, enable_search=True):
        self.eval()
        examples = self.memory_store.retrieve_examples(prompt, k=1, encoder_fn=self._get_sentence_embedding)
        if examples: prompt = self.format_few_shot_prompt(prompt, examples)

        if enable_search and any(kw in prompt.lower() for kw in ["what is", "who is", "when is"]):
            prompt = f"Context: {self.web_search_tool.search(prompt)}\nQuestion: {prompt}"

        tokens = self.tokenizer.encode(prompt, add_special_tokens=True)

        if task == Task.RCLM:
            logit_key = TASK_TO_LOGIT_NAMES.get(task, 'reverse_clm_logits')
            stop_token_id = self.tokenizer.bos_token_id
            tokens.reverse()
        elif task == Task.FIM:
            # FIM inference: the caller supplies a PSM-formatted prompt string OR
            # a plain prompt (treated as prefix-only → standard continuation).
            # The model generates tokens until EOS (the end of the middle span).
            logit_key = 'fim_logits'
            stop_token_id = self.tokenizer.eos_token_id
        else:
            logit_map = TASK_TO_LOGIT_NAMES.get(task, 'clm_logits')
            logit_key = logit_map['decoder'] if isinstance(logit_map, dict) else logit_map
            stop_token_id = self.tokenizer.eos_token_id

        tokens = tokens[:-1]  # For forward generation, remove EOS if present. For reverse, remove BOS.
        diagonal = 1
        generated_ids = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        encoder_past_kv, decoder_past_kv = None, None
        tokens_generated = 0
        for i in range(max_new_tokens):
            is_continuation_gen = encoder_past_kv is not None or decoder_past_kv is not None
            past_len = 0
            if is_continuation_gen:
                if encoder_past_kv:
                    past_len = encoder_past_kv[0][0].size(2)
                elif decoder_past_kv:
                    past_len = decoder_past_kv[0][0].size(2)

            current_input_ids = generated_ids if not is_continuation_gen else generated_ids[:, -1:]
            # Prepare batch for model forward pass
            batch = clm_collate_fn(current_input_ids, task=task)
            batch['input_ids'] = current_input_ids
            batch['task_ids'] = torch.full_like(current_input_ids, task.value)
            batch['task_class_ids'] = torch.full_like(current_input_ids, task.task_class.ordinal)
            batch['access_levels'] = torch.full_like(current_input_ids, user.access_level.value)
            batch['past_lengths'] = torch.tensor([[past_len]], device=self.device, dtype=torch.long)
            # Manually create a causal mask for generation.
            q_len = current_input_ids.shape[1]
            # Mask is `True` for positions that should be masked (upper triangle).
            causal_mask = torch.triu(torch.ones(q_len, q_len, device=self.device, dtype=torch.bool), diagonal)
            batch['attention_mask'] = causal_mask.unsqueeze(0)  # Add batch dimension
            outputs, encoder_past_kv, decoder_past_kv = self.forward(
                batch,
                encoder_past_kv=encoder_past_kv,
                decoder_past_kv=decoder_past_kv,
                task_class=task.task_class,
            )
            logits = outputs.get(logit_key)
            if logits is None:
                break
            last_logits = logits[:, -1, :]  # [1, vocab_size]
            next_token_id = torch.argmax(last_logits, dim=-1, keepdim=True)
            # Replace UNK with the most likely non-special token so generation
            # never produces <|unk|> in its output.
            _unk_id = self.tokenizer.vocab.get(UNK, 1)
            _special_ids_set = {
                self.tokenizer.pad_token_id,
                self.tokenizer.bos_token_id,
                self.tokenizer.eos_token_id,
                _unk_id,
            }
            if next_token_id.item() in _special_ids_set and next_token_id.item() == _unk_id:
                # Mask all special tokens and pick the next-best token.
                _masked = last_logits.clone()
                for _sid in _special_ids_set:
                    if 0 <= _sid < _masked.size(-1):
                        _masked[:, _sid] = float('-inf')
                _fallback = torch.argmax(_masked, dim=-1, keepdim=True)
                # Only substitute if a valid fallback exists.
                if _fallback.item() not in _special_ids_set:
                    next_token_id = _fallback
            tokens_generated += 1
            # Stop on EOS/PAD — but only after generating at least one real token
            # so that generation never returns an empty string when the model is
            # miscalibrated for the first step.
            if tokens_generated > 1 and next_token_id.item() in {stop_token_id, self.tokenizer.pad_token_id}:
                break
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        generated_tokens = generated_ids[0, len(tokens):].tolist()
        if task == Task.RCLM: generated_tokens.reverse()

        return self.tokenizer.decode(generated_tokens)

    @staticmethod
    def format_few_shot_prompt(prompt: str, examples: List[Tuple[str, str]]) -> str:
        formatted = "".join(f"Example Input:\n{p}\nExample Output:\n{r}\n---\n" for p, r in examples)
        return f"{formatted}Input:\n{prompt}\nOutput:\n"

    def shutdown(self):
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if hasattr(self, 'memory_store') and self.memory_store:
            paths = [self.memory_store.vectors_path, self.memory_store.meta_path]
            self.memory_store.close()
            del self.memory_store
            gc.collect()
            time.sleep(0.5)
            for path in paths:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        logger.error(f"Error removing '{path}': {e}")
        logger.info("Shutdown complete.")


def update_loss_weights(logits_names):
    """
    Compute normalised per-logit loss weights.

    Uses TASKCLASS_SCALE_FACTORS as the primary source (fast O(1) TaskClass
    lookup) with a fallback to the fine-grained SCALE_FACTORS table for logit
    keys that need task-level overrides.
    """
    scale = 0.0
    already_counted: set = set()

    for name in logits_names:
        task = LOGIT_NAMES_TO_TASK.get(name)
        if task is None:
            continue
        # Prefer TaskClass-level weight to avoid double-counting tasks in the
        # same TaskClass; fall back to fine-grained SCALE_FACTORS if available.
        tc_key = task.task_class.value
        task_key = task.name
        if tc_key not in already_counted:
            already_counted.add(tc_key)
            scale += TASKCLASS_SCALE_FACTORS.get(tc_key, SCALE_FACTORS.get(task_key, 1.0))

    if scale == 0.0:
        scale = max(len(logits_names), 1)

    weights: Dict[str, float] = {}
    for name in logits_names:
        task = LOGIT_NAMES_TO_TASK.get(name)
        if task is not None:
            tc_key = task.task_class.value
            raw = TASKCLASS_SCALE_FACTORS.get(tc_key, SCALE_FACTORS.get(task.name, 1.0))
        else:
            raw = 1.0
        weights[name] = raw / scale

    return weights


def calculate_loss(outputs: Dict[str, torch.Tensor],
                   batch: Dict[str, torch.Tensor],
                   args: ModelArgs) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Calculates the total weighted loss across all active task heads."""
    # Find a valid tensor to determine device
    device = args.device
    for v in outputs.values():
        if isinstance(v, torch.Tensor):
            device = v.device
            break
        elif isinstance(v, tuple) and hasattr(v, 'loss'):  # ASM Output
            device = v.loss.device
            break

    loss_dict = {}
    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    active_logit_keys = {k for k, v in outputs.items() if k in LOGIT_NAMES and v is not None}
    if set(args.weights.keys()) != active_logit_keys and len(active_logit_keys) > 1:
        args.weights = update_loss_weights(active_logit_keys)
        # logger.info(f"New weights: {args.weights}")

    def compute_and_add_loss(logit_key, labels, loss_name, is_generative=False, num_classes=None):
        nonlocal total_loss
        output_obj = outputs.get(logit_key)

        if output_obj is None:
            return

        # --- NEW: Handle Adaptive Softmax Output ---
        # If the output is an ASM named tuple, it already contains the calculated loss
        if isinstance(output_obj, tuple) and hasattr(output_obj, 'loss'):
            loss = output_obj.loss
            if torch.isfinite(loss) and loss.item() > 0:
                total_loss += loss * args.weights.get(logit_key, 1.0)
                loss_dict[loss_name] = loss.item()
            return
        # -------------------------------------------

        logits = output_obj
        if labels is None: return
        labels = labels.to(device)

        if is_generative:
            valid_mask = labels.view(-1) != -100
            if not valid_mask.any(): return
            vocab_size = logits.size(-1)
            selected_logits, selected_labels = logits.view(-1, vocab_size), labels.view(-1)
            loss = F.cross_entropy(selected_logits, selected_labels, ignore_index=-100)
        else:
            sequence_labels = labels[:, 0]
            valid_mask = sequence_labels != -100
            if not valid_mask.any() or logits.dim() != 2: return
            selected_logits = logits[valid_mask]
            selected_labels = sequence_labels[valid_mask]
            if selected_logits.size(0) == 0: return
            loss = F.cross_entropy(selected_logits, selected_labels)

        if torch.isfinite(loss) and loss.item() > 0:
            total_loss += loss * args.weights.get(logit_key, 1.0)
            loss_dict[loss_name] = loss.item()
        elif torch.isnan(loss):
            logger.warning(f"Detected NaN Loss for: {loss_name}.\n{logit_key}")

    compute_and_add_loss('reverse_clm_logits', batch.get('reverse_clm_labels'), 'RCLM', is_generative=True)
    compute_and_add_loss('clm_logits', batch.get('clm_labels'), 'CLM', is_generative=True)
    # FIM shares clm_labels (both use next-token prediction on the target span only)
    compute_and_add_loss('fim_logits', batch.get('clm_labels'), 'FIM', is_generative=True)
    clm_labels = batch.get('clm_labels')
    compute_and_add_loss('summary_abs_logits', clm_labels, 'SUM_ABS', is_generative=True)
    compute_and_add_loss('translation_logits', clm_labels, 'TRANSLATE', is_generative=True)
    compute_and_add_loss('code_logits', clm_labels, 'CODE', is_generative=True)
    compute_and_add_loss('qa_gen_logits', clm_labels, 'QA_GEN', is_generative=True)
    compute_and_add_loss('unclassified_skill_logits', clm_labels, 'UNCLASSIFIED', is_generative=True)

    if 'qa_ext_start_logits' in outputs and 'start_positions' in batch:
        start_logits, end_logits = outputs['qa_ext_start_logits'], outputs['qa_ext_end_logits']
        start_labels, end_labels = batch['start_positions'].to(device), batch['end_positions'].to(device)
        sub_seq_id = batch['sub_sequence_id']                        # [B, L]

        # is_first_in_subsequence: True at the first token of every packed sub-sequence.
        is_first_in_sub = (sub_seq_id != torch.roll(sub_seq_id, 1, 1))
        is_first_in_sub[:, 0] = True
        # valid_mask: positions that carry a span label AND start a sub-sequence.
        valid_mask = (start_labels != -100) & is_first_in_sub

        if valid_mask.any():
            # For each valid (batch_row, sub_seq_start_col) position, extract the
            # encoder logit slice corresponding to that sub-sequence window, then
            # apply the span label (which is a LOCAL offset within that sub-sequence).
            #
            # This correctly handles packed sequences: sequence 2 starting at
            # column C has its span label relative to column C, so we slide the
            # logit window to [C : C + seq_context_len] and index into it.
            batch_rows, seq_cols = torch.where(valid_mask)
            valid_start_labels = start_labels[valid_mask]   # local offsets
            valid_end_labels   = end_labels[valid_mask]

            seq_len = start_logits.size(1)                  # L
            start_losses, end_losses = [], []

            for b, col, sl, el in zip(
                batch_rows.tolist(), seq_cols.tolist(),
                valid_start_labels.tolist(), valid_end_labels.tolist()
            ):
                # Guard: skip invalid (-100) labels
                if sl < 0 or el < 0:
                    continue
                # Compute the end of this sub-sequence window.
                # Find the next sub-sequence start after `col`, or end of sequence.
                next_starts = (is_first_in_sub[b, col+1:]).nonzero(as_tuple=True)[0]
                win_end = (col + 1 + next_starts[0].item()) if next_starts.numel() > 0 else seq_len
                # Slice logits for this sub-sequence: [1, win_len]
                win_start_logits = start_logits[b:b+1, col:win_end]  # [1, win_len]
                win_end_logits   = end_logits[b:b+1, col:win_end]
                win_len = win_start_logits.size(1)
                # Labels are local offsets within this window
                s_lbl = torch.tensor([sl], device=device)
                e_lbl = torch.tensor([el], device=device)
                if sl >= win_len or el >= win_len:
                    continue  # label out of this window — skip (data bug, not crash)
                start_losses.append(F.cross_entropy(win_start_logits, s_lbl))
                end_losses.append(F.cross_entropy(win_end_logits, e_lbl))

            if start_losses:
                qa_loss = (torch.stack(start_losses).mean() + torch.stack(end_losses).mean()) / 2
                if torch.isfinite(qa_loss) and qa_loss.item() > 0:
                    weight = args.weights.get('qa_ext_start_logits', 1.0)
                    total_loss += qa_loss * weight
                    loss_dict['QA_EXT'] = qa_loss.item()

    if 'summary_ext_logits' in outputs and 'summary_ext_labels' in batch:
        logits = outputs['summary_ext_logits'].squeeze(-1)
        labels = batch['summary_ext_labels'].to(device, dtype=torch.float32)
        valid_mask = (batch['input_ids'] != args.pad_token_id) & (labels != -100)
        if valid_mask.any():
            loss = F.binary_cross_entropy_with_logits(logits[valid_mask], labels[valid_mask])
            if torch.isfinite(loss) and loss.item() > 0:
                weight = args.weights.get('summary_ext_logits', 1.0)
                total_loss += loss * weight
                loss_dict['SUM_EXT'] = loss.item()

    compute_and_add_loss('src_lang_logits', batch.get('src_lang_ids'), 'SrcLang', num_classes=args.num_languages)
    compute_and_add_loss('src_sensitivity_logits', batch.get('access_levels'), 'SrcSens',
                         num_classes=args.num_sensitivity_levels)
    compute_and_add_loss('tgt_sensitivity_logits', batch.get('access_levels'), 'TgtSens',
                         num_classes=args.num_sensitivity_levels)
    compute_and_add_loss('tgt_lang_logits', batch.get('tgt_lang_ids'), 'TgtLang', num_classes=args.num_languages)
    return total_loss, loss_dict


# ============================================================================
# MAIN: Demonstrate Three-Layer Architecture
# ============================================================================

def main():
    """Demonstrate the Three-Layer Architecture integration."""

    print("\n" + "=" * 80)
    print("THREE-LAYER ARCHITECTURE DEMONSTRATION")
    print("=" * 80)

    # Initialize Skill Registry (Layer 2)
    print("\nLayer 2: Initializing Skill Registry...")
    registry_path = os.path.join(REGISTRY_PATH, REGISTRY_FILE)
    registry = SkillRegistry(registry_path=registry_path)
    stats = registry.get_stats()
    print(f"  Total skills loaded: {stats['total_skills']}")
    print(f"  Task class distribution: {stats['by_task_class']}")
    print(f"  Average confidence: {stats['avg_confidence']:.2f}")

    # Demonstrate skill lookup and TaskClass resolution
    print("\nDemonstrating Layer 2 → Layer 3 Integration:")
    print("-" * 80)

    test_skills = [
        "spa_translation",
        "abstractive_summ",
        "sentiment_analysis",
        "lang_id",
        "reverse_clm"
    ]

    for skill_id in test_skills:
        skill = registry.get_skill(skill_id)
        if skill:
            print(f"\nSkill: {skill.name}")
            print(f"  ID: {skill.skill_id}")
            print(f"  TaskClass: {skill.task_class.value}")
            print(f"  Difficulty: {skill.difficulty}")
            print(f"  Confidence: {skill.confidence:.2f}")
            print(f"  Domain: {skill.domain}")
            print(f"  Tags: {', '.join(skill.tags)}")

    # Initialize Skill Executor
    print("\n" + "-" * 80)
    print("Layer 1 → Layer 2 → Layer 3 Execution Flow:")
    print("-" * 80)

    executor = SkillExecutor(registry)

    test_inputs = [
        ("spa_translation", "Hello, how are you?"),
        ("sentiment_analysis", "This product is amazing!"),
        ("lang_id", "Buenos días"),
    ]

    for skill_id, text in test_inputs:
        result = executor.execute(skill_id, text)
        print(f"\nExecuted: {result['skill_name']}")
        print(f"  TaskClass: {result['task_class']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Batches: {result['batches_processed']}")

    # Demonstrate TaskClass-specific behavior
    print("\n" + "=" * 80)
    print("TASKCLASS-SPECIFIC BEHAVIOR:")
    print("=" * 80)

    # Test padding behavior for GENERATIVE vs REVERSE_GENERATIVE
    print("\nPadding Direction Test:")
    test_tokens = [[1, 2, 3], [4, 5]]
    pad_id = 0
    max_len = 4

    # GENERATIVE: append padding
    gen_padded = pad_tokens(test_tokens, pad_id, max_len, task_class=TaskClass.GENERATIVE)
    print(f"  GENERATIVE (append): {gen_padded}")

    # REVERSE_GENERATIVE: prepend padding
    rev_padded = pad_tokens(test_tokens, pad_id, max_len, task_class=TaskClass.REVERSE_GENERATIVE)
    print(f"  REVERSE_GENERATIVE (prepend): {rev_padded}")

    print("\n" + "=" * 80)
    print("SUCCESS: Three-Layer Architecture Integrated!")
    print("=" * 80)
    print("\nSummary:")
    print("  ✓ Layer 1: Routing Network (learns which skill)")
    print("  ✓ Layer 2: Skill Registry (JSON-backed metadata)")
    print("  ✓ Layer 3: TaskClass Enum (controls model behavior)")
    print("  ✓ Integration: Routing → Registry → TaskClass → Data Pipeline")
    print("\n")


if __name__ == "__main__":
    main()
    # model = torch.compile(self.model)
    # run live model
    raise NotImplementedError()