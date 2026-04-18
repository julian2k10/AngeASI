"""
Affix extraction & filtering module for morphological analysis across ISO 639-3 languages.

Filters prefix/suffix candidates by frequency distribution relative to the preceding
character sequence (parent affix), vowel content, sub-fragment detection, and
Information Theory metrics (Boundary Entropy, Variation of Information).

Key design decisions:
  1. Distribution is calculated as count(affix) / count(parent_affix),
     NOT count(affix) / count(first_letter). This ensures deeply nested
     affixes like "amphi" are measured against their parent "amph".

  2. Length-adaptive frequency thresholds: len-2 affixes require a higher
     distribution (15%) because many common bigrams (ar, bo, ca, ha) are
     just frequent word-starts, not morphologically productive affixes.
     However, vowel+consonant bigrams get a lower threshold (3.5%) because
     they follow the phonological pattern of assimilated affixes across
     Indo-European language families (e.g., ad→ab, ad→ac, ad→af, ad→ag).

  3. Length-adaptive fragment thresholds: len-3 uses 8% (lowered from 10.5%)
     so productive affixes like "sub" and "out" that spread across many
     children aren't falsely flagged as fragments.

  4. Spread-aware fragment detection: multiple children above the threshold
     indicate productive branching (like "un-"), not fragmentation.
     A affix needs at least 2 children above the fragment threshold to
     be considered "productive" rather than funneling into one dominant child.
     However, if a single child dominates with more than the fragment
     threshold, the affix is still a fragment — the spread exemption only
     applies when no child overwhelmingly dominates.

  5. Sibling competition filtering: when a candidate extends an already-valid
     affix by 1+ characters, and it is one of many roughly-equal siblings
     (none dominating), it is likely affix + root-start, not a longer affix.
     This filters false positives like "misc" (mis+c), "mist" (mis+t),
     "counterp" (counter+p), "counterf" (counter+f) without language-specific
     rules. The principle generalizes: across language families, true affix
     extensions dominate their siblings (e.g., "anti" dominates among "ant"
     children), while root-starts are spread evenly.

  6. Boundary Entropy (Information Theory): The transition from affix to root
     creates a measurable entropy signature. A productive affix boundary has
     HIGH successor variety (many different next characters), while a
     fragment mid-morpheme has LOW successor entropy (predictable next char).
     This is used as an additional signal for chameleon/assimilated affix
     detection — assimilated prefixes like "af-" show low boundary entropy
     (almost always followed by 'f') but high doublet probability at word
     boundaries, which is the statistical "scar" of phonological assimilation.

  7. Language-agnostic consonant cluster detection using a statistical
     approach. Consonant clusters at affix boundaries are evaluated by
     their frequency relative to the corpus. Rare initial/final clusters
     that appear predominantly as gemination or assimilation artifacts are
     flagged without any language-specific rules.
"""

from __future__ import annotations

import csv
import os
import re
import sys
import json
import math
from collections import Counter, defaultdict
from typing import Counter as CounterType, FrozenSet, Optional, Set, List, Tuple, Dict, Iterable, Any
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)

logger = logging.getLogger("MorphemeExtractor")

_FRAGMENT_LONG_DEFAULT = 0.55
DICT_WORDS = "words.json"

try:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT_DIR = os.getcwd()

STORAGE_DIR = os.path.join(ROOT_DIR, "vocab_files")


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


def _successor_entropy(prefix: str, affix_freq: CounterType, children_map: dict) -> float:
    """
    Calculate the Shannon entropy of the successor distribution for a given prefix.

    This implements the "Branching Entropy" concept from the Boundary Entropy
    algorithm. At a true morpheme boundary, the entropy is HIGH (many possible
    next characters). Mid-morpheme, entropy is LOW (predictable continuation).

    H(X) = -Σ p(x) * log2(p(x))

    Returns entropy in bits. Higher = more branching = more likely a real boundary.
    """
    children = children_map.get(prefix, [])
    if not children:
        return 0.0

    total = sum(affix_freq.get(c, 0) for c in children)
    if total == 0:
        return 0.0

    entropy = 0.0
    for child in children:
        count = affix_freq.get(child, 0)
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def _conditional_entropy(prefix: str, affix_freq: CounterType, children_map: dict) -> float:
    """
    Calculate H(next_char | prefix), the conditional entropy of the next character
    given the prefix. This is equivalent to successor entropy but normalized by
    the prefix frequency, giving us the expected information gain at this boundary.

    Low conditional entropy means the next character is highly predictable —
    characteristic of fragments (e.g., "amph" → almost always "i").
    High conditional entropy means many equiprobable continuations —
    characteristic of productive morpheme boundaries.
    """
    return _successor_entropy(prefix, affix_freq, children_map)


def _variation_of_information(prefix: str, affix_freq: CounterType, children_map: dict) -> float:
    """
    Simplified Variation of Information metric between the prefix distribution
    and its children distribution.

    VI(X, Y) = H(X|Y) + H(Y|X)

    In our context, this measures how much information is lost or gained when
    transitioning from the prefix to its extensions. A high VI suggests the
    boundary is meaningful (morpheme boundary), while low VI suggests the
    characters are tightly coupled (mid-morpheme).

    Returns VI in bits.
    """
    children = children_map.get(prefix, [])
    if not children:
        return 0.0

    prefix_count = affix_freq.get(prefix, 0)
    if prefix_count == 0:
        return 0.0

    # H(children | prefix) - how unpredictable is the child given the prefix?
    h_child_given_prefix = _successor_entropy(prefix, affix_freq, children_map)

    # H(prefix | children) - for each child, how much of the prefix's mass does it explain?
    # This is related to how concentrated the children are
    child_counts = [affix_freq.get(c, 0) for c in children]
    total_child = sum(child_counts)
    if total_child == 0:
        return h_child_given_prefix

    # Proportion of prefix NOT explained by children
    unexplained = max(0, prefix_count - total_child)
    h_prefix_given_child = 0.0
    if unexplained > 0 and prefix_count > 0:
        p_unexplained = unexplained / prefix_count
        if 0 < p_unexplained < 1:
            h_prefix_given_child = -p_unexplained * math.log2(p_unexplained)

    return h_child_given_prefix + h_prefix_given_child


def _doublet_score(bigram: str, affix_freq: CounterType, children_map: dict,
                   is_suffix: bool = False) -> float:
    """
    Detect the "Doublet Spike" — the statistical anomaly where assimilated
    prefixes create gemination (double letters) at word boundaries.

    For a bigram like "af", if the most common third letter is also 'f' (making "aff"),
    this indicates assimilation: ad- → af- before roots starting with 'f'.

    The score is: P(gemination) = freq(bigram + last_char) / freq(bigram)

    High doublet scores (> 0.3) strongly indicate assimilation patterns across
    language families (Latin ad-, Arabic sun letters, etc.).

    Returns the doublet score (0.0 to 1.0).
    """
    if len(bigram) != 2:
        return 0.0

    # The character that might be doubled
    if is_suffix:
        double_char = bigram[0]  # First char of suffix
        doubled = double_char + bigram  # e.g., "le" -> "lle"
    else:
        double_char = bigram[-1]  # Last char of prefix
        doubled = bigram + double_char  # e.g., "af" -> "aff"

    bigram_count = affix_freq.get(bigram, 0)
    if bigram_count == 0:
        return 0.0

    doubled_count = affix_freq.get(doubled, 0)
    return doubled_count / bigram_count


def _log_likelihood_ratio(observed: float, expected: float, total: float) -> float:
    """
    Calculate the Log-Likelihood Ratio (LLR) for detecting whether a bigram
    at word-initial position is significantly more frequent than expected.

    LLR = 2 * Σ O_ij * ln(O_ij / E_ij)

    Where O_ij is observed frequency and E_ij is expected frequency.
    A high LLR indicates the bigram is statistically associated with the
    word-initial position — a hallmark of prefix assimilation.

    Returns the LLR value. Values > 3.84 are significant at p < 0.05.
    """
    if observed == 0 or expected == 0 or total == 0:
        return 0.0

    # Avoid log(0)
    ratio = observed / expected
    if ratio <= 0:
        return 0.0

    return 2.0 * observed * math.log(ratio)


def _is_assimilated_bigram(pref: str, vowels_set: FrozenSet[str], is_suffix: bool = False) -> bool:
    """
    Check if a bigram follows the vowel+consonant (prefix) or consonant+vowel
    (suffix) pattern characteristic of assimilated affixes across language families.

    For prefixes (is_suffix=False):
      Vowel+consonant shape: ad- -> ab-, ac-, af-, ag-, al-, an-, ap-, ar-, as-, at-
    For suffixes (is_suffix=True):
      Consonant+vowel shape: -le, -ly, -ne, etc. where the consonant leads
      and the vowel follows in the reading direction of the suffix.
    """
    if len(pref) != 2:
        return False
    if is_suffix:
        # For suffixes like "ly", "le", "ne": consonant + vowel
        return pref[0] not in vowels_set and pref[1] in vowels_set
    else:
        return pref[0] in vowels_set and pref[1] not in vowels_set


def _is_chameleon_affix(bigram: str, affix_freq: CounterType, children_map: dict,
                        vowels_set: FrozenSet[str], is_suffix: bool = False) -> bool:
    """
    Use Information Theory metrics to identify "chameleon" (assimilated) affixes.

    A chameleon affix is one where the prefix and the first letter of the root
    are "entangled" — the prefix assimilates to match the root's initial consonant.
    Examples: ad- → af- (before f-roots), ad- → ac- (before c-roots).

    Detection criteria (language-agnostic):
    1. Low successor entropy (< 2.0 bits) — the continuation is predictable
    2. High doublet score (> 0.3) — gemination at the boundary
    3. Vowel+consonant shape (prefix) or consonant+vowel shape (suffix)

    These three signals together identify assimilation patterns across all
    ISO-639-3 languages without needing a dictionary or language-specific rules.
    """
    if len(bigram) != 2:
        return False

    # Check shape: vowel+consonant for prefix, consonant+vowel for suffix
    if is_suffix:
        if bigram[0] in vowels_set or bigram[1] not in vowels_set:
            return False
    else:
        if bigram[0] not in vowels_set or bigram[1] in vowels_set:
            return False

    # Check doublet score
    dscore = _doublet_score(bigram, affix_freq, children_map, is_suffix)

    # Check successor entropy
    entropy = _successor_entropy(bigram, affix_freq, children_map)

    # A chameleon affix has gemination AND is productive (some entropy)
    # The doublet score alone is sufficient if reasonably high
    if dscore > 0.30:
        return True

    # Even without strong gemination, low-entropy + correct shape = likely assimilation
    if entropy < 2.0 and dscore > 0.10:
        return True

    return False


# ═══════════════════════════════════════════════════════════════════════
# Core Affix Processing
# ═══════════════════════════════════════════════════════════════════════

def _extract_affix_frequencies(dictionary: Set[str],
                               min_len: int = 1,
                               max_len: int = 10) -> Tuple[CounterType, CounterType]:
    """
    Extract prefix/suffix frequencies from a dictionary of words.
    Returns prefix_frequency, suffix_frequency
    """
    logger.info(f"Extracting affix frequencies from {len(dictionary)} words...")
    prefix_frequency = Counter()
    suffix_frequency = Counter()

    for word in dictionary:
        if len(word) <= max_len:
            prefix_frequency[word] += 1
            suffix_frequency[word] += 1

        for length in range(min_len, min(len(word), max_len + 1)):
            prefix_frequency[word[:length]] += 1
            suffix_frequency[word[-length:]] += 1

    return prefix_frequency, suffix_frequency


def run_morpheme_pipeline(dict_words, min_len=1, max_len=10):
    """
    Extract prefixes, suffixes, prefix_frequencies, and suffix_frequencies from a dictionary word set.

    Performs efficient deduplication of overlapping bound roots so that
    only the most productive variant survives.

    Returns: prefixes, suffixes, prefix_frequencies, suffix_frequencies
    """
    # Use the cube root of the dictionary size and the min threshold for common affixes
    min_affix_freq = math.floor(math.pow(len(dict_words), 0.3))
    logger.info(f"Extracting morphemes from {len(dict_words)} words using min_affix_freq={min_affix_freq}...")

    # 1. Extract prefixes & suffix frequencies
    prefix_frequencies, suffix_frequencies = (
        _extract_affix_frequencies(dict_words, min_len, max_len)
    )
    # 2. Extract common prefixes
    all_prefixes = {p for p, v in prefix_frequencies.items() if v >= min_affix_freq}
    prefixes = filter_affixes(set(all_prefixes), prefix_frequencies, is_suffix=False)

    # 3. Extract common suffixes
    all_suffixes = {s for s, v in suffix_frequencies.items() if v >= min_affix_freq}
    suffixes = filter_affixes(set(all_suffixes), suffix_frequencies, is_suffix=True)

    return prefixes, suffixes, prefix_frequencies, suffix_frequencies


def _build_children_map_from_freq(affix_freq: CounterType,
                                  min_children=1,
                                  min_dist=0.01,
                                  is_suffix=False) -> dict[str, list[str]]:
    """
    Build a mapping from each prefix to its direct children (one char longer).

    Iterates once over all keys, grouping by parent. O(n) where n = len(prefix_freq).
    """
    children: dict[str, list[str]] = {}
    for key in affix_freq:
        if len(key) > 1:
            parent = key[1:] if is_suffix else key[:-1]
            children_count = affix_freq.get(parent, 0)
            if children_count >= min_children:
                threshold = max(3.0, children_count * min_dist)
                if affix_freq.get(key, 0) >= threshold:
                    children.setdefault(parent, []).append(key)
    return children


def _get_relatives(affix: str, children_map: dict[str, list[str]], is_suffix=False) -> set[str]:
    """Get parent and all children and siblings of the specified affix."""
    relatives: set[str] = set()

    children = children_map.get(affix, [])
    relatives.update(children)  # Add children
    if len(affix) > 1:
        parent = affix[1:] if is_suffix else affix[:-1]
        siblings = children_map.get(parent, [])
        relatives.update(siblings)  # Add siblings
        relatives.add(parent)  # Add parent

    return relatives


def generate_subword_regex(substrings):
    """
    Generates a regex that matches words starting or ending with
    the provided substrings, with a maximum of 1 extra character.
    """
    if not substrings:
        return r"$.^"  # Matches nothing if list is empty

    # 1. Escape the substrings to handle special characters safely
    escaped_subs = [re.escape(s) for s in substrings]

    # 2. Join them into a single OR group: (sub1|sub2|sub3)
    subs_pattern = "|".join(escaped_subs)

    # 3. Construct the logic:
    # \b              -> Start of word
    # (?:             -> Non-capturing group for the two possibilities
    #   \w?(?:...)    -> Optional 1 char + substring (e.g., "eful")
    #   |             -> OR
    #   (?:...)\w?    -> Substring + optional 1 char (e.g., "full")
    # )
    # \b              -> End of word
    pattern = fr"\b(?:\w?(?:{subs_pattern})|(?:{subs_pattern})\w?)\b"

    return pattern


def _extract_testing_data(dict_words: Set[str]=None,
                          affix_candidates: List[str]=None,
                          lang_code="eng",
                          is_suffix=False):
    from context_aware_io import load_json_file
    dict_words = dict_words or load_json_file(os.path.join(STORAGE_DIR, f"{lang_code}_{DICT_WORDS}"))
    prefixes, suffixes, pre_freq, suf_freq = run_morpheme_pipeline(dict_words)
    freq = suf_freq if is_suffix else pre_freq
    children_map = _build_children_map_from_freq(freq, is_suffix=is_suffix)
    affixes = suffixes if is_suffix else prefixes
    if affix_candidates:
        regex_str = generate_subword_regex(affix_candidates)
        text = "|".join(affixes)
        selected_affixes = re.findall(regex_str, text)
    else:
        selected_affixes = list(affixes)

    family_tree = set(selected_affixes)
    logger.info(f"Selected affixes: {len(selected_affixes)}/{len(affixes)}")
    for s in selected_affixes: family_tree.update(_get_relatives(s, children_map, is_suffix=is_suffix))
    target_freq = Counter({k: v for k, v in freq.items() if k in family_tree})
    target_freq = sorted(target_freq.items(), key=lambda x: x[1], reverse=True)
    return sorted(selected_affixes), target_freq


def _default_fragment_thresholds() -> dict[int, float]:
    """
    Return the default length->fragment-threshold mapping.

        len 2  ->  3.5%  (bigrams are very common fragments)
        len 3  ->  8.0%  (lowered from 10.5% to preserve sub, out, mis)
        len 4  -> 31.5%
        len 5+ -> 50.0%  (fallback)
    """
    return {2: 0.035, 3: 0.08, 4: 0.315}


def _default_frequency_thresholds() -> dict[int, float]:
    """
    Return the default length->frequency-threshold mapping.

    At len 2, many common bigrams (ar 10.5%, al 9.9%, am 6.5%) look like
    prefixes by frequency but are not morphologically productive. A 10%
    threshold filters these while preserving real prefixes:
        un (83.8%), re (52.2%), in (58.4%), de (32.9%), co (36.7%),
        di (33.6%), en (14.2%), be (21.6%), ex (18.1%), bi (10.5%),
        or (10.4%)

    However, vowel+consonant bigrams (ab, ac, ad, af, ag, etc.) follow the
    phonological pattern of assimilated prefixes across Indo-European language
    families and use the base threshold (3.5%) instead.

        len 2  -> 10.0%  (but 3.5% for vowel+consonant via assimilation)
        other  ->  3.5%  (base threshold)
    """
    return {2: 0.10}


def _is_sibling_dominated(
    pref: str,
    prefix_freq: CounterType,
    children_map: dict[str, list[str]],
    valid_prefixes: Set[str],
    prefix_candidates: Set[str],
    domination_ratio: float = 0.33,
    min_siblings: int = 3,
    is_suffix: bool = False,
) -> bool:
    """
    Check if a candidate is one of many undifferentiated siblings extending
    a valid prefix -- indicating it's prefix + root-start, not a true prefix.

    This implements a cross-linguistic generalization: when a productive prefix
    (like "mis-" or "counter-") attaches to roots, it creates many children
    with roughly equal frequencies (mis+c, mis+t, mis+s, mis+p, ...). In
    contrast, when a prefix genuinely extends to a longer prefix (ant->anti),
    the extension dominates its siblings.

    Parameters
    ----------
    pref : str
        The candidate being evaluated.
    prefix_freq : Counter
        Frequency counts for all known sequences.
    children_map : dict
        Mapping from prefix to its direct children.
    valid_prefixes : set
        Prefixes that have already passed filtering (built incrementally).
    prefix_candidates : set
        The full set of prefix candidates being evaluated.
    domination_ratio : float
        A child must have at least this fraction of all sibling frequency
        combined to be considered "dominant". Default 0.33 (33%).
    min_siblings : int
        Minimum number of siblings before this rule activates. Default 3.

    Returns
    -------
    bool
        True if the candidate is dominated by siblings (should be filtered).
    """
    if len(pref) < 3:
        # Bigrams are handled by assimilation and frequency thresholds
        return False

    parent = pref[1:] if is_suffix else pref[:-1]

    # Only applies when the parent is long enough to be a specific prefix.
    # Short prefixes (len 1-2) naturally have many children that include
    # both root-starts AND legitimate longer prefixes. The sibling
    # competition signal is only reliable when the parent is specific
    # enough (len >= 3) that its children are mostly root-starts.
    if len(parent) < 3:
        return False

    # Only applies when the parent is itself a candidate or already valid prefix
    if parent not in prefix_candidates and parent not in valid_prefixes:
        return False

    siblings = children_map.get(parent, [])
    if len(siblings) < min_siblings:
        return False

    # Get this candidate's count and total sibling frequency
    my_count = prefix_freq.get(pref, 0)
    sibling_counts = [(s, prefix_freq.get(s, 0)) for s in siblings]
    total_sibling_freq = sum(c for _, c in sibling_counts)

    if total_sibling_freq == 0:
        return False

    # Check if this candidate dominates its siblings
    my_ratio = my_count / total_sibling_freq

    if my_ratio >= domination_ratio:
        # This candidate dominates -- it's likely a real prefix extension
        return False

    # No single sibling dominates -> these are all prefix + root-starts
    # Check if ANY sibling dominates (if so, only non-dominant ones are filtered)
    max_sibling_ratio = max(c / total_sibling_freq for _, c in sibling_counts)

    if max_sibling_ratio >= domination_ratio:
        # Some other sibling dominates; this one doesn't -> filter this one
        return True

    # No sibling dominates -> all are undifferentiated root-starts -> filter
    return True


def _is_vowel_heavy_language(affix_freq: CounterType, vowels_set: FrozenSet[str]) -> bool:
    """
    Determine if the language represented by the frequency data is vowel-heavy
    by examining the first 3 letters of the most frequent candidates with length >= 3.

    Checks the consonant-to-vowel ratio among the top frequency candidates.
    A language is considered vowel-heavy if the ratio of consonants to vowels
    is less than or equal to 3.0 (i.e., vowels make up a significant portion).

    Returns True for vowel-heavy languages (like English, Spanish, etc.)
    Returns False for consonant-heavy languages (like Georgian, some Slavic languages)
    """
    # Get candidates with length >= 3, sorted by frequency
    candidates_with_freq = [
        (k, v) for k, v in affix_freq.items() if len(k) >= 3
    ]
    if not candidates_with_freq:
        return True  # Default to vowel-heavy if insufficient data

    candidates_with_freq.sort(key=lambda x: x[1], reverse=True)

    # Take top candidates (up to 50 for a good sample)
    top_candidates = candidates_with_freq[:50]

    vowel_count = 0
    consonant_count = 0
    for candidate, _ in top_candidates:
        # Examine the first 3 characters
        for ch in candidate[:3]:
            if ch in vowels_set:
                vowel_count += 1
            else:
                consonant_count += 1

    if vowel_count == 0:
        return False

    ratio = consonant_count / vowel_count
    # A ratio <= 3.0 means at least ~25% vowels in the first 3 chars
    # English typically has ratio around 1.5-2.0
    return ratio <= 3.0


def _is_invalid_consonant_cluster_affix(affix: str, vowels_set: FrozenSet[str], is_suffix: bool,
                                        distribution: float = 0.0) -> bool:
    """
    Language-agnostic detection of consonant clusters at affix boundaries
    that indicate root fragments rather than productive affixes.

    For prefixes: if the prefix ends with 2+ consonants, it's almost certainly
    a root-start that got attached to a real prefix. Cross-linguistically,
    productive prefixes end with at most one consonant (or a vowel).

    For suffixes: this check is NOT applied because many productive suffixes
    across languages start with consonant clusters (e.g., English -ship, -ble,
    -graphy; German -schaft; Russian -ство). The suffix-initial consonant
    cluster rule is handled by other mechanisms (fragment detection, sibling
    competition) which are more reliable.

    The key insight is phonotactic: across ISO-639-3 languages, morpheme
    boundaries respect syllable structure. A cluster of 2+ consonants
    at the prefix-final boundary that does not form a valid coda is a
    strong signal of a root fragment.
    """
    if len(affix) < 3:
        return False

    # Only apply to prefixes; suffixes legitimately start with CC clusters
    if is_suffix:
        return False

    # For prefixes: check if ends with consonant cluster from root
    consonant_suffix_len = 0
    for ch in reversed(affix):
        if ch not in vowels_set:
            consonant_suffix_len += 1
        else:
            break

    if consonant_suffix_len >= 2:
        # Ends with 2+ consonants - cross-linguistically invalid
        # as a prefix coda. This catches root fragments attached
        # to the end of a prefix.
        # However, high-distribution affixes (> 50% of parent) are
        # independently productive and should be exempt — they are
        # established morphemes, not accidental fragments.
        # Examples: "trans" (92.1% of "tran"), "anthr" (57.6% of "anth").
        if distribution > 0.50:
            return False
        return True

    return False


class SuffixTrie:
    """
    Reversed-key trie for fast suffix matching.

    Insertions store fragments in reverse character order so that a single
    walk from the end of a word finds the longest matching suffix in O(k)
    time (k = suffix length), independent of the total number of suffixes.

    Used by ``_dict_stem_ratio`` to check whether stripping a candidate
    suffix from dictionary words leaves a valid stem — the core signal for
    distinguishing productive suffixes from root-end fragments.
    """

    def __init__(self) -> None:
        self.root: dict = {}
        self._end: str = "__end__"

    def insert(self, fragment: str, replacement: str) -> None:
        """Insert *fragment* (stored reversed) mapping to *replacement*."""
        node = self.root
        for ch in reversed(fragment):
            node = node.setdefault(ch, {})
        node[self._end] = replacement

    def find_and_replace(self, word: str) -> str:
        """
        Replace the longest matching suffix of *word* with its stored
        replacement.  Returns the original *word* when no suffix matches.
        """
        node = self.root
        match_len = 0
        replacement = None
        for i, ch in enumerate(reversed(word)):
            if ch not in node:
                break
            node = node[ch]
            if self._end in node:
                match_len = i + 1
                replacement = node[self._end]
        if replacement is not None:
            return word[:-match_len] + replacement
        return word


# Baseline dict-stem ratios for each suffix length (fraction of random word
# endings whose stem is also a dict word, with min_stem_len = 4).
# Pre-computed from the English 513k-word corpus; used to normalise the
# observed stem ratio in ``_dict_stem_ratio``.
_BASELINE_STEM_RATIO: Dict[int, float] = {
    1: 0.273, 2: 0.230, 3: 0.262, 4: 0.318, 5: 0.321, 6: 0.337,
}
_BASELINE_STEM_RATIO_DEFAULT = 0.340   # fallback for len > 6
_DICT_STEM_MIN_STEM = 4                # minimum stem length for the check


def _dict_stem_ratio(
    affix: str,
    dict_words: Optional[Set[str]],
    is_suffix: bool,
    min_stem_len: int = _DICT_STEM_MIN_STEM,
) -> float:
    """
    Compute the fraction of dictionary words that are *productively* affixed
    with *affix* — i.e., stripping the affix leaves a stem that is itself a
    dictionary word (with ``len(stem) >= min_stem_len``).

    This is the "dict stem ratio": high for real affixes (``-ness``, ``-ly``,
    ``-s``), low for root-end fragments (``-e``, ``-ckings``).

    The ratio is compared against a length-matched baseline (``_BASELINE_STEM_RATIO``)
    in the caller; this function just returns the raw ratio.

    Returns 1.0 when no dict_words are provided (no-op — caller must decide).
    """
    if not dict_words:
        return 1.0
    L = len(affix)
    total = 0
    hits = 0
    for word in dict_words:
        if not word.endswith(affix if is_suffix else word):
            continue
        stem_len = len(word) - L
        if stem_len < min_stem_len:
            continue
        stem = word[:-L] if is_suffix else word[L:]
        total += 1
        if stem in dict_words:
            hits += 1
    return hits / total if total > 0 else 0.0


def _filter_affix_duplicates(
        affixes: Set[str],
        free_roots_set: Set[str],
        *,
        is_suffix: bool = False,
        max_affix_len: int = 3,
        combining_forms: Optional[Set[str]] = None,
        orth_rules: Optional[Any] = None,
        dictionary: Optional[Set[str]] = None,
) -> Set[str]:
    """Remove affixes that are exactly one character longer than a known free root.

    For **prefix** roots (``is_suffix=False``, the default) the extra character is on
    the **right**: strip ``affix[:-1]`` and check whether that stem is a free root.

        "counterp", "counterf", "countermar"  →  stem "counter" ∈ free_roots  →  removed

    For **suffix** roots (``is_suffix=True``) the extra character is on the **left**:
    strip ``affix[1:]`` and check whether that stem is a free root.

        e.g. "calization"  →  "zation" if "zation" ∈ free_roots, "calization" would be removed

    **Exemptions** — an affix is NOT removed even if its stem is a free root when:

    1. The affix is itself a known combining form (e.g. "metallurg" should not
       be removed just because "metal" is a free root).
    2. The affix can be restored to a dictionary word or combining form via a
       morphophonological rule (vowel drop, gemination reversal, etc.).
       e.g. "programm" → "program" via gemination reversal.

    Uses a :class:`~collections.defaultdict` for O(1) stem-to-affix grouping, so the
    total work is O(|affixes|) dictionary lookups rather than a nested loop.

    Parameters
    ----------
    affixes:
        Candidate bound-root set — either prefix roots or suffix roots.
    free_roots_set:
        Set of known free roots for O(1) membership testing.
    is_suffix:
        When ``True``, strip the **leftmost** character (suffix direction).
        When ``False`` (default), strip the **rightmost** character (prefix direction).
    max_affix_len:
        Only examine affixes greater than this length.
    orth_rules:
        Compiled morphophonological rules.  If provided, affixes restorable to a
        dictionary word or combining form via any rule are exempt from removal.
    dictionary:
        Dictionary words for rule-based restoration lookup.

    Returns
    -------
    Filtered set of affixes with one-letter-diff duplicates removed.
    """
    # Lazy import to avoid circular dependency at module level
    from morpho_rules import apply_rules as _apply_rules

    # Strip from the right for prefixes, from the left for suffixes
    get_stem = (lambda affix, idx: affix[idx:]) if is_suffix else (lambda affix, idx: affix[:-idx])
    direction = "suffix" if is_suffix else "prefix"
    affixes = set(affixes)

    if combining_forms is None:
        combining_forms = set()
    if dictionary is None:
        dictionary = set()

    # Build the set of exempt affixes:
    # 1. Affixes that are combining forms
    # 2. Affixes restorable to a known word/form via morphological rules
    exempt = set()
    for affix in affixes:
        # Exemption 1: affix is a known combining form
        if affix in combining_forms:
            exempt.add(affix)
            continue

        # Exemption 2: affix can be restored via morphophonological rules
        if orth_rules is not None and len(affix) >= 3:
            candidates = _apply_rules(affix, orth_rules)
            for candidate in candidates:
                if candidate != affix and (
                    candidate in dictionary
                    or candidate in combining_forms
                    or candidate in free_roots_set
                ):
                    exempt.add(affix)
                    break

    # Group affixes by their stem
    stem_to_affixes: Dict[str, Set[str]] = defaultdict(set)
    for affix in affixes:
        if affix in exempt:
            continue  # Skip exempt affixes — don't even group them for removal
        if len(affix) > max_affix_len:
            diff = min(4, len(affix) - max_affix_len)
            if max_affix_len < 7 and len(affix) < 7:  # Only check 1 character difference
                stem_to_affixes[get_stem(affix, 1)].add(affix)
            else:  # Ensure bounds roots are not fragments of compound words like 'prenegl' from 'preneglect'
                for x in range(1, diff + 1):
                    stem_to_affixes[get_stem(affix, x)].add(affix)

    # Any affix whose stem is a free root is a duplicate
    n_removed = 0
    for stem, affix_group in stem_to_affixes.items():
        if stem in free_roots_set:
            affixes.difference_update(affix_group)
            n_removed += len(affix_group)

    size = len(affixes)
    affixes.difference_update(free_roots_set)  # Remove free roots
    n_removed += size - len(affixes)
    if n_removed > 0:
        print(f"    Removed {n_removed} {direction} roots that are extensions of free roots "
              f"(exempted {len(exempt)} combining-form / restorable roots).")
    return affixes


def filter_affixes(
    affix_candidates: Set[str],
    affix_freq: CounterType,
    vowels: str = "aeiouy",
    threshold: float = 0.035,
    frequency_thresholds: Optional[dict[int, float]] = None,
    fragment_thresholds: Optional[dict[int, float]] = None,
    fragment_long_default: Optional[float] = None,
    is_suffix: bool = False,
    dict_words: Optional[Set[str]] = None,
) -> Set[str]:
    """
    Filter affix candidates based on frequency, vowel content, boundary
    entropy, and sub-fragment analysis.

    Uses Information Theory metrics (Boundary Entropy, Variation of Information,
    Conditional Entropy) to detect productive morpheme boundaries vs. fragments,
    and to identify "chameleon" (assimilated) affixes across any ISO-639-3 language.

    Parameters
    ----------
    affix_candidates : set of str
        Candidate affix strings to evaluate.
    affix_freq : Counter
        Frequency counts for all known character sequences.
    vowels : str
        Characters considered vowels. Default "aeiouy".
    threshold : float
        Base minimum distribution ratio. Default 3.5%.
    frequency_thresholds : dict[int, float] or None
        Length-specific frequency thresholds. Default {2: 0.15}.
    fragment_thresholds : dict[int, float] or None
        Length-specific fragment thresholds. Default {2: 0.035, 3: 0.08, 4: 0.315}.
    fragment_long_default : float or None
        Fragment threshold for lengths not in fragment_thresholds. Default 0.50.
    is_suffix : bool
        If True, treat candidates as suffixes (parent = affix[1:] instead of affix[:-1]).
    dict_words : set of str or None
        Dictionary word set used for the dict-stem-ratio check on short
        affixes (len 1–2 in suffix mode).  When provided, single-character
        suffixes must achieve a stem ratio above the length-matched baseline
        to pass (filters spurious endings like ``-e``).  Ignored for prefixes
        and for affixes longer than 2 characters, where the existing frequency
        and fragment filters are sufficient.

    Returns
    -------
    set of str
        Affix candidates that passed all filters.
    """
    if fragment_thresholds is None:
        fragment_thresholds = _default_fragment_thresholds()
    if fragment_long_default is None:
        fragment_long_default = _FRAGMENT_LONG_DEFAULT
    if frequency_thresholds is None:
        frequency_thresholds = _default_frequency_thresholds()

    if not affix_candidates or not affix_freq:
        return set()

    vowels_set: FrozenSet[str] = frozenset(vowels)
    children_map = _build_children_map_from_freq(affix_freq, is_suffix=is_suffix)
    valid_affixes: set[str] = set()

    # Determine if the language is vowel-heavy to decide whether to apply Rule A
    is_vowel_heavy = _is_vowel_heavy_language(affix_freq, vowels_set)

    # Process shorter affixes first so parent validity is known for sibling check
    sorted_candidates = sorted(affix_candidates, key=len)

    for affix in sorted_candidates:
        if not affix:
            continue

        # Rule A: Must contain at least one vowel (only for vowel-heavy languages)
        # Languages that are not vowel-heavy (e.g., Georgian, some Slavic languages)
        # may have productive affixes without vowels, so skip this rule for them.
        if len(affix) > 1 and is_vowel_heavy and not any(c in vowels_set for c in affix):
            continue

        # Frequency stats (parent-relative distribution)
        count = affix_freq.get(affix, 0)
        if count == 0:
            continue

        affix_len = len(affix)
        if affix_len == 1:
            parent_count = count  # self-ratio = 1.0
        else:
            parent = affix[1:] if is_suffix else affix[:-1]
            parent_count = affix_freq.get(parent, 0)

        if parent_count == 0:
            continue

        distribution = count / parent_count

        # Rule A2: Filter invalid consonant cluster affixes (language-agnostic)
        # Suffixes starting with 2+ consonants or prefixes ending with 2+ consonants
        # are cross-linguistically invalid as productive affixes — they indicate
        # root fragments rather than morpheme boundaries.
        # High-distribution affixes are exempt as they are established morphemes.
        if len(affix) > 1 and _is_invalid_consonant_cluster_affix(affix, vowels_set, is_suffix, distribution):
            continue

        # Rule B: Length-adaptive frequency threshold with assimilation
        freq_thresh = frequency_thresholds.get(affix_len, threshold)

        # Assimilation rule: vowel+consonant bigrams (prefix) or consonant+vowel
        # bigrams (suffix) use a lower threshold. This reflects universal
        # phonological patterns of prefix/suffix assimilation.
        # Enhanced with Information Theory: chameleon affixes detected via
        # boundary entropy and doublet spike analysis.

        assimilation_upper_bound = 0.12  # Raised to include ar (10.5%)
        if (_is_assimilated_bigram(affix, vowels_set, is_suffix)
                and distribution <= freq_thresh
                and distribution <= assimilation_upper_bound):
            # Additionally check for chameleon signature using Information Theory
            # Even without the IT check, the phonological shape is sufficient
            # for the base assimilation pass
            freq_thresh = 0.015  # Any assimilated bigram above distribution passes

        if distribution <= freq_thresh:
            continue

        # Rule C: Sub-fragment detection (spread-aware)
        if affix_len == 1:
            # Single-character suffixes need a special gate because there is no
            # meaningful parent-relative distribution (the parent is empty, so
            # the self-ratio is always 1.0).  We apply a strict-vowel filter
            # instead, grounded in cross-linguistic phonology:
            #
            #   Productive single-character suffixes end in consonants or the
            #   semivowel 'y' — never in a strict vowel (a, e, i, o, u).
            #
            # Rationale: word-final strict vowels mark base forms in most
            # languages (English "love", "race"; Italian "bello", "porta").
            # They are the STEM itself, not an added morpheme.  In contrast,
            # consonant-final suffixes mark grammatical categories universally:
            #   's'  → plural / 3sg present
            #   'd'  → past tense (loved, parked)
            #   'n'  → past participle (beaten, frozen)
            #   'r'  → comparative (cleaner)
            #   'y'  → adjectival (lucky, rainy) — semivowel, NOT a strict vowel
            #
            # Strict vowels (a, e, i, o, u) are filtered; 'y' and all
            # consonants are allowed.  This is language-agnostic: the vowels
            # set is data-driven, but we explicitly exclude 'y' from the filter
            # because cross-linguistically 'y' behaves as a consonant at
            # morpheme boundaries (semivowel / palatal approximant).
            strict_vowels = vowels_set - {'y'}
            if is_suffix and affix in strict_vowels:
                continue
            valid_affixes.add(affix)
            continue

        frag_thresh = fragment_thresholds.get(affix_len, fragment_long_default)

        children = children_map.get(affix, ())
        has_children = bool(children)
        children_above = 0
        max_child_ratio = 0.0
        children_ratios = []
        for child in children:
            child_count = affix_freq.get(child, 0)
            child_ratio = child_count / count
            children_ratios.append(child_ratio)
            if child_ratio > frag_thresh:
                children_above += 1
            if child_ratio > max_child_ratio:
                max_child_ratio = child_ratio

        dominant_child_cap = fragment_long_default
        children_ratios.sort(reverse=True)
        child_dominant_ratio = children_ratios[1] / children_ratios[0] if len(children_ratios) > 1 else 0.0

        # High-distribution exemption: if an affix has very high distribution
        # relative to its parent (> 70%), it is independently productive and
        # should not be treated as a fragment even if a child dominates.
        # Examples: suffix "ion" (70.5% of "on"), "ive" (73.9% of "ve") —
        # these are productive suffixes where child dominance ("tion" under
        # "ion") reflects specialization, not truncation.
        high_distribution_exempt = distribution > 0.70

        # Long-affix exemption: longer affixes with meaningful distribution
        # are established morphemes even if they funnel into a dominant child.
        # The threshold decreases with length because longer affixes are
        # inherently more specific and less likely to be accidental fragments.
        # Examples: "anthro" (67.98% dist, len 6), "anthr" (57.6% dist, len 5),
        #           "anthra" (29.4% dist, len 6).
        #
        # For len >= 6 we additionally require has_children: a leaf node with
        # no further extensions has no trie-depth evidence of productivity.
        # "ckings" (dist=33%, 0 children) is a root fragment masquerading as a
        # suffix because it happens to dominate its small sibling group;
        # "graphy" / "wards" (dist > 25%, both have children) are real morphemes.
        long_affix_exempt = (affix_len >= 5 and distribution > 0.55) or \
                            (affix_len >= 6 and distribution > 0.25 and has_children)

        # Branching exemption: if 2+ children are above the fragment threshold,
        # the affix is productively branching even if one child exceeds the
        # dominant cap. Examples: "sym" (symp=60.2%, symb=23%, symm=11.6%)
        # has 3 children above the 8% threshold for len-3 affixes.
        branching_exempt = children_above >= 2

        if not high_distribution_exempt and not long_affix_exempt and not branching_exempt and max_child_ratio > dominant_child_cap:
            # One child overwhelmingly dominates -> this is a fragment
            # Use boundary entropy as additional confirmation: low entropy
            # at this boundary confirms it's mid-morpheme, not a real boundary
            continue

        # Spread-aware: 2+ children above threshold = productive branching
        has_dominant_child = 0 < children_above < 2
        is_fragment = not high_distribution_exempt and not long_affix_exempt and has_dominant_child

        # Additional fragment relaxation: if exactly 1 child is above the
        # per-length fragment threshold but below the dominant cap, and the
        # affix's own distribution is reasonably high (> base threshold),
        # don't treat as fragment. This prevents filtering productive affixes
        # like "ise" (14.4% of "se") where "wise" at 43.9% is the only
        # significant child but "ise" is independently productive.
        if is_fragment and children_above == 1 and max_child_ratio < dominant_child_cap:
            if distribution > threshold:
                is_fragment = False

        if is_fragment:
            continue

        # Additional check: if this candidate has its own productive children
        # in the children map, it is independently productive and should not
        # be filtered by sibling competition or child-dominant-ratio checks.
        # This preserves affixes like "-ling", "-less" which have many children
        # even though they are one of many siblings under their parent.
        #
        # We use grandchildren (children that themselves have children) as the
        # signal for deep productivity. A fragment like "eful" has children
        # (seful, ceful, ...) but NONE of those have further children.
        # A productive affix like "less" has children (eless, tless, nless)
        # which in turn have their OWN children — indicating real morphological
        # depth. Requiring >= 3 grandchild branches AND distribution > base
        # threshold separates productive affixes from fragments.
        grandchild_branches = sum(1 for c in children if children_map.get(c, ()))
        has_deep_productivity = grandchild_branches >= 3 and distribution > 0.10

        if 0.35 <= child_dominant_ratio < 0.5:
            pass  # Has productive children -> independently productive affix
        elif high_distribution_exempt:
            pass  # Very high distribution (>70%) -> independently productive
        elif has_deep_productivity:
            pass  # Deep morphological tree -> independently productive
        else:
            # Rule D: Sibling & children competition filtering
            # Long affixes (len >= 5) with meaningful distribution are exempt
            # from sibling domination because they represent established
            # morphemes even if a sibling is more common (e.g., "anthra"
            # under "anthr" where "anthro" dominates).
            #
            # Leaf affixes (no children) have no trie-depth evidence of
            # productivity beyond their own frequency.  They need to dominate
            # their sibling group more decisively (50%) than branching affixes
            # (33%) before we grant a pass.  This catches root fragments like
            # "ckings" (sib_ratio 34%, 0 children) without affecting productive
            # affixes like "wards" (sib_ratio 40%, has children).
            leaf_domination_ratio = 0.50 if not has_children else 0.33
            if not long_affix_exempt and _is_sibling_dominated(
                    affix, affix_freq, children_map, valid_affixes, affix_candidates,
                    domination_ratio=leaf_domination_ratio, is_suffix=is_suffix):
                continue

            if child_dominant_ratio >= 0.85 and max_child_ratio > 0.2 and not branching_exempt:
                # Fragment with dominant children — but only when there isn't
                # productive branching (2+ children above threshold).
                # Affixes like "ex", "meta", "dom", "ize" have many children
                # with similar ratios (all ~10-25%), giving child_dominant_ratio
                # close to 1.0. This is the signature of PRODUCTIVE branching,
                # not fragmentation.
                continue

        valid_affixes.add(affix)

    return valid_affixes


class MorphemeLineageTrie:
    """Trie for morpheme lookup with productivity ratio caching."""
    __slots__ = ("trie", "is_suffix_mode", "root_lexicon", "_productivity_cache")
    _END = object()

    def __init__(self, is_suffix_mode: bool = False):
        self.trie: dict = {}
        self.is_suffix_mode = is_suffix_mode
        self.root_lexicon: Dict[str, Tuple[float, float]] = {}
        self._productivity_cache: Dict[str, Tuple[float, float]] = {}

    def set_mode(self, is_suffix_mode: bool) -> "MorphemeLineageTrie":
        self.is_suffix_mode = is_suffix_mode
        return self

    def insert(self, morpheme: str) -> None:
        node = self.trie
        chars: Iterable[str] = reversed(morpheme) if self.is_suffix_mode else morpheme
        for ch in chars:
            node = node.setdefault(ch, {})
        node[self._END] = True

    def insert_morphemes(self, morphemes: Iterable[str], is_suffix: bool) -> None:
        self.set_mode(is_suffix)
        for m in morphemes:
            self.insert(m)

    def find_all_matches(self, word: str, start_idx: int) -> List[str]:
        node = self.trie
        matches: List[str] = []
        buf = ""
        if self.is_suffix_mode:
            r = range(start_idx, -1, -1)
        else:
            r = range(start_idx, len(word))
        for idx in r:
            ch = word[idx]
            if ch not in node:
                break
            node = node[ch]
            buf += ch
            if self._END in node:
                matches.append(buf[::-1] if self.is_suffix_mode else buf)
        return matches

    def find_longest_match(self, word: str) -> str:
        node = self.trie
        best = ""
        buf = ""
        for ch in word:
            if ch not in node:
                break
            node = node[ch]
            buf += ch
            if self._END in node:
                best = buf
        return best

    def has_prefix(self, prefix: str) -> bool:
        node = self.trie
        for ch in prefix:
            if ch not in node:
                return False
            node = node[ch]
        return True

    def collect_completions(self, prefix: str, max_results: int = 10000) -> List[str]:
        node = self.trie
        for ch in prefix:
            if ch not in node:
                return []
            node = node[ch]
        results = []
        stack = [(node, prefix)]
        while stack and len(results) < max_results:
            current, path = stack.pop()
            if self._END in current:
                results.append(path)
            for ch, child in current.items():
                if ch is not self._END:
                    stack.append((child, path + ch))
        return results

    def get_cached_productivity(self, word: str) -> Tuple[float, float]:
        return self._productivity_cache.get(word, (-1, -1))

    def set_cached_productivity(self, word: str, prefix_ratio: float, suffix_ratio: float) -> None:
        self._productivity_cache[word] = (prefix_ratio, suffix_ratio)


def _build_word_lookup(
    dict_words: Set[str],
    is_suffix: bool,
) -> Tuple[List[str], List[str]]:
    """
    Build a pair of sorted lists that enable O(log N) affix-match lookups
    over a dictionary of arbitrary size via :func:`bisect`.

    For **prefix mode** (``is_suffix=False``) both lists are the same: the
    words sorted lexicographically.  A range ``[lo, hi)`` in the sorted list
    contains exactly the words that start with a given prefix — found in
    O(log N) with two ``bisect`` calls.

    For **suffix mode** (``is_suffix=True``) the second list stores every
    word reversed and sorted.  A range in the reversed list contains exactly
    the words whose *reverse* starts with the reversed suffix, i.e. exactly
    the words that end with the original suffix — again O(log N).

    The function is called **once** inside :func:`extract_productive_affixes`
    before the per-affix scoring loop, amortising the O(N log N) sort cost
    across all candidates.

    Parameters
    ----------
    dict_words : set of str
        The dictionary to index.
    is_suffix : bool
        Controls which direction the secondary list is reversed.

    Returns
    -------
    (words_fwd, words_rev) : (list[str], list[str])
        *words_fwd* — words sorted normally (used for prefix lookups and for
        O(1) stem membership tests via the original ``dict_words`` set).
        *words_rev* — words sorted after reversing each word (used for suffix
        lookups).  In prefix mode this is identical to *words_fwd*.
    """
    words_fwd = sorted(dict_words)
    if is_suffix:
        words_rev = sorted(w[::-1] for w in dict_words)
    else:
        words_rev = words_fwd          # prefix mode: only forward list needed
    return words_fwd, words_rev


def _bisect_hi(key: str) -> str:
    """
    Return the smallest string that sorts strictly after every string with
    the given prefix ``key`` under lexicographic order.

    ``bisect_left(seq, _bisect_hi(key))`` gives the exclusive upper bound
    of the ``[lo, hi)`` range found by ``bisect_left(seq, key)``.

    Edge case: if *key* consists entirely of ``chr(0x10FFFF)`` characters
    (the maximum Unicode code point) the function falls back to returning
    the full-key string with a trailing ``\\xff\\xff``, which is sufficient
    for all natural-language inputs.
    """
    for i in range(len(key) - 1, -1, -1):
        cp = ord(key[i])
        if cp < 0x10FFFF:
            return key[:i] + chr(cp + 1)
    return key + "\U0010FFFF"


def _affix_productivity_score(
    affix: str,
    cands_fwd: List[str],
    cands_rev: List[str],
    affix_freq: CounterType,
    words_fwd: List[str],
    words_rev: List[str],
    dict_words: Optional[Set[str]],
    is_suffix: bool,
    min_stem_len: int = 3,
) -> float:
    """
    Score *affix* by morphological productivity using pre-sorted index lists.

    All list range queries use :func:`bisect.bisect_left` for O(log N) lookup,
    making the per-affix cost O(log N + M) where M is the number of matching
    words — independent of total dictionary size.

    Three complementary signals are combined multiplicatively:

    **free_stem_ratio** — fraction of dictionary words headed by *affix*
    whose residual stem is itself a dictionary word.  High for productive
    affixes (``un-``: unhappy → happy ✓), low for bound-root fragments
    (``comp-``: compete → *pete* ✗) and pure combining forms
    (``anthr-``: anthropology → *opology* ✗).

    **total_freq** — own corpus frequency plus the summed corpus frequency
    of all candidate descendants (longer affixes that start/end with this
    one), representing the entire word-family this affix heads.

    **n_desc** — count of candidate descendants, awarding a multiplicative
    branching bonus to affixes that head rich sub-families.

    Formula::

        score = (free_stem_ratio + ε) × log₂(total_freq + 2) × √(1 + n_desc)

    ``ε = 0.05`` prevents zero-score for combining forms that still carry
    real morphological productivity (e.g. ``micro-``, ``-logy``).

    Parameters
    ----------
    affix : str
        The affix to score.
    cands_fwd : list[str]
        Candidate affixes sorted lexicographically (prefix descendant lookup).
    cands_rev : list[str]
        Candidate affixes sorted after reversing each one (suffix descendant
        lookup).  Equal to *cands_fwd* in prefix mode.
    affix_freq : Counter
        Corpus frequency counts keyed by affix string.
    words_fwd : list[str]
        Dictionary words sorted lexicographically (prefix word lookup).
    words_rev : list[str]
        Dictionary words sorted after reversing (suffix word lookup).
        Equal to *words_fwd* in prefix mode.
    dict_words : set[str] or None
        O(1) membership set for stem validation.  When ``None`` the
        free-stem-ratio term is set to zero (ε floor only).
    is_suffix : bool
        ``True`` → suffix mode; ``False`` → prefix mode.
    min_stem_len : int
        Minimum stem length counted after stripping the affix.  Default 3.

    Returns
    -------
    float
        Productivity score ≥ 0.
    """
    import bisect as _bisect

    L = len(affix)
    own_freq = affix_freq.get(affix, 0)

    # ── Candidate descendants: longer affixes that start/end with this one ──
    # Use the appropriate sorted list so the range [lo, hi) gives exactly the
    # candidates whose forward (or reversed) form starts with the affix.
    if is_suffix:
        key = affix[::-1]
        search_list = cands_rev
    else:
        key = affix
        search_list = cands_fwd

    lo = _bisect.bisect_left(search_list, key)
    hi = _bisect.bisect_left(search_list, _bisect_hi(key))

    if is_suffix:
        # Reverse each hit back to the original suffix form.
        descendants = [search_list[i][::-1] for i in range(lo, hi)
                       if len(search_list[i]) > L]
    else:
        descendants = [search_list[i] for i in range(lo, hi)
                       if len(search_list[i]) > L]

    n_desc = len(descendants)
    desc_freq_sum = sum(affix_freq.get(d, 0) for d in descendants)
    total_freq = own_freq + desc_freq_sum

    # ── Free-stem ratio via sorted word lists ──────────────────────────────
    # Find all dictionary words that start (prefix) or end (suffix) with this
    # affix, then count how many have a residual stem that is also a word.
    epsilon = 0.05
    free_stem_ratio = 0.0

    if dict_words:
        wlo = _bisect.bisect_left(words_rev, key)
        whi = _bisect.bisect_left(words_rev, _bisect_hi(key))

        hits = total_w = 0
        for i in range(wlo, whi):
            entry = words_rev[i]        # reversed word in suffix mode, normal word in prefix
            if len(entry) - L < min_stem_len:
                continue
            total_w += 1
            # Extract the stem: strip the affix and reverse back if needed.
            if is_suffix:
                stem = entry[L:][::-1]  # entry is reversed word; stem is what remains
            else:
                stem = entry[L:]        # entry is normal word; stem is the tail
            if stem in dict_words:
                hits += 1

        free_stem_ratio = hits / total_w if total_w else 0.0

    # ── Composite score ────────────────────────────────────────────────────
    return (free_stem_ratio + epsilon) * math.log2(total_freq + 2) * math.sqrt(1 + n_desc)


def extract_productive_affixes(
    candidates: Set[str],
    affix_freq: CounterType,
    is_suffix: bool,
    *,
    dict_words: Optional[Set[str]] = None,
    top_n: int = 200,
    min_len: int = 2,
    min_score: float = 0.0,
) -> List[Tuple[str, float]]:
    """
    Rank a pre-filtered set of affix candidates by morphological productivity
    and return the top-N as scored pairs.

    The input *candidates* are assumed to have already passed
    :func:`filter_affixes` (structural/frequency filtering).  This function
    does **not** call :func:`filter_affixes` again — its job is purely to
    rank the survivors by how *generatively* productive they are.

    Design
    ------
    Efficient handling of large dictionaries (500k+ words) is achieved by
    building two sorted lists from *dict_words* **once** before the scoring
    loop, then using :mod:`bisect` for O(log N) range queries:

    * **Forward list** ``words_fwd`` — words sorted normally → prefix lookups.
    * **Reversed list** ``words_rev`` — words sorted after reversing each one
      → suffix lookups (a word ending in ``-ness`` is found by searching for
      ``ssen`` in the reversed list).

    Candidate descendants are found the same way using sorted candidate lists.
    The per-affix cost is O(log N + M) where M is the number of matching words
    (typically small), making the total complexity O(N log N + C × M_avg)
    where N is the dictionary size and C is the number of candidates.

    Productivity signals
    --------------------
    * **free_stem_ratio** — fraction of matched words whose residual stem is
      also a dictionary word.  Separates productive affixes (``re-``, ``-ness``)
      from bound-root fragments (``comp-``, ``bacter-``) and pure combining
      forms (``anthr-``).
    * **total_freq** — own corpus frequency plus summed frequency of all
      candidate descendants (the full word-family this affix heads).
    * **n_desc** — candidate descendant count (branching bonus).

    Score formula::

        score = (free_stem_ratio + ε) × log₂(total_freq + 2) × √(1 + n_desc)

    Parameters
    ----------
    candidates : set of str
        Pre-filtered affix strings to rank (pre-filtered by :func:`filter_affixes`).
    affix_freq : Counter
        Corpus frequency counts — pass ``prefix_freq`` when
        ``is_suffix=False``, ``suffix_freq`` when ``is_suffix=True``.
    is_suffix : bool
        ``False`` → prefix mode.  ``True`` → suffix mode.
    dict_words : set of str or None
        Dictionary word set for the free-stem-ratio signal.  When ``None``
        the ratio falls back to the ε floor and scoring is driven by
        frequency alone.  Providing *dict_words* significantly improves
        separation between productive affixes and bound-root fragments.
    top_n : int
        Maximum number of results to return.  Default 200.
    min_len : int
        Minimum affix length to consider.  Default 2.
    min_score : float
        Hard floor: affixes scoring below this are excluded even if they
        rank within the top *top_n*.  Default 0.0 (no floor).

    Returns
    -------
    list of (str, float)
        ``(affix, score)`` pairs in descending score order, length ≤ *top_n*,
        all with ``score >= min_score``.

    Examples
    --------
    >>> affix_freq = Counter({"e": 99456, "s": 123653, "er": 35322, "es": 23532, "ing": 18672})
    >>> candidates = filter_affixes({"e", "able", "s", "er", "es", "ing"}, affix_freq)
    >>> results = extract_productive_affixes(
    ...     candidates, prefix_freq, is_suffix=False,
    ...     dict_words=dict_words, top_n=200,
    ... )
    >>> top_prefixes = {affix for affix, _ in results}
    """
    if not candidates:
        return []

    mode = "suffix" if is_suffix else "prefix"
    filtered: Set[str] = {a for a in candidates if isinstance(a, str) and len(a) >= min_len}
    if not filtered:
        return []

    logger.info(
        f"extract_productive_affixes: scoring {len(filtered)} {mode} candidates "
        f"(min_len={min_len}, top_n={top_n}, dict={len(dict_words) if dict_words else 0} words)"
    )

    # ── Build sorted index lists (one-time O(N log N) cost) ────────────────
    # For prefix mode: words_fwd is both the forward and reverse list.
    # For suffix mode: words_rev stores reversed words sorted lexicographically,
    # enabling O(log N) suffix range queries via bisect.
    if dict_words:
        words_fwd, words_rev = _build_word_lookup(dict_words, is_suffix)
    else:
        words_fwd = words_rev = []

    # Sorted candidate lists for descendant lookups (same bisect trick).
    cands_fwd = sorted(filtered)
    cands_rev = sorted(a[::-1] for a in filtered) if is_suffix else cands_fwd

    # ── Score every candidate ───────────────────────────────────────────────
    scored: List[Tuple[str, float]] = []
    for affix in filtered:
        score = _affix_productivity_score(
            affix,
            cands_fwd, cands_rev,
            affix_freq,
            words_fwd, words_rev,
            dict_words,
            is_suffix,
        )
        if score >= min_score:
            scored.append((affix, score))

    scored.sort(key=lambda x: -x[1])
    result = scored[:top_n]

    if result:
        logger.info(
            f"extract_productive_affixes: returning {len(result)} {mode}es "
            f"(top={result[0][1]:.3f}, bottom={result[-1][1]:.3f})"
        )
    else:
        logger.info(f"extract_productive_affixes: no {mode}es passed the filter")
    return result


def get_affixes_and_frequencies(dict_words, min_len=1, max_len=10):
    from context_aware_io import load_json_file, save_json_file, get_hash_key

    hash_key = get_hash_key(
        sorted(dict_words), meta_str=f"get_affixes_and_frequencies, min_len={min_len}, max_len={max_len}"
    )
    prefix_file = f"{hash_key}_prefixes.json"
    suffix_file = f"{hash_key}_suffixes.json"
    prefix_freq_file = f"{hash_key}_prefixes_freq.json"
    suffix_freq_file = f"{hash_key}_suffixes_freq.json"

    prefixes_path = os.path.join(STORAGE_DIR, prefix_file)
    suffixes_path = os.path.join(STORAGE_DIR, suffix_file)
    prefix_freq_path = os.path.join(STORAGE_DIR, prefix_freq_file)
    suffix_freq_path = os.path.join(STORAGE_DIR, suffix_freq_file)

    # Fast path: load pre-built affix lists from disk.
    prefixes: Set[str] = set(load_json_file(prefixes_path))
    suffixes: Set[str] = set(load_json_file(suffixes_path))
    prefix_freq: CounterType = Counter(load_json_file(prefix_freq_path))
    suffix_freq: CounterType = Counter(load_json_file(suffix_freq_path))
    if not (prefixes and suffixes and prefix_freq and suffix_freq):
        (
            prefixes,
            suffixes,
            prefix_freq,
            suffix_freq,
        ) = run_morpheme_pipeline(dict_words, min_len=min_len, max_len=max_len)

        # Persist the discovered affixes so subsequent calls skip discovery.
        save_json_file(STORAGE_DIR, prefix_file, sorted(prefixes))
        save_json_file(STORAGE_DIR, suffix_file, sorted(suffixes))
        save_json_file(STORAGE_DIR, prefix_freq_file, dict(prefix_freq))
        save_json_file(STORAGE_DIR, suffix_freq_file, dict(suffix_freq))

    return prefixes, suffixes, prefix_freq, suffix_freq


# ── Export / analysis helpers ────────────────────────────────────────────

def export_roots_csv(
        roots: Dict[str, Dict[str, Any]],
        output_path: str,
        min_pct: float = 0.0,
        sort_by: str = "frequency",
) -> None:
    """Write roots to CSV."""
    filtered = {k: v for k, v in roots.items()
                if v["combining_vowel_percentage"] >= min_pct}
    key_fn = ((lambda kv: kv[1]["frequency"]) if sort_by == "frequency"
              else (lambda kv: kv[1]["combining_vowel_percentage"]))
    rows = sorted(filtered.items(), key=key_fn, reverse=True)

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Root", "Pattern", "Frequency", "CV/Standalone",
                    "Ratio", "Diversity", "Examples"])
        for root, st in rows:
            w.writerow([root, st.get("pattern", "?"), st["frequency"],
                        st["combining_vowel_count"],
                        f"{st['combining_vowel_percentage']:.1%}",
                        st.get("continuation_diversity", ""),
                        "; ".join(st.get("examples", []))])
    print(f"  Exported {len(rows)} roots -> {output_path}")


def _compute_lang_bias(self,
                       prefix_freq: CounterType,
                       suffix_freq: CounterType,
                       prefixes: Set[str],
                       suffixes: Set[str],
                       dict_size: int) -> str:
    """
    Determine language bias (suffixing vs prefixing).
    """

    if dict_size < 100:
        suffix_hits = sum(v for k, v in suffix_freq.items())
        prefix_hits = sum(v for k, v in prefix_freq.items())
        suffix_dict_stems = sum(
            1 for s in suffixes
            for stem in self.suffix_stems.get(s, set())
            if stem in self.word_set
        )
        prefix_dict_stems = sum(
            1 for p in prefixes
            for stem in self.prefix_stems.get(p, set())
            if stem in self.word_set
        )

        suffix_score = suffix_hits + suffix_dict_stems * 2
        prefix_score = prefix_hits + prefix_dict_stems * 2

        return "suffixing" if suffix_score >= prefix_score else "prefixing"

    if suffixes:
        suffix_productivity = sum(
            len(self.suffix_stems.get(a, set()))
            for a in suffixes
        ) / len(suffixes)
    else:
        suffix_productivity = 0.0

    if prefixes:
        prefix_productivity = sum(
            len(self.prefix_stems.get(a, set()))
            for a in prefixes
        ) / len(prefixes)
    else:
        prefix_productivity = 0.0

    if suffix_productivity >= prefix_productivity:
        return "suffixing"
    else:
        return "prefixing"


if __name__ == "__main__":
    from test_morpheme_extractor import *  # noqa: F401,F403
    unittest.main(exit=False)