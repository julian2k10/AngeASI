"""
Language-agnostic root extractor module.
"""

import csv
import json
import math
import os
import time
from collections import defaultdict, Counter
from functools import wraps
from typing import Any, Dict, List, Optional, Set, Tuple, Counter as CounterType

def log_time(func):
    """Decorator that prints wall-clock execution time."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        print(f"  [{func.__name__}] {dt:.4f}s")
        return result

    return wrapper

# ---------------------------------------------------------------------------
# Morpheme frequency extraction
# ---------------------------------------------------------------------------

def _extract_morpheme_frequencies(dictionary: Set[str],
                                  min_len: int = 1,
                                  max_len: int = 20,
                                  min_coverage: int = 0.4,
                                  max_coverage: int = 0.6) -> Tuple[Counter, Counter, Counter, Counter]:
    """
    Extract substring frequencies from a dictionary of words.
    Returns (subword_freq, morpheme_freq, prefix_freq, suffix_freq).
    """
    morpheme_freq, subword_freq, prefix_freq, suffix_freq = Counter(), Counter(), Counter(), Counter()

    for word in dictionary:
        wlen = len(word)
        prefixes, suffixes, subwords = [], [], []
        is_compound = False
        for length in range(min_len, min(wlen, max_len + 1)):
            coverage = length / wlen
            if wlen > 6 and min_coverage <= coverage <= max_coverage:
                prefix = word[:length]
                suffix = word[length:]
                prefixes.append(prefix)
                suffixes.append(word[-length:])
                if prefix in dictionary and suffix in dictionary:
                    is_compound = True
                    break
            else:
                prefixes.append(word[:length])
                suffixes.append(word[-length:])

            for start in range(1, wlen - length):
                subwords.append(word[start:start + length])

        if not is_compound:  # Only count frequencies of non-compound words
            prefix_freq.update(prefixes)
            suffix_freq.update(suffixes)
            subword_freq.update(subwords)

    # Update morpheme_freq with prefix/suffix counts
    morpheme_freq.update(prefix_freq)
    morpheme_freq.update(suffix_freq)
    subword_freq.update(morpheme_freq)

    return subword_freq, morpheme_freq, prefix_freq, suffix_freq


def extract_bound_roots(affix_freq,
                        morpheme_freq,
                        min_root_freq,
                        desired_root_len,
                        max_bound_root_len,
                        minimum_variance,
                        prod_freq_len_multi,
                        is_suffix):
    direction = "suffix" if is_suffix else "prefix"
    print(f"Extracting {direction} bound roots...")
    bound_roots = set()
    affixes = [k for k, v in affix_freq.items() if v >= min_root_freq and len(k) <= max_bound_root_len]
    sort_func = (lambda x: x[::-1]) if is_suffix else (lambda x: x)
    affixes.sort(key=sort_func)

    # Create buckets and group affixes
    groups, bucket = [], []
    prev_cand = ""
    for affix in affixes:
        if prev_cand and len(affix) > len(prev_cand):
            bucket.append(affix)
        else:
            if bucket:
                groups.append(bucket)
                bucket = []

            bucket.append(affix)

        prev_cand = affix

    if bucket:  # Add final bucket
        groups.append(bucket)

    upper_root_limit = min_root_freq * max(2, prod_freq_len_multi)
    for group in groups:
        prev_cand = ""
        prev_freq = 0
        for affix in sorted(group, key=len, reverse=True):
            freq = morpheme_freq[affix]
            if prev_cand:
                if freq > prev_freq + minimum_variance:
                    if len(affix) > desired_root_len:
                        if len(affix) <= max_bound_root_len and freq >= upper_root_limit:
                            bound_roots.add(affix)
                            prev_cand = affix
                            prev_freq = freq
                    else:
                        bound_roots.add(affix)
                        prev_cand = affix
                        prev_freq = freq
            else:
                if len(affix) > desired_root_len:
                    if len(affix) <= max_bound_root_len and freq >= upper_root_limit:
                        bound_roots.add(affix)
                        prev_cand = affix
                        prev_freq = freq
                else:
                    bound_roots.add(affix)
                    prev_cand = affix
                    prev_freq = freq

    return bound_roots


@log_time
def extract_productive_roots(
        lang_code: str,
        dict_words: Set[str],
        suffixes: Set[str],
        prefixes: Set[str],
        script: str,
        min_root_freq=4,
        desired_root_len=6,
        max_bound_root_len=10,
        max_free_root_len=15,
        prod_freq_len_multi=2,
        is_suffixing=True,  # Language bias
        combining_forms: Optional[Set[str]] = None,
        orth_rules: Optional[Any] = None,
) -> Tuple[Set[str], Set[str], Set[str], Set[str], CounterType, CounterType, CounterType]:
    """
    returns (subword_freq, morpheme_freq, prefix_freq, suffix_freq)
    """
    print(f"Extracting productive roots...")
    if not isinstance(dict_words, set):
        dict_words = set(dict_words)

    if is_suffixing:
        # Suffixes need higher variance
        minimum_prefix_variance = 1
        minimum_suffix_variance = 3
    else:
        # Prefixes need higher variance
        minimum_prefix_variance = 3
        minimum_suffix_variance = 1

    # 1. Extract frequency tables
    subword_freq, morpheme_freq, prefix_freq, suffix_freq = _extract_morpheme_frequencies(dict_words)

    # 2. Extract prefix roots
    prefix_roots = extract_bound_roots(prefix_freq,
                                       subword_freq,  # Use subword freq to detect bound roots like 'metallurg'
                                       min_root_freq,
                                       desired_root_len,
                                       max_bound_root_len,
                                       minimum_prefix_variance,
                                       prod_freq_len_multi,
                                       is_suffix=False)

    # 3. Extract suffix roots
    suffix_roots = extract_bound_roots(suffix_freq,
                                       subword_freq,
                                       min_root_freq,
                                       desired_root_len,
                                       max_bound_root_len,
                                       minimum_suffix_variance,
                                       prod_freq_len_multi,
                                       is_suffix=True)

    # 4. Extract free roots the meet length and frequency requirements
    print("Extracting free roots...")
    free_roots = set()
    for word in dict_words:
        if len(word) > max_free_root_len:
            continue

        freq = morpheme_freq.get(word, 0)

        if freq >= min_root_freq * 2:
            free_roots.add(word)
        elif len(word) <= desired_root_len and freq >= min_root_freq:
            # Short words with moderate freq are productive enough as free roots
            free_roots.add(word)

    print(f"Extracted {len(free_roots)} free roots. "
          f"{len(prefix_roots)} prefix bound roots & {len(suffix_roots)} suffix bound roots.")

    # Lazy imports to avoid circular dependency at module level
    if orth_rules is None:
        from morpho_rules import discover_orth_rules
        orth_rules = discover_orth_rules(dict_words, suffixes, lang_code=lang_code)

    if combining_forms is None:
        from latin_greek_root_extractor import extract_latin__greek_roots, normalize_combining_forms
        roots, srs = extract_latin__greek_roots(dict_words, suffixes, suffix_freq, lang_code=lang_code, script=script)
        combining_forms = set(roots.keys())
        # exc_forms = normalize_combining_forms(combining_forms, subword_freq, script=script)

    # 5. Filter 1 letter (diff) duplicates & affixes > 7 char that fragments any free roots.
    #    Exempt combining forms and roots restorable via morphophonological rules.
    prefix_roots = _filter_affix_duplicates(
        prefix_roots, free_roots, is_suffix=False,
        combining_forms=combining_forms, orth_rules=orth_rules, dictionary=dict_words,
    )
    suffix_roots = _filter_affix_duplicates(
        suffix_roots, free_roots, is_suffix=True,
        combining_forms=combining_forms, orth_rules=orth_rules, dictionary=dict_words,
    )

    print(f"After step-5 filters: {len(free_roots)} free roots, "
          f"{len(prefix_roots)} prefix bound roots & {len(suffix_roots)} suffix bound roots.")

    return free_roots, prefix_roots, suffix_roots, combining_forms, morpheme_freq, prefix_freq, suffix_freq


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


def vowel_band_analysis(roots: Dict[str, Dict[str, Any]]) -> None:
    """Print histogram of ratio bands."""
    bands = {"100%": 0, "90-99%": 0, "80-89%": 0, "70-79%": 0,
             "60-69%": 0, "50-59%": 0, "<50%": 0}
    for st in roots.values():
        p = st["combining_vowel_percentage"]
        if p >= 1.0:
            bands["100%"] += 1
        elif p >= 0.90:
            bands["90-99%"] += 1
        elif p >= 0.80:
            bands["80-89%"] += 1
        elif p >= 0.70:
            bands["70-79%"] += 1
        elif p >= 0.60:
            bands["60-69%"] += 1
        elif p >= 0.50:
            bands["50-59%"] += 1
        else:
            bands["<50%"] += 1
    print("\n  Ratio distribution:")
    for band, cnt in bands.items():
        pct = cnt / max(len(roots), 1) * 100
        print(f"    {band:>7}: {cnt:>5}  ({pct:5.1f}%)")


def top_productive_roots(
        roots: Dict[str, Dict[str, Any]], n: int = 20,
) -> List[Tuple[str, int, float, float]]:
    """Return *n* roots with highest productivity score (freq * ratio)."""
    scored = [(r, s["frequency"], s["combining_vowel_percentage"],
               s["frequency"] * s["combining_vowel_percentage"])
              for r, s in roots.items()]
    scored.sort(key=lambda x: x[3], reverse=True)
    return scored[:n]

MAX_VOCAB_PER_LANGUAGE      = 200_000  # ---- Maximum vocab budget per language ----
MIN_FREE_ROOT_LEN: int      = 4   # minimum length for a word to be considered a "free root"
# --- Constants and Special Tokens ---
SUBWORD_PREFIX = "#"
PAD = "<|pad|>"
UNK = "<|unk|>"
BOS = "<|bos|>"
EOS = "<|eos|>"
MASK = "<|mask|>"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS, MASK]

DICT_WORDS = "words.json"
WORD_FREQ_FILE_NAME = 'word_frequencies.json'
MORPHEME_FREQ_FILE_NAME = 'morpheme_frequencies.json'
VOCAB_FILE_NAME = 'vocab.json'

try:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT_DIR = os.getcwd()

LOG_DIR = os.path.join(ROOT_DIR, "logs")
STORAGE_DIR = os.path.join(ROOT_DIR, "vocab_files")
VOCAB_FILE = os.path.join(STORAGE_DIR, VOCAB_FILE_NAME)
MORPHEME_FILE = os.path.join(STORAGE_DIR, 'morphemes.json')
FILTERED_WORD_FREQ_FILE = os.path.join(STORAGE_DIR, 'dict_word_frequencies.json')

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


def _compute_bias(self,
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
    print(f"running {__file__}")