"""
Language-Agnostic Morphophonological Rule Discovery
====================================================

Given a set of dictionary words for ANY ISO 639-3 language, discovers
orthographic alternation rules (vowel deletion, consonant doubling,
character mutations, assimilation, etc.) via statistical analysis at
morpheme boundaries, guided by universal phonetic feature groups.

Output: List[Tuple[str, List[str]]]
    Each: (surface_end_pattern_regex, [possible_lexical_restoration, ...])

Nothing is hard-coded for a specific language.  The only built-in
knowledge is cross-family phonetic feature groups and attested
phonological bridge pairs (voicing, lenition, spirantization, etc.)
drawn from IPA / comparative linguistics.

Algorithm
---------
1.  Discover (or accept) productive suffixes.
2.  For every suffixed word whose naive stem ∉ dict, try a small set of
    *universal* boundary edits (vowel append, geminate undo, char swap
    via phonetic bridges, digraph reduction).  If an edited stem ∈ dict
    → record the (surface_boundary, lexical_boundary) alternation.
3.  Generalize concrete pairs into regex rules, rank by frequency.
"""

from __future__ import annotations

import json
import logging
import math
import os
import os.path
import re
import sys
import time
from collections import Counter, defaultdict, deque
from typing import (
    Any, Counter as CounterType, Dict, FrozenSet, List,
    Optional, Set, Tuple,
)
from context_aware_io import load_json_file, save_json_file

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)

logger = logging.getLogger("MorphoRulesPipeline")

# ═════════════════════════════════════════════════════════════════════════════
# §1  Phonetic Feature Groups & Cross-Family Bridges
#
#     These are NOT language-specific.  They encode universal (IPA-grounded)
#     articulatory relationships attested across Germanic, Romance, Slavic,
#     Semitic, Turkic, Finno-Ugric, Indo-Iranian, and other families.
# ═════════════════════════════════════════════════════════════════════════════

PHONETIC_GROUPS: Dict[str, Set[str]] = {
    # ── Vowels ──
    'front_vowels':      set('iɪeɛæyʏøœ' + 'іеєяюиэ'),
    'back_vowels':       set('uʊoɔɑɒ' + 'уоаыъ'),
    'central_vowels':    set('əɐɨʉɵ'),
    'vowels_all':        set('aeiouyæøœɐɨɯʊɵɶ' + 'аеёиоуыэюяіїєґ'),
    # ── Stops ──
    'bilabial_stops':    set('pbпб'),
    'alveolar_stops':    set('tdтд'),
    'velar_stops':       set('kgкг' + 'qɢ'),
    'palatal_stops':     set('cɟ'),
    # ── Fricatives ──
    'labiodental_fric':  set('fvфв'),
    'dental_fric':       set('θð'),
    'alveolar_fric':     set('szсз'),
    'postalveolar_fric': set('ʃʒšžшжщ'),
    'velar_fric':        set('xɣхғ'),
    'glottal_fric':      set('hɦ'),
    'pharyngeal_fric':   set('ħʕ'),
    # ── Affricates ──
    'alveolar_affric':   set('ʦʣцдз'),
    'postalv_affric':    set('ʧʤčǰчджћђ'),
    # ── Sonorants ──
    'nasals':            set('mnŋɲɳмн' + 'ñ'),
    'laterals':          set('lɫɬʎлљ'),
    'rhotics':           set('rɾɹʁʀр'),
    'glides':            set('wjɥй' + 'ŭ'),
    # ── Cross-cutting manner ──
    'sibilants':         set('szʃʒʦʣʧʤ' + 'сзшжщцч'),
}

# Cross-family phonological bridges: attested sound alternations.
# Each (set_A, set_B) means a↔b is a plausible morphophonological change.
VALID_BRIDGES: List[Tuple[Set[str], Set[str]]] = [
    # Spirantization: stop → fricative (Germanic, Celtic, Romance)
    ({'t', 'd', 'т', 'д'},   {'s', 'z', 'θ', 'ð', 'с', 'з'}),
    # Latinate softening: velar stop → sibilant (Latin, Romance)
    ({'c', 'k', 'к'},        {'s', 'ʃ', 'с', 'ш'}),
    # Semivowel alternation (cross-family)
    ({'y', 'й'},              {'i', 'e', 'и', 'е'}),
    # Voicing alternation (Germanic, Slavic, Turkic)
    ({'f', 'ф'},              {'v', 'в'}),
    ({'s', 'с'},              {'z', 'з'}),
    ({'p', 'п'},              {'b', 'б'}),
    ({'t', 'т'},              {'d', 'д'}),
    ({'k', 'к'},              {'g', 'г'}),
    # Velar–palatal shift (Slavic)
    ({'k', 'к'},              {'č', 'ч', 'c', 'ц'}),
    # Grimm's law echoes (Germanic)
    ({'p', 'п'},              {'f', 'ф'}),
    ({'t', 'т'},              {'ts', 'ц'}),
    # Rhotacism (Latin, Germanic)
    ({'s', 'с'},              {'r', 'р'}),
    # Lenition (Celtic, Iberian, Romance)
    ({'b', 'б'},              {'v', 'β', 'в'}),
    ({'d', 'д'},              {'ð', 'з'}),
    ({'g', 'г'},              {'ɣ', 'х'}),
    # Nasal place assimilation (universal)
    ({'n', 'н'},              {'m', 'ŋ', 'м'}),
    # Liquid alternation (cross-family)
    ({'l', 'л'},              {'r', 'р'}),
    # Deaffrication (Slavic, Romance)
    ({'ʧ', 'ч'},              {'ʃ', 'ш'}),
    ({'ʦ', 'ц'},              {'s', 'с'}),
]

# Pre-compute fast bridge lookup: frozenset(a, b) → True
_BRIDGE_PAIRS: Set[FrozenSet[str]] = set()
for _src, _tgt in VALID_BRIDGES:
    for _s in _src:
        for _t in _tgt:
            if len(_t) == 1:  # only single-char targets for swap edits
                _BRIDGE_PAIRS.add(frozenset((_s, _t)))

# Pre-compute char → set of bridge partners
_BRIDGE_PARTNERS: Dict[str, Set[str]] = defaultdict(set)
for pair in _BRIDGE_PAIRS:
    items = list(pair)
    if len(items) == 2:
        _BRIDGE_PARTNERS[items[0]].add(items[1])
        _BRIDGE_PARTNERS[items[1]].add(items[0])

# Shared-group lookup
_CHAR_TO_GROUPS: Dict[str, Set[str]] = defaultdict(set)
for _gn, _gc in PHONETIC_GROUPS.items():
    for _ch in _gc:
        _CHAR_TO_GROUPS[_ch].add(_gn)


def is_phonetically_plausible(a: str, b: str) -> bool:
    """Is a↔b an attested phonological alternation?"""
    if a == b:
        return True
    if frozenset((a, b)) in _BRIDGE_PAIRS:
        return True
    return bool(_CHAR_TO_GROUPS.get(a, set()) & _CHAR_TO_GROUPS.get(b, set()))


# ═════════════════════════════════════════════════════════════════════════════
# §2  Character Classification (data-driven)
# ═════════════════════════════════════════════════════════════════════════════

_KNOWN_VOWELS = PHONETIC_GROUPS['vowels_all'] | set(
    'aeiouyàáâãäåæèéêëìíîïòóôõöøùúûüýÿ'
)


def classify_alphabet(words: Set[str]) -> Tuple[Set[str], Set[str]]:
    """Data-driven vowel / consonant classification from the word list."""
    chars: Set[str] = set()
    for w in words:
        for ch in w:
            if ch.isalpha():
                chars.add(ch.lower())
    vowels = chars & _KNOWN_VOWELS
    consonants = chars - vowels
    # Fallback for unknown scripts: top-frequency chars → vowels
    if not vowels and consonants:
        freq: Counter = Counter()
        for w in words:
            for ch in w:
                if ch.isalpha():
                    freq[ch.lower()] += 1
        total = sum(freq.values())
        for ch, cnt in freq.most_common():
            if cnt / total > 0.04:
                vowels.add(ch)
                consonants.discard(ch)
            else:
                break
    return vowels, consonants


def _cc(chars: Set[str]) -> str:
    """Build regex char class [abc…] from a set."""
    return '[' + ''.join(sorted(chars)) + ']'


# ═════════════════════════════════════════════════════════════════════════════
# §3  Suffix Discovery
# ═════════════════════════════════════════════════════════════════════════════

def discover_suffixes(words: Set[str], min_freq: int = 40, max_len: int = 7) -> Set[str]:
    """Frequency-based: S is a suffix if many words end in S with stem ∈ dict."""
    logger.info(f"Discovering suffixes from {len(words)} words with min_freq={min_freq}, max_len={max_len}...")
    counter: Counter = Counter()
    for w in words:
        wlen = len(w)
        for slen in range(1, min(max_len + 1, wlen - 1)):
            if wlen - slen >= 2 and w[:-slen] in words:
                counter[w[-slen:]] += 1
    return {s for s, c in counter.items() if c >= min_freq}


# ═════════════════════════════════════════════════════════════════════════════
# §4  Universal Boundary Edits
#
#     Every edit here is grounded in a cross-family phonological process.
#     No language-specific swap tables.
# ═════════════════════════════════════════════════════════════════════════════

def _boundary_edits(
    stem: str,
    vowels: Set[str],
    consonants: Set[str],
) -> List[Tuple[str, str, str]]:
    """
    Generate plausible restored stems for a stem NOT found in the dictionary.
    Returns: [(restored_stem, surface_tail, lexical_tail), ...]

    Processes (all cross-family):
    1. Vowel-deletion restoration: append each vowel of the language
    2. Gemination reversal:        de-duplicate final doubled consonant
    3. Phonetic bridge swap:       swap final char via attested bridge
    4. Digraph reduction:          remove inserted char in 2-char cluster
    """
    if len(stem) < 2:
        return []

    results: List[Tuple[str, str, str]] = []
    last = stem[-1]
    prev = stem[-2]

    # ── 1. Vowel-deletion restoration ───────────────────────────────────
    #    Universal: many languages delete a stem-final vowel before a
    #    vocalic suffix.  Restore by appending each vowel in the language.
    #    e.g., English bak→bake, Italian bell→bello, Turkish ev→evi
    if last in consonants:
        for v in sorted(vowels):
            results.append((stem + v, last, last + v))

    # ── 2. Gemination (doubling) reversal ───────────────────────────────
    #    Universal: consonant doubling to preserve vowel quality.
    #    e.g., English runn→run, Finnish taakk→taak, Arabic saddad→sadad
    if last == prev and last in consonants:
        results.append((stem[:-1], last + last, last))

    # ── 3. Phonetic bridge character swaps ──────────────────────────────
    #    Swap the final char with any attested phonological partner.
    #    Covers voicing (f↔v, s↔z, p↔b, t↔d, k↔g), lenition (b→v),
    #    spirantization (t→s), rhotacism (s→r), semivowel (y↔i), etc.
    #    This is the ONLY mutation mechanism — purely data-driven via
    #    the cross-family VALID_BRIDGES table.
    partners = _BRIDGE_PARTNERS.get(last, set())
    for partner in sorted(partners):
        restored = stem[:-1] + partner
        if restored != stem:
            results.append((restored, last, partner))

    # ── 4. Digraph / cluster reduction ──────────────────────────────────
    #    Some languages insert a char at a morpheme boundary to preserve
    #    pronunciation.  Reverse by removing the last char of a 2-char
    #    consonant cluster at stem end.
    #    e.g., English panick→panic (k inserted after c)
    #    Also handles: French doublements, Turkish buffer consonants, etc.
    if last in consonants and prev in consonants and last != prev:
        results.append((stem[:-1], prev + last, prev))

    # ── 5. Vowel harmony / vowel-to-vowel mutation ──────────────────────
    #    Some languages mutate stem-final vowel before suffix (Turkish
    #    vowel harmony, Germanic umlaut, etc.)
    if last in vowels:
        for v in sorted(vowels):
            if v != last:
                results.append((stem[:-1] + v, last, v))

    return results


# ═════════════════════════════════════════════════════════════════════════════
# §5  Rule Generalization (concrete alternation → regex)
# ═════════════════════════════════════════════════════════════════════════════

def _generalize(
    surf: str,
    lex: str,
    consonants: Set[str],
    vowels: Set[str],
    C: str,
    V: str,
) -> Tuple[str, str]:
    """
    Convert (surface_boundary, lexical_boundary) → (regex_pattern, restoration).

    Generalization strategy:
    - Gemination (XX→X) gets a consonant-class back-reference.
    - Everything else stays as a literal match on the boundary chars.
      This avoids over-generalization and keeps rules precise.
    """
    # ── Gemination: XX → X (generalize across all consonants) ──
    if (len(surf) == 2 and surf[0] == surf[1]
            and len(lex) == 1 and surf[0] == lex[0]
            and surf[0] in consonants):
        return (f"({C})\\1", r"\1")

    # ── All other patterns: literal boundary match ──
    # The surface tail as-is anchored to end of stem.
    return (f"{re.escape(surf)}$", lex)


# ═════════════════════════════════════════════════════════════════════════════
# §6  Main Discovery Entry Point
# ═════════════════════════════════════════════════════════════════════════════

def discover_alternation_rules(
    dict_words: Set[str],
    suffixes: Optional[Set[str]] = None,
    min_evidence: int = 5,
    max_suffix_count: int = 150,
    top_n: int = 60,
) -> List[Tuple[str, List[str]]]:
    """
    Discover morphophonological alternation rules from a word list.

    Parameters
    ----------
    dict_words : Set[str]
        Dictionary words for ANY ISO 639-3 language.
    suffixes : set[str] | None
        Known suffixes.  Auto-discovered if None.
    min_evidence : int
        Minimum word-pair count to emit a rule.
    max_suffix_count : int
        Max suffixes to scan (speed vs coverage trade-off).
    top_n : int
        Target number of top-scoring rules to return.  All rules
        tied at the boundary score are included, so the actual count
        may exceed ``top_n``.  This ensures deterministic output
        regardless of dict iteration order.

    Returns
    -------
    List[Tuple[str, List[str]]]
        (surface_end_regex, [lexical_restoration, ...])
    """
    words = {w.lower() for w in dict_words}
    vowels, consonants = classify_alphabet(words)
    C = _cc(consonants)
    V = _cc(vowels)

    if suffixes is None:
        suffixes = discover_suffixes(words)

    logger.info(f"Discovering rules using {len(suffixes)} suffixes, min_evidence={min_evidence}, "
                f"max_suffix={max_suffix_count}...")
    sorted_sfx = sorted(suffixes, key=lambda s: (-len(s), s))[:max_suffix_count]

    # ── Build suffix → word-list index ──
    # Optimized: instead of checking every word against every suffix,
    # build a set of target suffixes and check each word's endings once.
    sfx_set = set(sorted_sfx)
    max_sfx_len = max(len(s) for s in sorted_sfx) if sorted_sfx else 0
    sfx_words: Dict[str, List[str]] = defaultdict(list)
    for w in words:
        wlen = len(w)
        for slen in range(1, min(max_sfx_len + 1, wlen - 2)):
            ending = w[-slen:]
            if ending in sfx_set:
                sfx_words[ending].append(w)
    # Sort each word list for deterministic evidence counts regardless
    # of set iteration order (Python's hash randomization).
    for sfx in sfx_words:
        sfx_words[sfx].sort()

    # ── Collect alternation evidence ──
    evidence: Counter = Counter()
    examples: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = defaultdict(list)

    for sfx in sorted_sfx:
        slen = len(sfx)
        for w in sfx_words[sfx]:
            stem = w[:-slen]

            if stem in words:
                # Even when stem is in dict, check for structural patterns
                # (gemination, digraph) where BOTH surface and restored exist.
                # This handles large dictionaries with variant spellings.
                for restored, surf_tail, lex_tail in _boundary_edits(stem, vowels, consonants):
                    if restored in words and restored != stem and len(restored) >= 2:
                        is_structural = (
                            # Gemination: XX→X
                            (len(surf_tail) == 2 and surf_tail[0] == surf_tail[1]
                             and len(lex_tail) == 1 and surf_tail[0] == lex_tail[0])
                            or
                            # Digraph reduction: XY→X
                            (len(surf_tail) == 2 and len(lex_tail) == 1
                             and surf_tail[0] == lex_tail[0]
                             and surf_tail[0] in consonants and surf_tail[1] in consonants)
                        )
                        if is_structural:
                            key = (surf_tail, lex_tail)
                            evidence[key] += 1
                            if len(examples[key]) < 5:
                                examples[key].append((w, stem, restored))
                continue

            # Stem NOT in dict → try all boundary edits
            for restored, surf_tail, lex_tail in _boundary_edits(stem, vowels, consonants):
                if restored not in words or restored == stem or len(restored) < 2:
                    continue
                key = (surf_tail, lex_tail)
                evidence[key] += 1
                if len(examples[key]) < 5:
                    examples[key].append((w, stem, restored))

    # ── Generalize into regex rules ──
    rule_map: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rule_examples: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)

    for (surf, lex), count in evidence.items():
        if count < min_evidence:
            continue
        pat, rest = _generalize(surf, lex, consonants, vowels, C, V)
        rule_map[pat][rest] += count
        if len(rule_examples[pat]) < 5:
            rule_examples[pat].extend(examples[(surf, lex)][:3])

    # ── Score, rank, deduplicate ──
    scored: List[Tuple[str, List[str], int]] = []
    for pat, rests in rule_map.items():
        sorted_r = sorted(rests.items(), key=lambda x: -x[1])
        scored.append((pat, [r for r, _ in sorted_r], sum(c for _, c in sorted_r)))

    scored.sort(key=lambda x: -x[2])

    # ── Top-N with tie-inclusion ──
    # Take the top_n rules by score, then include every additional rule
    # whose score equals the boundary score.  This guarantees deterministic
    # output: all rules tied at the cutoff are included regardless of
    # dict iteration order.
    if not scored:
        return []

    boundary_score = scored[min(top_n - 1, len(scored) - 1)][2]

    final: List[Tuple[str, List[str]]] = []
    seen: Set[str] = set()
    for pat, rests, score in scored:
        if score < boundary_score:
            break
        if pat not in seen:
            final.append((pat, rests))
            seen.add(pat)

    return final


# ═════════════════════════════════════════════════════════════════════════════
# §7  Compile & Apply  (suffix-dict, no regex iteration)
# ═════════════════════════════════════════════════════════════════════════════

# Minimum candidate length — avoids wasting time on 1–2 char fragments
# that would never be valid stems anyway.
MIN_STEM = 3


class CompiledRules:
    """
    Fast suffix-dict representation of morphophonological alternation rules.

    Instead of iterating N compiled regexes per stem, rules are stored as a
    dict[str, set[str]] keyed by literal suffix → set of replacements.
    Lookup is O(max_suffix_len) dict probes — effectively constant time.

    The only special case is the gemination (doubled-character) rule
    ``([C])\\1$`` which cannot be expressed as a literal suffix.  Instead
    of a hardcoded regex, the character class is extracted from the
    discovered pattern and stored as ``gemination_chars`` — a frozenset of
    characters that participate in doubling.  This keeps the module fully
    language-agnostic: the set will contain Latin consonants for English,
    Cyrillic consonants for Russian, etc.

    Tier classifications are computed once at compile time and stored in
    ``rule_tiers`` for use in scoring without re-classifying per call.
    """
    __slots__ = (
        'suffix_rules', 'max_suffix_len', 'gemination_chars',
        'vowels', 'consonants', 'rule_tiers', 'restoration_tiers',
    )

    def __init__(
        self,
        suffix_rules: Dict[str, Set[str]],
        max_suffix_len: int,
        gemination_chars: FrozenSet[str],
        vowels: FrozenSet[str],
        consonants: FrozenSet[str],
        rule_tiers: Dict[str, str],
        restoration_tiers: Dict[str, Dict[str, str]],
    ):
        self.suffix_rules = suffix_rules
        self.max_suffix_len = max_suffix_len
        self.gemination_chars = gemination_chars
        self.vowels = vowels
        self.consonants = consonants
        self.rule_tiers = rule_tiers
        self.restoration_tiers = restoration_tiers


# ── Rule Tier Classification ────────────────────────────────────────────────
#
# Each suffix rule is classified into a confidence tier for bonus weighting
# in pair scoring.  Classification is done once at compile time and stored
# in ``CompiledRules.rule_tiers``.  All character class checks use the
# data-driven vowels/consonants sets — nothing is hardcoded per language.

# Bonus multipliers per tier: higher = more trusted rule
RULE_TIER_BONUS: Dict[str, float] = {
    "high":   0.50,   # up to 1.5x bonus
    "medium": 0.25,   # up to 1.25x bonus
    "low":    0.10,   # up to 1.1x bonus
}


def _classify_rule_tier(
    suffix: str,
    restorations: Set[str],
    vowels: FrozenSet[str],
    consonants: FrozenSet[str],
) -> str:
    """
    Classify a suffix rule into a confidence tier for bonus weighting.

    Tier 1 (HIGH): Well-established, highly productive morphophonological
    processes — gemination (CC→C), vowel deletion (C→Ce), y↔i bridge.
    These cover the majority of morphophonological alternations.

    Tier 2 (MEDIUM): Consonant cluster simplification at morpheme
    boundaries (th→t, st→s, nd→n, etc.) and consonant-to-consonant
    voicing alternations (s→z, f→v, etc.).  Also digraph reduction
    (ck→c).

    Tier 3 (LOW): Vowel-to-vowel mutations and less common alternations.

    All character checks use the data-driven vowels/consonants sets from
    ``classify_alphabet`` — fully language-agnostic.
    """
    # ── Tier 1: vowel deletion (single consonant → consonant+vowel) ──
    # Pattern is a single consonant, restoration is consonant+vowel
    if (len(suffix) == 1 and suffix in consonants
            and any(r.startswith(suffix) and len(r) == 2
                    and r[1] in vowels for r in restorations)):
        return "high"

    # ── Tier 1: y↔i bridge (semivowel alternation) ──
    if suffix == 'i' and 'y' in restorations:
        return "high"
    if suffix == 'y' and 'i' in restorations:
        return "high"

    # ── Tier 2: consonant cluster simplification (2-char → 1-char) ──
    if (len(suffix) == 2
            and all(c in consonants for c in suffix)
            and any(len(r) == 1 and r in consonants for r in restorations)):
        return "medium"

    # ── Tier 3: everything else (vowel-vowel swaps, etc.) ──
    return "low"


def _classify_all_rule_tiers(
    suffix_rules: Dict[str, Set[str]],
    has_gemination: bool,
    vowels: FrozenSet[str],
    consonants: FrozenSet[str],
) -> Dict[str, str]:
    """
    Classify all rules into tiers at compile time.

    Returns a dict mapping each suffix (or the special key
    ``"__gemination__"``) to its tier string.
    """
    tiers: Dict[str, str] = {}

    if has_gemination:
        tiers["__gemination__"] = "high"

    for suffix, restorations in suffix_rules.items():
        tiers[suffix] = _classify_rule_tier(
            suffix, restorations, vowels, consonants,
        )

    return tiers


def _classify_restoration_tier(
    suffix: str,
    restoration: str,
    vowels: FrozenSet[str],
    consonants: FrozenSet[str],
) -> str:
    """
    Classify the tier of a SPECIFIC suffix→restoration pair.

    This is more granular than ``_classify_rule_tier`` which classifies
    the entire rule.  For example, rule ``c$ → {s, ce, cy, k}`` has
    rule-level tier "high" (because ce is vowel deletion), but the
    specific restoration ``c→s`` is "medium" (voicing alternation),
    while ``c→ce`` is "high" (vowel deletion).

    All character checks use data-driven vowels/consonants sets —
    fully language-agnostic.
    """
    # ── Tier 1: vowel deletion (C → C+vowel) ──
    if (len(suffix) == 1 and suffix in consonants
            and restoration.startswith(suffix) and len(restoration) == 2
            and restoration[1] in vowels):
        return "high"

    # ── Tier 1: y↔i bridge ──
    if suffix == 'i' and restoration == 'y':
        return "high"
    if suffix == 'y' and restoration == 'i':
        return "high"

    # ── Tier 2: consonant cluster simplification (2-char suffix → 1-char) ──
    if (len(suffix) == 2
            and all(c in consonants for c in suffix)
            and len(restoration) == 1 and restoration in consonants):
        return "medium"

    # ── Tier 2: single consonant → different single consonant (voicing) ──
    if (len(suffix) == 1 and suffix in consonants
            and len(restoration) == 1 and restoration in consonants):
        return "medium"

    # ── Tier 3: everything else ──
    return "low"


def _classify_all_restoration_tiers(
    suffix_rules: Dict[str, Set[str]],
    has_gemination: bool,
    vowels: FrozenSet[str],
    consonants: FrozenSet[str],
) -> Dict[str, Dict[str, str]]:
    """
    Classify every (suffix, restoration) pair into a tier at compile time.

    Returns a nested dict: ``suffix → {restoration → tier}``.
    The special key ``"__gemination__"`` maps to ``{"__dedup__": "high"}``.
    """
    tiers: Dict[str, Dict[str, str]] = {}

    if has_gemination:
        tiers["__gemination__"] = {"__dedup__": "high"}

    for suffix, restorations in suffix_rules.items():
        tiers[suffix] = {
            r: _classify_restoration_tier(suffix, r, vowels, consonants)
            for r in restorations
        }

    return tiers


# Pre-classified tier cache: maps lang_code → CompiledRules.rule_tiers
# Built once per lang_code, avoids re-classifying on every scoring call.
_RULE_TIER_CACHE: Dict[str, Dict[str, str]] = {}


def build_rule_tier_cache(
    lang_code: str,
    compiled_rules: CompiledRules,
) -> Dict[str, str]:
    """
    Return the tier classification dict for the given language.

    On first call for a lang_code, caches ``compiled_rules.rule_tiers``.
    Subsequent calls return the cached version.
    """
    if lang_code in _RULE_TIER_CACHE:
        return _RULE_TIER_CACHE[lang_code]
    _RULE_TIER_CACHE[lang_code] = compiled_rules.rule_tiers
    return compiled_rules.rule_tiers


def compile_rules(
    rules: List[Tuple[str, List[str]]],
    vowels: Set[str] = frozenset(),
    consonants: Set[str] = frozenset(),
) -> CompiledRules:
    """
    Convert discovered rules into a fast suffix-dict lookup structure.

    Every rule whose pattern is a literal ``xyz$`` becomes a dict entry.
    The gemination back-reference rule ``([C])\\1$`` has its character class
    extracted into a frozenset for O(1) membership checks — no regex needed
    at apply time, and no hardcoded alphabet.

    If ``vowels`` and ``consonants`` are provided (as returned by
    ``classify_alphabet``), each rule is also classified into a confidence
    tier (high / medium / low) and stored in ``rule_tiers``.
    """
    suffix_rules: Dict[str, Set[str]] = defaultdict(set)
    gemination_chars: Set[str] = set()

    # Pattern to extract the character class from a gemination rule like
    # ``([bcdfg…])\1`` — captures the bracket contents.
    _gemination_class_rx = re.compile(r'\(\[([^\]]+)\]\)\\1')

    has_gemination = False

    for pat, rests in rules:
        stripped = pat.rstrip('$')

        # Detect the gemination back-reference pattern: ([...])\\1
        if '\\1' in stripped:
            has_gemination = True
            m = _gemination_class_rx.search(stripped)
            if m:
                # Extract every character from the bracket class.
                # Handles plain chars; ranges like a-z are expanded.
                bracket_content = m.group(1)
                i = 0
                while i < len(bracket_content):
                    if (i + 2 < len(bracket_content)
                            and bracket_content[i + 1] == '-'):
                        # Range: e.g. a-z
                        start = ord(bracket_content[i])
                        end = ord(bracket_content[i + 2])
                        for code in range(start, end + 1):
                            gemination_chars.add(chr(code))
                        i += 3
                    else:
                        gemination_chars.add(bracket_content[i])
                        i += 1
            continue

        # Everything else is a literal suffix — unescape any regex escapes
        # (discover_alternation_rules uses re.escape on literals).
        try:
            literal = re.sub(r'\\(.)', r'\1', stripped)
        except Exception:
            continue

        # Only keep if the suffix is purely literal characters
        if literal and not any(c in literal for c in r'()[]{}*+?|^.'):
            suffix_rules[literal].update(rests)

    max_len = max((len(k) for k in suffix_rules), default=0)
    frozen_vowels = frozenset(vowels)
    frozen_consonants = frozenset(consonants)
    suffix_dict = dict(suffix_rules)

    # ── Classify each rule into a confidence tier ──
    rule_tiers = _classify_all_rule_tiers(
        suffix_dict, has_gemination, frozen_vowels, frozen_consonants,
    )

    # ── Classify each (suffix, restoration) pair into a tier ──
    restoration_tiers = _classify_all_restoration_tiers(
        suffix_dict, has_gemination, frozen_vowels, frozen_consonants,
    )

    return CompiledRules(
        suffix_dict, max_len, frozenset(gemination_chars),
        frozen_vowels, frozen_consonants, rule_tiers, restoration_tiers,
    )


def apply_rules(
    stem: str,
    compiled: CompiledRules,
) -> List[str]:
    """
    Generate candidate base forms by applying compiled rules to a stem.

    Uses O(max_suffix_len) dict lookups instead of iterating compiled
    regexes.  Deduplication is handled by a set — no ``not in`` list scans.
    """
    if len(stem) < MIN_STEM:
        return [stem]

    candidates: Set[str] = {stem}
    suffix_rules = compiled.suffix_rules

    for length in range(compiled.max_suffix_len, 0, -1):
        if len(stem) < length:
            continue
        tail = stem[-length:]
        rests = suffix_rules.get(tail)
        if rests is not None:
            prefix = stem[:-length]
            for r in rests:
                if len(prefix) + length >= MIN_STEM:
                    candidates.add(prefix + r)

    # Gemination: doubled character → single  (e.g. runn → run)
    if (compiled.gemination_chars
            and len(stem) >= 2
            and stem[-1] == stem[-2]
            and stem[-1] in compiled.gemination_chars
            and len(stem) - 1 >= MIN_STEM):
        candidates.add(stem[:-1])

    return list(candidates)


def fst_stem_lookup(
    stem: str,
    dictionary: Set[str],
    combining_forms: Set[str],
    prefixes: Set[str],
    suffixes: Set[str],
    rules: CompiledRules,
) -> Tuple[bool, str]:
    """
    Given a surface stem, return (found, canonical_form) where ``found`` is
    True if the stem or any orthographic variant is a known dictionary word /
    combining form / prefix / suffix.

    Lookup order (fastest to slowest):

    1. **Direct lookup** — O(1) set membership.  The most common path for
       stems that are already canonical.

    2. **SuffixTrie restoration** — O(k) single trie walk on the module-level
       ``_RESTORATION_TRIE``.  Covers morphophonological alternations that
       were pre-indexed by ``build_restoration_trie`` / ``ensure_restoration_trie``
       (e.g. "happi" → "happy", "adenoviru" → "adenovirus").  Only fires when
       the trie has been populated; costs nothing otherwise.

    3. **Suffix-dict restoration rules** — O(max_suffix_len) dict probes via
       ``CompiledRules``.  Covers rules discovered by ``discover_alternation_rules``
       that were not pre-indexed into the SuffixTrie (vowel deletion, gemination,
       phonetic bridges, digraph reduction).

    4. **Gemination reversal** — O(1) check for doubled final consonants.
    """
    # ── Step 1: Direct lookup — O(1) ────────────────────────────────────────
    if (stem in dictionary or stem in combining_forms
            or stem in prefixes or stem in suffixes):
        return True, stem

    slen = len(stem)
    if slen < MIN_STEM:
        return False, stem

    # ── Step 2: SuffixTrie fast path — O(k) single trie walk ────────────────
    # Only attempt when the trie has been populated (dictionary_ids non-empty).
    # find_and_replace returns the original stem when no suffix matches, so
    # there is no additional branch cost on a miss.
    if _RESTORATION_TRIE.dictionary_ids:
        restored = _RESTORATION_TRIE.find_and_replace(stem)
        if restored != stem and len(restored) >= MIN_STEM and (
            restored in dictionary or restored in combining_forms
            or restored in prefixes or restored in suffixes
        ):
            return True, restored

    # ── Step 3: Suffix-dict restoration rules — O(max_suffix_len) probes ────
    suffix_rules = rules.suffix_rules
    for length in range(rules.max_suffix_len, 0, -1):
        if slen < length:
            continue
        tail = stem[-length:]
        rests = suffix_rules.get(tail)
        if rests is not None:
            prefix = stem[:-length]
            for r in rests:
                candidate = prefix + r
                if len(candidate) >= MIN_STEM and (
                    candidate in dictionary or candidate in combining_forms
                    or candidate in prefixes or candidate in suffixes
                ):
                    return True, candidate

    # ── Step 4: Gemination reversal — O(1) ───────────────────────────────────
    if (rules.gemination_chars
            and slen >= 2
            and stem[-1] == stem[-2]
            and stem[-1] in rules.gemination_chars
            and slen - 1 >= MIN_STEM):
        candidate = stem[:-1]
        if (candidate in dictionary or candidate in combining_forms
                or candidate in prefixes or candidate in suffixes):
            return True, candidate

    return False, stem


# ── High-tier bonus value, cached for fast early-exit comparison ──
_HIGH_TIER_BONUS = RULE_TIER_BONUS["high"]


def check_morphophonological_rule_bonus(
    subword_split: List[str],
    dictionary: Set[str],
    compiled_rules: CompiledRules,
) -> float:
    """
    Check if a split follows known morphophonological alternation rules.

    Uses tiered bonuses: high-confidence rules (gemination, vowel deletion,
    y↔i) get stronger bonuses than low-confidence rules.  This ensures that
    common, well-established alternation patterns reliably beat fragments
    and false-positive rules.

    The tier is determined by the SPECIFIC restoration that matched, not
    just the rule pattern.  e.g. if ``c$`` has restorations ``{s, ce, cy}``,
    and ``danc`` → ``dance`` matched via ``ce``, the tier is "high" (vowel
    deletion), but ``danc`` → ``dans`` via ``s`` would be "medium" (voicing).

    Uses the suffix-dict lookup from ``CompiledRules`` for O(max_suffix_len)
    probes per token instead of iterating compiled regexes.

    Parameters
    ----------
    subword_split : list[str]
        The candidate morphological split (list of subword tokens).
    dictionary : set[str]
        Known dictionary words for the language.
    compiled_rules : CompiledRules
        Pre-compiled rules with per-restoration tier classifications.

    Returns
    -------
    float
        Multiplier ≥ 1.0.  Higher values indicate the split better matches
        known morphophonological patterns.
    """
    if len(subword_split) < 2:
        return 1.0

    total_bonus = 0.0
    total_candidates = 0

    suffix_rules = compiled_rules.suffix_rules
    restoration_tiers = compiled_rules.restoration_tiers
    max_suffix_len = compiled_rules.max_suffix_len
    gemination_chars = compiled_rules.gemination_chars

    for tok in subword_split:
        t = tok.lower()
        if t in dictionary:
            continue  # already a word, no rule needed

        total_candidates += 1
        best_tier_bonus = 0.0
        tlen = len(t)

        if tlen < MIN_STEM:
            continue

        # ── Suffix-dict lookup: O(max_suffix_len) probes ──
        for length in range(max_suffix_len, 0, -1):
            if tlen < length:
                continue
            tail = t[-length:]
            rests = suffix_rules.get(tail)
            if rests is None:
                continue

            prefix = t[:-length]
            tier_map = restoration_tiers.get(tail)

            for r in rests:
                candidate = prefix + r
                if len(candidate) >= MIN_STEM and candidate in dictionary:
                    # Look up pre-computed tier for this specific restoration
                    tier = tier_map.get(r, "low") if tier_map else "low"
                    bonus = RULE_TIER_BONUS[tier]
                    if bonus > best_tier_bonus:
                        best_tier_bonus = bonus
                        if best_tier_bonus >= _HIGH_TIER_BONUS:
                            break  # can't do better than high
            if best_tier_bonus >= _HIGH_TIER_BONUS:
                break

        # ── Gemination check ──
        if (best_tier_bonus < _HIGH_TIER_BONUS
                and gemination_chars
                and tlen >= 2
                and t[-1] == t[-2]
                and t[-1] in gemination_chars
                and tlen - 1 >= MIN_STEM
                and t[:-1] in dictionary):
            best_tier_bonus = _HIGH_TIER_BONUS

        total_bonus += best_tier_bonus

    if total_candidates == 0:
        return 1.0

    return 1.0 + total_bonus / total_candidates

# ═════════════════════════════════════════════════════════════════════════════
# §9  SuffixTrie — Fast Suffix-Based Word Restoration
#
#     A trie keyed on reversed characters so that suffix lookups are O(k)
#     where k = length of the matched suffix, regardless of trie size.
#     Designed for restoring morphophonologically altered words back to their
#     canonical dictionary form, e.g. "viru" → "virus", "happi" → "happy".
#
#     Longest-match semantics: when multiple inserted fragments share a common
#     prefix of the reversed key, the deepest (longest) terminal wins.
# ═════════════════════════════════════════════════════════════════════════════

class SuffixTrie:
    """
    Trie keyed on reversed character sequences for O(k) suffix lookup.

    Supports fast lookup and in-place replacement of morphophonologically
    altered word endings.  The trie stores (altered_fragment → restored_fragment)
    pairs; ``find_and_replace`` scans from the end of a word and returns the
    word with the longest matching suffix replaced.

    Example
    -------
    >>> trie = SuffixTrie()
    >>> trie.insert("viru", "virus")    # altered → restored
    >>> trie.find_and_replace("adenoviru")
    'adenovirus'
    >>> trie.find_and_replace("unrelated")
    'unrelated'
    """

    def __init__(self) -> None:
        self.root: Dict[str, Any] = {}
        self.end_symbol: str = "__end__"
        # Tracks id(dict_words) for every dictionary whose restoration pairs
        # have been indexed into this trie.  Used by ensure_restoration_trie
        # to skip redundant rebuilds when the same dictionary object is passed
        # more than once.
        self.dictionary_ids: Set[int] = set()

    def insert(self, fragment: str, replacement: str) -> None:
        """
        Insert an (altered_fragment, replacement) pair into the trie.

        The fragment is stored in reverse so that suffix traversal from
        the end of a word maps naturally onto a top-down trie walk.

        Parameters
        ----------
        fragment    : The altered (surface) word ending to match.
        replacement : The canonical form to substitute when the fragment
                      is found at the end of a query word.
        """
        node = self.root
        for char in reversed(fragment):
            node = node.setdefault(char, {})
        # Store the replacement at the terminal node.
        node[self.end_symbol] = replacement

    def find_and_replace(self, word: str) -> str:
        """
        Return ``word`` with its longest matching suffix replaced.

        Traverses the word from right to left.  At each step where a
        terminal node (``end_symbol``) is found, the current match is
        recorded.  The *longest* such match is applied — greedy,
        non-overlapping.  If no suffix matches, ``word`` is returned
        unchanged.

        Parameters
        ----------
        word : The surface form to restore.

        Returns
        -------
        str
            Restored form if a suffix match was found; original word
            otherwise.
        """
        node = self.root
        match_len = 0
        replacement = None

        for i, char in enumerate(reversed(word)):
            if char in node:
                node = node[char]
                if self.end_symbol in node:
                    # Record the longest terminal seen so far.
                    match_len = i + 1
                    replacement = node[self.end_symbol]
            else:
                break

        if replacement is not None:
            return word[:-match_len] + replacement
        return word

    def __len__(self) -> int:
        """Return the number of (fragment, replacement) pairs stored."""
        count = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            if self.end_symbol in node:
                count += 1
            for key, child in node.items():
                if key != self.end_symbol:
                    stack.append(child)
        return count

    def __repr__(self) -> str:
        return f"SuffixTrie(entries={len(self)})"


# ── Module-level SuffixTrie instance ────────────────────────────────────────
#
# Built (or rebuilt) by ``build_restoration_trie``.  Once populated, callers
# can restore any altered word in O(k) time:
#
#   restored_word = _RESTORATION_TRIE.find_and_replace(altered_word)
#
_RESTORATION_TRIE: SuffixTrie = SuffixTrie()


def build_restoration_dict(
    alteration_rules: List[Dict[str, Any]],
    dict_words: Set[str],
) -> Tuple[Dict[str, str], int]:
    """
    Build a mapping of {altered_word: restored_word} from alteration rules.

    For each word in ``dict_words``, each alteration rule is applied to
    generate one or more surface (altered) forms.  When a surface form is
    NOT already in the dictionary (i.e. it is genuinely altered), it is
    recorded as a key with the original dictionary word as the value.

    Parameters
    ----------
    alteration_rules : list of dicts, each with keys:
        ``"pattern"``      – regex pattern matching the end of a word
                             (e.g. ``"i$"`` or ``"r$"``).
        ``"restorations"`` – iterable of suffix strings that, when re-appended
                             to an altered stem, recover the canonical form.

    dict_words : set of canonical dictionary words (lowercased).

    Returns
    -------
    Tuple[Dict[str, str], int]
        A 2-tuple of:
        - ``restoration_dict`` : ``{altered_form: canonical_form}``
          e.g. ``{"happi": "happy", "easi": "easy"}``.
        - ``dict_id`` : ``id(dict_words)`` — the unique memory identity of the
          input set, stored in ``SuffixTrie.dictionary_ids`` so that
          ``ensure_restoration_trie`` can skip redundant rebuilds when the
          exact same set object is passed more than once.

    Notes
    -----
    *   Only altered forms that are absent from the dictionary are included,
        preventing false positives where the altered form happens to be a
        valid word in its own right.
    *   When multiple rules produce the same altered form (pointing to
        different canonical words), the first rule wins (stable mapping).

    Example
    -------
    >>> rules = [{"pattern": "i$", "restorations": {"y"}}]
    >>> words = {"happy", "easy", "body"}
    >>> rd, did = build_restoration_dict(rules, words)
    >>> rd
    {'happi': 'happy', 'easi': 'easy', 'bodi': 'body'}
    >>> did == id(words)
    True
    """
    dict_id: int = id(dict_words)
    words_lower: Set[str] = {w.lower() for w in dict_words}
    result: Dict[str, str] = {}

    for rule in alteration_rules:
        pattern: str = rule.get("pattern", "")
        restorations: Set[str] = set(rule.get("restorations", []))

        # Extract the literal suffix that the ALTERED form ends with.
        # The pattern is anchored to $, e.g. "i$" means altered ends in "i".
        # Strip the anchor and any regex metacharacters for a literal match.
        # Only simple literal patterns (no character classes, alternations, etc.)
        # are handled here; complex patterns are skipped.
        literal_pattern = pattern.rstrip("$")
        if any(c in literal_pattern for c in r"()[]{}*+?|^.\\"):
            logger.debug(
                "build_restoration_dict: skipping non-literal pattern %r", pattern
            )
            continue

        altered_suffix = literal_pattern  # e.g. "i" for pattern "i$"

        for word in sorted(words_lower):  # sorted for deterministic output
            # A canonical word can produce an altered form if the word ends
            # with any of the restoration suffixes.  The altered form is
            # produced by stripping the restoration suffix and appending the
            # altered suffix.
            #
            # Example (rule: "i$", restorations={"y", ...}):
            #   canonical = "happy"  (ends in "y")
            #   restoration = "y"
            #   altered = "happ" + "i" = "happi"
            #   SuffixTrie entry: "happi" → "happy"

            for restoration in sorted(restorations):
                if not word.endswith(restoration):
                    continue

                # Build the altered form: strip restoration, append altered_suffix
                stem = word[: -len(restoration)]
                if not stem:
                    continue

                altered = stem + altered_suffix

                # Unchanged or trivially short → skip
                if altered == word or len(altered) < 2:
                    continue

                # Only record if the altered form is absent from the dictionary
                # and not already claimed by an earlier rule (stable mapping).
                if altered not in words_lower and altered not in result:
                    result[altered] = word

    return result, dict_id


def build_restoration_trie(
    restoration_dict: Dict[str, str],
    dict_id: Optional[int] = None,
) -> SuffixTrie:
    """
    Populate the module-level ``_RESTORATION_TRIE`` from a restoration dict.

    After calling this function, any altered word can be restored in O(k)
    time by calling ``_RESTORATION_TRIE.find_and_replace(word)``.

    Parameters
    ----------
    restoration_dict : dict returned by ``build_restoration_dict``,
        mapping ``{altered_form: canonical_form}``.
    dict_id : int | None
        The ``id(dict_words)`` value returned alongside the restoration dict
        by ``build_restoration_dict``.  When provided, it is added to
        ``_RESTORATION_TRIE.dictionary_ids`` so that
        ``ensure_restoration_trie`` can detect previously indexed
        dictionaries without rebuilding.  Pass ``None`` to skip registration
        (e.g. when populating from a manually constructed dict).

    Returns
    -------
    SuffixTrie
        The populated module-level trie (same object as
        ``_RESTORATION_TRIE``).

    Notes
    -----
    This function always rebuilds the trie from scratch — it clears any
    previously inserted entries and previously registered dictionary IDs.
    To *incrementally* add a new dictionary without clearing existing
    entries, use ``ensure_restoration_trie`` instead.

    Example
    -------
    >>> rd, did = build_restoration_dict(rules, words)
    >>> trie = build_restoration_trie(rd, did)
    >>> trie.find_and_replace("adenoviru")
    'adenovirus'
    >>> trie.find_and_replace("happi")
    'happy'
    >>> did in trie.dictionary_ids
    True
    """
    global _RESTORATION_TRIE
    _RESTORATION_TRIE = SuffixTrie()
    for altered, canonical in restoration_dict.items():
        _RESTORATION_TRIE.insert(altered, canonical)
    if dict_id is not None:
        _RESTORATION_TRIE.dictionary_ids.add(dict_id)
    logger.info(
        "build_restoration_trie: loaded %d entries into module-level SuffixTrie "
        "(dict_ids=%s).",
        len(_RESTORATION_TRIE),
        _RESTORATION_TRIE.dictionary_ids,
    )
    return _RESTORATION_TRIE


def ensure_restoration_trie(
    alteration_rules: List[Dict[str, Any]],
    dict_words: Set[str],
) -> SuffixTrie:
    """
    Ensure the module-level ``_RESTORATION_TRIE`` covers ``dict_words``.

    Checks whether ``id(dict_words)`` is already recorded in
    ``_RESTORATION_TRIE.dictionary_ids``.  If so, the trie is returned
    immediately — no work is done.  If not, ``build_restoration_dict`` is
    called for ``dict_words`` and the resulting pairs are **merged** into
    the existing trie (existing entries are preserved), then
    ``id(dict_words)`` is added to ``dictionary_ids``.

    This is the preferred call site for production code because it avoids
    redundant re-indexing when the same dictionary object is passed across
    multiple processing steps.

    Parameters
    ----------
    alteration_rules : list[dict]
        Same format as accepted by ``build_restoration_dict``.
    dict_words : set[str]
        The canonical dictionary to index.  The object's ``id()`` is used
        as the deduplication key, so passing a *different* set object with
        identical contents will trigger a fresh build.

    Returns
    -------
    SuffixTrie
        The module-level ``_RESTORATION_TRIE``, guaranteed to contain
        restorations for ``dict_words``.

    Example
    -------
    >>> rules = [{"pattern": "i$", "restorations": {"y"}}]
    >>> dict_a = {"happy", "easy"}
    >>> trie = ensure_restoration_trie(rules, dict_a)
    >>> id(dict_a) in trie.dictionary_ids
    True
    >>> # Second call with the same object → no rebuild
    >>> trie2 = ensure_restoration_trie(rules, dict_a)
    >>> trie2 is trie
    True

    >>> # Different object, same contents → new entries merged in
    >>> dict_b = {"body", "copy"}
    >>> trie3 = ensure_restoration_trie(rules, dict_b)
    >>> id(dict_b) in trie3.dictionary_ids
    True
    >>> len(trie3) > len(trie2)   # dict_b entries added on top
    True
    """
    dict_id = id(dict_words)

    if dict_id in _RESTORATION_TRIE.dictionary_ids:
        logger.debug(
            "ensure_restoration_trie: dict_id=%d already indexed — skipping.",
            dict_id,
        )
        return _RESTORATION_TRIE

    restoration_dict, did = build_restoration_dict(alteration_rules, dict_words)

    # Merge into the existing trie — do NOT clear previous entries.
    for altered, canonical in restoration_dict.items():
        _RESTORATION_TRIE.insert(altered, canonical)
    _RESTORATION_TRIE.dictionary_ids.add(did)

    logger.info(
        "ensure_restoration_trie: merged %d new entries for dict_id=%d "
        "(trie now has %d entries, dict_ids=%s).",
        len(restoration_dict),
        did,
        len(_RESTORATION_TRIE),
        _RESTORATION_TRIE.dictionary_ids,
    )
    return _RESTORATION_TRIE

# ═════════════════════════════════════════════════════════════════════════════
# §11  Orthographic Rule Discovery Cache
# ═════════════════════════════════════════════════════════════════════════════

# Module-level cache for discovered rules (per lang_code).
_DISCOVERED_RULES_CACHE: Dict[str, CompiledRules] = {}
_DISCOVERED_RAW_RULES_CACHE: Dict[str, List[Tuple[str, List[str]]]] = {}
_CACHE_DIR = os.path.join(os.getcwd(), 'cache')
ORTH_RULES_FILE = "orth_rules.json"

# Convenience aliases matching the old pipeline import names.
compile_orth_rules = compile_rules
apply_orth_rules = apply_rules


def discover_orth_rules(
    dictionary: Set[str],
    suffixes: Set[str],
    lang_code: str = "eng",
    min_evidence: int = 3,
    max_suffix_count: int = 120,
    top_n: int = 60,
    cache_dir: Optional[str] = _CACHE_DIR,
) -> CompiledRules:
    """
    Discover morphophonological alternation rules from dictionary words.

    Uses the morpho_rules module to statistically discover rules like
    vowel deletion, consonant doubling, y->i, ck-insertion, etc. from
    the actual word list -- no hard-coded patterns.

    Results are cached per lang_code both in memory and on disk
    (as ``{lang_code}_orth_rules.json``).

    Returns compiled regex rules suitable for fst_stem_lookup.
    """
    # 1. Check memory cache
    if lang_code in _DISCOVERED_RULES_CACHE:
        return _DISCOVERED_RULES_CACHE[lang_code]

    # 2. Check disk cache
    cache_path = os.path.join(cache_dir, f"{lang_code}_{ORTH_RULES_FILE}")
    if os.path.exists(cache_path):
        logger.info(f"Loading cached orth rules from {cache_path}")
        raw = load_json_file(cache_path)
        if raw:
            raw_rules = [(entry["pattern"], entry["restorations"]) for entry in raw]
            compiled = compile_orth_rules(raw_rules)
            _DISCOVERED_RULES_CACHE[lang_code] = compiled
            _DISCOVERED_RAW_RULES_CACHE[lang_code] = raw_rules
            build_rule_tier_cache(lang_code, compiled)
            logger.info(f"Loaded {len(raw_rules)} orth rules from cache.")
            return compiled

    # 3. Discover from scratch
    logger.info(f"Discovering orth rules for lang_code={lang_code} from {len(dictionary)} words...")
    t0 = time.time()
    raw_rules = discover_alternation_rules(
        dictionary,
        suffixes=suffixes,
        min_evidence=min_evidence,
        max_suffix_count=max_suffix_count,
        top_n=top_n,
    )
    elapsed = time.time() - t0
    logger.info(f"Discovered {len(raw_rules)} orth rules in {elapsed:.2f}s")

    # 4. Save to disk
    save_data = [{"pattern": pat, "restorations": rests} for pat, rests in raw_rules]
    save_json_file(cache_dir, f"{lang_code}_{ORTH_RULES_FILE}", save_data)

    # 5. Compile and cache
    compiled = compile_orth_rules(raw_rules)
    _DISCOVERED_RULES_CACHE[lang_code] = compiled
    _DISCOVERED_RAW_RULES_CACHE[lang_code] = raw_rules

    # 6. Pre-classify rule tiers for efficient scoring
    build_rule_tier_cache(lang_code, compiled)

    return compiled


# ═════════════════════════════════════════════════════════════════════════════
# §12  Morphological Role Classifier
# ═════════════════════════════════════════════════════════════════════════════

class MorphRole:
    PREFIX   = "prefix"
    SUFFIX   = "suffix"
    ROOT     = "root"         # free root in dictionary
    CFORM    = "cform"        # combining form (neoclassical)
    FRAGMENT = "fragment"     # unrecognised piece

    # Canonical template structures and their base bonus scores.
    #
    # Priority principles:
    #   1. Fewer tokens = higher score (single root > 2-part > 3-part > 4-part)
    #      because fewer vocab lookups is better tokenization.
    #   2. Within the same part-count, cform (combining form) is the most
    #      linguistically specific identification and gets highest priority.
    #   3. Recognized morphological patterns (prefix+root+suffix) score
    #      higher than ambiguous patterns (root+root+root).
    #   4. Fragment-containing patterns are NOT in this table and get 0.
    #
    # Score bands are compressed so that quality weighting and frequency
    # can override a 2-part template when the 3-part split has much better
    # morpheme quality.  The band gaps:
    #   1-part: 1000     (best)
    #   2-part: 800-900  (good, narrow band)
    #   3-part: 600-780  (mid-range)
    #   4-part: 400-580  (acceptable)
    TEMPLATES: List[Tuple[Tuple[str, ...], float]] = [
        # ── 1-part: single token covers the whole word (best possible) ──
        (("root",),                      1000.0),
        (("cform",),                      980.0),

        # ── 2-part: minimal split ──
        (("cform",  "cform"),             900.0),
        (("cform",  "root"),              890.0),
        (("cform",  "suffix"),            880.0),
        (("root",   "cform"),             870.0),
        (("root",   "suffix"),            860.0),
        (("prefix", "root"),              850.0),
        (("prefix", "cform"),             840.0),
        (("prefix", "suffix"),            830.0),
        (("root",   "root"),              820.0),

        # ── 3-part: common morphological patterns ──
        (("cform",  "cform",  "suffix"),  780.0),
        (("cform",  "cform",  "root"),    770.0),
        (("cform",  "cform",  "cform"),   760.0),
        (("cform",  "root",   "suffix"),  750.0),
        (("prefix", "cform",  "suffix"),  740.0),
        (("prefix", "cform",  "cform"),   730.0),
        (("prefix", "cform",  "root"),    720.0),
        (("root",   "cform",  "suffix"),  710.0),
        (("prefix", "root",   "suffix"),  700.0),
        (("root",   "root",   "suffix"),  690.0),
        (("prefix", "root",   "root"),    680.0),
        (("prefix", "root",   "cform"),   670.0),
        (("root",   "root",   "root"),    660.0),
        (("prefix", "prefix", "root"),    650.0),
        (("root",   "root",   "cform"),   640.0),
        (("root",   "cform",  "root"),    630.0),
        (("root",   "cform",  "cform"),   620.0),
        (("prefix", "prefix", "suffix"),  610.0),

        # ── 4-part: deeper decomposition ──
        (("cform",  "cform",  "cform",  "suffix"),  580.0),
        (("prefix", "cform",  "cform",  "suffix"),  570.0),
        (("prefix", "root",   "cform",  "suffix"),  560.0),
        (("prefix", "root",   "root",   "suffix"),  550.0),
        (("cform",  "root",   "cform",  "suffix"),  540.0),
        (("prefix", "cform",  "root",   "suffix"),  530.0),
        (("cform",  "cform",  "root",   "suffix"),  520.0),
        (("root",   "root",   "root",   "suffix"),  510.0),
    ]

    # Build a lookup dict for O(1) template matching
    _TEMPLATE_SCORES: Dict[Tuple[str, ...], float] = {t: s for t, s in TEMPLATES}


# ═════════════════════════════════════════════════════════════════════════════
# §13  WordProfile: pre-computed morphological analysis cached per token
# ═════════════════════════════════════════════════════════════════════════════

class WordProfile:
    """
    Pre-computed morphological profile for a vocabulary token.

    Stores:
      - role: the morphological role (prefix/suffix/root/cform/fragment)
      - found: whether the token is found via FST lookup
      - restored: the canonical (FST-restored) form
      - freq: morpheme frequency (max of surface and restored forms)
      - freq_len_score: log(freq) * len^2.5 (pre-computed)

    WordProfiles are built once per token and cached. This avoids
    redundant classify_role + fst_stem_lookup calls when the same
    token appears in multiple candidate splits across words.
    """

    __slots__ = ("token", "role", "found", "restored", "freq", "freq_len_score")

    def __init__(self, token: str, role: str, found: bool, restored: str,
                 freq: int, freq_len_score: float):
        self.token = token
        self.role = role
        self.found = found
        self.restored = restored
        self.freq = freq
        self.freq_len_score = freq_len_score

    def to_dict(self) -> dict:
        return {
            "token": self.token,
            "role": self.role,
            "found": self.found,
            "restored": self.restored,
            "freq": self.freq,
            "freq_len_score": round(self.freq_len_score, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WordProfile":
        return cls(
            token=d["token"],
            role=d["role"],
            found=d["found"],
            restored=d["restored"],
            freq=d["freq"],
            freq_len_score=d["freq_len_score"],
        )


class WordProfileCache:
    """
    Global cache for WordProfile objects.

    Profiles are keyed by (token, position_class) where position_class is
    one of "first", "last", "middle", "only" -- since a token's morphological
    role can depend on its position in the split.

    Supports JSON serialization for persistence across runs.
    """
    def __init__(self):
        self._cache: Dict[Tuple[str, str], WordProfile] = {}
        # Split-level cache: word -> list of top scored splits with profiles
        self._split_cache: Dict[str, List[Tuple[List[str], float]]] = {}

    @staticmethod
    def _pos_class(pos: int, n_parts: int) -> str:
        if n_parts == 1:
            return "only"
        if pos == 0:
            return "first"
        if pos == n_parts - 1:
            return "last"
        return "middle"

    def get_profile(
        self,
        token: str,
        pos: int,
        n_parts: int,
        prefixes: Set[str],
        suffixes: Set[str],
        combining_forms: Set[str],
        dictionary: Set[str],
        morphemes_freq: Counter,
        orth_rules: CompiledRules,
    ) -> WordProfile:
        """Get or create a WordProfile for a token at a given position class."""
        pc = self._pos_class(pos, n_parts)
        t = token
        key = (t, pc)

        if key in self._cache:
            return self._cache[key]

        is_first = pos == 0
        is_last = pos == n_parts - 1

        # Classify role (same logic as the old classify_role function)
        #
        # Short-token guard: tokens of 1-2 characters are almost never
        # independent morphemes unless they are explicitly listed as a
        # prefix, suffix, or combining form.  Large dictionaries contain
        # many 1-2 char entries (abbreviations, letter names, archaic
        # spellings) that would otherwise be promoted to ROOT via the
        # FST lookup or frequency fallback, creating false positives
        # that pollute morphological splits.
        #
        # The guard ensures that short tokens can only be classified as
        # PREFIX (if in prefixes AND first position), SUFFIX (if in
        # suffixes AND last position), or CFORM (if in combining_forms).
        # Everything else stays FRAGMENT.
        is_short = len(t) <= 2

        role = MorphRole.FRAGMENT
        if t in prefixes and is_first:
            role = MorphRole.PREFIX
        elif t in suffixes and is_last:
            role = MorphRole.SUFFIX
        elif t in combining_forms:
            role = MorphRole.CFORM
        elif not is_short:
            # Only attempt FST / frequency promotion for tokens > 2 chars.
            # Short tokens that didn't match prefix/suffix/cform above
            # remain FRAGMENT — the dictionary and frequency counts are
            # too noisy at 1-2 characters to be reliable evidence of
            # morpheme status.
            found_fst, _ = fst_stem_lookup(
                t, dictionary, combining_forms, prefixes, suffixes, orth_rules
            )
            if found_fst:
                if t in prefixes:
                    role = MorphRole.PREFIX if is_first else MorphRole.ROOT
                elif t in suffixes:
                    role = MorphRole.SUFFIX if is_last else MorphRole.ROOT
                else:
                    role = MorphRole.ROOT
            else:
                freq_check = morphemes_freq.get(t, 0)
                if freq_check >= 5:
                    role = MorphRole.CFORM if len(t) <= 5 else MorphRole.ROOT

        # FST lookup for canonical form
        found, restored = fst_stem_lookup(
            t, dictionary, combining_forms, prefixes, suffixes, orth_rules
        )
        # Compute frequency (max of surface and restored forms)
        freq = morphemes_freq.get(t, 0)
        freq = max(freq, morphemes_freq.get(token, freq))
        if found and restored != t:
            freq = max(freq, morphemes_freq.get(restored, 0))

        # Pre-compute freq_len_score
        freq_score = math.log(max(freq, 2))
        length_weight = len(t) ** 2.5
        freq_len_score = freq_score * length_weight

        profile = WordProfile(
            token=t,
            role=role,
            found=found,
            restored=restored,
            freq=freq,
            freq_len_score=freq_len_score,
        )
        self._cache[key] = profile
        return profile

    def get_split_cache(self, word: str) -> Optional[List[Tuple[List[str], float]]]:
        """Get cached scored splits for a word (for child-consistency lookup)."""
        return self._split_cache.get(word)

    def set_split_cache(self, word: str, scored_splits: List[Tuple[List[str], float]]):
        """Cache the top scored splits for a word."""
        self._split_cache[word] = scored_splits

    def save_profiles(self, file_path: str):
        """Save all profiles to a JSON file."""
        data = {}
        for (token, pc), profile in self._cache.items():
            key_str = f"{token}|{pc}"
            data[key_str] = profile.to_dict()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(data)} word profiles to {file_path}")

    def load_profiles(self, file_path: str):
        """Load profiles from a JSON file."""
        if not os.path.exists(file_path):
            return
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        count = 0
        for key_str, prof_dict in data.items():
            parts = key_str.rsplit("|", 1)
            if len(parts) != 2:
                continue
            token, pc = parts
            profile = WordProfile.from_dict(prof_dict)
            self._cache[(token, pc)] = profile
            count += 1
        logger.info(f"Loaded {count} word profiles from {file_path}")

    def save_split_cache(self, file_path: str):
        """Save split cache to JSON."""
        data = {}
        for word, splits in self._split_cache.items():
            data[word] = [{"split": s, "score": round(sc, 4)} for s, sc in splits]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved split cache for {len(data)} words to {file_path}")

    def load_split_cache(self, file_path: str):
        """Load split cache from JSON."""
        if not os.path.exists(file_path):
            return
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for word, entries in data.items():
            self._split_cache[word] = [(e["split"], e["score"]) for e in entries]
        logger.info(f"Loaded split cache for {len(data)} words")

    def __len__(self):
        return len(self._cache)


# Global profile cache instance
_PROFILE_CACHE = WordProfileCache()

WORD_PROFILES_FILE = "word_profiles.json"
SPLIT_CACHE_FILE = "split_cache.json"


def get_profile_cache() -> WordProfileCache:
    """Get the global WordProfileCache instance."""
    return _PROFILE_CACHE

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
        if wlen <= max_len:
            prefix_freq[word] += 1
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


MIN_STEM_COVERAGE: float    = 0.40
MIN_STEM_LENGTH: int        = 3
MIN_BOUND_ROOT_LEN: int     = 3
MIN_BOUND_ROOT_FREQ: int    = 4

def calculate_improved_stem_score(
        stem: str,
        frequency: int,
        stem_position_length: float,  # ratio of stem to word (p_ratio or s_ratio)
        is_prefixes: bool = False,
        is_suffixes: bool = False,
        is_combining_forms: bool = False,
        is_dict_words: bool = False,
) -> float:
    """
    Improved scoring formula that properly weighs frequency, length, and type.

    PRINCIPLE:
    ----------
    1. Frequency (logarithmic): log(freq) - exponential growth with real freq
    2. Length (multiplicative): 1 + (len-3) * 0.25 - longer stems get bonus
    3. Coverage (ratio): stem_position_length - longer stems cover more word
    4. Type (multiplicative): 1.5x for combining_forms, 1.1x for dict_words

    This ensures:
    - thiev (len=5, freq=10) > thie (len=4, freq=100) > thi (len=3, freq=1000)
      Score relationship: 10*1.6*ratio > 100*1.3*ratio > 1000*1.0*ratio
                         Is NOT true with frequency alone

    - But WITH proper length weighting, the formula adapts:
      thiev: log(10) * 1.5 * ratio = 2.3 * 1.5 = 3.45
      thie:  log(100) * 1.3 * ratio = 4.6 * 1.3 = 5.98
      thi:   log(1000) * 1.0 * ratio = 6.9 * 1.0 = 6.9

      This shows that even log-weighted, frequency dominates.
      BUT with proper frequency selection (only freq >= MIN_BOUND_ROOT_FREQ * 2),
      the stems should have similar frequencies:

      thiev: log(10) * 1.5 = 3.45 ✓
      thie:  log(9) * 1.3 = 3.94 ✓
      thi:   log(8) * 1.0 = 2.08 ✗ doesn't qualify (freq < MIN_BOUND_ROOT_FREQ * 2)

      So thi should be filtered out, and thiev > thie wins.
    """

    if frequency < 1:
        return 0.0

    # STEP 1: Base frequency score (log scale)
    # Uses log instead of log(sqrt(freq)) to reduce exponential effect
    freq_score = math.log(max(2, frequency))  # min 2 to avoid log(1)=0

    # STEP 2: Length bonus (multiplicative)
    # Each char above MIN_BOUND_ROOT_LEN adds 25% to score
    # len=3 (min): 1.0
    # len=4: 1.25
    # len=5: 1.50
    # len=6: 1.75
    length_bonus = 1.0 + (len(stem) - MIN_BOUND_ROOT_LEN) * 0.25

    # STEP 3: Coverage ratio bonus (how much of word this stem covers)
    # Stems covering >50% of word get additional boost
    coverage_bonus = 1.0 + max(0.0, stem_position_length - MIN_STEM_COVERAGE) * 1.5

    # STEP 4: Combine base components multiplicatively
    base_score = freq_score * length_bonus * coverage_bonus

    # STEP 5: Type-based multipliers (combining_forms >> affixes)
    # Note: is_dict_words intentionally receives NO bonus (1.0x).  Being a
    # standalone dictionary word should not inflate a stem's score when it is
    # used as a bound morpheme prefix/suffix — e.g. 'thie' (a rare dict word)
    # should not outscore 'thiev' (the productive bound stem) inside 'thieving'.
    # The combining-form bonus (1.5x) already rewards morphologically
    # meaningful roots, and the length / coverage bonuses handle the rest.
    type_multiplier = 1.0

    if is_combining_forms:
        # Combining forms: strongest priority (1.5x)
        type_multiplier = 1.5
    elif is_prefixes or is_suffixes:
        # Known affixes: minor boost (1.05x)
        type_multiplier = 1.05

    # STEP 6: Apply type multiplier
    final_score = base_score * type_multiplier

    # STEP 7: Sanity check and return
    assert final_score >= 0, f"Score should never be negative: {final_score}"
    return final_score


if __name__ == "__main__":
    file_path = os.path.join(os.getcwd(), "eng_words.json")
    dict_words = set(load_json_file(file_path))
    ENG_PREFIXES = load_json_file(os.path.join(os.getcwd(), "eng_prefixes.json"))
    ENG_SUFFIXES = load_json_file(os.path.join(os.getcwd(), "eng_suffixes.json"))