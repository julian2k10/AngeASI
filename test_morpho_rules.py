"""
Test Harness for Morphophonological Rule Discovery
====================================================

Validates auto-discovered rules against known English alternation patterns.
Target: ≥90% of common test cases.

Also validates get_profile() fragment classification:
  - 1-2 letter fragments found in word middles should be FRAGMENT, not ROOT.
  - Known prefixes/suffixes/roots should be classified correctly by position.

Extracted from morpho_rules.py §9 (original).
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from typing import List, Tuple, Set

from asi.morpho_rules import (
    classify_alphabet,
    compile_rules,
    apply_rules,
    discover_alternation_rules,
    load_json_file,
    _extract_morpheme_frequencies,
    fst_stem_lookup,
    MorphRole,
    WordProfileCache,
    CompiledRules,
    discover_orth_rules,
    SuffixTrie,
    build_restoration_dict,
    build_restoration_trie,
    ensure_restoration_trie,
    _RESTORATION_TRIE,
)

# ═════════════════════════════════════════════════════════════════════════════
# §8  Reference ENG_RULES (validation target)
# ═════════════════════════════════════════════════════════════════════════════

ENG_RULES_REFERENCE: List[Tuple[str, List[str]]] = [
    (r"([bcdfghjklmnprstvwxz])t",   [r"\1te"]),       # 1.  t → te
    (r"([bcdfghjklmnprstvwxz])k",   [r"\1ke"]),       # 2.  k → ke
    (r"([bcdfghjklmnprstvwxz])v",   [r"\1ve"]),       # 3.  v → ve
    (r"([bcdfghjklmnprstvwxz])s",   [r"\1se"]),       # 4.  s → se
    (r"([bcdfghjklmnprstvwxz])z",   [r"\1ze"]),       # 5.  z → ze
    (r"([bcdfghjklmnprstvwxz])c",   [r"\1ce"]),       # 6.  c → ce
    (r"([bcdfghjklmnprstvwxz])g",   [r"\1ge"]),       # 7.  g → ge
    (r"([bcdfghjklmnprstvwxz])p",   [r"\1pe"]),       # 8.  p → pe
    (r"([bcdfghjklmnprstvwxz])d",   [r"\1de"]),       # 9.  d → de
    (r"([bcdfghjklmnprstvwxz])m",   [r"\1me"]),       # 10. m → me
    (r"([bcdfghjklmnprstvwxz])r",   [r"\1re"]),       # 11. r → re
    (r"([bcdfghjklmnprstvwxz])b",   [r"\1be"]),       # 12. b → be
    (r"([bcdfghjklmnprstvwxz])f",   [r"\1fe"]),       # 13. f → fe
    (r"([bcdfghjklmnprstvwxz])l",   [r"\1le"]),       # 14. l → le
    (r"([bcdfghjklmnprstvwxz])n",   [r"\1ne"]),       # 15. n → ne
    (r"([bcdfghjklmnprstvwxz])\1",  [r"\1"]),         # 16. CC → C (gemination)
    (r"i$",                         ["y"]),            # 17. i → y (semivowel)
    (r"ick$",                       ["ic"]),           # 18. ck → c (digraph)
]

# ═════════════════════════════════════════════════════════════════════════════
# Test cases for apply_rules — every ENG_RULES_REFERENCE pattern covered
#
# Format: (stem_after_suffix_strip, expected_base, description)
# Each ENG_RULES_REFERENCE pattern has at least 2-3 test cases.
# ═════════════════════════════════════════════════════════════════════════════

ENGLISH_TEST_CASES = [
    # ── Pattern 1: Vowel-deletion t → te ──
    ("skat",    "skate",    "vowel-del: t→te"),
    ("writ",    "write",    "vowel-del: t→te"),
    ("lat",     "late",     "vowel-del: t→te"),
    # ── Pattern 2: Vowel-deletion k → ke ──
    ("bak",     "bake",     "vowel-del: k→ke"),
    ("pok",     "poke",     "vowel-del: k→ke"),
    ("mistak",  "mistake",  "vowel-del: k→ke"),
    ("strok",   "stroke",   "vowel-del: k→ke"),
    # ── Pattern 3: Vowel-deletion v → ve ──
    ("driv",    "drive",    "vowel-del: v→ve"),
    ("serv",    "serve",    "vowel-del: v→ve"),
    ("grav",    "grave",    "vowel-del: v→ve"),
    # ── Pattern 4: Vowel-deletion s → se ──
    ("clos",    "close",    "vowel-del: s→se"),
    ("chas",    "chase",    "vowel-del: s→se"),
    ("caus",    "cause",    "vowel-del: s→se"),
    # ── Pattern 5: Vowel-deletion z → ze ──
    ("freez",   "freeze",   "vowel-del: z→ze"),
    ("graz",    "graze",    "vowel-del: z→ze"),
    ("glaz",    "glaze",    "vowel-del: z→ze"),
    # ── Pattern 6: Vowel-deletion c → ce ──
    ("danc",    "dance",    "vowel-del: c→ce"),
    ("forc",    "force",    "vowel-del: c→ce"),
    ("plac",    "place",    "vowel-del: c→ce"),
    # ── Pattern 7: Vowel-deletion g → ge ──
    ("arrang",  "arrange",  "vowel-del: g→ge"),
    ("charg",   "charge",   "vowel-del: g→ge"),
    ("judg",    "judge",    "vowel-del: g→ge"),
    # ── Pattern 8: Vowel-deletion p → pe ──
    ("hop",     "hope",     "vowel-del: p→pe"),
    ("shap",    "shape",    "vowel-del: p→pe"),
    ("typ",     "type",     "vowel-del: p→pe"),
    # ── Pattern 9: Vowel-deletion d → de ──
    ("rid",     "ride",     "vowel-del: d→de"),
    ("guid",    "guide",    "vowel-del: d→de"),
    ("trad",    "trade",    "vowel-del: d→de"),
    # ── Pattern 10: Vowel-deletion m → me ──
    ("gam",     "game",     "vowel-del: m→me"),
    ("nam",     "name",     "vowel-del: m→me"),
    ("flam",    "flame",    "vowel-del: m→me"),
    # ── Pattern 11: Vowel-deletion r → re ──
    ("stor",    "store",    "vowel-del: r→re"),
    ("shar",    "share",    "vowel-del: r→re"),
    ("car",     "care",     "vowel-del: r→re"),
    # ── Pattern 12: Vowel-deletion b → be ──
    ("describ", "describe", "vowel-del: b→be"),
    ("prob",    "probe",    "vowel-del: b→be"),
    ("glob",    "globe",    "vowel-del: b→be"),
    # ── Pattern 13: Vowel-deletion f → fe ──
    ("lif",     "life",     "vowel-del: f→fe"),
    ("saf",     "safe",     "vowel-del: f→fe"),
    ("wif",     "wife",     "vowel-del: f→fe"),
    # ── Pattern 14: Vowel-deletion l → le ──
    ("handl",   "handle",   "vowel-del: l→le"),
    ("troubl",  "trouble",  "vowel-del: l→le"),
    ("simpl",   "simple",   "vowel-del: l→le"),
    # ── Pattern 15: Vowel-deletion n → ne ──
    ("zon",     "zone",     "vowel-del: n→ne"),
    ("bon",     "bone",     "vowel-del: n→ne"),
    ("phon",    "phone",    "vowel-del: n→ne"),
    ("ton",     "tone",     "vowel-del: n→ne"),
    # ── Pattern 16: Consonant doubling / gemination CC → C ──
    ("runn",     "run",      "gemination: CC→C"),
    ("bigg",     "big",      "gemination: CC→C"),
    ("sitt",     "sit",      "gemination: CC→C"),
    ("hopp",     "hop",      "gemination: CC→C"),
    ("swimm",    "swim",     "gemination: CC→C"),
    ("programm", "program",  "gemination: CC→C"),
    ("cutt",     "cut",      "gemination: CC→C"),
    ("bedd",     "bed",      "gemination: CC→C"),
    ("robb",     "rob",      "gemination: CC→C"),
    ("hitt",     "hit",      "gemination: CC→C"),
    # ── Pattern 17: Semivowel bridge i → y ──
    ("happi",   "happy",    "bridge: i→y"),
    ("easi",    "easy",     "bridge: i→y"),
    ("heavi",   "heavy",    "bridge: i→y"),
    ("bodi",    "body",     "bridge: i→y"),
    ("copi",    "copy",     "bridge: i→y"),
    ("empti",   "empty",    "bridge: i→y"),
    # ── Pattern 18: Digraph reduction ck → c ──
    ("panick",  "panic",    "digraph: ck→c"),
    ("mimick",  "mimic",    "digraph: ck→c"),
    ("traffick","traffic",  "digraph: ck→c"),
    ("frolick", "frolic",   "digraph: ck→c"),
]


def test_english_coverage():
    """
    Validate auto-discovered rules against known English alternation patterns.
    Target: ≥90% of common test cases.
    """
    eng_list = load_json_file("eng_words.json")
    eng_suffixes = load_json_file("eng_suffixes.json")

    dict_words = set(w.lower() for w in eng_list)

    print("=" * 72)
    print("  MORPHOPHONOLOGICAL RULE DISCOVERY — English Validation")
    print("=" * 72)

    t0 = time.time()
    discovered = discover_alternation_rules(
        dict_words,
        suffixes=eng_suffixes,
        min_evidence=3,
        max_suffix_count=120,
        top_n=60,
    )
    elapsed = time.time() - t0
    print(f"\n  Dictionary : {len(dict_words):>10,} words")
    print(f"  Suffixes   : {len(eng_suffixes):>10,}")
    print(f"  Rules found: {len(discovered):>10}")
    print(f"  Time       : {elapsed:>10.1f}s")

    vowels, consonants = classify_alphabet(dict_words)
    compiled = compile_rules(discovered, vowels, consonants)

    # ── Print discovered rules ──
    print(f"\n{'─' * 72}")
    print("  Discovered Rules")
    print(f"{'─' * 72}")
    for i, (pat, rests) in enumerate(discovered):
        rests_str = ', '.join(rests[:5])
        if len(rests) > 5:
            rests_str += f", … (+{len(rests)-5})"
        print(f"  {i+1:3d}. {pat:42s} → [{rests_str}]")

    # ── Print compiled suffix-dict summary ──
    print(f"\n{'─' * 72}")
    print("  Compiled Suffix-Dict Summary")
    print(f"{'─' * 72}")
    print(f"  Suffix entries : {len(compiled.suffix_rules)}")
    print(f"  Max suffix len : {compiled.max_suffix_len}")
    print(f"  Has gemination : {bool(compiled.gemination_chars)}")
    if compiled.gemination_chars:
        chars_preview = ''.join(sorted(compiled.gemination_chars))
        print(f"  Gemination set : [{chars_preview}]")

    # ── Print rule tier distribution ──
    tier_counts = Counter(compiled.rule_tiers.values())
    print(f"  Rule tiers     : high={tier_counts.get('high', 0)}, "
          f"medium={tier_counts.get('medium', 0)}, "
          f"low={tier_counts.get('low', 0)}")

    # ── Run test cases ──
    print(f"\n{'─' * 72}")
    print("  Test Cases")
    print(f"{'─' * 72}")

    # Track coverage per ENG_RULES_REFERENCE pattern
    pattern_labels = [
        "t→te", "k→ke", "v→ve", "s→se", "z→ze", "c→ce", "g→ge",
        "p→pe", "d→de", "m→me", "r→re", "b→be", "f→fe", "l→le",
        "n→ne", "CC→C", "i→y", "ck→c",
    ]
    pattern_tested = {label: False for label in pattern_labels}

    passed = failed = 0
    for stem, expected, desc in ENGLISH_TEST_CASES:
        candidates = apply_rules(stem, compiled)
        ok = expected in candidates
        passed += ok
        failed += not ok
        print(f"  {'✓' if ok else '✗'}  {stem:12s} → {expected:12s}  ({desc})")
        if not ok:
            print(f"       candidates: {candidates[:8]}")

        # Track which patterns were successfully tested
        if ok:
            for label in pattern_labels:
                if label in desc:
                    pattern_tested[label] = True

    pct = passed / (passed + failed) * 100
    print(f"\n{'═' * 72}")
    print(f"  Results: {passed}/{passed + failed} passed  ({pct:.0f}%)")
    print(f"{'═' * 72}")

    # ── Verify all ENG_RULES_REFERENCE patterns have passing tests ──
    print(f"\n  ENG_RULES_REFERENCE pattern coverage:")
    all_covered = True
    for label in pattern_labels:
        covered = pattern_tested[label]
        all_covered = all_covered and covered
        print(f"    {'✓' if covered else '✗'} {label}")
    if all_covered:
        print(f"    → All {len(pattern_labels)} patterns have passing test cases")
    else:
        missing = [l for l, v in pattern_tested.items() if not v]
        print(f"    → MISSING coverage for: {', '.join(missing)}")

    # ── Verify alternation class coverage ──
    classes = {
        'vowel-deletion (C→Ce)':        False,
        'gemination (CC→C)':            False,
        'semivowel bridge (i↔y)':       False,
        'digraph reduction (ck→c)':     False,
    }
    for pat, rests in discovered:
        pat_stripped = pat.rstrip('$')
        if (len(pat_stripped) == 1 and pat_stripped.isalpha()
                and any(r.startswith(pat_stripped) and len(r) == 2 for r in rests)):
            classes['vowel-deletion (C→Ce)'] = True
        if '\\1' in pat and rests == ['\\1']:
            classes['gemination (CC→C)'] = True
        if pat_stripped == 'i' and 'y' in rests:
            classes['semivowel bridge (i↔y)'] = True
        if 'ck' in pat and 'c' in rests:
            classes['digraph reduction (ck→c)'] = True

    print(f"\n  Alternation class coverage:")
    for cls, found in classes.items():
        print(f"    {'✓' if found else '✗'} {cls}")

    return discovered


# ═════════════════════════════════════════════════════════════════════════════
# §9  get_profile Tests — Fragment vs ROOT classification
# ═════════════════════════════════════════════════════════════════════════════

def _get_combining_forms(lang_code='eng'):
    data = set(load_json_file(os.path.join(os.getcwd(), f"{lang_code}_combining_forms.json")))
    return data if data else None


def _setup_profile_test_fixtures():
    """Load all data needed for get_profile tests."""
    ENG_PREFIXES = load_json_file(os.path.join(os.getcwd(), "eng_prefixes.json"))
    ENG_SUFFIXES = load_json_file(os.path.join(os.getcwd(), "eng_suffixes.json"))
    combining_forms = _get_combining_forms('eng')

    file_path = os.path.join(os.getcwd(), "eng_words.json")
    dict_words = set(w.lower() for w in load_json_file(file_path))

    subword_freq, morpheme_freq, prefix_freq, suffix_freq = _extract_morpheme_frequencies(dict_words)

    orth_rules = discover_orth_rules(
        dict_words,
        suffixes=ENG_SUFFIXES,
        lang_code="eng",
        min_evidence=3,
        max_suffix_count=120,
        top_n=60,
    )

    return {
        "prefixes": set(ENG_PREFIXES),
        "suffixes": set(ENG_SUFFIXES),
        "combining_forms": combining_forms or set(),
        "dictionary": dict_words,
        "morphemes_freq": morpheme_freq,
        "orth_rules": orth_rules,
        "subword_freq": subword_freq,
        "prefix_freq": prefix_freq,
        "suffix_freq": suffix_freq,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Fragment test cases: 1-2 char tokens that should be FRAGMENT, not ROOT
#
# These tokens are substrings commonly found in the MIDDLE of English words.
# Even though some appear in the dictionary (the dictionary includes many
# abbreviations and letter combos), they are not real morphemes and should
# not be classified as ROOT when they appear in a morphological split.
#
# The short-token guard in get_profile ensures that tokens of ≤2 chars
# that aren't in the prefix/suffix/combining_forms lists stay FRAGMENT.
#
# Format: (token, pos, n_parts, expected_role, description)
# ═════════════════════════════════════════════════════════════════════════════

FRAGMENT_TEST_CASES = [
    # ── Pure consonant clusters: never standalone morphemes ──
    ("bl", 1, 3, MorphRole.FRAGMENT, "consonant cluster 'bl' mid-word → fragment"),
    ("cr", 1, 3, MorphRole.FRAGMENT, "consonant cluster 'cr' mid-word → fragment"),
    ("dr", 1, 3, MorphRole.FRAGMENT, "consonant cluster 'dr' mid-word → fragment"),
    ("fl", 1, 3, MorphRole.FRAGMENT, "consonant cluster 'fl' mid-word → fragment"),
    ("gr", 1, 3, MorphRole.FRAGMENT, "consonant cluster 'gr' mid-word → fragment"),
    ("pl", 1, 3, MorphRole.FRAGMENT, "consonant cluster 'pl' mid-word → fragment"),
    ("pr", 1, 3, MorphRole.FRAGMENT, "consonant cluster 'pr' mid-word → fragment"),
    ("sl", 1, 3, MorphRole.FRAGMENT, "consonant cluster 'sl' mid-word → fragment"),
    ("sp", 1, 3, MorphRole.FRAGMENT, "consonant cluster 'sp' mid-word → fragment"),
    ("st", 1, 3, MorphRole.FRAGMENT, "consonant cluster 'st' mid-word → fragment"),
    ("tr", 1, 3, MorphRole.FRAGMENT, "consonant cluster 'tr' mid-word → fragment"),
    ("br", 1, 3, MorphRole.FRAGMENT, "consonant cluster 'br' mid-word → fragment"),
    ("cl", 1, 3, MorphRole.FRAGMENT, "consonant cluster 'cl' mid-word → fragment"),
    ("fr", 1, 3, MorphRole.FRAGMENT, "consonant cluster 'fr' mid-word → fragment"),
    ("gl", 1, 3, MorphRole.FRAGMENT, "consonant cluster 'gl' mid-word → fragment"),
    ("sk", 1, 3, MorphRole.FRAGMENT, "consonant cluster 'sk' mid-word → fragment"),
    ("sm", 1, 3, MorphRole.FRAGMENT, "consonant cluster 'sm' mid-word → fragment"),
    ("sn", 1, 3, MorphRole.FRAGMENT, "consonant cluster 'sn' mid-word → fragment"),
    ("sw", 1, 3, MorphRole.FRAGMENT, "consonant cluster 'sw' mid-word → fragment"),
    ("th", 1, 3, MorphRole.FRAGMENT, "consonant cluster 'th' mid-word → fragment"),
    ("wr", 1, 3, MorphRole.FRAGMENT, "consonant cluster 'wr' mid-word → fragment"),
    ("wh", 1, 3, MorphRole.FRAGMENT, "consonant cluster 'wh' mid-word → fragment"),
    # ── Doubled consonants: artifacts of spelling, not morphemes ──
    ("bb", 1, 3, MorphRole.FRAGMENT, "doubled consonant 'bb' mid-word → fragment"),
    ("cc", 1, 3, MorphRole.FRAGMENT, "doubled consonant 'cc' mid-word → fragment"),
    ("dd", 1, 3, MorphRole.FRAGMENT, "doubled consonant 'dd' mid-word → fragment"),
    ("ff", 1, 3, MorphRole.FRAGMENT, "doubled consonant 'ff' mid-word → fragment"),
    ("gg", 1, 3, MorphRole.FRAGMENT, "doubled consonant 'gg' mid-word → fragment"),
    ("ll", 1, 3, MorphRole.FRAGMENT, "doubled consonant 'll' mid-word → fragment"),
    ("mm", 1, 3, MorphRole.FRAGMENT, "doubled consonant 'mm' mid-word → fragment"),
    ("nn", 1, 3, MorphRole.FRAGMENT, "doubled consonant 'nn' mid-word → fragment"),
    ("pp", 1, 3, MorphRole.FRAGMENT, "doubled consonant 'pp' mid-word → fragment"),
    ("rr", 1, 3, MorphRole.FRAGMENT, "doubled consonant 'rr' mid-word → fragment"),
    ("ss", 1, 3, MorphRole.FRAGMENT, "doubled consonant 'ss' mid-word → fragment"),
    ("tt", 1, 3, MorphRole.FRAGMENT, "doubled consonant 'tt' mid-word → fragment"),
    # ── Single consonants: never standalone morphemes in a split ──
    ("b", 1, 3, MorphRole.FRAGMENT, "single consonant 'b' mid-word → fragment"),
    ("c", 1, 3, MorphRole.FRAGMENT, "single consonant 'c' mid-word → fragment"),
    ("d", 1, 3, MorphRole.FRAGMENT, "single consonant 'd' mid-word → fragment"),
    ("f", 1, 3, MorphRole.FRAGMENT, "single consonant 'f' mid-word → fragment"),
    ("g", 1, 3, MorphRole.FRAGMENT, "single consonant 'g' mid-word → fragment"),
    ("h", 1, 3, MorphRole.FRAGMENT, "single consonant 'h' mid-word → fragment"),
    ("j", 1, 3, MorphRole.FRAGMENT, "single consonant 'j' mid-word → fragment"),
    ("k", 1, 3, MorphRole.FRAGMENT, "single consonant 'k' mid-word → fragment"),
    ("l", 1, 3, MorphRole.FRAGMENT, "single consonant 'l' mid-word → fragment"),
    ("m", 1, 3, MorphRole.FRAGMENT, "single consonant 'm' mid-word → fragment"),
    ("n", 1, 3, MorphRole.FRAGMENT, "single consonant 'n' mid-word → fragment"),
    ("p", 1, 3, MorphRole.FRAGMENT, "single consonant 'p' mid-word → fragment"),
    ("q", 1, 3, MorphRole.FRAGMENT, "single consonant 'q' mid-word → fragment"),
    ("r", 1, 3, MorphRole.FRAGMENT, "single consonant 'r' mid-word → fragment"),
    ("s", 1, 3, MorphRole.FRAGMENT, "single consonant 's' mid-word → fragment"),
    ("t", 1, 3, MorphRole.FRAGMENT, "single consonant 't' mid-word → fragment"),
    ("v", 1, 3, MorphRole.FRAGMENT, "single consonant 'v' mid-word → fragment"),
    ("w", 1, 3, MorphRole.FRAGMENT, "single consonant 'w' mid-word → fragment"),
    ("x", 1, 3, MorphRole.FRAGMENT, "single consonant 'x' mid-word → fragment"),
    ("z", 1, 3, MorphRole.FRAGMENT, "single consonant 'z' mid-word → fragment"),
    # ── Vowel pairs: phonological artifacts, not morphemes ──
    ("ae", 1, 3, MorphRole.FRAGMENT, "vowel pair 'ae' mid-word → fragment"),
    ("ai", 1, 3, MorphRole.FRAGMENT, "vowel pair 'ai' mid-word → fragment"),
    ("ao", 1, 3, MorphRole.FRAGMENT, "vowel pair 'ao' mid-word → fragment"),
    ("oe", 1, 3, MorphRole.FRAGMENT, "vowel pair 'oe' mid-word → fragment"),
    ("oo", 1, 3, MorphRole.FRAGMENT, "vowel pair 'oo' mid-word → fragment"),
    ("ou", 1, 3, MorphRole.FRAGMENT, "vowel pair 'ou' mid-word → fragment"),
    # ── Mixed bigrams that are not morphemes ──
    ("ak", 1, 3, MorphRole.FRAGMENT, "'ak' mid-word → fragment"),
    ("av", 1, 3, MorphRole.FRAGMENT, "'av' mid-word → fragment"),
    ("az", 1, 3, MorphRole.FRAGMENT, "'az' mid-word → fragment"),
    ("bk", 1, 3, MorphRole.FRAGMENT, "'bk' mid-word → fragment"),
    ("cf", 1, 3, MorphRole.FRAGMENT, "'cf' mid-word → fragment"),
    ("dg", 1, 3, MorphRole.FRAGMENT, "'dg' mid-word → fragment"),
    ("ek", 1, 3, MorphRole.FRAGMENT, "'ek' mid-word → fragment"),
    ("gh", 1, 3, MorphRole.FRAGMENT, "'gh' mid-word → fragment"),
    ("ht", 1, 3, MorphRole.FRAGMENT, "'ht' mid-word → fragment"),
    ("ix", 1, 3, MorphRole.FRAGMENT, "'ix' mid-word → fragment"),
    ("lk", 1, 3, MorphRole.FRAGMENT, "'lk' mid-word → fragment"),
    ("mp", 1, 3, MorphRole.FRAGMENT, "'mp' mid-word → fragment"),
    ("nk", 1, 3, MorphRole.FRAGMENT, "'nk' mid-word → fragment"),
    ("ph", 1, 3, MorphRole.FRAGMENT, "'ph' mid-word → fragment"),
    ("rk", 1, 3, MorphRole.FRAGMENT, "'rk' mid-word → fragment"),
    ("sh", 1, 3, MorphRole.FRAGMENT, "'sh' mid-word → fragment"),
    ("ub", 1, 3, MorphRole.FRAGMENT, "'ub' mid-word → fragment"),
    ("wn", 1, 3, MorphRole.FRAGMENT, "'wn' mid-word → fragment"),
    ("xt", 1, 3, MorphRole.FRAGMENT, "'xt' mid-word → fragment"),
    ("yz", 1, 3, MorphRole.FRAGMENT, "'yz' mid-word → fragment"),
    # ── Single vowels as mid-word fragments ──
    ("a", 1, 3, MorphRole.FRAGMENT, "single vowel 'a' mid-word → fragment"),
    ("e", 1, 3, MorphRole.FRAGMENT, "single vowel 'e' mid-word → fragment"),
    ("i", 1, 3, MorphRole.FRAGMENT, "single vowel 'i' mid-word → fragment"),
    ("o", 1, 3, MorphRole.FRAGMENT, "single vowel 'o' mid-word → fragment"),
    ("u", 1, 3, MorphRole.FRAGMENT, "single vowel 'u' mid-word → fragment"),
    # ── Short tokens in first/last position that are NOT prefix/suffix ──
    # These should still be FRAGMENT despite being positionally first/last,
    # because they aren't in the prefix/suffix lists.
    ("bl", 0, 2, MorphRole.FRAGMENT, "consonant cluster 'bl' first pos, not prefix → fragment"),
    ("cr", 2, 3, MorphRole.FRAGMENT, "consonant cluster 'cr' last pos, not suffix → fragment"),
    ("th", 0, 2, MorphRole.FRAGMENT, "consonant cluster 'th' first pos, not prefix → fragment"),
    ("gh", 2, 3, MorphRole.FRAGMENT, "consonant cluster 'gh' last pos, not suffix → fragment"),
    ("bb", 0, 2, MorphRole.FRAGMENT, "doubled 'bb' first pos, not prefix → fragment"),
    ("tt", 2, 3, MorphRole.FRAGMENT, "doubled 'tt' last pos, not suffix → fragment"),
]

# ═════════════════════════════════════════════════════════════════════════════
# Correct role classification tests for legitimate morphemes
# ═════════════════════════════════════════════════════════════════════════════

CORRECT_ROLE_TEST_CASES = [
    # ── Prefixes in first position ──
    ("un",    0, 2, MorphRole.PREFIX, "prefix 'un' in first pos → PREFIX"),
    ("re",    0, 2, MorphRole.PREFIX, "prefix 're' in first pos → PREFIX"),
    ("pre",   0, 2, MorphRole.PREFIX, "prefix 'pre' in first pos → PREFIX"),
    ("dis",   0, 2, MorphRole.PREFIX, "prefix 'dis' in first pos → PREFIX"),
    ("mis",   0, 2, MorphRole.PREFIX, "prefix 'mis' in first pos → PREFIX"),
    ("non",   0, 2, MorphRole.PREFIX, "prefix 'non' in first pos → PREFIX"),
    ("sub",   0, 2, MorphRole.PREFIX, "prefix 'sub' in first pos → PREFIX"),
    ("over",  0, 2, MorphRole.PREFIX, "prefix 'over' in first pos → PREFIX"),
    ("anti",  0, 2, MorphRole.PREFIX, "prefix 'anti' in first pos → PREFIX"),
    ("inter", 0, 2, MorphRole.PREFIX, "prefix 'inter' in first pos → PREFIX"),
    ("super", 0, 2, MorphRole.PREFIX, "prefix 'super' in first pos → PREFIX"),
    ("under", 0, 2, MorphRole.PREFIX, "prefix 'under' in first pos → PREFIX"),
    ("trans", 0, 2, MorphRole.PREFIX, "prefix 'trans' in first pos → PREFIX"),
    ("counter", 0, 2, MorphRole.PREFIX, "prefix 'counter' in first pos → PREFIX"),
    # ── Short prefixes in first position (≤2 chars, in prefix list) ──
    ("ab",    0, 2, MorphRole.PREFIX, "short prefix 'ab' in first pos → PREFIX"),
    ("ad",    0, 2, MorphRole.PREFIX, "short prefix 'ad' in first pos → PREFIX"),
    ("be",    0, 2, MorphRole.PREFIX, "short prefix 'be' in first pos → PREFIX"),
    ("de",    0, 2, MorphRole.PREFIX, "short prefix 'de' in first pos → PREFIX"),
    ("en",    0, 2, MorphRole.PREFIX, "short prefix 'en' in first pos → PREFIX"),
    ("ex",    0, 2, MorphRole.PREFIX, "short prefix 'ex' in first pos → PREFIX"),
    # ── Suffixes in last position ──
    ("ly",    1, 2, MorphRole.SUFFIX, "suffix 'ly' in last pos → SUFFIX"),
    ("ness",  1, 2, MorphRole.SUFFIX, "suffix 'ness' in last pos → SUFFIX"),
    ("ment",  1, 2, MorphRole.SUFFIX, "suffix 'ment' in last pos → SUFFIX"),
    ("able",  1, 2, MorphRole.SUFFIX, "suffix 'able' in last pos → SUFFIX"),
    ("ible",  1, 2, MorphRole.SUFFIX, "suffix 'ible' in last pos → SUFFIX"),
    ("less",  1, 2, MorphRole.SUFFIX, "suffix 'less' in last pos → SUFFIX"),
    ("ful",   1, 2, MorphRole.SUFFIX, "suffix 'ful' in last pos → SUFFIX"),
    ("tion",  1, 2, MorphRole.SUFFIX, "suffix 'tion' in last pos → SUFFIX"),
    ("ing",   1, 2, MorphRole.SUFFIX, "suffix 'ing' in last pos → SUFFIX"),
    ("ous",   1, 2, MorphRole.SUFFIX, "suffix 'ous' in last pos → SUFFIX"),
    ("ive",   1, 2, MorphRole.SUFFIX, "suffix 'ive' in last pos → SUFFIX"),
    ("ism",   1, 2, MorphRole.SUFFIX, "suffix 'ism' in last pos → SUFFIX"),
    ("ist",   1, 2, MorphRole.SUFFIX, "suffix 'ist' in last pos → SUFFIX"),
    ("ity",   1, 2, MorphRole.SUFFIX, "suffix 'ity' in last pos → SUFFIX"),
    # ── Short suffixes in last position (≤2 chars, in suffix list) ──
    ("ed",    1, 2, MorphRole.SUFFIX, "short suffix 'ed' in last pos → SUFFIX"),
    ("er",    1, 2, MorphRole.SUFFIX, "short suffix 'er' in last pos → SUFFIX"),
    ("al",    1, 2, MorphRole.SUFFIX, "short suffix 'al' in last pos → SUFFIX"),
    # ── Roots: well-known dictionary words (3+ chars) ──
    ("play",    0, 1, MorphRole.ROOT, "dict word 'play' sole token → ROOT"),
    ("work",    0, 2, MorphRole.ROOT, "dict word 'work' in first pos → ROOT"),
    ("rain",    0, 2, MorphRole.PREFIX, "dict word 'rain' in first pos (also prefix) → PREFIX"),
    ("moon",    1, 2, MorphRole.ROOT, "dict word 'moon' in last pos → ROOT"),
    ("fish",    1, 2, MorphRole.ROOT, "dict word 'fish' in last pos → ROOT"),
    ("star",    0, 2, MorphRole.ROOT, "dict word 'star' in first pos → ROOT"),
    ("port",    1, 2, MorphRole.CFORM, "dict word 'port' in last pos (also cform) → CFORM"),
    # ── Roots via FST restoration (vowel-deleted stems, 3+ chars) ──
    ("bak",     0, 2, MorphRole.ROOT, "FST-restored 'bak'→'bake' → ROOT"),
    ("driv",    0, 2, MorphRole.ROOT, "FST-restored 'driv'→'drive' → ROOT"),
    ("danc",    0, 2, MorphRole.ROOT, "FST-restored 'danc'→'dance' → ROOT"),
    ("clos",    0, 2, MorphRole.ROOT, "FST-restored 'clos'→'close' → ROOT"),
    # ── Roots via FST restoration (gemination, 3+ chars) ──
    ("runn",    0, 2, MorphRole.ROOT, "FST-restored 'runn'→'run' → ROOT"),
    ("bigg",    0, 2, MorphRole.ROOT, "FST-restored 'bigg'→'big' → ROOT"),
    ("sitt",    0, 2, MorphRole.ROOT, "FST-restored 'sitt'→'sit' → ROOT"),
    ("hopp",    0, 2, MorphRole.ROOT, "FST-restored 'hopp'→'hop' → ROOT"),
]

# ═════════════════════════════════════════════════════════════════════════════
# Positional interaction tests: tokens in prefix AND suffix lists
# ═════════════════════════════════════════════════════════════════════════════

POSITIONAL_TEST_CASES = [
    # 'over' is prefix AND dict word — in non-first position → ROOT
    ("over", 1, 2, MorphRole.ROOT, "'over' in last pos: prefix + dict word → ROOT"),
    # 'ed' is prefix AND suffix — first position → PREFIX
    ("ed",   0, 2, MorphRole.PREFIX, "'ed' in first pos: prefix + suffix → PREFIX"),
    # 'er' is prefix AND suffix — first position → PREFIX
    ("er",   0, 2, MorphRole.PREFIX, "'er' in first pos: prefix + suffix → PREFIX"),
    # 'en' is prefix AND suffix — last position → SUFFIX
    ("en",   1, 2, MorphRole.SUFFIX, "'en' in last pos: prefix + suffix → SUFFIX"),
    # 'al' is prefix AND suffix — first position → PREFIX
    ("al",   0, 2, MorphRole.PREFIX, "'al' in first pos: prefix + suffix → PREFIX"),
    # 'al' is prefix AND suffix — last position → SUFFIX
    ("al",   1, 2, MorphRole.SUFFIX, "'al' in last pos: prefix + suffix → SUFFIX"),
]


def test_get_profile():
    """
    Comprehensive tests for WordProfileCache.get_profile() to ensure:

    1. 1-2 letter fragments found in the middle of words are classified
       as MorphRole.FRAGMENT, not MorphRole.ROOT.  The dictionary contains
       many 1-2 char entries (abbreviations, letter names) that should not
       be promoted to ROOT.

    2. Known prefixes, suffixes, roots, and combining forms are correctly
       classified based on their position in the split.

    3. Positional interactions work correctly when a token appears in
       both the prefix and suffix lists.
    """
    fixtures = _setup_profile_test_fixtures()
    cache = WordProfileCache()

    print("\n" + "=" * 72)
    print("  get_profile — Fragment vs ROOT Classification Tests")
    print("=" * 72)

    all_cases = (
        ("FRAGMENT classification (1-2 char tokens should NOT be ROOT)",
         FRAGMENT_TEST_CASES),
        ("Correct role classification (prefixes, suffixes, roots)",
         CORRECT_ROLE_TEST_CASES),
        ("Positional interaction tests (prefix+suffix overlap)",
         POSITIONAL_TEST_CASES),
    )

    total_passed = 0
    total_failed = 0

    for section_name, cases in all_cases:
        print(f"\n{'─' * 72}")
        print(f"  {section_name}")
        print(f"{'─' * 72}")

        passed = failed = 0
        for token, pos, n_parts, expected_role, desc in cases:
            profile = cache.get_profile(
                token=token,
                pos=pos,
                n_parts=n_parts,
                prefixes=fixtures["prefixes"],
                suffixes=fixtures["suffixes"],
                combining_forms=fixtures["combining_forms"],
                dictionary=fixtures["dictionary"],
                morphemes_freq=fixtures["morphemes_freq"],
                orth_rules=fixtures["orth_rules"],
            )

            ok = profile.role == expected_role
            passed += ok
            failed += not ok
            status = "✓" if ok else "✗"
            print(f"  {status}  {token:12s} pos={pos} n={n_parts}  "
                  f"expected={expected_role:8s}  got={profile.role:8s}  ({desc})")
            if not ok:
                print(f"       freq={profile.freq}, found={profile.found}, "
                      f"restored={profile.restored}")

        pct = passed / (passed + failed) * 100 if (passed + failed) > 0 else 0
        print(f"  Section: {passed}/{passed + failed} passed ({pct:.0f}%)")
        total_passed += passed
        total_failed += failed

    total = total_passed + total_failed
    total_pct = total_passed / total * 100 if total > 0 else 0
    print(f"\n{'═' * 72}")
    print(f"  get_profile Results: {total_passed}/{total} passed ({total_pct:.0f}%)")
    print(f"{'═' * 72}")

    return total_passed, total_failed



# ═════════════════════════════════════════════════════════════════════════════
# §10  SuffixTrie Tests
#
#     Validates SuffixTrie.insert / find_and_replace, build_restoration_dict,
#     and build_restoration_trie (module-level trie).
# ═════════════════════════════════════════════════════════════════════════════

# Alteration rules used across SuffixTrie tests.
# "restorations" = suffixes that, when re-appended, recover the canonical word.
_ALTERATION_RULES_FOR_TEST = [
    {
        "pattern": "i$",
        "restorations": {"y", "e", "o", "a", "u"},
    },
    {
        "pattern": "r$",
        "restorations": {"re", "ro", "s", "ry", "ra", "l", "ri"},
    },
]


# ── SuffixTrie unit tests ────────────────────────────────────────────────────

SUFFIX_TRIE_INSERT_FIND_CASES = [
    # (word_to_query, expected_result, description)
    # Basic virus family — altered forms ending in "viru"
    ("viru",          "virus",       "direct match: viru → virus"),
    ("adenoviru",     "adenovirus",  "prefix + match: adenoviru → adenovirus"),
    ("retroviru",     "retrovirus",  "prefix + match: retroviru → retrovirus"),
    ("protoviru",     "protovirus",  "prefix + match: protoviru → protovirus"),
    # Words that should NOT be altered (no suffix match)
    ("stemvirutetr",  "stemvirutetr", "no suffix match: interior 'viru' not at end"),
    ("virus",         "virus",        "no match: 'virus' itself (no 'viru' suffix)"),
    ("",              "",             "empty string → unchanged"),
    # y↔i bridge — "happi" → "happy"
    ("happi",         "happy",       "bridge: happi → happy"),
    ("easi",          "easy",        "bridge: easi → easy"),
    # Longest-match semantics — insert both "vi" and "viru"; "viru" should win
    # (tested in test_suffix_trie_longest_match below)
]


def test_suffix_trie_basic():
    """Unit tests for SuffixTrie.insert and find_and_replace."""
    print("\n" + "=" * 72)
    print("  SuffixTrie — Basic insert / find_and_replace Tests")
    print("=" * 72)

    trie = SuffixTrie()
    # Populate with a small, known restoration dict
    known = {
        "viru":  "virus",
        "happi": "happy",
        "easi":  "easy",
    }
    for frag, repl in known.items():
        trie.insert(frag, repl)

    passed = failed = 0
    for word, expected, desc in SUFFIX_TRIE_INSERT_FIND_CASES:
        result = trie.find_and_replace(word)
        ok = result == expected
        passed += ok
        failed += not ok
        print(f"  {'✓' if ok else '✗'}  {word!r:20s} → {result!r:20s}  ({desc})")
        if not ok:
            print(f"       expected: {expected!r}")

    print(f"\n  SuffixTrie basic: {passed}/{passed + failed} passed")
    return passed, failed


def test_suffix_trie_longest_match():
    """Verify that the longest matching suffix wins (greedy matching)."""
    print("\n" + "-" * 72)
    print("  SuffixTrie — Longest-match semantics")
    print("-" * 72)

    trie = SuffixTrie()
    # Insert a shorter and a longer overlapping fragment
    trie.insert("i",    "y")      # short:  ends-in-i  → append y
    trie.insert("viru", "virus")  # longer: ends-in-viru → replace with virus

    cases = [
        # "adenoviru" ends in "viru" — the longer match must win
        ("adenoviru", "adenovirus", "longer 'viru' beats shorter 'u'"),
        # "happi" ends in "i" — only the short match applies (no "...ppi" entry)
        ("happi",     "happy",      "short 'i' match when no longer match exists"),
    ]

    passed = failed = 0
    for word, expected, desc in cases:
        result = trie.find_and_replace(word)
        ok = result == expected
        passed += ok
        failed += not ok
        print(f"  {'✓' if ok else '✗'}  {word!r:20s} → {result!r:20s}  ({desc})")

    print(f"\n  Longest-match: {passed}/{passed + failed} passed")
    return passed, failed


def test_suffix_trie_len_and_repr():
    """Verify __len__ and __repr__ work correctly."""
    print("\n" + "-" * 72)
    print("  SuffixTrie — __len__ and __repr__")
    print("-" * 72)

    trie = SuffixTrie()
    assert len(trie) == 0, f"Empty trie should have length 0, got {len(trie)}"

    pairs = {"viru": "virus", "happi": "happy", "easi": "easy", "bodi": "body"}
    for frag, repl in pairs.items():
        trie.insert(frag, repl)

    assert len(trie) == len(pairs), (
        f"Expected {len(pairs)} entries, got {len(trie)}"
    )
    repr_str = repr(trie)
    assert "SuffixTrie" in repr_str, f"repr missing 'SuffixTrie': {repr_str}"

    print(f"  ✓  len(trie) == {len(trie)} after inserting {len(pairs)} pairs")
    print(f"  ✓  repr: {repr_str}")
    return 2, 0  # 2 assertions


# ── build_restoration_dict tests ─────────────────────────────────────────────

def test_build_restoration_dict():
    """
    Validate build_restoration_dict against known (altered, canonical) pairs.

    Uses a small, hand-crafted dictionary so the test is self-contained and
    does not depend on an external eng_words.json file.

    Semantics recap
    ---------------
    Rule ``{"pattern": "i$", "restorations": {"y", "e", ...}}`` means:
      - The ALTERED surface form ends in ``"i"``.
      - To restore, replace the ``"i"`` with a restoration suffix.
    So for canonical "happy" (ends in "y", which IS a restoration):
      altered = "happ" + "i" = "happi"   (strip "y", append pattern suffix "i")
    """
    print("\n" + "=" * 72)
    print("  build_restoration_dict Tests")
    print("=" * 72)

    # Small, self-contained dictionary
    small_dict: Set[str] = {
        "happy", "easy", "body", "copy", "empty",
        "virus", "bonus", "status", "campus", "cactus",
        "store", "share", "care",
    }

    result, dict_id = build_restoration_dict(_ALTERATION_RULES_FOR_TEST, small_dict)

    # Verify the returned dict_id matches id(small_dict)
    ok_id = dict_id == id(small_dict)
    passed = int(ok_id); failed = int(not ok_id)
    print(f"  {'✓' if ok_id else '✗'}  returned dict_id={dict_id} == id(small_dict)={id(small_dict)}")

    cases = [
        ("happi", "happy",  "i$+y: happy → happi"),
        ("easi",  "easy",   "i$+y: easy  → easi"),
        ("bodi",  "body",   "i$+y: body  → bodi"),
        ("copi",  "copy",   "i$+y: copy  → copi"),
        ("empti", "empty",  "i$+y: empty → empti"),
    ]

    for altered, canonical, desc in cases:
        ok = result.get(altered) == canonical
        passed += ok
        failed += not ok
        print(f"  {'✓' if ok else '✗'}  {altered!r:12s} → {result.get(altered)!r:12s}"
              f"  (expected {canonical!r})  ({desc})")

    # Verify that canonical words themselves are NOT in the result
    canon_in_result = [w for w in small_dict if w in result]
    if canon_in_result:
        print(f"  ✗  Canonical words found in result (should not be): {canon_in_result}")
        failed += 1
    else:
        print(f"  ✓  No canonical dictionary words appear as altered keys")
        passed += 1

    print(f"\n  build_restoration_dict: {passed}/{passed + failed} passed")
    return passed, failed


# ── build_restoration_trie tests ─────────────────────────────────────────────

def test_build_restoration_trie():
    """
    Validate build_restoration_trie populates the module-level trie and
    that the trie can restore all entries via find_and_replace.
    Also checks that dict_id is registered in trie.dictionary_ids.
    """
    print("\n" + "=" * 72)
    print("  build_restoration_trie / module-level _RESTORATION_TRIE Tests")
    print("=" * 72)

    import asi.morpho_rules as _mr

    restoration_dict = {
        "viru":  "virus",
        "happi": "happy",
        "easi":  "easy",
    }
    # Simulate the dict_id as returned by build_restoration_dict
    sentinel_set: Set[str] = {"virus", "happy", "easy"}
    fake_dict_id = id(sentinel_set)

    trie = build_restoration_trie(restoration_dict, dict_id=fake_dict_id)

    # Verify the returned trie IS the module-level _RESTORATION_TRIE
    assert trie is _mr._RESTORATION_TRIE, (
        "build_restoration_trie must update the module-level _RESTORATION_TRIE"
    )

    cases = [
        # Exact fragment matches
        ("viru",          "virus",          "exact: viru → virus"),
        ("happi",         "happy",          "exact: happi → happy"),
        ("easi",          "easy",           "exact: easi → easy"),
        # Prefix + fragment
        ("adenoviru",     "adenovirus",     "prefix+frag: adenoviru → adenovirus"),
        ("retroviru",     "retrovirus",     "prefix+frag: retroviru → retrovirus"),
        # Words not in trie — must be returned unchanged
        ("stemvirutetr",  "stemvirutetr",   "no match: interior fragment not at end"),
        ("virus",         "virus",          "no match: canonical form unchanged"),
        ("unknown",       "unknown",        "no match: unrelated word unchanged"),
    ]

    passed = failed = 0
    for word, expected, desc in cases:
        result = trie.find_and_replace(word)
        ok = result == expected
        passed += ok
        failed += not ok
        print(f"  {'✓' if ok else '✗'}  {word!r:20s} → {result!r:20s}  ({desc})")
        if not ok:
            print(f"       expected: {expected!r}")

    # Verify __len__
    ok_len = len(trie) == len(restoration_dict)
    passed += ok_len; failed += not ok_len
    print(f"  {'✓' if ok_len else '✗'}  trie has {len(trie)} entries (matches input dict)")

    # Verify dict_id was registered
    ok_did = fake_dict_id in trie.dictionary_ids
    passed += ok_did; failed += not ok_did
    print(f"  {'✓' if ok_did else '✗'}  dict_id={fake_dict_id} in trie.dictionary_ids: "
          f"{trie.dictionary_ids}")

    print(f"\n  build_restoration_trie: {passed}/{passed + failed} passed")
    return passed, failed


def test_suffix_trie_set_comprehension_usage():
    """
    End-to-end smoke test matching the usage example from the docstring:

        forms = {'adenoviru', 'retroviru', 'protoviru', 'viru', 'stemvirutetr'}
        restored = {'viru': 'virus'}
        trie = SuffixTrie(); [trie.insert(f, r) for f, r in restored.items()]
        final = {trie.find_and_replace(w) for w in forms}
        # → {'adenovirus', 'retrovirus', 'protovirus', 'virus', 'stemvirutetr'}
    """
    print("\n" + "-" * 72)
    print("  SuffixTrie — set-comprehension end-to-end smoke test")
    print("-" * 72)

    forms = {"adenoviru", "retroviru", "protoviru", "viru", "stemvirutetr"}
    restoration_map = {"viru": "virus"}

    trie = SuffixTrie()
    for frag, repl in restoration_map.items():
        trie.insert(frag, repl)

    final_output = {trie.find_and_replace(word) for word in forms}
    expected_output = {"adenovirus", "retrovirus", "protovirus", "virus", "stemvirutetr"}

    ok = final_output == expected_output
    print(f"  {'✓' if ok else '✗'}  Output: {sorted(final_output)}")
    if not ok:
        print(f"       Expected: {sorted(expected_output)}")
        missing = expected_output - final_output
        extra = final_output - expected_output
        if missing:
            print(f"       Missing : {sorted(missing)}")
        if extra:
            print(f"       Extra   : {sorted(extra)}")

    passed = int(ok)
    failed = 1 - passed
    print(f"\n  Smoke test: {passed}/1 passed")
    return passed, failed


def test_ensure_restoration_trie():
    """
    Validate ensure_restoration_trie:

    1. First call for a new dict_words object → build_restoration_dict is
       called, pairs are merged into the module-level trie, and
       id(dict_words) is added to trie.dictionary_ids.

    2. Second call with the SAME dict_words object → no rebuild; trie is
       returned immediately and unchanged.

    3. Call with a DIFFERENT dict_words object (even same contents) →
       treated as a new dictionary; new pairs are merged and the new id is
       registered.

    4. Multiple distinct dictionaries accumulate in dictionary_ids; all
       their altered forms are resolvable via find_and_replace.

    5. fst_stem_lookup uses the SuffixTrie fast path (step 2) once the
       trie is populated: a stem restored via the trie is found and
       returned as (True, canonical) without falling through to the
       suffix-dict loop.
    """
    import asi.morpho_rules as _mr

    print("\n" + "=" * 72)
    print("  ensure_restoration_trie Tests")
    print("=" * 72)

    # Reset module-level trie to a known empty state for test isolation.
    _mr._RESTORATION_TRIE = _mr.SuffixTrie()

    RULES = [
        {"pattern": "i$", "restorations": {"y", "e", "o", "a", "u"}},
    ]

    passed = failed = 0

    # ── 1. First call for dict_a ──────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("  1. First call for dict_a → should build and register")
    print(f"{'─' * 72}")

    dict_a: Set[str] = {"happy", "easy", "body"}
    id_a = id(dict_a)

    trie = _mr.ensure_restoration_trie(RULES, dict_a)

    ok_id_registered = id_a in trie.dictionary_ids
    ok_entries_added = len(trie) > 0
    ok_happi = trie.find_and_replace("happi") == "happy"
    ok_easi  = trie.find_and_replace("easi")  == "easy"
    ok_bodi  = trie.find_and_replace("bodi")  == "body"
    ok_same_obj = trie is _mr._RESTORATION_TRIE

    for ok, desc in [
        (ok_id_registered, f"id(dict_a)={id_a} registered in dictionary_ids"),
        (ok_entries_added,  f"trie has {len(trie)} entries after first call"),
        (ok_happi,          "find_and_replace('happi') == 'happy'"),
        (ok_easi,           "find_and_replace('easi')  == 'easy'"),
        (ok_bodi,           "find_and_replace('bodi')  == 'body'"),
        (ok_same_obj,       "returned trie IS _RESTORATION_TRIE"),
    ]:
        passed += ok; failed += not ok
        print(f"  {'✓' if ok else '✗'}  {desc}")

    len_after_a = len(trie)

    # ── 2. Second call with the SAME dict_a object ────────────────────────────
    print(f"\n{'─' * 72}")
    print("  2. Second call with same dict_a object → no rebuild")
    print(f"{'─' * 72}")

    trie2 = _mr.ensure_restoration_trie(RULES, dict_a)

    ok_same_trie  = trie2 is trie
    ok_len_stable = len(trie2) == len_after_a
    ok_ids_stable = trie2.dictionary_ids == {id_a}

    for ok, desc in [
        (ok_same_trie,  "same trie object returned (no rebuild)"),
        (ok_len_stable, f"entry count unchanged ({len(trie2)})"),
        (ok_ids_stable, f"dictionary_ids unchanged: {trie2.dictionary_ids}"),
    ]:
        passed += ok; failed += not ok
        print(f"  {'✓' if ok else '✗'}  {desc}")

    # ── 3. New object, different contents (dict_b) ───────────────────────────
    print(f"\n{'─' * 72}")
    print("  3. New dict_b object → merges new entries, registers id(dict_b)")
    print(f"{'─' * 72}")

    dict_b: Set[str] = {"copy", "empty", "berry"}
    id_b = id(dict_b)
    assert id_b != id_a, "Test requires dict_b to be a distinct object from dict_a"

    trie3 = _mr.ensure_restoration_trie(RULES, dict_b)
    id_b = id(dict_b)  # re-read after ensure call (id must not change)

    ok_id_b_registered = id_b in trie3.dictionary_ids
    ok_id_a_preserved  = id_a in trie3.dictionary_ids
    ok_len_grew        = len(trie3) > len_after_a
    ok_copi            = trie3.find_and_replace("copi")  == "copy"
    ok_empti           = trie3.find_and_replace("empti") == "empty"
    ok_berri           = trie3.find_and_replace("berri") == "berry"
    # dict_a entries still present
    ok_happi_still     = trie3.find_and_replace("happi") == "happy"

    for ok, desc in [
        (ok_id_b_registered, f"id(dict_b)={id_b} registered in dictionary_ids"),
        (ok_id_a_preserved,  f"id(dict_a)={id_a} still in dictionary_ids"),
        (ok_len_grew,        f"entry count grew: {len_after_a} → {len(trie3)}"),
        (ok_copi,            "dict_b: find_and_replace('copi')  == 'copy'"),
        (ok_empti,           "dict_b: find_and_replace('empti') == 'empty'"),
        (ok_berri,           "dict_b: find_and_replace('berri') == 'berry'"),
        (ok_happi_still,     "dict_a entries preserved: 'happi' → 'happy'"),
    ]:
        passed += ok; failed += not ok
        print(f"  {'✓' if ok else '✗'}  {desc}")

    # ── 4. Same contents, NEW object → treated as fresh (different id) ───────
    print(f"\n{'─' * 72}")
    print("  4. dict_c == dict_a contents, new object → treated as fresh dict")
    print(f"{'─' * 72}")

    dict_c: Set[str] = {"happy", "easy", "body"}   # identical contents to dict_a
    id_c = id(dict_c)
    assert id_c != id_a, "dict_c must be a distinct object from dict_a"

    len_before_c = len(trie3)
    trie4 = _mr.ensure_restoration_trie(RULES, dict_c)
    id_c = id(dict_c)

    ok_id_c_registered = id_c in trie4.dictionary_ids
    # Entry count may not grow (all altered forms already in trie from dict_a),
    # but the id must still be registered.
    ok_both_ids = {id_a, id_b, id_c}.issubset(trie4.dictionary_ids)

    for ok, desc in [
        (ok_id_c_registered, f"id(dict_c)={id_c} registered even though contents == dict_a"),
        (ok_both_ids,        f"all three ids in dictionary_ids: {trie4.dictionary_ids}"),
    ]:
        passed += ok; failed += not ok
        print(f"  {'✓' if ok else '✗'}  {desc}")

    # ── 5. fst_stem_lookup uses the SuffixTrie fast path ─────────────────────
    print(f"\n{'─' * 72}")
    print("  5. fst_stem_lookup step-2 SuffixTrie fast path")
    print(f"{'─' * 72}")

    # Build a minimal CompiledRules with no suffix_rules (empty) so that
    # only the SuffixTrie (step 2) can succeed for trie-restorable stems.
    empty_rules = _mr.CompiledRules(
        suffix_rules={},
        max_suffix_len=0,
        gemination_chars=frozenset(),
        vowels=frozenset(),
        consonants=frozenset(),
        rule_tiers={},
        restoration_tiers={},
    )

    # dict_a words as the lookup dictionary
    lookup_dict = {"happy", "easy", "body", "copy", "empty", "berry"}

    fst_cases = [
        # (stem, expected_found, expected_canonical, description)
        ("happi", True,  "happy", "trie fast path: happi → happy"),
        ("easi",  True,  "easy",  "trie fast path: easi  → easy"),
        ("bodi",  True,  "body",  "trie fast path: bodi  → body"),
        ("copi",  True,  "copy",  "trie fast path: copi  → copy"),
        ("empti", True,  "empty", "trie fast path: empti → empty"),
        ("berri", True,  "berry", "trie fast path: berri → berry"),
        # Direct hit (step 1, before trie)
        ("happy", True,  "happy", "direct lookup: canonical unchanged"),
        # No trie match AND no suffix-dict match → not found
        ("xyzzy", False, "xyzzy", "no match in trie or dict → (False, stem)"),
    ]

    for stem, exp_found, exp_canon, desc in fst_cases:
        found, canon = _mr.fst_stem_lookup(
            stem,
            dictionary=lookup_dict,
            combining_forms=set(),
            prefixes=set(),
            suffixes=set(),
            rules=empty_rules,
        )
        ok = (found == exp_found and canon == exp_canon)
        passed += ok; failed += not ok
        print(f"  {'✓' if ok else '✗'}  fst_stem_lookup({stem!r}) → "
              f"({found}, {canon!r})  (expected ({exp_found}, {exp_canon!r}))  [{desc}]")

    total = passed + failed
    pct = passed / total * 100 if total else 0
    print(f"\n  ensure_restoration_trie: {passed}/{total} passed ({pct:.0f}%)")
    return passed, failed


def test_suffix_trie_all():
    """
    Run all SuffixTrie-related tests and print a combined summary.
    """
    print("\n" + "═" * 72)
    print("  SuffixTrie — Full Test Suite")
    print("═" * 72)

    total_passed = total_failed = 0

    for fn in [
        test_suffix_trie_basic,
        test_suffix_trie_longest_match,
        test_suffix_trie_len_and_repr,
        test_build_restoration_dict,
        test_build_restoration_trie,
        test_suffix_trie_set_comprehension_usage,
        test_ensure_restoration_trie,
    ]:
        p, f = fn()
        total_passed += p
        total_failed += f

    total = total_passed + total_failed
    pct = total_passed / total * 100 if total else 0
    print(f"\n{'═' * 72}")
    print(f"  SuffixTrie Suite Total: {total_passed}/{total} passed ({pct:.0f}%)")
    print(f"{'═' * 72}")
    return total_passed, total_failed


if __name__ == "__main__":
    test_english_coverage()
    test_get_profile()
    test_suffix_trie_all()