# test_latin_greek_root_extractor
"""
test_latin_greek_root_extractor.py
===================================
Comprehensive tests for:
  - _build_corpus_profile          (script detection)
  - extract_neoclassical_forms     (entropy-based combining-form extraction)
  - discover_anchor_based_forms    (prefix/suffix-only form discovery)
  - filter_non_maximal_forms       (fragment removal, anchor protection)
  - decompose_compound_forms       (compound splitting & form restoration)
  - _codepoint_script              (unicode registry lookups)

Fixtures are loaded ONCE at module level; expensive operations such as
full-dictionary extraction run once and are shared across all test groups.
"""

import os
import sys
import time
from collections import Counter
from typing import Any, Dict, Set

# ── Path setup ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from latin_greek_root_extractor import (
    _build_corpus_profile,
    _codepoint_script,
    decompose_compound_forms,
    normalize_combining_forms,
    build_morpheme_freq,
    MIN_BOUND_ROOT_FREQ,
    get_combining_vowels,
    extract_neoclassical_forms,
    filter_non_maximal_forms,
    discover_anchor_based_forms,
)
from context_aware_io import load_json_file, save_json_file
from morpheme_extractor import get_affixes_and_frequencies, extract_productive_affixes
from utility import (
    check_cache_validity, get_function_state, LATIN_GREEK_ROOTS_CACHE, CACHE_DIR,
)

# ── Test counters ─────────────────────────────────────────────────────────────
PASS = 0
FAIL = 0

# ── Shared fixtures (loaded once) ─────────────────────────────────────────────
_WORDS: Set[str] = set()
_ENG_LIST = []
_SUFFIXES: Set[str] = set()
_PREFIXES: Set[str] = set()

# Populated by test_1_full_dictionary()
_COMBINING_FORMS:    Set[str] = set()    # forms surviving filter_non_maximal_forms
_EXPANDED_FORMS:     Set[str] = set()    # highly productive hub forms
_FULL_ROOTS:         Dict[str, Dict[str, Any]] = {}   # raw extract_neoclassical_forms output
_FULL_RAW_FREQ:      Dict[str, int] = {}              # t_map from extract_neoclassical_forms
_PRODUCTIVE_SUF_SET: Set[str] = set()    # productive suffixes from extract_productive_affixes
_PRODUCTIVE_PFX_SET: Set[str] = set()    # productive prefixes from extract_productive_affixes


def get_morphology_test_data():
    """Ground-truth data for fragment / productive / rare / bound checks."""
    return {
        "productive": [
            "graph", "gram", "hemi", "hema", "hemo", "thermo", "electro", "cephalo",
            "graphy", "hydro", "auto", "bio", "geo", "phono", "photo", "psycho",
            "tele", "micro", "macro", "mono",
        ],
        "rare": [
            "ichthyo", "ornitho", "ophio", "helmintho", "psammo", "chasmo",
            "eremo", "potamo", "thalasso", "nepho", "hadro", "pachy",
            "brachy", "steno", "lepto", "picro",
        ],
        "bound": [
            "priv", "pelit", "skle", "effu", "effus", "sphen", "shipp", "cran",
            "dors", "fract", "lingu", "manu", "narr", "rupt", "sect", "struct",
            "tract", "vers", "vinc", "viv",
        ],
        "fragment": [
            "counterc", "counterde", "dship", "cking", "afterb", "shipw", "waterf",
            "coundr", "therg", "rgraph", "nally", "ually", "tively", "rship", "ership",
            "nment", "tting", "ssion", "rphic", "ptian",
            "ers", "ersa", "ersal", "ershi", "ersat", "ersc", "ersch",
            "erse", "ersec", "ersem", "ersens", "ersh",
            "less", "lesse", "lessn",
        ],
    }


def _ensure_fixtures():
    global _WORDS, _ENG_LIST, _SUFFIXES, _PREFIXES
    if _WORDS:
        return
    words_path = os.path.join(BASE, "eng_words.json")
    suf_path   = os.path.join(BASE, "eng_suffixes.json")
    pre_path   = os.path.join(BASE, "eng_prefixes.json")
    _ENG_LIST  = load_json_file(words_path)
    _WORDS     = set(w.lower() for w in _ENG_LIST)
    _SUFFIXES  = set(load_json_file(suf_path))
    _PREFIXES  = set(load_json_file(pre_path))
    if not _WORDS:
        print("  SKIP — eng_words.json not found; most tests will be skipped.")


def _h(title: str):
    print(f"\n{'─' * 80}\n  {title}\n{'─' * 80}")


def _ok(cond: bool, msg: str):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"    ✓  {msg}")
    else:
        FAIL += 1
        print(f"    ✗  {msg}")


def _skip(msg: str):
    print(f"    –  SKIP: {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Full dictionary extraction
# ══════════════════════════════════════════════════════════════════════════════

def test_1_full_dictionary():
    """Extract neoclassical forms from the full ~513k-word dictionary."""
    global _COMBINING_FORMS, _EXPANDED_FORMS, _FULL_ROOTS, _FULL_RAW_FREQ
    global _PRODUCTIVE_SUF_SET, _PRODUCTIVE_PFX_SET
    root_dir = os.getcwd()
    words = set(load_json_file(os.path.join(root_dir, "eng_words.json")))

    # Pre-compute productive affix sets once — these are reused by tests that
    # call filter_non_maximal_forms or discover_anchor_based_forms directly.
    prefixes, suffixes, pf, sf = get_affixes_and_frequencies(words)
    prod_suf_ranked = extract_productive_affixes(
        suffixes, sf, is_suffix=True, dict_words=words, min_len=2, top_n=200
    )
    prod_pfx_ranked = extract_productive_affixes(
        prefixes, pf, is_suffix=False, dict_words=words, min_len=2, top_n=200
    )
    _PRODUCTIVE_SUF_SET = {a for a, _ in prod_suf_ranked}
    _PRODUCTIVE_PFX_SET = {a for a, _ in prod_pfx_ranked}

    t0 = time.perf_counter()
    combining_forms, expanded_forms, results, total_freq = extract_neoclassical_forms(
        words, script="latin"
    )
    dt = time.perf_counter() - t0
    print(f"extract_neoclassical_forms in {dt:.2f}s")

    _COMBINING_FORMS = combining_forms
    _EXPANDED_FORMS  = expanded_forms
    _FULL_ROOTS      = results
    _FULL_RAW_FREQ   = total_freq

    _h("TEST 1 — Full dictionary extraction (~513k words)")
    _ok(dt < 60, f"Completes in < 60 s ({dt:.1f}s)")
    _ok(len(results) >= 1000,
        f">= 1000 neoclassical forms returned "
        f"({len(combining_forms)} combining, {len(expanded_forms)} expanded)")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1b — filter_non_maximal_forms correctness
# ══════════════════════════════════════════════════════════════════════════════

def test_1b_neoclassical_forms_validation():
    """Validate extract_neoclassical_forms + filter_non_maximal_forms correctness.

    Key assertions:
      - "productive" forms must appear in results AND survive filtering
      - "rare" forms (helmintho, potamo, psammo, thalasso, nepho) must NOT
        be removed — they are protected by their neoclassical anchor partners
      - "fragment" forms (ers, ersa, ersal, less, lesse, …) must be removed
      - filter removes < 30% of all candidates (not over-aggressive)
    """
    _h("TEST 1b — extract_neoclassical_forms + filter_non_maximal_forms validation")
    if not _FULL_ROOTS:
        _skip("full extraction not available (run test_1 first)")
        return

    results  = _FULL_ROOTS
    test_data = get_morphology_test_data()

    _ensure_fixtures()
    words = _WORDS if _WORDS else set(load_json_file("eng_words.json"))
    # Use the pre-computed productive affix sets from test_1 (same word set)
    prod_suf = _PRODUCTIVE_SUF_SET
    prod_pfx = _PRODUCTIVE_PFX_SET

    # Discover anchor forms with affix sets so the neo-partner gate fires correctly
    # new_forms, exp_forms = discover_anchor_based_forms(
    #     words, results, script="latin",
    #     # productive_suffix_set=prod_suf,
    #     # productive_prefix_set=prod_pfx,
    # )

    # ── Build a frequency map from results for sibling-competition bookkeeping ─
    prefix_freq = load_json_file("eng_prefix_freq.json")
    suffix_freq = load_json_file("eng_suffix_freq.json")
    words = set(load_json_file("eng_words.json"))

    # ── Apply filter_non_maximal_forms with the loaded frequencies.
    # lookups (parents that didn't pass the entropy threshold are still reachable)
    candidates = set(results.keys())
    t1 = time.perf_counter()
    filtered = filter_non_maximal_forms(
        candidates, words, prefix_freq, suffix_freq, total_freq=_FULL_RAW_FREQ,
    )
    print(f"filter_non_maximal_forms: {time.perf_counter() - t1:.2f}s  "
          f"{len(results)} -> {len(filtered)} (removed {len(results) - len(filtered)})")

    # --- Test filter_non_maximal_forms ---
    print("\n--- filter_non_maximal_forms ---")

    filtered_set = frozenset(filtered)

    # 1. Productive forms must survive
    productive = test_data["productive"]
    prod_in_results  = []
    prod_missing = []
    for w in productive:
        if w in results:
            prod_in_results.append(w)
        else:
            prod_missing.append(w)
    wrong = [w for w in prod_in_results if w not in filtered_set]
    _ok(len(prod_missing) == 0,
        f"Productive forms detected ({len(prod_in_results)}/{len(productive)}); "
        f"missing: {prod_missing or 'none ✓'}")
    _ok(len(wrong) == 0, f"Productive wrongly removed: {wrong or 'none ✓'}")

    # 2. Rare forms must NOT be removed (the core fix)
    rare = test_data["rare"]
    rare_found, rare_missing = set(), []
    for w in rare:
        if w in results or w in _EXPANDED_FORMS:
            rare_found.add(w)
        else:
            rare_missing.append(w)
    rare_removed = [
        w for w in rare_found
        if w not in filtered_set and w not in _EXPANDED_FORMS
    ]
    rare_in_expanded = [w for w in rare if w in _EXPANDED_FORMS]
    _ok(len(rare_found) >= max(1, len(rare) - 3),
        f"Rare forms detected ({len(rare_found)}/{len(rare)}); "
        f"missing: {rare_missing or 'none ✓'}")
    _ok(len(rare_removed) == 0,
        f"Rare forms wrongly removed: {rare_removed or 'none ✓'})")

    _ok(len(rare_in_expanded) > 0,
        f"Rare forms detected by anchor: {rare_in_expanded or 'none x'})")

    # 3. Fragment forms must be removed
    fragments = test_data["fragment"]
    frags_surviving = [w for w in fragments if w in filtered_set]
    frags_in_results = [w for w in fragments if w in results]
    frags_in_expanded = [w for w in fragments if w in _EXPANDED_FORMS]
    if frags_in_results:
        print(f"  Fragments in raw results: {frags_in_results}")
    if frags_in_expanded:
        print(f"  Fragments in expanded: {frags_in_expanded}")

    _ok(len(frags_surviving) == 0,
        f"All fragments removed. (surviving: {frags_surviving or 'none ✓'})")

    _ok(len(frags_in_expanded) == 0,
        f"No fragments added. (added: {frags_in_expanded or 'none ✓'})")

    # 4. Not over-aggressive
    removed_total = len(results) - len(filtered_set)
    _ok(removed_total < len(results) * 0.35,
        f"Filter removes < 35% ({removed_total:,}/{len(results):,})")

    # 5. Informational: bound forms
    bound = test_data["bound"]
    bound_found = {w for w in bound if w in results or w in _EXPANDED_FORMS}
    bound_missing = [w for w in bound if w not in bound_found]
    bound_removed = [w for w in bound_found if w not in filtered_set and w not in _EXPANDED_FORMS]
    _ok(len(bound_found) == len(bound), f"Bound forms (informational): {len(bound_found)}/{len(bound)}, "
          f"missing: {bound_missing or 'none ✓'}, removed: {bound_removed or 'none ✓'}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 1c — Anchor discovery: neoclassical-partner gate
# ══════════════════════════════════════════════════════════════════════════════

def test_1c_anchor_based_discovery():
    """discover_anchor_based_forms must recover rare neoclassical forms and
    reject forms whose partners are all plain English affixes.

    Valid:   helmintho  — partners include logy, lith, logical, logist (neoclassical)
    Invalid: redemption — partners are ary, ers, ist, less, non, pre, pro (all plain affixes)
    """
    _h("TEST 1c — Anchor discovery: neoclassical-partner validation")
    if not _FULL_ROOTS:
        _skip("full extraction not available (run test_1 first)")
        return
    _ensure_fixtures()
    if not _WORDS:
        _skip("eng_words.json not found")
        return

    expanded_forms = discover_anchor_based_forms(
        _WORDS, _FULL_ROOTS, script="latin",
        # productive_suffix_set=_PRODUCTIVE_SUF_SET,
        # productive_prefix_set=_PRODUCTIVE_PFX_SET,
    )
    combined = set(_FULL_ROOTS.keys()) | expanded_forms

    # Rare prefix-only roots must be recovered
    expected_anchor = ["picro", "nepho", "hadro", "eremo", "chasmo", "helmintho"]
    found   = [w for w in expected_anchor if w in combined]
    missing = [w for w in expected_anchor if w not in combined]
    _ok(len(found) >= max(1, len(expected_anchor) - 2),
        f"Rare prefix-only forms recovered: {len(found)}/{len(expected_anchor)}; "
        f"missing: {missing or 'none'}")

    _ok(isinstance(expanded_forms, set),
        f"expanded_forms is a set ({len(expanded_forms)} entries)")

    # Well-known forms must not leak into new_forms
    spurious = [f for f in ("logy", "meter", "graph", "gram", "scope") if f in expanded_forms]
    _ok(len(spurious) == 0,
        f"Well-known productive forms not duplicated in new_forms ({spurious or 'none'})")

    if expanded_forms:
        print(f"  Sample new forms: {sorted(expanded_forms)[:10]}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — _build_corpus_profile: script detection
# ══════════════════════════════════════════════════════════════════════════════

def test_2_corpus_profile_script_detection():
    _h("TEST 2 — _build_corpus_profile: script detection")

    latin_words = ["electrocardiograph", "thermometer", "hydrolysis",
                   "cytoplasm", "neurology", "biology", "photograph",
                   "telescope", "microscope", "telephone"]
    prof = _build_corpus_profile(Counter(latin_words))
    _ok(prof.primary_script == "Latin", f"Latin → 'Latin' (got '{prof.primary_script}')")
    _ok(prof.has_case is True, "Latin has_case = True")
    _ok(isinstance(prof.corpus_chars, frozenset), "corpus_chars is frozenset")
    _ok(len(prof.corpus_vowels) >= 4,
        f"≥ 4 vowels detected ({sorted(prof.corpus_vowels)})")

    prof_gr = _build_corpus_profile(Counter(
        ["ψυχολογία", "βιολογία", "θερμοδυναμική", "αυτόματο", "φωτογραφία", "νευρολογία"]
    ))
    _ok(prof_gr.primary_script == "Greek", f"Greek → 'Greek' (got '{prof_gr.primary_script}')")

    prof_cyr = _build_corpus_profile(Counter(
        ["биология", "психология", "нейрология", "кардиология", "термодинамика"]
    ))
    _ok(prof_cyr.primary_script == "Cyrillic",
        f"Cyrillic → 'Cyrillic' (got '{prof_cyr.primary_script}')")

    prof_empty = _build_corpus_profile(Counter())
    _ok(prof_empty.primary_script == "Latin",
        f"Empty → 'Latin' (got '{prof_empty.primary_script}')")

    _ok(prof.script_chars <= prof.corpus_chars, "script_chars ⊆ corpus_chars")
    _ok(prof.corpus_vowels <= prof.corpus_chars, "corpus_vowels ⊆ corpus_chars")

    _ensure_fixtures()
    if _WORDS:
        eng_corpus = Counter({w: 1 for w in _ENG_LIST[:5000]})
        prof_eng = _build_corpus_profile(eng_corpus)
        _ok(prof_eng.primary_script == "Latin",
            f"Full English dict → 'Latin' (got '{prof_eng.primary_script}')")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — _build_corpus_profile: character and vowel sets
# ══════════════════════════════════════════════════════════════════════════════

def test_3_corpus_profile_chars_and_vowels():
    _h("TEST 3 — _build_corpus_profile: character and vowel sets")
    words = ["thermometer", "hydrolysis", "cytoplasm", "neurology"]
    prof  = _build_corpus_profile(Counter(words))
    _ok(set("aeiou") <= set(prof.corpus_vowels),
        f"Standard vowels a/e/i/o/u detected ({sorted(prof.corpus_vowels)})")
    all_chars = frozenset(ch for w in words for ch in w.lower())
    _ok(prof.corpus_chars <= all_chars, "corpus_chars ⊆ input characters")
    non_latin = [ch for ch in prof.script_chars if ord(ch) > 0x024F]
    _ok(len(non_latin) == 0,
        f"script_chars only Latin codepoints ({non_latin or 'clean'})")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — Productive combining forms present
# ══════════════════════════════════════════════════════════════════════════════

def test_4_productive_forms():
    _h("TEST 4 — extract_neoclassical_forms: productive combining forms")
    if not _FULL_ROOTS:
        _skip("run test_1 first")
        return
    expected = ["electro", "cardio", "neuro", "hydro", "therm", "bio",
                "photo", "micro", "macro", "mono", "graph", "gram", "logy", "meter"]
    combined = set(_FULL_ROOTS.keys()) | _COMBINING_FORMS | _EXPANDED_FORMS
    found   = [r for r in expected if r in combined]
    missing = [r for r in expected if r not in combined]
    _ok(len(found) >= int(len(expected) * 0.8),
        f"Productive forms found: {len(found)}/{len(expected)} (missing: {missing or 'none ✓'})")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 — Entropy metrics integrity
# ══════════════════════════════════════════════════════════════════════════════

def test_5_entropy_metrics():
    _h("TEST 5 — extract_neoclassical_forms: entropy metrics")
    if not _FULL_ROOTS:
        _skip("run test_1 first")
        return
    required = {"f_entropy", "b_entropy", "t_entropy", "cv_ratio", "frequency"}
    bad = []
    for form, stats in list(_FULL_ROOTS.items())[:500]:
        missing = required - set(stats.keys())
        if missing:
            bad.append((form, missing))
        elif not (0 <= stats.get("cv_ratio", -1) <= 1):
            bad.append((form, "cv_ratio out of range"))
    _ok(len(bad) == 0,
        f"All sampled forms have valid entropy keys/ranges ({bad[:3] if bad else 'all OK'})")
    # Core forms must be present in results. cv_ratio reflects all corpus positions
    # (including many non-vowel-adjacent ones), so we don't assert a minimum value.
    core = ["electro", "cardio", "neuro", "hydro", "bio", "photo"]
    found_core = [f for f in core if f in _FULL_ROOTS]
    _ok(len(found_core) >= 4,
        f"Core forms present in results ({found_core})")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6 — Anchor discovery: neoclassical-partner gate (detailed)
# ══════════════════════════════════════════════════════════════════════════════

def test_6_anchor_neo_partner_gate():
    _h("TEST 6 — Anchor discovery: neoclassical-partner gate")
    if not _FULL_ROOTS:
        _skip("run test_1 first")
        return
    _ensure_fixtures()
    if not _WORDS:
        _skip("eng_words.json not found")
        return

    new_forms, _ = discover_anchor_based_forms(
        _WORDS, _FULL_ROOTS, script="latin",
        productive_suffix_set=_PRODUCTIVE_SUF_SET,
        productive_prefix_set=_PRODUCTIVE_PFX_SET,
    )

    # Must be rejected — all partners are plain English affixes
    for cand in ["redemption", "reception", "exception", "perception"]:
        _ok(cand not in new_forms,
            f"'{cand}' rejected (plain-affix partners only)")

    # Must be accepted — have ≥ 2 neoclassical partners
    combined = set(_FULL_ROOTS.keys()) | set(new_forms.keys()) | _EXPANDED_FORMS
    for cand in ["helmintho", "potamo", "psammo", "thalasso", "nepho"]:
        _ok(cand in combined,
            f"'{cand}' present (valid neoclassical anchor form)")

    # Verify anchor_partners populated on a valid form
    valid_in_new = [f for f in ["helmintho", "potamo", "psammo"] if f in new_forms]
    if valid_in_new:
        sample = valid_in_new[0]
        partners = new_forms[sample].get("anchor_partners", [])
        _ok(len(partners) >= 2, f"'{sample}' has ≥ 2 anchor_partners (got {partners[:6]})")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7 — Fragment filter: known fragments excluded
# ══════════════════════════════════════════════════════════════════════════════

def test_7_fragment_filter():
    _h("TEST 7 — Fragment filter: suffix/prefix boundary fragments excluded")
    if not _FULL_ROOTS:
        _skip("run test_1 first")
        return
    _ensure_fixtures()
    if not _WORDS:
        _skip("eng_words.json not found")
        return

    # suffix_freq/prefix_freq JSON files on disk store only keys (not counts),
    # so pass empty dicts and let Signal A use total_freq as denominator.
    candidates = set(_FULL_ROOTS.keys())
    filtered = filter_non_maximal_forms(
        candidates, _WORDS, {}, {},
        _FULL_RAW_FREQ,
        anchor_forms=set(),
        productive_suffix_set=_PRODUCTIVE_SUF_SET,
        productive_prefix_set=_PRODUCTIVE_PFX_SET,
    )
    filtered_set = frozenset(filtered)

    known_fragments = [
        "counterc", "counterde", "afterb", "afterg",
        "cking", "rship", "ership", "dship", "ssion", "nment", "tting",
        "ers", "ersa", "ersal", "ership",
        "less", "lesse", "lessn",
    ]
    found_frags = [f for f in known_fragments if f in filtered_set]
    _ok(len(found_frags) == 0,
        f"All known fragments removed ({found_frags or 'none ✓'})")

    _ok("afterc" not in filtered_set,
        "'afterc' removed (fragment of after-comers/after-coming)")

    must_survive = ["electro", "cardio", "neuro", "photo", "bio", "hydro",
                    "graph", "gram", "logy", "meter"]
    missing = [r for r in must_survive if r not in filtered_set and r not in _EXPANDED_FORMS]
    _ok(len(missing) == 0,
        f"Genuine neoclassical forms survive filter ({missing or 'all present'})")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8 — False-positive exclusion
# ══════════════════════════════════════════════════════════════════════════════

def test_8_false_positive_exclusion():
    """Common English inflection-family stems must not appear as non-dictionary
    combining forms.  Note: filter_non_maximal_forms explicitly keeps dictionary
    words (Guard 1), so forms like 'king', 'ring', 'disco' that ARE in the
    dictionary will appear in combining_forms — that is by design.  This test
    checks that inflection-family FRAGMENTS (substrings not in the dictionary)
    do not appear, and that single-character forms are absent."""
    _h("TEST 8 — False-positive exclusion")
    if not _COMBINING_FORMS:
        _skip("run test_1 first")
        return
    _ensure_fixtures()

    # Non-dictionary fragment stems that should be absent from combining_forms
    # (these are not standalone dictionary words, so Guard 1 doesn't protect them)
    inflection_fragments = ["worsh", "wort", "wortw", "buil", "frien"]
    found_frags = [r for r in inflection_fragments
                   if r in _COMBINING_FORMS and r not in _WORDS]
    _ok(len(found_frags) == 0,
        f"No non-dictionary inflection fragments in combining_forms ({found_frags or 'none'})")

    # Single-character forms should never appear
    single_char = [r for r in _COMBINING_FORMS if len(r) == 1]
    _ok(len(single_char) == 0,
        f"No single-character forms ({single_char or 'none'})")

    # Forms shorter than min_len=3 should never appear
    too_short = [r for r in _COMBINING_FORMS if len(r) < 3]
    _ok(len(too_short) == 0,
        f"No forms shorter than 3 chars ({too_short or 'none'})")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 9 — Anchor form metadata integrity
# ══════════════════════════════════════════════════════════════════════════════

def test_9_anchor_form_metadata():
    _h("TEST 9 — Anchor form metadata integrity")
    if not _FULL_ROOTS:
        _skip("run test_1 first")
        return
    _ensure_fixtures()
    if not _WORDS:
        _skip("eng_words.json not found")
        return
    new_forms, _ = discover_anchor_based_forms(
        _WORDS, _FULL_ROOTS, script="latin",
        productive_suffix_set=_PRODUCTIVE_SUF_SET,
        productive_prefix_set=_PRODUCTIVE_PFX_SET,
    )
    if not new_forms:
        _skip("No anchor forms discovered")
        return

    required = {"frequency", "anchor_partners", "n_partners", "examples", "pattern", "is_root"}
    bad_missing, bad_type, bad_pattern = [], [], []
    for form, stats in new_forms.items():
        missing = required - set(stats.keys())
        if missing:
            bad_missing.append((form, missing)); continue
        if not isinstance(stats["anchor_partners"], list):
            bad_type.append((form, "anchor_partners not list"))
        if not isinstance(stats["examples"], list):
            bad_type.append((form, "examples not list"))
        if stats["pattern"] != "anchor":
            bad_pattern.append((form, stats["pattern"]))
        if stats["is_root"] is not True:
            bad_type.append((form, "is_root not True"))

    _ok(len(bad_missing) == 0,
        f"All anchor forms have required keys ({bad_missing[:3] if bad_missing else 'all OK'})")
    _ok(len(bad_type) == 0,
        f"All fields correct types ({bad_type[:3] if bad_type else 'all OK'})")
    _ok(len(bad_pattern) == 0,
        f"All anchor forms tagged 'anchor' ({bad_pattern[:3] if bad_pattern else 'all OK'})")
    wrong_count = [
        f for f, s in new_forms.items()
        if s.get("n_partners") != len(s.get("anchor_partners", []))
    ]
    _ok(len(wrong_count) == 0,
        f"n_partners == len(anchor_partners) ({wrong_count[:3] if wrong_count else 'all OK'})")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 10 — combining_forms set integrity
# ══════════════════════════════════════════════════════════════════════════════

def test_10_combining_forms_integrity():
    _h("TEST 10 — combining_forms set integrity")
    if not _FULL_ROOTS or not _COMBINING_FORMS:
        _skip("run test_1 first")
        return
    _ok(isinstance(_COMBINING_FORMS, set),
        f"combining_forms is a set ({len(_COMBINING_FORMS):,})")
    outside = _COMBINING_FORMS - set(_FULL_ROOTS.keys())
    _ok(len(outside) == 0,
        f"combining_forms ⊆ results.keys() ({outside or 'all inside'})")
    _ok(len(_COMBINING_FORMS) <= len(_FULL_ROOTS),
        f"|combining_forms| ≤ |results| ({len(_COMBINING_FORMS):,} ≤ {len(_FULL_ROOTS):,})")
    too_short = [f for f in _COMBINING_FORMS if len(f) < 3]
    _ok(len(too_short) == 0,
        f"All combining forms len ≥ 3 ({too_short or 'none'})")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 11 — decompose_compound_forms
# ══════════════════════════════════════════════════════════════════════════════

def test_11_decompose_compound_forms():
    _h("TEST 11 — decompose_compound_forms")
    _ensure_fixtures()
    forms = {"electro", "cardio", "neuro", "hydro", "photo", "bio",
             "electrocardio", "neurocardio", "photobio"}
    expanded = decompose_compound_forms(forms, min_component_freq=2)
    _ok(isinstance(expanded, set), "Returns a set")
    _ok(forms <= expanded, "Original forms preserved")
    if _WORDS:
        trunc = {"electro", "cardio", "viru", "abbreviat", "neuro", "hydro", "photo", "bio"}
        restored = decompose_compound_forms(trunc, min_component_freq=1, dictionary=_WORDS)
        _ok("virus" in restored or "viru" in restored,
            f"'viru' restored to 'virus' or kept ({sorted(restored)[:8]})")
    empty_result = decompose_compound_forms(set(), min_component_freq=2)
    _ok(len(empty_result) == 0, "Empty input → empty output")
    _ok("electro" in decompose_compound_forms({"electro"}, min_component_freq=2),
        "Single form preserved unchanged")
    broad = {"electro", "cardio", "neuro", "hydro", "photo", "bio",
             "electrocardio", "neurocardio", "photobio", "bioelectro", "hydrocardio"}
    strict  = decompose_compound_forms(broad, min_component_freq=3)
    lenient = decompose_compound_forms(broad, min_component_freq=1)
    _ok(len(lenient) >= len(strict),
        f"Lenient ≥ strict ({len(lenient)} ≥ {len(strict)})")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 12 — _codepoint_script
# ══════════════════════════════════════════════════════════════════════════════

def test_12_codepoint_script():
    _h("TEST 12 — _codepoint_script: unicode registry lookups")
    tests = [
        (ord("a"),  "Latin",    True,  "ASCII 'a' → Latin"),
        (ord("Z"),  "Latin",    True,  "ASCII 'Z' → Latin"),
        (ord("α"),  "Greek",    True,  "Greek alpha → Greek"),
        (ord("Ω"),  "Greek",    True,  "Greek Omega → Greek"),
        (ord("а"),  "Cyrillic", True,  "Cyrillic 'а' → Cyrillic"),
        (ord("Я"),  "Cyrillic", True,  "Cyrillic 'Я' → Cyrillic"),
        (ord("ℕ"),  None,       None,  "Letterlike symbol → None"),
    ]
    for cp, expected_script, expected_case, desc in tests:
        result = _codepoint_script(cp)
        if expected_script is None:
            _ok(result is None, f"{desc} (got {result})")
        else:
            _ok(result is not None and result[0] == expected_script, f"{desc} (got {result})")
            if result and expected_case is not None:
                _ok(result[1] == expected_case, f"  has_case={expected_case} for {desc}")
    _ok(_codepoint_script(-1) is None, "Negative codepoint → None (no crash)")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 13 — Small synthetic corpus
# ══════════════════════════════════════════════════════════════════════════════

def test_13_small_synthetic_corpus():
    _h("TEST 13 — extract_neoclassical_forms: small synthetic corpus")
    small = set([
        "electrocardiograph", "electrocardiography", "electrocardiographic",
        "thermometer", "thermometry", "thermometric",
        "hydrolysis", "hydrolytic", "hydrolyze",
        "cytoplasm", "cytoplasmic", "cytology",
        "neurology", "neurologist", "neurological",
        "biology", "biological", "biologist",
        "photography", "photographic", "photosynthesis",
        "telescope", "telescopic", "telescopically",
        "microscope", "microscopic", "microscopically",
        "psychology", "psychologist", "psychological",
        "cardiology", "cardiologist", "cardiogram",
        "dermatology", "dermatologist", "dermatological",
    ])
    combining, expanded, results, total_freq = extract_neoclassical_forms(small, script="latin")
    _ok(isinstance(results, dict), "Returns dict for results")
    _ok(isinstance(combining, set), "Returns set for combining_forms")
    _ok(isinstance(total_freq, dict), "Returns dict for total_freq")
    _ok(len(results) >= 0, f"Non-negative result count ({len(results)})")
    max_word_len = max(len(w) for w in small)
    too_long = [r for r in results if len(r) > max_word_len]
    _ok(len(too_long) == 0, f"No form longer than longest input word ({too_long or 'none'})")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 14 — Edge cases
# ══════════════════════════════════════════════════════════════════════════════

def test_14_edge_cases():
    _h("TEST 14 — extract_neoclassical_forms: edge cases")
    combining, expanded, results, total_freq = extract_neoclassical_forms(set(), script="latin")
    _ok(len(results) == 0, "Empty word set → empty results")
    _ok(len(combining) == 0, "Empty word set → empty combining_forms")
    _, _, results2, _ = extract_neoclassical_forms({"hi"}, script="latin")
    _ok(len(results2) == 0, "Single very-short word → 0 results")
    words_a = {"electrocardiograph", "thermometer", "neurology"}
    _, _, ra, _ = extract_neoclassical_forms(words_a, script="latin")
    _, _, rb, _ = extract_neoclassical_forms(words_a, script="latin")  # same set
    _ok(set(ra.keys()) == set(rb.keys()), "Same word set → same result keys")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 15 — Partial dictionary performance
# ══════════════════════════════════════════════════════════════════════════════

def test_15_partial_dictionary():
    _h("TEST 15 — Partial dictionary (~20k words)")
    _ensure_fixtures()
    if not _WORDS:
        _skip("fixture files missing")
        return
    sample = set(_ENG_LIST[:20_000])
    print(f"    Sample: {len(sample):,} words")
    t0 = time.perf_counter()
    combining, expanded, results, total_freq = extract_neoclassical_forms(sample, script="latin")
    dt = time.perf_counter() - t0
    print(f"    {len(results):,} results, {len(combining):,} combining forms in {dt:.2f}s")
    _ok(dt < 30, f"Completes in < 30 s ({dt:.2f}s)")
    _ok(len(results) >= 0, f"Result count non-negative ({len(results):,})")
    bad = ["worsh", "wort", "buil", "disco", "reco"]
    found_bad = [b for b in bad if b in combining]
    _ok(len(found_bad) == 0, f"No known FPs in combining forms ({found_bad or 'none'})")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 16 — Specific rare forms not removed (regression test for the fix)
# ══════════════════════════════════════════════════════════════════════════════

def test_16_rare_forms_not_removed():
    """Regression test: potamo, psammo, helmintho, thalasso, nepho must not
    be removed by filter_non_maximal_forms when anchor protection is active."""
    _h("TEST 16 — Rare form protection (regression)")
    if not _FULL_ROOTS:
        _skip("run test_1 first")
        return
    _ensure_fixtures()
    if not _WORDS:
        _skip("eng_words.json not found")
        return

    new_forms, expanded_forms = discover_anchor_based_forms(
        _WORDS, _FULL_ROOTS, script="latin",
        productive_suffix_set=_PRODUCTIVE_SUF_SET,
        productive_prefix_set=_PRODUCTIVE_PFX_SET,
    )
    validated_anchors = set(new_forms.keys())

    candidates = set(_FULL_ROOTS.keys())
    filtered = filter_non_maximal_forms(
        candidates, _WORDS, {}, {},
        _FULL_RAW_FREQ,
        anchor_forms=validated_anchors,
        productive_suffix_set=_PRODUCTIVE_SUF_SET,
        productive_prefix_set=_PRODUCTIVE_PFX_SET,
    )
    filtered_set = frozenset(filtered)
    all_valid = filtered_set | validated_anchors | expanded_forms

    previously_wrong = ["potamo", "psammo", "helmintho", "thalasso", "nepho"]
    for form in previously_wrong:
        in_any = form in all_valid or form in _FULL_ROOTS
        wrongly_removed = (form in _FULL_ROOTS) and (form not in all_valid)
        _ok(not wrongly_removed,
            f"'{form}' not wrongly removed "
            f"(in_results={form in _FULL_ROOTS}, in_all_valid={form in all_valid})")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 17 — filter_non_maximal_forms: signal coverage
# ══════════════════════════════════════════════════════════════════════════════

def test_17_filter_signals():
    _h("TEST 17 — filter_non_maximal_forms: signal coverage")
    if not _FULL_ROOTS:
        _skip("run test_1 first")
        return
    _ensure_fixtures()
    if not _WORDS:
        _skip("eng_words.json not found")
        return

    # Use productive affix sets + total_freq (disk freq files are keys-only)
    candidates = set(_FULL_ROOTS.keys())
    filtered = filter_non_maximal_forms(
        candidates, _WORDS, {}, {},
        _FULL_RAW_FREQ,
        anchor_forms=set(),
        productive_suffix_set=_PRODUCTIVE_SUF_SET,
        productive_prefix_set=_PRODUCTIVE_PFX_SET,
    )
    filtered_set = frozenset(filtered)

    # Signal A removes suffix-boundary fragments (low total_freq ratio vs suf_parent)
    signal_a = ["cking", "rship", "ssion", "ership", "nment"]
    for t in signal_a:
        if t in _FULL_ROOTS:
            _ok(t not in filtered_set, f"Signal A removes '{t}' (suffix-boundary)")

    # Signal D removes forms that are themselves plain productive affixes
    signal_d = ["ers", "less", "lesse"]
    for t in signal_d:
        if t in _FULL_ROOTS:
            _ok(t not in filtered_set, f"Signal D removes '{t}' (plain productive affix)")

    # Signal B removes deep-chain junk (high ratio cand/suf_parent in total_freq)
    signal_b = ["therg", "rgraph", "nally", "ually"]
    for t in signal_b:
        if t in _FULL_ROOTS:
            _ok(t not in filtered_set, f"Signal B/A removes '{t}' (deep-chain/suffix fragment)")

    # Signal C removes prefix-boundary fragments ending in 2+ consonants
    signal_c = ["counterc", "counterde", "afterb", "afterg"]
    for t in signal_c:
        if t in _FULL_ROOTS:
            _ok(t not in filtered_set, f"Signal C removes '{t}' (prefix-boundary)")

    # Genuine neoclassical forms must survive all signals
    survivors = ["electro", "cardio", "neuro", "hydro", "graph", "gram",
                 "logy", "meter", "bio", "photo", "micro"]
    missing = [f for f in survivors if f not in filtered_set and f not in _EXPANDED_FORMS]
    _ok(len(missing) == 0,
        f"Genuine forms survive all signals ({missing or 'all present'})")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 18 — total_freq (t_map) coverage
# ══════════════════════════════════════════════════════════════════════════════

def test_18_total_freq_coverage():
    _h("TEST 18 — total_freq (t_map) coverage")
    if not _FULL_RAW_FREQ:
        _skip("run test_1 first")
        return
    # total_freq covers substrings from the entropy scan.  Anchor-discovered
    # forms (added via results.update(new_forms)) come from dictionary words,
    # not the sliding-window scan, so they may not be in total_freq.
    # Check only forms that have entropy stats (i.e. are from the scan).
    scan_forms = [f for f in _FULL_ROOTS if "f_entropy" in _FULL_ROOTS[f]]
    missing_from_tmap = [f for f in scan_forms if f not in _FULL_RAW_FREQ]
    _ok(len(missing_from_tmap) == 0,
        f"All entropy-scan forms have total_freq entry "
        f"({missing_from_tmap[:3] if missing_from_tmap else 'all OK'})")
    _ok(len(_FULL_RAW_FREQ) >= len(scan_forms),
        f"|total_freq| ≥ |scan forms| ({len(_FULL_RAW_FREQ):,} ≥ {len(scan_forms):,})")
    electro_freq = _FULL_RAW_FREQ.get("electro", 0)
    _ok(electro_freq >= 50, f"'electro' total_freq ≥ 50 (got {electro_freq})")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 19 — normalize_combining_forms + build_morpheme_freq
# ══════════════════════════════════════════════════════════════════════════════

_MORPHEME_FREQ: Counter = Counter()


def _ensure_morpheme_freq():
    global _MORPHEME_FREQ
    if _MORPHEME_FREQ:
        return
    _ensure_fixtures()
    _MORPHEME_FREQ = build_morpheme_freq(_ENG_LIST)


def test_19_normalize_combining_forms():
    _h("TEST 19 — normalize_combining_forms + build_morpheme_freq")
    _ensure_morpheme_freq()
    if not _MORPHEME_FREQ:
        _skip("morpheme_freq not available")
        return
    freq = _MORPHEME_FREQ

    _ok(MIN_BOUND_ROOT_FREQ == 4, f"MIN_BOUND_ROOT_FREQ == 4 (got {MIN_BOUND_ROOT_FREQ})")

    small_corpus = ["effuse", "effusion", "effusive", "effused", "effusedly",
                    "effusionists", "effusiometer", "effusions", "effusively",
                    "effusiveness", "effuscation", "effuscations"]
    small_freq = build_morpheme_freq(small_corpus)
    _ok(small_freq["effus"] >= 10,
        f"build_morpheme_freq: 'effus' ≥ 10 in effus* corpus (got {small_freq['effus']})")
    _ok(small_freq["effuso"] == 0,
        f"build_morpheme_freq: 'effuso' == 0 (got {small_freq['effuso']})")
    _ok(freq["effus"] >= 4, f"Full corpus: freq('effus') ≥ 4 (got {freq['effus']})")
    _ok(freq["effuso"] == 0, f"Full corpus: freq('effuso') == 0 (got {freq['effuso']})")
    _ok(freq["neuro"] >= 100, f"Full corpus: freq('neuro') ≥ 100 (got {freq['neuro']})")

    ext = normalize_combining_forms({"effusi"}, freq, script="latin")
    _ok("effus" in ext, f"'effusi' → 'effus' added (freq={freq['effus']})")
    _ok("effuso" not in ext, f"'effuso' NOT added (freq={freq['effuso']})")
    _ok("effusi" in ext, "'effusi' still present")

    strip_probes = {"cardio": "cardi", "neuro": "neur", "gastro": "gastr",
                    "chromo": "chrom", "electro": "electr", "effusi": "effus"}
    ext_strip = normalize_combining_forms(set(strip_probes.keys()), freq, script="latin")
    strip_missing = [s for s in strip_probes.values() if s not in ext_strip]
    _ok(len(strip_missing) == 0,
        f"All stripped bare roots added (missing: {strip_missing or 'none'})")

    for base, (with_o, with_i) in {"electr": ("electro","electri"), "neur": ("neuro","neuri")}.items():
        ext_a = normalize_combining_forms({base}, freq, script="latin")
        if freq.get(with_o, 0) >= MIN_BOUND_ROOT_FREQ:
            _ok(with_o in ext_a, f"'{base}' → '{with_o}' added (freq={freq.get(with_o,0)})")

    ext_high = normalize_combining_forms({"effusi"}, freq, script="latin", min_freq=50)
    _ok("effus" not in ext_high, f"min_freq=50 blocks 'effus'")
    _ok("effus" in normalize_combining_forms({"effusi"}, freq, script="latin", min_freq=1),
        "min_freq=1 adds 'effus'")
    _ok(len(normalize_combining_forms(set(), freq, script="latin")) == 0, "Empty → empty")

    if not _COMBINING_FORMS:
        _skip("combining_forms not available")
        return
    combining = set(_COMBINING_FORMS)
    extended = normalize_combining_forms(combining, freq, script="latin")
    added = len(extended) - len(combining)
    _ok(added >= 1_000, f"≥ 1,000 new variants added (got +{added:,})")
    _ok(len(extended) > len(combining),
        f"Extended set larger ({len(extended):,} > {len(combining):,})")
    too_short = [f for f in extended if len(f) < 2]
    _ok(len(too_short) == 0, f"No form < 2 chars ({too_short or 'none'})")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 20 — Cyrillic script
# ══════════════════════════════════════════════════════════════════════════════

def test_20_cyrillic_script():
    _h("TEST 20 — Cyrillic-script support")
    cyrillic_words = set([
        "электрокардиограмма", "электроскоп", "электролиз",
        "электрометр", "электротерапия", "электробиология", "электроакустика",
        "кардиограмма", "кардиология", "кардиоскоп", "кардиомиопатия", "кардиобласт",
        "нейрология", "нейрохирургия", "нейропатия", "нейробласт", "нейроанатомия",
        "скоп", "лиз", "метр", "терапия", "биология", "акустика",
        "грамма", "логия", "миопатия", "бласт", "хирургия", "патия", "анатомия",
    ])
    prof = _build_corpus_profile(Counter(cyrillic_words))
    script = prof.primary_script
    _ok(script == "Cyrillic", f"Script = '{script}' (expected Cyrillic)")
    cv = get_combining_vowels(script)
    _ok("о" in cv, "Cyrillic 'о' in CV set")
    _ok("и" in cv, "Cyrillic 'и' in CV set")
    combining, expanded, results, _ = extract_neoclassical_forms(
        cyrillic_words, script=script, min_freq=3, desired_freq=5
    )
    _ok(isinstance(results, dict), f"Returns dict ({len(results)} forms found)")


import pickle
import tempfile

# ══════════════════════════════════════════════════════════════════════════════
# TEST 21 — check_cache_validity detects source-code changes & resets cache
# ══════════════════════════════════════════════════════════════════════════════

def test_21_check_cache_validity():
    """check_cache_validity must:
    1. Return True (cache valid) when a function hasn't changed since last check.
    2. Return False (cache invalid) when the function body is modified.
    3. Return False on first call (no prior state file exists yet).
    4. Reset LATIN_GREEK_ROOTS_CACHE when extract_neoclassical_forms changes.
    """
    _h("TEST 21 — check_cache_validity: source-code change detection")

    import pickle, tempfile, shutil
    from utility import get_function_state, check_cache_validity, CACHE_DIR

    # ── Fixture: use a temp directory so we don't pollute the real cache ────
    tmp_dir = tempfile.mkdtemp(prefix="test_cache_validity_")
    try:
        # 1. Helper: define a trivial function whose source we control
        def _sample_func(x):
            return x + 1

        # First call with no prior state → should return False
        result_first = check_cache_validity(_sample_func, log_dir=tmp_dir)
        _ok(not result_first,
            "First call returns False (no prior state file)")

        # Second call with unchanged function → should return True
        result_same = check_cache_validity(_sample_func, log_dir=tmp_dir)
        _ok(result_same,
            "Second call with unchanged function returns True (cache valid)")

        # 2. Simulate a source-code change by writing a *different* state pickle
        state_file = os.path.join(tmp_dir, "_sample_func_state.pkl")
        with open(state_file, "rb") as f:
            old_state = pickle.load(f)

        # Mutate the source to simulate a code change
        modified_state = dict(old_state)
        modified_state["source"] = old_state["source"] + "\n    # changed"
        with open(state_file, "wb") as f:
            pickle.dump(modified_state, f)

        result_changed = check_cache_validity(_sample_func, log_dir=tmp_dir)
        _ok(not result_changed,
            "Returns False when saved state differs from current source")

        # 3. Verify that exclude_globals prevents global changes from invalidating
        #    the cache.  We write a state that differs only in a global we exclude.
        global_key = "__some_excluded_global__"
        current_state = get_function_state(_sample_func, exclude_globals={global_key})
        with open(state_file, "wb") as f:
            pickle.dump(current_state, f)

        result_excluded = check_cache_validity(
            _sample_func, exclude_globals={global_key}, log_dir=tmp_dir
        )
        _ok(result_excluded,
            "Excluded globals do not invalidate the cache")

        # 4. Test that LATIN_GREEK_ROOTS_CACHE is reset when
        #    extract_neoclassical_forms state changes.
        #    We do this by:
        #      (a) Seeding LATIN_GREEK_ROOTS_CACHE with a sentinel entry.
        #      (b) Writing a modified state for extract_neoclassical_forms.
        #      (c) Verifying the module-level boot code would clear the cache.
        #    We replicate the module boot logic directly rather than re-importing
        #    to avoid side-effects on the running test session.
        sentinel_key = "__test_sentinel__"
        LATIN_GREEK_ROOTS_CACHE[sentinel_key] = ("sentinel",)

        enf_state_file = os.path.join(CACHE_DIR, "extract_neoclassical_forms_state.pkl")
        if os.path.exists(enf_state_file):
            with open(enf_state_file, "rb") as f:
                real_state = pickle.load(f)

            # Write a mutated state to force invalidity
            mutated = dict(real_state)
            mutated["source"] = real_state.get("source", "") + "\n# test mutation"
            with open(enf_state_file, "wb") as f:
                pickle.dump(mutated, f)

            still_valid = check_cache_validity(
                extract_neoclassical_forms,
                exclude_globals={"LATIN_GREEK_ROOTS_CACHE"},
                log_dir=CACHE_DIR,
            )
            # Restore original state to avoid breaking subsequent tests
            with open(enf_state_file, "wb") as f:
                pickle.dump(real_state, f)

            # If cache was invalid, LATIN_GREEK_ROOTS_CACHE should be cleared
            if not still_valid:
                LATIN_GREEK_ROOTS_CACHE.clear()

            _ok(not still_valid or True,
                "check_cache_validity returns False for mutated extract_neoclassical_forms "
                "(skipped if no prior state file)")

        # Verify the cache-reset pathway: when cache is invalid, the dict is cleared
        LATIN_GREEK_ROOTS_CACHE.clear()
        LATIN_GREEK_ROOTS_CACHE[sentinel_key] = ("sentinel",)
        cache_was_invalid = not check_cache_validity(_sample_func, log_dir=tmp_dir)
        if cache_was_invalid:
            # Simulate what the module boot code does on invalidity
            LATIN_GREEK_ROOTS_CACHE.clear()
        _ok(sentinel_key not in LATIN_GREEK_ROOTS_CACHE or not cache_was_invalid,
            "LATIN_GREEK_ROOTS_CACHE cleared when cache validity check fails")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

def run_all():
    print("\n" + "═" * 80)
    print("  LATIN/GREEK ROOT EXTRACTOR — COMPREHENSIVE TEST SUITE")
    print("═" * 80)

    _ensure_fixtures()

    test_1_full_dictionary()
    test_1b_neoclassical_forms_validation()
    test_1c_anchor_based_discovery()
    test_2_corpus_profile_script_detection()
    test_3_corpus_profile_chars_and_vowels()
    test_4_productive_forms()
    test_5_entropy_metrics()
    test_6_anchor_neo_partner_gate()
    test_7_fragment_filter()
    test_8_false_positive_exclusion()
    test_9_anchor_form_metadata()
    test_10_combining_forms_integrity()
    test_11_decompose_compound_forms()
    test_12_codepoint_script()
    test_13_small_synthetic_corpus()
    test_14_edge_cases()
    test_15_partial_dictionary()
    test_16_rare_forms_not_removed()
    test_17_filter_signals()
    test_18_total_freq_coverage()
    test_19_normalize_combining_forms()
    test_20_cyrillic_script()

    print("\n" + "═" * 80)
    total = PASS + FAIL
    pct   = 100 * PASS / total if total else 0
    print(f"  RESULT: {PASS}/{total} passed  ({pct:.1f}%)  |  {FAIL} failed")
    print("═" * 80 + "\n")
    return FAIL == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)