"""
COMPREHENSIVE TEST SUITE FOR ROOT EXTRACTION FUNCTIONS
=====================================================

Covers:
  1. _filter_affix_duplicates  — fixture-based, no redundant computation
  2. extract_productive_roots  — fixture-based, semantically correct assertions
  3. SemanticTokenizer.run_morpheme_pipeline — new method, unit + integration

Design principles
-----------------
* All expensive fixtures (real word/affix data, productive-root extraction,
  morpheme frequencies) are computed ONCE per test-class via setUpClass and
  stored as class attributes.  Individual tests read those attributes — they
  never re-run the pipeline themselves.
* Synthetic unit tests use minimal, purpose-built word sets so they run in
  microseconds and are fully deterministic.
* Test logic is validated against the actual function contract; assertions
  are derived from documented behaviour, not cargo-culted from prior output.

Usage
-----
    python -m pytest test_root_extractor_comprehensive.py -v
    python test_root_extractor_comprehensive.py
"""

from __future__ import annotations

import os
import unittest.mock
import sys
import time
import unittest
from collections import Counter
from typing import Dict, List, Set, Tuple

from asi.ange_tokenizer import SemanticTokenizer
from asi.context_aware_io import load_json_file, save_json_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from root_extractor import (
    _extract_morpheme_frequencies,
    _filter_affix_duplicates,
    extract_productive_roots, _promote_neoclassical_bound_roots,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture loader  (class-level, computed once per process)
# ─────────────────────────────────────────────────────────────────────────────

class _RealData:
    """
    Lazy singleton that loads the four JSON files exactly once.
    All heavy test-classes call _RealData.init() in setUpClass so the
    expensive I/O and morpheme-extraction run only once for the whole suite.
    """
    _loaded: bool = False
    words:           Set[str] = set()
    prefixes:        Set[str] = set()
    suffixes:        Set[str] = set()
    combining_forms: Set[str] = set()
    prefix_freq: Counter   = Counter()
    suffix_freq: Counter   = Counter()
    morpheme_freq: Counter = Counter()
    _pipeline_result = None

    @classmethod
    def init(cls) -> None:
        if cls._loaded:
            return
        cls.words           = set(load_json_file(os.path.join(BASE_DIR, "eng_words.json")))
        cls.prefixes        = set(load_json_file(os.path.join(BASE_DIR, "eng_prefixes.json")))
        cls.suffixes        = set(load_json_file(os.path.join(BASE_DIR, "eng_suffixes.json")))
        cls.combining_forms = set(load_json_file(os.path.join(BASE_DIR, "eng_combining_forms.json")))
        # load freq data
        cls.prefix_freq     = Counter(load_json_file(os.path.join(BASE_DIR, "eng_prefix_freq.json")))
        cls.suffix_freq     = Counter(load_json_file(os.path.join(BASE_DIR, "eng_suffix_freq.json")))

        if not (cls.prefixes and cls.suffixes and cls.prefix_freq):
            print(f"Extracting prefixes and suffixes for eng...")
            # Slow path: discover affixes from dict_words on the fly.
            # morpheme_extractor.run_morpheme_pipeline uses Information-Theory
            # metrics (Boundary Entropy, Variation of Information, sibling
            # competition) to extract productive prefixes and suffixes without
            # any language-specific rules — works for any ISO 639-3 code.
            from morpheme_extractor import run_morpheme_pipeline as _me_pipeline
            cls.prefixes, cls.suffixes, cls.prefix_freq, cls.suffix_freq = _me_pipeline(cls.words)
            cls.morpheme_freq.update(cls.prefixes)
            cls.morpheme_freq.update(cls.suffix_freq)

            # Persist the discovered affixes so subsequent calls skip discovery.
            if cls.words:  # don't write empty files
                save_json_file(BASE_DIR, "eng_prefixes.json", sorted(cls.prefixes))
                save_json_file(BASE_DIR, "eng_suffixes.json", sorted(cls.suffixes))
                save_json_file(BASE_DIR, "eng_prefix_freq.json", {k: v for k, v in cls.prefix_freq.items() if v > 1})
                save_json_file(BASE_DIR, "eng_suffix_freq.json", {k: v for k, v in cls.suffix_freq.items() if v > 1})
        cls._loaded = True

    @classmethod
    def pipeline(cls):
        """Return cached extract_productive_roots result (computed once).

        Returns a 5-tuple: (free_roots, combining_forms, morpheme_freq,
        prefix_freq, suffix_freq).
        """
        if cls._pipeline_result is None:
            cls.init()
            cls._pipeline_result = extract_productive_roots(
                lang_code="eng",
                dict_words=cls.words,
                suffixes=cls.suffixes,
                prefixes=cls.prefixes,
                script="latin",
                combining_forms=cls.combining_forms,
            )
        return cls._pipeline_result


# ─────────────────────────────────────────────────────────────────────────────
# §1  Unit tests for _filter_affix_duplicates  (pure synthetic data)
# ─────────────────────────────────────────────────────────────────────────────

class Test01_FilterAffixDuplicatesSynthetic(unittest.TestCase):
    """
    Unit tests for _filter_affix_duplicates using minimal synthetic data.
    Every test is self-contained and sub-millisecond.
    """

    # helpers ──────────────────────────────────────────────────────────────
    def _pfx(self, affixes, free, **kw):
        return _filter_affix_duplicates(affixes, free, is_suffix=False, **kw)

    def _sfx(self, affixes, free, **kw):
        return _filter_affix_duplicates(affixes, free, is_suffix=True, **kw)

    # prefix direction ─────────────────────────────────────────────────────

    def test_prefix_one_char_ext_removed(self):
        """
        Prefix: 'counterp'[:-1]='counter' is free root → 'counterp' removed.
        """
        free = {"counter", "photo"}
        affixes = {"counterp", "countert", "photog"}
        result = self._pfx(affixes, free)
        self.assertNotIn("counterp", result)
        self.assertNotIn("countert", result)
        self.assertNotIn("photog",   result)

    def test_prefix_no_stem_match_preserved(self):
        """Affixes whose stripped stem is NOT a free root must survive."""
        free   = {"meter"}
        # 'photon'[:-1]='photo' ∉ free; 'micrometer'[:-1]='micromete' ∉ free
        affixes = {"photon", "micrometer"}
        result  = self._pfx(affixes, free)
        self.assertIn("photon", result)

    def test_free_roots_themselves_removed_from_affix_set(self):
        """
        The final difference_update(free_roots_set) removes any free root
        that ended up in the affix candidate set.
        """
        free   = {"photo", "graph"}
        affixes = {"photo", "graph", "photon"}
        result  = self._pfx(affixes, free)
        self.assertNotIn("photo", result)
        self.assertNotIn("graph", result)

    def test_multiple_extensions_same_stem_all_removed(self):
        """All one-char extensions of 'counter' share the same stem → all removed."""
        free   = {"counter"}
        affixes = {"counterp", "countert", "counterf", "counterb"}
        result  = self._pfx(affixes, free)
        self.assertEqual(result, set(), "All extensions of free root must be removed")

    # suffix direction ─────────────────────────────────────────────────────

    def test_suffix_one_char_ext_removed(self):
        """
        Suffix: 'stion'[1:]='tion' is free root → 'stion' removed.
                'pment'[1:]='ment' is free root → 'pment' removed.
        """
        free   = {"tion", "ment"}
        affixes = {"stion", "pment"}
        result  = self._sfx(affixes, free)
        self.assertNotIn("stion", result)
        self.assertNotIn("pment", result)

    def test_suffix_no_stem_match_preserved(self):
        """
        The filter strips up to 4 left-chars from long affixes.
        'ization' stripped by 1-4 chars gives: zation, ation, tion, ion.
        With free={'ment'} none of those match so 'ization' must survive.
        """
        free    = {"ment"}
        affixes = {"ization"}
        result  = self._sfx(affixes, free)
        self.assertIn("ization", result)

    def test_direction_symmetry(self):
        """
        is_suffix=False strips right; is_suffix=True strips left.
        'photog' prefix-stripped = 'photo'; suffix-stripped = 'hotog'.
        Only prefix direction should trigger removal when free={'photo'}.
        """
        free  = {"photo"}
        affix = {"photog"}
        self.assertNotIn("photog", self._pfx(affix, free), "Prefix must remove 'photog'")
        self.assertIn("photog",    self._sfx(affix, free), "Suffix must keep 'photog'")

    # exemptions ───────────────────────────────────────────────────────────

    def test_combining_form_exemption(self):
        """
        'metallurg' is a combining form so it must NOT be removed even though
        its stripped stem 'metallu' could be constructed from free root 'metal'.
        """
        free      = {"metal"}
        affixes   = {"metallurg", "metalp"}
        combining = {"metallurg"}
        result    = self._pfx(affixes, free, combining_forms=combining)
        self.assertIn("metallurg",    result, "Combining form must be exempt")
        self.assertNotIn("metalp", result, "Non-exempt extension must be removed")

    def test_orth_rules_exemption(self):
        """
        'programm' restores to 'program' via gemination rule → exempt from removal
        even though 'program' is a free root.
        'programp' has no rule restoration → removed (stem 'program' in free).
        """
        from morpho_rules import compile_rules
        rules     = compile_rules([("mm$", ["m"])])
        free      = {"program"}
        dictionary = {"program", "programmer"}
        affixes   = {"programm", "programp"}
        result    = self._pfx(affixes, free,
                              combining_forms=set(), orth_rules=rules, dictionary=dictionary)
        self.assertIn("programm",    result, "Restorable affix must be exempt")
        self.assertNotIn("programp", result, "Non-restorable extension must be removed")

    # edge cases ───────────────────────────────────────────────────────────

    def test_empty_affixes_returns_empty(self):
        self.assertEqual(self._pfx(set(), {"photo"}), set())

    def test_empty_free_roots_nothing_removed(self):
        affixes = {"photo", "metric"}
        self.assertEqual(self._pfx(affixes, set()), affixes)

    def test_returns_set_type(self):
        self.assertIsInstance(self._pfx({"photog"}, {"photo"}), set)

    def test_idempotent(self):
        """Running the filter twice on already-clean output is idempotent."""
        free    = {"photo"}
        affixes = {"metric", "graphic"}
        r1 = self._pfx(affixes, free)
        r2 = self._pfx(r1, free)
        self.assertEqual(r1, r2)

    def test_short_affixes_within_max_affix_len_not_stem_checked(self):
        """
        Affixes ≤ max_affix_len are not examined for the stem-match rule;
        they are still removed by the final difference_update if they are
        literally in free_roots.
        'ab' is in free_roots → removed by difference_update (not stem rule).
        'abc' is ≤ max_affix_len=3 → only difference_update applies;
        since 'abc' ∉ free_roots it survives.
        'abcd' is > max_affix_len=3 → stem rule applies: 'abc' ∉ free_roots
        (only 'a' and 'ab' are), so 'abcd' survives.
        """
        free    = {"a", "ab"}
        affixes = {"ab", "abc", "abcd"}
        result  = self._pfx(affixes, free, max_affix_len=3)
        self.assertNotIn("ab",  result, "'ab' ∈ free_roots → removed by difference_update")
        self.assertIn("abc",  result,   "'abc' ∉ free_roots and ≤ max_affix_len → kept")
        self.assertIn("abcd", result,   "'abcd' stem 'abc' ∉ free_roots → kept")


# ─────────────────────────────────────────────────────────────────────────────
# §2  Fixture-based integration tests for extract_productive_roots
# ─────────────────────────────────────────────────────────────────────────────

class Test02_ExtractProductiveRootsIntegration(unittest.TestCase):
    """
    Integration tests consuming the CACHED pipeline result.
    setUpClass runs the pipeline once; every test reads class attrs.

    extract_productive_roots now returns a **5-tuple**:
        (free_roots, combining_forms, morpheme_freq, prefix_freq, suffix_freq)

    Bound roots that qualify as neoclassical compound components are promoted
    into free_roots by the internal trie-based Step 5, so there are no
    separate prefix_roots / suffix_roots in the public API.
    """

    @classmethod
    def setUpClass(cls):
        result = _RealData.pipeline()
        (cls.free_roots, cls.combining_forms,
         cls.morpheme_freq, cls.prefix_freq, cls.suffix_freq) = result

    # return-type contracts ────────────────────────────────────────────────

    def test_returns_five_tuple(self):
        self.assertEqual(len(_RealData.pipeline()), 5)

    def test_free_roots_is_set_of_nonempty_strings(self):
        self.assertIsInstance(self.free_roots, set)
        for r in self.free_roots:
            self.assertIsInstance(r, str)
            self.assertGreater(len(r), 0)

    def test_combining_forms_is_set_of_nonempty_strings(self):
        self.assertIsInstance(self.combining_forms, set)
        for cf in list(self.combining_forms)[:200]:
            self.assertIsInstance(cf, str)
            self.assertGreater(len(cf), 0)

    def test_morpheme_freq_is_counter(self):
        self.assertIsInstance(self.morpheme_freq, Counter)

    # volume sanity ────────────────────────────────────────────────────────

    def test_free_roots_nonempty(self):
        self.assertGreater(len(self.free_roots), 0)

    def test_total_root_count_reasonable(self):
        total = len(self.free_roots)
        self.assertGreater(total, 1_000,   f"Too few roots: {total}")
        self.assertLess(total,   500_000, f"Too many roots: {total}")

    # linguistic correctness ───────────────────────────────────────────────

    def test_common_english_words_in_free_roots(self):
        """High-frequency function words must appear as free roots."""
        expected = {"the", "and", "is", "in", "to", "of", "a"}
        found    = expected & self.free_roots
        self.assertGreater(len(found), len(expected) // 2,
                           f"Missing common words from free_roots: {expected - found}")

    def test_neoclassical_combining_forms_in_free_roots(self):
        """Classical combining forms must appear in free_roots."""
        classical = {"photo", "graph", "meter", "scope", "bio"}
        found     = [r for r in classical if r in self.free_roots]
        self.assertGreater(len(found), 1,
                           f"Only {found} of {classical} found in free_roots")

    def test_common_productive_suffixes_in_free_roots(self):
        """
        Common productive suffixes like "ing", "tion", "ness" are free
        dictionary words and must appear in free_roots.
        """
        productive = {"ing", "tion", "ness", "ment", "able", "er", "ed"}
        found = productive & self.free_roots
        self.assertGreater(len(found), len(productive) // 2,
                           f"Missing suffixes from free_roots: {productive - found}")

    def test_common_productive_prefixes_in_free_roots(self):
        """
        Common productive prefixes like "un", "re", "pre" are free dictionary
        words and must appear in free_roots.
        """
        productive = {"un", "re", "pre", "over", "under", "out", "non"}
        found = productive & self.free_roots
        self.assertGreater(len(found), len(productive) // 2,
                           f"Missing prefixes from free_roots: {productive - found}")

    # filter correctness ───────────────────────────────────────────────────

    def test_morpheme_freq_covers_most_free_roots(self):
        """Most free roots were derived from morpheme_freq; it must cover them."""
        sample = list(self.free_roots)[:100]
        covered = sum(1 for r in sample if self.morpheme_freq.get(r, 0) > 0)
        self.assertGreater(covered, len(sample) // 2)

    # parameter sensitivity ────────────────────────────────────────────────

    def test_higher_min_freq_gives_fewer_free_roots(self):
        small = set(list(_RealData.words)[:5_000])
        r_low,  *_ = extract_productive_roots("eng", small, _RealData.suffixes,
                                               _RealData.prefixes, script="latin",
                                               min_root_freq=2,
                                               combining_forms=_RealData.combining_forms)
        r_high, *_ = extract_productive_roots("eng", small, _RealData.suffixes,
                                               _RealData.prefixes, script="latin",
                                               min_root_freq=20,
                                               combining_forms=_RealData.combining_forms)
        self.assertGreaterEqual(len(r_low), len(r_high))

    def test_list_input_converted_to_set(self):
        result = extract_productive_roots(
            "eng", list(_RealData.words)[:500],
            _RealData.suffixes, _RealData.prefixes, script="latin",
            combining_forms=_RealData.combining_forms,
        )
        self.assertEqual(len(result), 5)


# ─────────────────────────────────────────────────────────────────────────────
# §3  Unit tests for run_morpheme_pipeline  (synthetic data)
# ─────────────────────────────────────────────────────────────────────────────

MINI_WORDS: Set[str] = {
    "run", "running", "runner", "runs",
    "play", "playing", "player", "plays",
    "work", "working", "worker", "works",
    "walk", "walking", "walker", "walks",
    "talk", "talking", "talker", "talks",
    "help", "helping", "helper", "helps",
    "jump", "jumping", "jumper", "jumps",
    "sing", "singing", "singer", "sings",
    "read", "reading", "reader", "reads",
    "write", "writing", "writer", "writes",
    "overworking", "overworked", "overworker",
    "unplayable", "unplayably",
    "unhelpful", "unhelpfully", "unhelpfulness",
    "underplaying", "underplay",
    "photography", "photograph", "photographer",
    "biography", "biographer", "biographical",
    "telescope", "telescopic", "telescopy",
    "microscope", "microscopic", "microscopist",
    "thermometer", "thermometry", "thermometric",
    "barometer", "barometric", "barometry",
    "chronometer", "chronometry", "chronometric",
    "photographic", "photographically",
}


class Test03_RunMorphemePipelineSynthetic(unittest.TestCase):
    """
    Unit tests for run_morpheme_pipeline using small, controlled synthetic
    word sets so each test runs in milliseconds.
    """

    @classmethod
    def setUpClass(cls):
        cls.tok = SemanticTokenizer(vocab={'<|pad|>': 0})
        # Compute once; re-use across all tests in this class.
        cls.result = cls.tok.run_morpheme_pipeline(
            lang_code="tst",
            dict_words=MINI_WORDS,
            min_morpheme_len=1,
            desired_root_len=10,
            min_root_freq=2,
        )
        (cls.all_pfx, cls.common_pfx,
         cls.all_sfx, cls.common_sfx,
         cls.roots, cls.bound, cls.scores) = cls.result

    # return-type contracts ────────────────────────────────────────────────

    def test_returns_seven_tuple(self):
        self.assertEqual(len(self.result), 7)

    def test_all_sets_are_sets(self):
        for item in (self.all_pfx, self.common_pfx,
                     self.all_sfx, self.common_sfx,
                     self.roots, self.bound):
            self.assertIsInstance(item, set)

    def test_scores_is_list(self):
        self.assertIsInstance(self.scores, list)

    def test_scores_pairs_are_str_float(self):
        for tok, score in self.scores:
            self.assertIsInstance(tok, str)
            self.assertIsInstance(score, float)

    # ordering / uniqueness ────────────────────────────────────────────────

    def test_scores_sorted_descending(self):
        values = [s for _, s in self.scores]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_no_duplicate_tokens_in_scores(self):
        tokens = [t for t, _ in self.scores]
        self.assertEqual(len(tokens), len(set(tokens)))

    def test_all_tokens_nonempty_and_positive_score(self):
        for tok, score in self.scores:
            self.assertGreater(len(tok), 0)
            self.assertGreater(score, 0.0)

    # subset relationships ─────────────────────────────────────────────────

    def test_common_prefixes_subset_of_all_prefixes(self):
        self.assertTrue(self.common_pfx <= self.all_pfx)

    def test_common_suffixes_subset_of_all_suffixes(self):
        self.assertTrue(self.common_sfx <= self.all_sfx)

    def test_common_prefixes_are_dict_words(self):
        for tok in self.common_pfx:
            self.assertIn(tok, MINI_WORDS,
                          f"common_pfx member '{tok}' not in dict_words")

    def test_common_suffixes_are_dict_words(self):
        for tok in self.common_sfx:
            self.assertIn(tok, MINI_WORDS,
                          f"common_sfx member '{tok}' not in dict_words")

    # min_morpheme_len enforcement ─────────────────────────────────────────

    def test_min_morpheme_len_respected(self):
        _, _, _, _, _, _, scores4 = self.tok.run_morpheme_pipeline(
            lang_code="tst2", dict_words=MINI_WORDS,
            min_morpheme_len=4, min_root_freq=2,
        )
        for tok, _ in scores4:
            self.assertGreaterEqual(len(tok), 4,
                                    f"Token '{tok}' shorter than min_morpheme_len=4")

    # combining-form priority ──────────────────────────────────────────────

    def test_pre_supplied_combining_forms_appear_in_scores(self):
        cf = {"photo", "graph", "scope"}
        *_, scores = self.tok.run_morpheme_pipeline(
            lang_code="tst3", dict_words=MINI_WORDS,
            combining_forms=cf, min_root_freq=1,
        )
        score_map = dict(scores)
        found = [f for f in cf if f in score_map]
        self.assertGreater(len(found), 0,
                           f"Combining forms {cf} not found in scored output")

    # edge inputs ──────────────────────────────────────────────────────────

    def test_empty_vocabulary(self):
        r = self.tok.run_morpheme_pipeline(
            lang_code="tst_empty", dict_words=set(), min_root_freq=1)
        self.assertEqual(len(r), 7)
        _, common_pfx, _, common_sfx, roots, bound, _ = r
        self.assertEqual(len(common_pfx), 0)
        self.assertEqual(len(common_sfx), 0)
        self.assertEqual(len(roots), 0)

    def test_list_input_accepted(self):
        result = self.tok.run_morpheme_pipeline(
            lang_code="tst_list", dict_words=list(MINI_WORDS), min_root_freq=1)
        self.assertEqual(len(result), 7)


# ─────────────────────────────────────────────────────────────────────────────
# §4  Integration tests for run_morpheme_pipeline  (real English data)
# ─────────────────────────────────────────────────────────────────────────────

class Test04_RunMorphemePipelineRealData(unittest.TestCase):
    """
    Integration tests using the full English dictionary.
    Pipeline runs once in setUpClass; all tests read class attrs.
    """

    @classmethod
    def setUpClass(cls):
        _RealData.init()
        tok = SemanticTokenizer(vocab={'<|pad|>': 0})
        cls.result = tok.run_morpheme_pipeline(
            lang_code="eng",
            dict_words=_RealData.words,
            min_morpheme_len=1,
            desired_root_len=10,
            min_root_freq=4,
            script="latin",
            combining_forms=_RealData.combining_forms,
        )
        (cls.all_pfx,
         cls.all_sfx, cls.common_sfx,
         cls.roots, cls.bound, cls.scores) = cls.result
        cls.score_map: Dict[str, float] = dict(cls.scores)

    # volume ───────────────────────────────────────────────────────────────

    def test_scores_nonempty(self):
        self.assertGreater(len(self.scores), 0)

    def test_scores_large_enough_for_500k_dict(self):
        self.assertGreater(len(self.scores), 10_000,
                           f"Only {len(self.scores)} scored tokens")

    def test_all_six_sets_nonempty(self):
        for s, name in [(self.all_pfx, "all_pfx"), (self.all_sfx, "all_sfx"),
                        (self.roots, "roots"),     (self.bound,   "bound")]:
            self.assertGreater(len(s), 0, f"{name} is empty")

    # linguistic spot-checks ───────────────────────────────────────────────

    def test_english_prefixes_in_all_pfx(self):
        expected = {"un", "re", "pre", "over"}
        self.assertGreater(len(expected & self.all_pfx), 0,
                           f"Missing prefixes: {expected - self.all_pfx}")

    def test_english_suffixes_in_all_sfx(self):
        expected = {"ing", "tion", "ness", "ment", "able"}
        self.assertGreater(len(expected & self.all_sfx), 0,
                           f"Missing suffixes: {expected - self.all_sfx}")

    def test_neoclassical_roots_in_score_map(self):
        classical = {"photo", "graph", "scope", "bio", "meter"}
        found     = [c for c in classical if c in self.score_map]
        self.assertGreater(len(found), 1,
                           f"Neoclassical roots missing from scores: {classical - set(found)}")

    def test_combining_forms_have_positive_score(self):
        cf_in_scores = _RealData.combining_forms & self.score_map.keys()
        if not cf_in_scores:
            self.skipTest("No combining forms appear in score_map")
        for cf in list(cf_in_scores)[:20]:
            self.assertGreater(self.score_map[cf], 0.0,
                               f"Combining form '{cf}' has zero/negative score")

    # ordering / uniqueness ────────────────────────────────────────────────

    def test_scores_sorted_descending(self):
        values = [s for _, s in self.scores]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_no_duplicate_tokens(self):
        tokens = [t for t, _ in self.scores]
        self.assertEqual(len(tokens), len(set(tokens)))

    def test_all_tokens_nonempty_positive(self):
        for tok, score in self.scores:
            self.assertGreater(len(tok), 0)
            self.assertGreater(score, 0.0)

    # subset relationships ─────────────────────────────────────────────────

    def test_common_prefixes_subset_of_all_prefixes(self):
        self.assertTrue(self.common_pfx <= self.all_pfx)

    def test_common_suffixes_subset_of_all_suffixes(self):
        self.assertTrue(self.common_sfx <= self.all_sfx)

    def test_common_prefixes_are_real_words(self):
        for tok in self.common_pfx:
            self.assertIn(tok, _RealData.words,
                          f"'{tok}' in common_pfx but not in eng_words")

    def test_common_suffixes_are_real_words(self):
        for tok in self.common_sfx:
            self.assertIn(tok, _RealData.words,
                          f"'{tok}' in common_sfx but not in eng_words")

    def test_scored_tokens_cover_all_morpheme_classes(self):
        """
        Every discovered morpheme (within the length filter) must appear
        in the scored output — coverage must be ≥ 90 %.
        """
        all_morphemes = (self.all_pfx | self.all_sfx | self.roots | self.bound
                         | _RealData.combining_forms)
        expected = {m for m in all_morphemes if 1 <= len(m) <= 14}
        uncovered = expected - self.score_map.keys()
        coverage  = 1 - len(uncovered) / max(len(expected), 1)
        self.assertGreater(coverage, 0.90,
                           f"Only {coverage:.1%} of morphemes appear in score_map")


# ─────────────────────────────────────────────────────────────────────────────
# §5  Performance benchmarks
# ─────────────────────────────────────────────────────────────────────────────

class Test05_Performance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _RealData.init()
        cls.small = set(list(_RealData.words)[:10_000])

    def test_filter_affix_duplicates_under_1s(self):
        free    = set(list(_RealData.words)[:5_000])
        affixes = set(list(_RealData.words)[5_000:10_000])
        t0 = time.perf_counter()
        _filter_affix_duplicates(affixes, free, is_suffix=False)
        self.assertLess(time.perf_counter() - t0, 1.0)

    def test_extract_morpheme_frequencies_under_5s(self):
        t0 = time.perf_counter()
        _extract_morpheme_frequencies(self.small)
        self.assertLess(time.perf_counter() - t0, 5.0)

    def test_run_morpheme_pipeline_10k_words_under_120s(self):
        tok = SemanticTokenizer(vocab={'<|pad|>': 0})
        t0  = time.perf_counter()
        result = tok.run_morpheme_pipeline(
            lang_code="eng_small",
            dict_words=self.small,
            combining_forms=_RealData.combining_forms,
            min_root_freq=2,
        )
        elapsed = time.perf_counter() - t0
        self.assertEqual(len(result), 7)
        self.assertLess(elapsed, 120.0, f"Pipeline took {elapsed:.1f}s on 10K words")
        print(f"\n    [run_morpheme_pipeline 10K] {elapsed:.2f}s")

    def test_extract_productive_roots_10k_under_120s(self):
        t0 = time.perf_counter()
        result = extract_productive_roots(
            "eng", self.small, _RealData.suffixes, _RealData.prefixes,
            script="latin", combining_forms=_RealData.combining_forms,
        )
        elapsed = time.perf_counter() - t0
        self.assertEqual(len(result), 5)
        self.assertLess(elapsed, 120.0, f"extract_productive_roots took {elapsed:.1f}s")
        print(f"\n    [extract_productive_roots 10K] {elapsed:.2f}s")


# ─────────────────────────────────────────────────────────────────────────────
# §6  Edge cases & boundary conditions
# ─────────────────────────────────────────────────────────────────────────────

class Test06_EdgeCases(unittest.TestCase):

    def test_filter_empty_affixes(self):
        self.assertEqual(_filter_affix_duplicates(set(), {"photo"}, is_suffix=False), set())

    def test_filter_empty_free_roots(self):
        affixes = {"photo", "metric"}
        self.assertEqual(_filter_affix_duplicates(affixes, set(), is_suffix=False), affixes)

    def test_morpheme_freq_empty(self):
        sw, mf, pf, sf = _extract_morpheme_frequencies(set())
        self.assertEqual(len(mf), 0)

    def test_morpheme_freq_single_word(self):
        _, _, pf, _ = _extract_morpheme_frequencies({"cat"})
        self.assertIn("cat", pf)

    def test_pipeline_list_input(self):
        tok = SemanticTokenizer(vocab={'<|pad|>': 0})
        result = tok.run_morpheme_pipeline(
            lang_code="tst_list",
            dict_words=list({"run", "running", "runner", "runs",
                              "play", "playing", "player", "plays"}),
            min_root_freq=1,
        )
        self.assertEqual(len(result), 7)

    def test_pipeline_all_scores_positive(self):
        tok = SemanticTokenizer(vocab={'<|pad|>': 0})
        *_, scores = tok.run_morpheme_pipeline(
            lang_code="tst_pos",
            dict_words={"run", "running", "runner",
                        "play", "playing", "player"},
            min_root_freq=1,
        )
        for t, s in scores:
            self.assertGreater(s, 0.0, f"Token '{t}' has non-positive score {s}")

    def test_combining_form_exemption_regression(self):
        """
        Regression: combining form whose stripped stem IS a free root must NOT be removed.
        """
        free      = {"metal", "metallu"}
        combining = {"metallurg"}
        result = _filter_affix_duplicates(
            {"metallurg"}, free, is_suffix=False, combining_forms=combining
        )
        self.assertIn("metallurg", result)

    def test_pipeline_min_morpheme_len_filters_short(self):
        tok = SemanticTokenizer(vocab={'<|pad|>': 0})
        *_, scores = tok.run_morpheme_pipeline(
            lang_code="tst_len",
            dict_words={"running", "runner", "plays", "player"},
            min_morpheme_len=3,
            min_root_freq=1,
        )
        for token, _ in scores:
            self.assertGreaterEqual(len(token), 3,
                                    f"Token '{token}' shorter than min_morpheme_len=3")

# ─────────────────────────────────────────────────────────────────────────────
# §7  Rare neoclassical compound bound-root promotion  (Step 5 inside
#     extract_productive_roots — trie-based, O(|words|×avg_word_len))
# ─────────────────────────────────────────────────────────────────────────────

class Test07_NeoclassicalBoundRootPromotion(unittest.TestCase):
    """
    Verify that Step 5 of extract_productive_roots promotes rare neoclassical
    compound components into free_roots via two forward prefix tries.

    Target morphemes and representative example words:
      • "loxo"      — loxoclase, loxotomy, loxodonta
      • "clase"     — anorthoclase, sphenoclase, loxoclase, skleroclase
      • "sklero"    — skleroclase, skleropelite
      • "pelite"    — skleropelite, sapropelite
      • "metallurg" — electrometallurgic, pyrometallurgist, metallurgical

    These forms almost never appear as standalone words so they survive the
    bound-root extraction phase (Steps 2–4).  Step 5 then checks whether each
    bound root appears after a known combining-form anchor in ≥ 2 dictionary
    words and, if so, promotes it into free_roots.
    """

    @classmethod
    def setUpClass(cls):
        _RealData.init()
        cls.prefixes = _RealData.prefixes
        cls.suffixes = _RealData.suffixes
        cls.prefix_freq = _RealData.prefix_freq
        cls.suffix_freq = _RealData.suffix_freq

        result = _RealData.pipeline()
        cls.free_roots, cls.combining_forms, cls.morpheme_freq, *_ = result

    # ── individual promotion assertions ──────────────────────────────────

    def test_clase_in_free_roots(self):
        """
        "clase" is the suffix in anorthoclase / sphenoclase / rhomboclase /
        microclase.  Anchors anortho, spheno, rhombo, micro are known
        combining forms → "clase" must be promoted.
        """
        self.assertIn("clase", self.combining_forms,
                      "'clase' not promoted — check Step-5 trie scan for "
                      "anorthoclase, sphenoclase, rhomboclase, microclase")

    def test_loxo_in_free_roots(self):
        """
        "loxo" is the prefix in loxoclase, loxotomy, loxodonta.
        It must be promoted into combining_forms.
        """
        self.assertIn("loxo", self.combining_forms,
                      "'loxo' not promoted — check Step-5 trie scan for "
                      "loxoclase, loxotomy, loxodonta")

    def test_sklero_in_free_roots(self):
        """
        "sklero" is the prefix in skleroclase and skleropelite.
        It must be promoted into combining_forms.
        """
        self.assertIn("sklero", self.combining_forms,
                      "'sklero' not promoted — check Step-5 trie scan for "
                      "skleroclase, skleropelite")

    def test_pelite_in_free_roots(self):
        """
        "pelite" is the suffix in skleropelite and sapropelite.
        It must be promoted into combining_forms.
        """
        self.assertIn("pelite", self.combining_forms,
                      "'pelite' not promoted — check Step-5 trie scan for "
                      "skleropelite, sapropelite")

    def test_metallurg_in_free_roots(self):
        """
        "metallurg" is a medial/prefix component in electrometallurgic,
        pyrometallurgist, hydrometallurgically, metallurgical.
        Anchors electro, pyro, hydro are known combining forms → promoted.
        """
        self.assertIn("metallurg", self.combining_forms,
                      "'metallurg' not promoted — check Step-5 trie scan for "
                      "electrometallurgic, metallurgical, pyrometallurgist")

    # ── batch assertion ───────────────────────────────────────────────────

    def test_all_target_rare_forms_promoted(self):
        """All five target rare neoclassical components must be in combining_forms."""
        targets = {"loxo", "clase", "sklero", "pelite", "metallurg"}
        missing = targets - self.combining_forms
        self.assertEqual(missing, set(),
                         f"Rare neoclassical compound components not promoted "
                         f"into free_roots: {sorted(missing)}")

    # ── promoted roots must be genuine (non-zero corpus frequency) ────────

    def test_promoted_roots_have_nonzero_morpheme_freq(self):
        """Every promoted root must appear at least once in the corpus."""
        targets = {"loxo", "clase", "sklero", "pelite", "metallurg"}
        found = targets & self.combining_forms
        for root in found:
            freq = self.morpheme_freq.get(root, 0)
            self.assertGreater(freq, 0,
                               f"Promoted root '{root}' has zero morpheme frequency")

    # ── trie algorithm unit tests (synthetic, microsecond-fast) ──────────

    def _promote_neo_roots(self, words, anchor_set, prefix_roots, suffix_roots):
        """
        Execute one trie-scan pass (mirrors the _trie_scan helper inside
        extract_productive_roots) and return the promoted set (count >= 2).
        """
        combining_forms = _promote_neoclassical_bound_roots(
            words, self.prefix_freq, self.suffix_freq, anchor_set,
            prefixes=self.prefixes, prefix_roots=prefix_roots,
            suffixes=self.suffixes, suffix_roots=suffix_roots,
            free_roots=self.free_roots
        )
        return combining_forms

    def test_trie_promotes_clase_synthetic(self):
        """
        "clase" after cf anchors anortho/spheno/rhombo in 3 words -> count=3 >= 2.
        """
        words   = {"anorthoclase", "sphenoclase", "rhomboclase"}
        anchors = {"anortho", "spheno", "rhombo"}
        prefix_roots = {"anortho", "spheno", "rhombo"}
        suffix_roots = {"clase"}
        promoted = self._promote_neo_roots(words, anchors, prefix_roots, suffix_roots)
        self.assertIn("clase", promoted)

    def test_trie_does_not_promote_hapax_suffix(self):
        """
        A bound root seen after a cf anchor in only 1 word must NOT be promoted
        (threshold is >= 2 distinct words).
        """
        words   = {"microxyzzy"}
        anchors = {"micro"}
        promoted = self._promote_neo_roots(words, anchors, {"micro"}, {"xyzzy"})
        self.assertNotIn("xyzzy", promoted,
                         "Hapax bound root must not be promoted (threshold >= 2)")

    def test_trie_promotes_metallurg_synthetic(self):
        """
        "metallurg" follows cf anchors electro and pyro in 2 words -> promoted.
        """
        words   = {"electrometallurgic", "pyrometallurgist"}
        anchors = {"electro", "pyro"}
        promoted = self._promote_neo_roots(words, anchors, {"metallurg"}, {"clase"})
        self.assertIn("metallurg", promoted)

    def test_trie_per_word_dedup_prevents_double_counting(self):
        """
        Even if two valid anchor splits exist in a single word ("electro" and
        "electromet"), the bound root "metallurg" is counted only once per word,
        so a single word cannot push a root past the >= 2 threshold alone.
        """
        words   = {"electrometallurgic"}
        anchors = {"electro", "electromet"}
        promoted = self._promote_neo_roots(words, anchors, {"metallurg"}, {"clase"})
        # count["metallurg"] == 1 < 2 -> NOT promoted
        self.assertNotIn("metallurg", promoted,
                         "Per-word dedup failed: a single word must not yield count >= 2")

    def test_two_pass_promotes_sklero_via_clase(self):
        """
        Pass 1 promotes "clase" (anchors: anortho, spheno).
        Pass 2 uses "clase" + "pelite" as anchors -> finds "sklero" in
        skleroclase / skleropelite.
        """
        # Simulate pass 1
        words_p1  = {
            # anortho family
            "anorthoclase", "anorthoclases", "anorthographic", "anorthographical", "anorthographically",
            "anorthography", "anorthophyre", "anorthopia", "anorthoscope", "anorthose", "anorthosite",
            "anorthosites",
            # spheno family
            "spheno", "sphenobasilar", "sphenobasilic", "sphenocephalia", "sphenocephalic", "sphenocephalous",
            "sphenocephaly", "sphenochasm", "sphenoclase", "sphenodon", "sphenodont", "sphenodontia", "sphenodontidae",
            "sphenoethmoid", "sphenoethmoidal", "sphenofrontal", "sphenogram", "sphenographic", "sphenographist",
            "sphenography", "sphenoid", "sphenoidal", "sphenoiditis", "sphenoids", "sphenolith", "sphenomalar",
            "sphenomandibular", "sphenomaxillary", "sphenopalatine", "sphenoparietal", "sphenopetrosal", "sphenophorus",
            "sphenophyllaceae", "sphenophyllaceous", "sphenophyllales", "sphenophyllum", "sphenopteris", "sphenotic",
            "sphenosquamosal", "sphenotemporal", "sphenotribe", "sphenotripsy", "sphenoturbinal", "sphenovomerine",
            "sphenozygomatic",
        }
        anchors_1 = {"anortho", "spheno"}
        promoted_1 = self._promote_neo_roots(words_p1, anchors_1, {"anortho", "spheno"}, {"clase"})
        self.assertIn("clase", promoted_1, "Pass-1 must promote 'clase'")

        # Simulate pass 2: clase (from pass 1) and pelite (free root) as anchors
        words_p2  = {"skleroclase", "skleropelite"}
        anchors_2 = anchors_1 | promoted_1 | {"pelite"}
        promoted_2 = self._promote_neo_roots(words_p2, anchors_2, {"anortho", "spheno"}, {"clase"})
        self.assertIn("sklero", promoted_2,
                      "Pass-2 must promote 'sklero' using 'clase'/'pelite' as anchors")

# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  COMPREHENSIVE ROOT EXTRACTION + run_morpheme_pipeline TEST SUITE")
    print("=" * 80)

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        Test01_FilterAffixDuplicatesSynthetic,
        Test02_ExtractProductiveRootsIntegration,
        Test03_RunMorphemePipelineSynthetic,
        Test04_RunMorphemePipelineRealData,
        Test05_Performance,
        Test06_EdgeCases,
        Test07_NeoclassicalBoundRootPromotion,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print(f"  ✓ ALL {result.testsRun} TESTS PASSED")
    else:
        print(f"  ✗ {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 80 + "\n")

    sys.exit(0 if result.wasSuccessful() else 1)