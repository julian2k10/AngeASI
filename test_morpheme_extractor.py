"""
Test suite for morpheme_extractor.py

Extracted from morpheme_extractor.py.  Run with:
    python -m pytest test_morpheme_extractor.py -v
or:
    python test_morpheme_extractor.py
"""

import os
import sys
import unittest
from collections import Counter
from typing import Set, List, Tuple

# Ensure the package is importable from the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from morpheme_extractor import (
    filter_affixes, extract_productive_affixes, _extract_testing_data, logger
)
from asi.context_aware_io import save_json_file, load_json_file

PRODUCTIVE_PREFIXES = [
    # Productive/Negative
    ('un', 'productive negative prefix'), ('re', 'productive repetition prefix'),
    ('in', 'productive negation prefix'), ('de', 'productive reversal prefix'),
    ('dis', 'productive negation'), ('non', 'productive negation'),
    ('mis', 'productive wrong'), ('anti', 'productive against'),
    ('counter', 'productive against/opposite'),

    # Spatial/Directional
    ('sub', 'productive under'), ('super', 'productive above'),
    ('inter', 'productive between'), ('trans', 'productive across'),
    ('ex', 'productive out-of prefix'), ('pre', 'productive before'),
    ('pro', 'productive forward'), ('fore', 'productive before'),
    ('circum', 'productive around'), ('extra', 'productive beyond'),
    ('intra', 'productive within'), ('retro', 'productive backward'),
    ('dia', 'productive through'), ('peri', 'productive around'),
    ('epi', 'productive upon'), ('para', 'productive beside'),
    ('amphi', 'productive both/around'),

    # Quantitative/Number
    ('mono', 'productive one'), ('uni', 'productive one'),
    ('bi', 'productive two'), ('di', 'productive two/apart prefix'),
    ('tri', 'productive three'), ('multi', 'productive many'),
    ('poly', 'productive many'), ('semi', 'productive half'),
    ('hemi', 'productive half'), ('mega', 'productive large'),
    ('macro', 'productive large'), ('micro', 'productive small'),

    # Intensive/Degree
    ('ultra', 'productive beyond'), ('hyper', 'productive excess'),
    ('hypo', 'productive under'), ('over', 'productive above'),
    ('under', 'productive below'), ('out', 'productive surpass'),

    # Scientific/Technical/Abstract
    ('auto', 'productive self'), ('co', 'productive together'),
    ('pseudo', 'productive false'), ('iso', 'productive equal'),
    ('meta', 'productive beyond'), ('neo', 'productive new'),
    ('proto', 'productive first'), ('syn', 'productive with/together'),
    ('sym', 'productive with/together'),

    # Human/Organic/Substance
    ('anthro', 'productive human'), ('anthr', 'productive human/carbon'),
    ('anthra', 'productive coal/carbon'),

    # Assimilated (Fossilized in OED roots)
    ('ad', 'to/toward prefix'), ('ac', 'assimilated ad- prefix'),
    ('af', 'assimilated ad- prefix'), ('ag', 'assimilated ad- prefix'),
    ('al', 'assimilated ad- prefix'), ('ap', 'assimilated ad- prefix'),
    ('ar', 'assimilated ad- prefix'), ('as', 'assimilated ad- prefix'),
    ('at', 'assimilated ad- prefix'), ('be', 'productive causative prefix')
]

INVALID_PREFIXES = ['counterf', 'counterp', 'counters']

PREFIX_CANDIDATES = [
    'ab', 'aba', 'abb', 'abd', 'abe', 'abi', 'abl', 'abo', 'abr', 'abs', 'abu', 'ac', 'aca', 'acc', 'ace', 'ach',
    'aci', 'aco', 'acq', 'acr', 'act', 'acu', 'ad', 'ada', 'add', 'ade', 'adh', 'adi', 'adj', 'adm', 'ado', 'adr',
    'adu', 'adv', 'ae', 'aer', 'af', 'aff', 'afr', 'aft', 'ag', 'aga', 'age', 'agg', 'agi', 'agn', 'ago', 'agr', 'ah',
    'ai', 'air', 'ak', 'al', 'ala', 'alb', 'alc', 'ald', 'ale', 'alg', 'ali', 'alk', 'all', 'alm', 'alo', 'alp', 'alt',
    'alu', 'am', 'ama', 'amb', 'ame', 'ami', 'amm', 'amo', 'amp', 'amph', 'amphi', 'amphib', 'ampl', 'amy', 'anthr',
    'anthra', 'anthro', 'anthrop', 'bab', 'bac', 'bad', 'bag', 'bai', 'bak', 'bal', 'bi', 'bia', 'bib', 'bic', 'bid',
    'bie', 'bif', 'big', 'bil', 'bim', 'bin', 'bio', 'bip', 'bir', 'bis', 'bit', 'cab', 'cac', 'cad', 'cae', 'cai',
    'cal', 'cam', 'camp', 'counterf', 'counterp', 'counters', 'dac', 'dag', 'dai', 'dal', 'dam', 'dex', 'ex', 'exa',
    'exc', 'exe', 'exh', 'exi', 'exo', 'exp', 'exs', 'ext', 'exu', 'fab', 'fac', 'fad', 'fag', 'fai', 'fal', 'fam',
    'gab', 'gad', 'gai', 'gal', 'gam', 'hab', 'hac', 'had', 'hae', 'hag', 'hai', 'hal', 'ham', 'hex', 'jac', 'jag',
    'jam', 'kai', 'kal', 'kam', 'lab', 'lac', 'lad', 'lag', 'lai', 'lak', 'lam', 'lamp', 'lex', 'mac', 'macro',
    'macroc', 'mad', 'mag', 'mah', 'mai', 'maj', 'mak', 'mal', 'mam', 'meta', 'metac', 'metal', 'metam', 'metap',
    'metas', 'metat', 'nab', 'nag', 'nai', 'nam', 'pac', 'pad', 'pag', 'pai', 'pal', 'pam', 'pamp', 'peri', 'peria',
    'peric', 'perio', 'perip', 'peris', 'perit', 'rab', 'rac', 'rad', 'raf', 'rag', 'rai', 'rak', 'ram', 'ramp', 'sab',
    'sac', 'sad', 'saf', 'sag', 'sai', 'sak', 'sal', 'sam', 'sex', 'sym', 'symb', 'symp', 'tab', 'tac', 'tag', 'tai',
    'tak', 'tal', 'tam', 'tex', 'trans', 'transa', 'transc', 'transf', 'transi', 'transl', 'transm', 'transp',
    'transu', 'transv', 'vac', 'vag', 'vai', 'val', 'vam', 'wad', 'waf', 'wag', 'wai', 'wak', 'wal', 'wam'
]

PREFIX_FREQ = load_json_file(os.path.join(os.getcwd(), "prefix_freq_data.json"))

PRODUCTIVE_SUFFIXES = [
    # Inflectional (Grammatical markers)
    ('s', 'inflectional plural/present'), ('es', 'inflectional plural/present'),
    ('d', 'inflectional past tense short form'), ('ed', 'inflectional past tense'),
    ('y', 'derivational adjective-forming'), ('ing', 'inflectional progressive'),
    ('er', 'inflectional comparative'), ('est', 'inflectional superlative'),

    # Noun-forming (Derivational)
    ('ness', 'productive state/quality noun'), ('ism', 'productive belief/system'),
    ('ist', 'productive person/agent'), ('er', 'productive agent noun'),
    ('or', 'productive agent noun'), ('ship', 'productive status noun'),
    ('ment', 'productive action/result noun'), ('tion', 'productive state noun'),
    ('ity', 'productive quality noun'), ('hood', 'semi-productive status noun'),
    ('dom', 'semi-productive state noun'), ('acy', 'state/quality noun'),

    # Adjective-forming (Derivational)
    ('able', 'productive ability adjective'), ('ible', 'productive ability adjective'),
    ('ish', 'productive similarity adjective'), ('y', 'productive quality adjective'),
    ('less', 'productive privative adjective'), ('ful', 'productive quality adjective'),
    ('ive', 'productive tendency adjective'), ('ous', 'productive quality adjective'),
    ('esque', 'productive style adjective'), ('like', 'productive similarity adjective'),
    ('ic', 'productive nature adjective'), ('al', 'productive relation adjective'),

    # Verb-forming (Derivational)
    ('ize', 'productive causative verb'), ('ise', 'productive causative verb'),
    ('ify', 'productive causative verb'), ('ate', 'productive causative verb'),
    ('en', 'productive causative verb'),

    # Adverb-forming
    ('ly', 'productive manner adverb'), ('wise', 'productive direction/manner'),
    ('ward', 'productive direction'), ('wards', 'productive direction'),

    # Diminutives/Technical
    ('ling', 'productive smallness/connection'), ('oid', 'productive resemblance'),
    ('logy', 'productive study of'), ('graphy', 'productive writing/recording')
]

INVALID_SUFFIXES = [
    'dful', 'eful', 'efull', 'efully', 'stful', 'tful', "ckings", "cket", "king", "e"
]

SUFFIX_CANDIDATES = [
    'acy', 'aes', 'aking', 'as', 'ber', 'bes', 'bing', 'bs', 'cer', 'ces', 'cible', 'cing', 'cize', 'cket', 'cking',
    'ckings', 'cs', 'd', 'den', 'der', 'des', 'dful', 'dible', 'ding', 'dize', 'dless', 'dom', 'dome', 'dor', 'ds', 'dship',
    'e', 'edom', 'een', 'eer', 'ees', 'eful', 'efull', 'efully', 'efulness', 'eing', 'eless', 'en', 'ena', 'end', 'ene',
    'eng', 'enn', 'eno', 'ens', 'ent', 'eny', 'er', 'era', 'erd', 'ere', 'erg', 'eri', 'erk', 'erm', 'ern', 'ero',
    'ers', 'ert', 'ery', 'es', 'ese', 'esh', 'eship', 'ess', 'est', 'esy', 'eward', 'fen', 'fer', 'fes', 'fing', 'fs',
    'ful', 'full', 'fully', 'fulness', 'fulnesse', 'fuls', 'gen', 'ger', 'ges', 'gible', 'ging', 'gize', 'gless',
    'graphy', 'gs', 'hen', 'her', 'hes', 'hful', 'hing', 'hize', 'hless', 'hor', 'hs', 'ible', 'ibles', 'ien', 'ier',
    'ies', 'ing', 'inge', 'ings', 'ior', 'is', 'ize', 'ized', 'izer', 'izes', 'ken', 'ker', 'kes', 'kful', 'king',
    'kinge', 'kings', 'kless', 'ks', 'ldom', 'len', 'ler', 'les', 'less', 'lesse', 'lful', 'ling', 'lize', 'lless',
    'lor', 'ls', 'lship', 'men', 'mer', 'mes', 'ming', 'mize', 'mless', 'ms', 'nen', 'ner', 'nes', 'nful', 'ning',
    'nize', 'nking', 'nless', 'nor', 'ns', 'nship', 'oer', 'oes', 'ography', 'oing', 'oking', 'oor', 'or', 'ora',
    'ord', 'ore', 'ori', 'ork', 'orm', 'orn', 'oro', 'ors', 'ort', 'ory', 'os', 'pen', 'per', 'pes', 'ping', 'pless',
    'ps', 'racy', 'rdom', 'ren', 'rer', 'res', 'rful', 'ring', 'rize', 'rking', 'rless', 'rs', 'rship', 'rward', 's',
    'sa', 'sc', 'sd', 'se', 'sen', 'ser', 'ses', 'sh', 'ship', 'ships', 'si', 'sible', 'sing', 'sk', 'sm', 'so', 'sor',
    'sp', 'ss', 'st', 'stful', 'su', 'sy', 'ten', 'ter', 'tes', 'tful', 'tfull', 'tfully', 'tfulness', 'tible', 'ting',
    'tize', 'tless', 'tor', 'ts', 'tship', 'uen', 'uer', 'ues', 'uing', 'us', 'ven', 'ver', 'ves', 'ving', 'ward',
    'warde', 'wards', 'wen', 'wer', 'wes', 'wing', 'ws', 'xes', 'xing', 'y', 'ydom', 'yen', 'yer', 'yes', 'ying',
    'yless', 'ys', 'yship', 'zen', 'zer', 'zes', 'zing'
]

SUFFIX_FREQ = load_json_file(os.path.join(os.getcwd(), "suffix_freq_data.json"))

# Build Frequency Counters
prefix_freq = Counter(dict(PREFIX_FREQ))
suffix_freq = Counter(dict(SUFFIX_FREQ))

# Default file names for the pre-filtered English affix lists produced by
# run_morpheme_pipeline / filter_affixes and saved to disk.
ENG_PREFIXES_FILE = "eng_prefixes.json"
ENG_SUFFIXES_FILE = "eng_suffixes.json"

# ═══════════════════════════════════════════════════════════════════════
# Test Suite
# ═══════════════════════════════════════════════════════════════════════

class TestDistributionCalculation(unittest.TestCase):
    """Verify the core fix: distribution is parent-relative, not root-relative."""

    def test_parent_relative_distribution(self):
        """'amp' should be measured against 'am' (21.8%), not 'a' (1.4%)."""
        freq = Counter({"a": 29426, "am": 1915, "amp": 418, "amph": 285, "amphi": 248})
        candidates = {"a", "am", "amp", "amph", "amphi"}
        result = filter_affixes(candidates, freq, threshold=0.035)
        self.assertNotIn("amp", result)

    def test_deep_chain_amphi(self):
        """'amphi' = 248/285 = 87% relative to 'amph', should pass (no children)."""
        freq = Counter({"a": 29426, "am": 1915, "amp": 418, "amph": 285, "amphi": 248})
        candidates = {"amphi"}
        result = filter_affixes(candidates, freq, threshold=0.035)
        self.assertIn("amphi", result)

    def test_single_char_root(self):
        """Single-char vowel prefix 'a' passes with nonzero count (no parent)."""
        freq = Counter({"a": 29426})
        candidates = {"a"}
        result = filter_affixes(candidates, freq, threshold=0.035)
        self.assertIn("a", result)


class TestVowelRule(unittest.TestCase):
    """Rule A: prefix must contain at least one vowel."""

    def test_no_vowel_rejected(self):
        freq = Counter({"b": 1000, "br": 500})
        candidates = {"br"}
        result = filter_affixes(candidates, freq, threshold=0.01)
        self.assertEqual(result, set())

    def test_vowel_present_accepted(self):
        freq = Counter({"b": 1000, "ba": 500})
        candidates = {"ba"}
        result = filter_affixes(candidates, freq, threshold=0.01)
        self.assertIn("ba", result)

    def test_custom_vowels(self):
        """Override vowels for a different language/script."""
        freq = Counter({"k": 500, "kö": 200})
        candidates = {"kö"}
        result = filter_affixes(candidates, freq, vowels="aeiouöüä", threshold=0.01)
        self.assertIn("kö", result)

    def test_custom_vowels_rejects_without_match(self):
        freq = Counter({"k": 500, "ky": 200})
        candidates = {"ky"}
        result = filter_affixes(candidates, freq, vowels="aeiou", threshold=0.01)
        self.assertEqual(result, set())


class TestAssimilationRule(unittest.TestCase):
    """
    Vowel+consonant bigrams use the base threshold (3.5%) instead of the
    stricter len-2 threshold (15%). This reflects the universal phonological
    pattern of prefix assimilation across language families.
    """

    def test_assimilated_prefix_below_threshold(self):
        """'af' = 808/29426 = 2.7% > 3.5% base threshold via assimilation."""
        freq = Counter({"a": 29426, "af": 808})
        candidates = {"af"}
        result = filter_affixes(candidates, freq, threshold=0.035)
        self.assertIn("af", result)

    def test_assimilation_only_for_bigrams(self):
        """Longer prefixes don't get the assimilation pass."""
        freq = Counter({"a": 10000, "af": 808, "aff": 50})
        candidates = {"aff"}
        result = filter_affixes(candidates, freq, threshold=0.10)
        self.assertNotIn("aff", result)

    def test_vowel_vowel_not_assimilated(self):
        """'ae' is vowel+vowel, not vowel+consonant -> no assimilation pass."""
        freq = Counter({"a": 29426, "ae": 100})
        candidates = {"ae"}
        result = filter_affixes(candidates, freq, threshold=0.035)
        self.assertNotIn("ae", result)


class TestDynamicFragmentThreshold(unittest.TestCase):
    """Fragment threshold scales with prefix length."""

    def test_len2_below_fragment_threshold(self):
        """Bigram 'al': child 'ale' at 2% < 3.5% -> al is NOT a fragment."""
        freq = Counter({"a": 29426, "al": 2920, "ale": 58})
        candidates = {"al"}
        result = filter_affixes(candidates, freq, threshold=0.035)
        self.assertIn("al", result)

    def test_len3_threshold(self):
        """Trigram 'amp': child 'amph' at 68.2% > 8% -> amp is a fragment."""
        freq = Counter({"a": 29426, "am": 1915, "amp": 418, "amph": 285, "amphi": 248})
        candidates = {"amp"}
        result = filter_affixes(candidates, freq, threshold=0.035)
        self.assertNotIn("amp", result)

    def test_len3_below_fragment_threshold(self):
        """Trigram 'pre': child 'pref' at 8% < 8% (not strictly above) -> pre is NOT a fragment."""
        freq = Counter({"p": 10000, "pr": 3000, "pre": 1000, "pref": 80})
        candidates = {"pre"}
        result = filter_affixes(candidates, freq, threshold=0.035)
        self.assertIn("pre", result)

    def test_len4_threshold(self):
        """4-letter 'amph': child 'amphi' at 87% > 50% dominant cap -> amph is a fragment."""
        freq = Counter({"a": 29426, "am": 1915, "amp": 418, "amph": 285, "amphi": 248})
        candidates = {"amph"}
        result = filter_affixes(candidates, freq, threshold=0.035)
        self.assertNotIn("amph", result)

    def test_len4_below_fragment_threshold(self):
        """4-letter 'over': child 'overt' at 20% < 31.5% -> over is NOT a fragment."""
        freq = Counter({"o": 20000, "ov": 5000, "ove": 2000, "over": 1000, "overt": 200})
        candidates = {"over"}
        result = filter_affixes(candidates, freq, threshold=0.035)
        self.assertIn("over", result)

    def test_len5_no_children_passes(self):
        """5-letter 'amphi': no dominant child -> passes."""
        freq = Counter({"a": 29426, "am": 1915, "amp": 418, "amph": 285, "amphi": 248})
        candidates = {"amphi"}
        result = filter_affixes(candidates, freq, threshold=0.035)
        self.assertIn("amphi", result)

    def test_len5_above_fragment_threshold(self):
        """5-letter prefix with a child at 60% > 50% -> fragment."""
        freq = Counter({
            "s": 50000, "su": 10000, "sub": 5000, "supe": 2000,
            "super": 1000, "superv": 600,
        })
        candidates = {"super"}
        result = filter_affixes(candidates, freq, threshold=0.035)
        self.assertNotIn("super", result)

    def test_custom_fragment_thresholds(self):
        """Caller can override the length->threshold mapping."""
        freq = Counter({"a": 10000, "am": 1915, "amp": 418})
        candidates = {"am"}
        # With default thresholds, 'am' has child 'amp' at 21.8%.
        # len-2 fragment threshold is 3.5%, so amp > 3.5% means 'am' has
        # 1 child above threshold. But spread-aware logic + fragment relaxation
        # means 'am' passes (max_child_ratio 21.8% < 50% dominant cap,
        # and distribution > base threshold). So we use stricter custom
        # thresholds to force filtering.
        result_strict = filter_affixes(
            candidates, freq, threshold=0.035,
            fragment_thresholds={2: 0.035, 3: 0.08, 4: 0.315},
            fragment_long_default=0.20,  # Lower dominant cap to catch 21.8%
        )
        self.assertNotIn("am", result_strict)
        result_custom = filter_affixes(
            candidates, freq, threshold=0.035,
            fragment_thresholds={2: 0.25, 3: 0.25, 4: 0.40},
        )
        self.assertIn("am", result_custom)

    def test_custom_long_default(self):
        """Caller can override the fallback for long prefixes."""
        freq = Counter({"s": 50000, "su": 10000, "sub": 5000, "supe": 2000, "super": 1000, "superv": 600})
        candidates = {"super"}
        result_strict = filter_affixes(candidates, freq, threshold=0.035)
        self.assertNotIn("super", result_strict)
        result_loose = filter_affixes(
            candidates, freq, threshold=0.035, fragment_long_default=0.70,
        )
        self.assertIn("super", result_loose)

    def test_single_char_never_fragment(self):
        """Single-char prefix (len 1) is a root -- fragment check is N/A."""
        freq = Counter({"a": 29426, "ab": 10000})
        candidates = {"a"}
        result = filter_affixes(candidates, freq, threshold=0.035)
        self.assertIn("a", result)


class TestSiblingCompetition(unittest.TestCase):
    """
    Sibling competition filtering: when a candidate extends an already-valid
    prefix and is one of many undifferentiated siblings, it's prefix + root-start.
    """
    def setUp(self):
        self.filtered_prefixes = filter_affixes(set(PREFIX_CANDIDATES), prefix_freq)

    def test_misc_filtered_as_sibling(self):
        """'misc' is one of many children of 'mis' -> filtered."""
        self.assertNotIn("misc", self.filtered_prefixes)

    def test_mist_filtered_as_sibling(self):
        """'mist' is one of many children of 'mis' -> filtered."""
        self.assertNotIn("mist", self.filtered_prefixes)

    def test_counterp_filtered_as_sibling(self):
        """'counterp' is one of many children of 'counter' -> filtered."""
        self.assertNotIn("counterp", self.filtered_prefixes)

    def test_counterf_filtered_as_sibling(self):
        """'counterf' is one of many children of 'counter' -> filtered."""
        self.assertNotIn("counterf", self.filtered_prefixes)

    def test_dominant_sibling_passes(self):
        """'amphi' dominates among 'amph' children -> passes."""
        self.assertIn("amphi", self.filtered_prefixes)

    def test_miso_filtered_as_sibling(self):
        """'miso' is one of many children of 'mis' -> filtered."""
        self.assertNotIn("miso", self.filtered_prefixes)


class TestEdgeCases(unittest.TestCase):
    """Edge cases for robustness across ISO 639-3 language families."""

    def test_empty_candidates(self):
        freq = Counter({"a": 100})
        self.assertEqual(filter_affixes(set(), freq), set())

    def test_empty_freq(self):
        self.assertEqual(filter_affixes({"ab"}, Counter()), set())

    def test_both_empty(self):
        self.assertEqual(filter_affixes(set(), Counter()), set())

    def test_candidate_not_in_freq(self):
        freq = Counter({"a": 100, "ab": 50})
        candidates = {"ac"}
        self.assertEqual(filter_affixes(candidates, freq), set())

    def test_parent_not_in_freq(self):
        freq = Counter({"abc": 50})
        candidates = {"abc"}
        self.assertEqual(filter_affixes(candidates, freq), set())

    def test_empty_string_candidate(self):
        freq = Counter({"a": 100, "": 50})
        candidates = {"", "a"}
        result = filter_affixes(candidates, freq)
        self.assertNotIn("", result)

    def test_case_sensitivity(self):
        freq = Counter({"A": 1000, "Ab": 500})
        candidates = {"Ab"}
        result = filter_affixes(candidates, freq, threshold=0.01)
        self.assertEqual(result, set())
        result2 = filter_affixes(candidates, freq, vowels="aeiouyAEIOUY", threshold=0.01)
        self.assertIn("Ab", result2)

    def test_unicode_prefixes(self):
        """Non-Latin scripts (Cyrillic) with custom vowels."""
        freq = Counter({"а": 5000, "аб": 1000})
        candidates = {"аб"}
        result = filter_affixes(candidates, freq, vowels="аеиоуыэюя", threshold=0.035)
        self.assertIn("аб", result)

    def test_large_scale_performance(self):
        import string
        import itertools
        chars = string.ascii_lowercase
        freq = Counter()
        for c in chars:
            freq[c] = 10000
        for a, b in itertools.product(chars, repeat=2):
            freq[a + b] = 100
        candidates = {a + b for a, b in itertools.product(chars, repeat=2)}
        result = filter_affixes(candidates, freq, threshold=0.035)
        self.assertIsInstance(result, set)

    def test_threshold_boundary_exactly_at(self):
        """Value exactly at threshold should NOT pass (strict >)."""
        freq2 = Counter({"a": 10000, "ab": 1000, "abc": 35})
        candidates2 = {"abc"}
        result2 = filter_affixes(candidates2, freq2, threshold=0.035)
        self.assertNotIn("abc", result2)

    def test_zero_count_prefix_skipped(self):
        """Prefix in candidates with 0 count -> skipped safely."""
        freq = Counter({"a": 100, "ab": 0})
        candidates = {"ab"}
        result = filter_affixes(candidates, freq)
        self.assertEqual(result, set())

    def test_multiple_children_only_max_matters(self):
        """Only the single largest child matters for fragment detection."""
        freq = Counter({"a": 10000, "ab": 1000, "abc": 20, "abd": 20, "abe": 20})
        candidates = {"ab"}
        # Each child is 2% < 3.5% (len-2 threshold) -> not a fragment
        result = filter_affixes(candidates, freq, threshold=0.035)
        self.assertIn("ab", result)


class TestFullExampleFromSpec(unittest.TestCase):
    """End-to-end test using the example data from the specification."""

    def test_ab_passes(self):
        """ab: 1400/29426 = 4.7% > 3.5% (assimilated bigram uses base threshold)"""
        result = filter_affixes({"ab"}, prefix_freq, threshold=0.035)
        self.assertIn("ab", result)

    def test_ac_passes(self):
        """ac: 2421/29426 = 8.2% > 3.5% (assimilated bigram uses base threshold)"""
        result = filter_affixes({"ac"}, prefix_freq, threshold=0.035)
        self.assertIn("ac", result)

    def test_ae_rejected(self):
        """ae: vowel+vowel, 1.4% < 15% (no assimilation) -> rejected."""
        result = filter_affixes({"ae"}, prefix_freq, threshold=0.035)
        self.assertNotIn("ae", result)

    def test_af_passes_via_assimilation(self):
        """af: 808/29426=2.74% which is below 3.5% but passes via assimilation
        (vowel+consonant bigram gets freq_thresh=0)."""
        result = filter_affixes({"af"}, prefix_freq, threshold=0.035)
        self.assertIn("af", result)

    def test_am_passes_via_assimilation(self):
        """am (len 2): vowel+consonant bigram passes via assimilation.
        Although child amp=21.8% > 3.5% fragment threshold, 'am' has 8+
        children above threshold (productive branching), same pattern as
        ab, ac, ad, af, ag, al. The filter correctly treats these the same."""
        result = filter_affixes({"am"}, prefix_freq, threshold=0.035)
        self.assertIn("am", result)

    def test_amp_is_fragment(self):
        """amp (len 3): child amph = 285/418 = 68.2% > 31.5% (dominant child cap) -> fragment."""
        result = filter_affixes({"amp"}, prefix_freq, threshold=0.035)
        self.assertNotIn("amp", result)

    def test_amph_is_fragment(self):
        """amph (len 4): child amphi = 248/285 = 87% > 31.5% -> fragment."""
        result = filter_affixes({"amph"}, prefix_freq, threshold=0.035)
        self.assertNotIn("amph", result)

    def test_amphi_passes(self):
        """amphi (len 5): no children -> passes (87% of parent, above threshold)."""
        result = filter_affixes({"amphi"}, prefix_freq, threshold=0.035)
        self.assertIn("amphi", result)

    def test_batch_evaluation(self):
        """Evaluate all candidates at once with dynamic fragment thresholds."""
        candidates = {"ab", "ac", "ad", "ae", "af", "ag", "ah", "ai",
                       "aj", "ak", "al", "am", "amp", "amph", "amphi"}
        result = filter_affixes(candidates, prefix_freq, threshold=0.035)

        self.assertIn("ab", result)
        self.assertIn("ac", result)
        self.assertIn("ad", result)
        self.assertNotIn("ae", result)   # vowel+vowel, no assimilation, 1.4% < 15%
        self.assertIn("af", result)
        self.assertIn("ag", result)
        # ah, aj, ak have count 0 in freq -> filtered
        self.assertNotIn("ah", result)
        self.assertNotIn("ai", result)   # vowel+vowel, no assimilation
        self.assertNotIn("aj", result)
        self.assertNotIn("ak", result)
        self.assertIn("al", result)
        self.assertIn("am", result)      # vowel+consonant, productive branching like ab/ac
        self.assertNotIn("amp", result)  # fragment: amph=68% > 50% dominant cap
        self.assertNotIn("amph", result) # fragment: amphi=87% > 50% dominant cap
        self.assertIn("amphi", result)

    def test_comprehensive_prefix_detection(self, full_regression=False):
        if full_regression:
            all_affixes, affix_freq = _extract_testing_data(is_suffix=False)
            save_json_file(os.getcwd(), "prefix_freq_data.json", affix_freq)
            affix_candidates = set(all_affixes)
            affix_freq = Counter(dict(affix_freq))
        else:
            affix_candidates = set(PREFIX_CANDIDATES)
            affix_freq = prefix_freq

        result = filter_affixes(affix_candidates, affix_freq, is_suffix=False)
        logger.info(f"Filter results: {len(result)} passed out of {len(affix_candidates)} prefix candidates\n")
        # save_json_file(os.getcwd(), "eng_prefixes.json", sorted(result))

        # -- TEST 1: Productive prefixes must pass --
        logger.info("=" * 65)
        logger.info("TEST 1: Productive prefixes must PASS the filter")
        logger.info("=" * 65)
        pass_failures = []
        for prefix, desc in PRODUCTIVE_PREFIXES:
            if prefix not in affix_candidates:
                continue
            if prefix in result:
                logger.info(f"  PASS  {prefix:15s} - {desc}")
            else:
                logger.info(f"  FAIL  {prefix:15s} - {desc} [WRONGLY FILTERED]")
                pass_failures.append(prefix)

        # -- TEST 2: False positive root-starts must be filtered --
        must_filter = [
            ('misc', 'not a valid prefix'),
            ('mist', 'not a valid prefix'),
            ('counterp', 'not a valid prefix'),
            ('counterf', 'not a valid prefix'),
        ]

        logger.info("=" * 65)
        logger.info("TEST 2: False positive root-starts must be FILTERED")
        logger.info("=" * 65)
        filter_failures = []
        for prefix, desc in must_filter:
            if prefix not in affix_candidates:
                continue
            if prefix not in result:
                logger.info(f"  PASS  {prefix:15s} filtered - {desc}")
            else:
                logger.info(f"  FAIL  {prefix:15s} NOT filtered - {desc}")
                filter_failures.append(prefix)

        # -- Summary --
        total_failures = len(pass_failures) + len(filter_failures)
        logger.info("=" * 65)
        if total_failures == 0:
            logger.info("ALL PREFIX TESTS PASSED!")
        else:
            logger.info(f"FAILURES: {total_failures}")
            if pass_failures:
                print(f"  Productive prefixes wrongly filtered: {pass_failures}")
            if filter_failures:
                print(f"  False positives not filtered: {filter_failures}")
        logger.info("=" * 65)

        self.assertEqual(total_failures, 0)

    def test_comprehensive_suffix_detection(self, full_regression=False):
        """
        Suffix filtering tests using real-world English suffix frequency data.
        """
        if full_regression:
            all_affixes, affix_freq = _extract_testing_data(is_suffix=True)
            save_json_file(os.getcwd(), "suffix_freq_data.json", affix_freq)
            affix_candidates = set(all_affixes)
            affix_freq = Counter(dict(affix_freq))
        else:
            affix_candidates = set(SUFFIX_CANDIDATES)
            affix_freq = suffix_freq

        result = filter_affixes(affix_candidates, affix_freq, is_suffix=True)
        logger.info(f"Filter results: {len(result)} passed out of {len(affix_candidates)} suffix candidates\n")
        # save_json_file(os.getcwd(), "eng_suffixes.json", sorted(result))

        # -- TEST 1: Productive suffixes must pass --
        logger.info("=" * 65)
        logger.info("TEST 1: Productive suffixes must PASS the filter")
        logger.info("=" * 65)
        pass_failures = []
        for suffix, desc in PRODUCTIVE_SUFFIXES:
            if suffix not in affix_candidates:
                continue
            if suffix in result:
                logger.info(f"  PASS  {suffix:15s} - {desc}")
            else:
                logger.info(f"  FAIL  {suffix:15s} - {desc} [WRONGLY FILTERED]")
                pass_failures.append(suffix)

        # -- TEST 2: False positive root-starts must be filtered --
        must_filter = [(x, 'not a valid suffix') for x in INVALID_SUFFIXES]

        logger.info("=" * 65)
        logger.info("TEST 2: False positive root-starts must be FILTERED")
        logger.info("=" * 65)
        filter_failures = []
        for suffix, desc in must_filter:
            if suffix not in affix_candidates:
                continue
            if suffix not in result:
                logger.info(f"  PASS  {suffix:15s} filtered - {desc}")
            else:
                logger.info(f"  FAIL  {suffix:15s} NOT filtered - {desc}")
                filter_failures.append(suffix)

        # -- Summary --
        total_failures = len(pass_failures) + len(filter_failures)
        logger.info("=" * 65)
        if total_failures == 0:
            logger.info("ALL SUFFIX TESTS PASSED!")
        else:
            logger.info(f"FAILURES: {total_failures}")
            if pass_failures:
                print(f"  Productive suffixes wrongly filtered: {pass_failures}")
            if filter_failures:
                print(f"  False positives not filtered: {filter_failures}")
        logger.info("=" * 65)

        self.assertEqual(0, total_failures)


class TestExtractProductiveAffixes(unittest.TestCase):
    """
    Tests for ``extract_productive_affixes``.

    The function accepts a *set* of pre-filtered affix candidates, scores each
    one by morphological productivity (free-stem ratio × log(total_freq) ×
    √(1 + n_desc)), and returns the top-N as a ranked list of (affix, score)
    pairs.  It does NOT call ``filter_affixes`` internally — the candidates
    are assumed to have already been structurally filtered.
    """

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def _load_json_set(cls, filename: str) -> Set[str]:
        """Locate *filename* and return its contents as a set of strings."""
        import json
        candidates = [
            os.path.join(os.getcwd(), filename),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
        ]
        for path in candidates:
            if os.path.exists(path):
                with open(path) as f:
                    return set(json.load(f))
        raise FileNotFoundError(
            f"{filename!r} not found. Looked in: {candidates}"
        )

    @classmethod
    def _load_dict_words(cls) -> Set[str]:
        return cls._load_json_set("eng_words.json")

    @classmethod
    def setUpClass(cls):
        cls.prefix_candidates = cls._load_json_set(ENG_PREFIXES_FILE)
        cls.suffix_candidates = cls._load_json_set(ENG_SUFFIXES_FILE)
        cls.dict_words = cls._load_dict_words()

    # ------------------------------------------------------------------ #
    # Contract / edge cases                                                #
    # ------------------------------------------------------------------ #

    def test_empty_input_returns_empty(self):
        """Empty candidate set → empty result, no crash."""
        result = extract_productive_affixes(set(), prefix_freq, is_suffix=False)
        self.assertEqual(result, [])

    def test_returns_list_of_tuples(self):
        """Return type is list of (str, float) tuples."""
        result = extract_productive_affixes(
            {"re", "un", "pre"}, prefix_freq, is_suffix=False
        )
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], str)
            self.assertIsInstance(item[1], float)

    def test_result_is_subset_of_input(self):
        """Every returned affix must have been in the input set."""
        sample = {"re", "un", "pre", "ness", "ing", "zzz"}
        result = extract_productive_affixes(sample, prefix_freq, is_suffix=False)
        returned_affixes = {affix for affix, _ in result}
        self.assertTrue(returned_affixes.issubset(sample))

    def test_sorted_descending(self):
        """Results are sorted by score descending."""
        result = extract_productive_affixes(
            self.prefix_candidates, prefix_freq,
            is_suffix=False, dict_words=self.dict_words, top_n=50,
        )
        scores = [s for _, s in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_n_respected(self):
        """Result length never exceeds top_n."""
        for n in [10, 50, 100, 500]:
            result = extract_productive_affixes(
                self.prefix_candidates, prefix_freq,
                is_suffix=False, top_n=n,
            )
            self.assertLessEqual(len(result), n)

    def test_min_len_filters_short(self):
        """Affixes shorter than min_len are excluded."""
        result = extract_productive_affixes(
            {"a", "re", "un", "pre"}, prefix_freq,
            is_suffix=False, min_len=3,
        )
        returned = {a for a, _ in result}
        self.assertNotIn("a", returned)
        self.assertNotIn("re", returned)

    def test_result_smaller_than_input(self):
        """Result must be strictly smaller than the full candidate pool."""
        result_p = extract_productive_affixes(
            self.prefix_candidates, prefix_freq,
            is_suffix=False, dict_words=self.dict_words, top_n=500,
        )
        result_s = extract_productive_affixes(
            self.suffix_candidates, suffix_freq,
            is_suffix=True, dict_words=self.dict_words, top_n=500,
        )
        self.assertLess(len(result_p), len(self.prefix_candidates))
        self.assertLess(len(result_s), len(self.suffix_candidates))

    # ------------------------------------------------------------------ #
    # Prefix mode — productive prefixes must rank in the top results      #
    # ------------------------------------------------------------------ #

    def test_prefix_core_productive_in_top200(self):
        """
        Core productive prefixes (un-, re-, pre-, dis-, mis-, non-, over-,
        sub-, inter-, counter-) must all appear in the top-200 results when
        dict_words is provided.
        """
        result = extract_productive_affixes(
            self.prefix_candidates, prefix_freq,
            is_suffix=False, dict_words=self.dict_words, top_n=200,
        )
        returned = {a for a, _ in result}
        must_pass = [
            "un", "re", "pre", "dis", "mis", "non", "over",
            "sub", "inter", "counter",
        ]
        failures = [p for p in must_pass if p not in returned]
        self.assertEqual(
            failures, [],
            f"Core productive prefixes missing from top-200: {failures}",
        )

    def test_prefix_productive_outrank_nonproductive(self):
        """
        Every core productive prefix must score higher than clearly
        non-productive root fragments present in the actual candidate file.
        No injection — all candidates come from ``eng_prefixes.json`` as-is.
        """
        result = extract_productive_affixes(
            self.prefix_candidates, prefix_freq,
            is_suffix=False, dict_words=self.dict_words, top_n=2691,
        )
        score_map = {a: s for a, s in result}

        productive = ["un", "re", "pre", "dis", "non", "over", "sub"]
        # Low-scoring root fragments verified to be present in eng_prefixes.json
        non_productive = [p for p in ["staph", "intell", "sesqu", "blepha", "ophtha"]
                          if p in score_map]

        for prod in productive:
            prod_score = score_map.get(prod, 0.0)
            for nonprod in non_productive:
                nonprod_score = score_map[nonprod]
                self.assertGreater(
                    prod_score, nonprod_score,
                    f"{prod!r} (score={prod_score:.3f}) should outscore "
                    f"non-productive fragment {nonprod!r} (score={nonprod_score:.3f})",
                )

    def test_comp_fragments_rank_below_productive(self):
        """
        compo- (the comp- fragment present in eng_prefixes.json) must score
        below core productive prefixes because it attaches only to bound
        Latin roots (compose, composure…) where the residual stem is not
        an independent English word.
        """
        result = extract_productive_affixes(
            self.prefix_candidates, prefix_freq,
            is_suffix=False, dict_words=self.dict_words, top_n=2691,
        )
        score_map = {a: s for a, s in result}

        productive = ["un", "re", "pre", "dis", "mis", "non", "over", "sub", "counter"]
        # compo is verified to be in eng_prefixes.json; comp/compr are not
        comp_frags = [f for f in ["compo", "compr"] if f in score_map]
        self.assertTrue(comp_frags, "Expected at least one comp- fragment in score_map")

        for prod in productive:
            prod_score = score_map.get(prod, 0.0)
            for frag in comp_frags:
                frag_score = score_map[frag]
                self.assertGreater(
                    prod_score, frag_score,
                    f"{prod!r} (score={prod_score:.3f}) should outscore "
                    f"comp-fragment {frag!r} (score={frag_score:.3f})",
                )

    def test_prefix_result_count_in_expected_range(self):
        """Result size is in the 100–500 range for top_n=500."""
        result = extract_productive_affixes(
            self.prefix_candidates, prefix_freq,
            is_suffix=False, dict_words=self.dict_words, top_n=500,
        )
        self.assertGreaterEqual(len(result), 100)
        self.assertLessEqual(len(result), 500)

    # ------------------------------------------------------------------ #
    # Suffix mode — productive suffixes must rank in the top results      #
    # ------------------------------------------------------------------ #

    def test_suffix_core_productive_in_top200(self):
        """
        Core productive suffixes (-s, -ed, -ing, -er, -ness, -ly, -less,
        -ful, -able, -tion, -ment, -ize) must all appear in the top-200
        results when dict_words is provided.  min_len=1 to include single-char
        suffixes like 's' and 'd'.
        """
        result = extract_productive_affixes(
            self.suffix_candidates, suffix_freq,
            is_suffix=True, dict_words=self.dict_words, top_n=200, min_len=1,
        )
        returned = {a for a, _ in result}
        must_pass = ["s", "ed", "ing", "er", "ness", "ly", "less", "ful", "able", "tion", "ment", "ize"]
        failures = [s for s in must_pass if s not in returned]
        self.assertEqual(
            failures, [],
            f"Core productive suffixes missing from top-200: {failures}",
        )

    def test_suffix_productive_outrank_nonproductive(self):
        """
        Core productive suffixes must score higher than clearly non-productive
        root-end fragments that are present in the pre-filtered candidate file.

        These fragments passed ``filter_affixes`` (structural filtering) but
        are not truly morphologically productive — they should rank near the
        bottom when scored by ``extract_productive_affixes``.  This verifies
        the productivity scoring separates them from real suffixes without any
        injection of synthetic bad data.
        """
        result = extract_productive_affixes(
            self.suffix_candidates, suffix_freq,
            is_suffix=True, dict_words=self.dict_words, top_n=500, min_len=1,
        )
        score_map = {a: s for a, s in result}

        productive = ["s", "ed", "ing", "er", "ness", "ly", "less", "ful", "able"]
        # Non-productive fragments that survived structural filtering but are
        # present in the actual eng_suffixes.json file (verified at test design time)
        non_productive = [s for s in ["acking", "alistic", "aldehyde", "rking", "nking"]
                          if s in score_map]

        for prod in productive:
            prod_score = score_map.get(prod, 0.0)
            for nonprod in non_productive:
                nonprod_score = score_map[nonprod]
                self.assertGreater(
                    prod_score, nonprod_score,
                    f"{prod!r} (score={prod_score:.3f}) should outscore "
                    f"non-productive suffix {nonprod!r} (score={nonprod_score:.3f})",
                )

    def test_suffix_result_count_in_expected_range(self):
        """Result size is in the 100–500 range for top_n=500."""
        result = extract_productive_affixes(
            self.suffix_candidates, suffix_freq,
            is_suffix=True, dict_words=self.dict_words, top_n=500,
        )
        self.assertGreaterEqual(len(result), 100)
        self.assertLessEqual(len(result), 500)

    # ------------------------------------------------------------------ #
    # Mode separation                                                      #
    # ------------------------------------------------------------------ #

    def test_is_suffix_flag_changes_result(self):
        """
        Running the same candidates with is_suffix=True vs False must
        produce different results, confirming the flag is respected.
        """
        prefix_result = extract_productive_affixes(
            self.prefix_candidates, prefix_freq, is_suffix=False,
        )
        suffix_result = extract_productive_affixes(
            self.prefix_candidates, suffix_freq, is_suffix=True,
        )
        p_affixes = {a for a, _ in prefix_result}
        s_affixes = {a for a, _ in suffix_result}
        self.assertNotEqual(
            p_affixes, s_affixes,
            "is_suffix flag had no effect — results are identical",
        )

    def test_dict_words_improves_comp_separation(self):
        """
        Providing dict_words must increase the score ratio between core
        productive prefixes and comp-fragments.  Without dict_words, scoring
        is driven purely by frequency (compo benefits from the large 'comp-'
        word family).  With dict_words, the free-stem-ratio signal reveals that
        compo attaches mostly to bound Latin roots, widening the score gap.
        """
        result_no_dict = extract_productive_affixes(
            self.prefix_candidates, prefix_freq, is_suffix=False, top_n=2691,
        )
        result_with_dict = extract_productive_affixes(
            self.prefix_candidates, prefix_freq, is_suffix=False,
            dict_words=self.dict_words, top_n=2691,
        )

        def score_of(affix, result):
            for a, s in result:
                if a == affix:
                    return s
            return 0.0

        un_no,    compo_no    = score_of("un", result_no_dict),    score_of("compo", result_no_dict)
        un_with,  compo_with  = score_of("un", result_with_dict),  score_of("compo", result_with_dict)

        ratio_no   = un_no   / compo_no   if compo_no   else float("inf")
        ratio_with = un_with / compo_with if compo_with else float("inf")

        self.assertGreater(
            ratio_with, ratio_no,
            f"dict_words should widen the un/compo score ratio "
            f"(no_dict: {ratio_no:.2f}x, with_dict: {ratio_with:.2f}x)",
        )


if __name__ == "__main__":
    unittest.main(exit=False)