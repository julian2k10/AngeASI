import bisect
import math
import os
import time
import unicodedata
from functools import wraps
from collections import Counter, defaultdict
from typing import List, Set, Optional, Tuple, Dict, Any, FrozenSet, Counter as CounterType
from context_aware_io import get_affix_by_length, get_hash_key, load_json_file, save_json_file, STORAGE_DIR
from morpheme_extractor import filter_affixes
from utility import (
    CACHE_DIR, LATIN_GREEK_ROOTS_CACHE, check_cache_validity, find_files_by_pattern, delete_files_from_list,
)

_VOWELS_ALL: FrozenSet[str] = frozenset(
    "aeiouAEIOU"
    + "yYwW"
    + "æÆøØœŒ"
    + "ɐɑɒɓɔɕɛɜɝɞɟɠɡɢɣɤɥɨɩɪɫɬɭɯɰɱɲɳɴɵɶɷɸɹɺɻɼɽɾɿ"
    + "ʀʁʂʃʄʅʆʇʈʉʊʋʌʍʎʏʐʑʒʓ"
    + "əƏı"
    + "ÀàÁáÂâÃãÄäÅåĀāĂăĄąǍǎȀȁȂȃ"
    + "ẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặ"
    + "ÈèÉéÊêËëĒēĔĕĘęĚěȄȅȆȇ"
    + "ẸẹẺẻẼẽẾếỀềỂểỄễỆệ"
    + "ÌìÍíÎîÏïĨĩĪīĬĭĮįİȈȉȊȋỈỉỊị"
    + "ÒòÓóÔôÕõÖöŌōŎŏŐőǑǒȌȍȎȏ"
    + "ỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợ"
    + "ÙùÚúÛûÜüŨũŪūŬŭŮůŰűŲųǓǔȔȕȖȗ"
    + "ỤụỦủỨứỪừỬửỮữỰựŸÿÝýŶŷ"
    + "аАеЕёЁиИоОуУыЫэЭюЮяЯіІїЇєЄөӨүҮұҰәӘӣӢӯӮӑӐӓӒӏ"
    + "αΑεΕηΗιΙοΟυΥωΩάέήίόύώΆΈΉΊΌΎΏ"
    + "ἀἁἂἃἄἅἆἇἈἉἊἋἌἍἎἏἐἑἒἓἔἕἘἙἚἛἜἝ"
    + "ἠἡἢἣἤἥἦἧἨἩἪἫἬἭἮἯἰἱἲἳἴἵἶἷἸἹἺἻἼἽἾἿ"
    + "ὀὁὂὃὄὅὈὉὊὋὌὍὐὑὒὓὔὕὖὗὙὛὝὟ"
    + "ὠὡὢὣὤὥὦὧὨὩὪὫὬὭὮὯὰάὲέὴήὶίὸόὺύὼώ"
    + "ᾀᾁᾂᾃᾄᾅᾆᾇᾐᾑᾒᾓᾔᾕᾖᾗᾠᾡᾢᾣᾤᾥᾦᾧ"
    + "ᾰᾱᾲᾳᾴᾶᾷᾸᾹᾺΆᾼῂῃῄῆῇῊΉῌ"
    + "ῐῑῒΐῖῗῘῙῚΊῠῡῢΰῤῥῦῧῨῩῪΎῬῲῳῴῶῷῺΏῼ"
    + "եԵէԷըԸիԻոՈ\u0531\u0561"
    + "აეიოუ"
    + "\u0627\u0622\u0623\u0625\u0671\u0648\u064A\u0649"
    + "\u05D0\u05D4\u05D5\u05D9\u05E2"
    + "अआइईउऊएऐओऔऋॠऌॡ"
    + "\u093E\u093F\u0940\u0941\u0942\u0943\u0947\u0948\u094B\u094C"
    + "অআইঈউঊএঐওঔঋৠঌৡ"
    + "\u09BE\u09BF\u09C0\u09C1\u09C2\u09C3\u09C7\u09C8\u09CB\u09CC"
    + "ਅਆਇਈਉਊਏਐਓਔ"
    + "અઆઇઈઉઊઋઌઍએઐઑઓઔ"
    + "ଅଆଇଈଉଊଋଌଏଐଓଔ"
    + "அஆஇஈஉஊஏஐஒஓஔ"
    + "అఆఇఈఉఊఋఌఎఏఐఒఓఔ"
    + "ಅಆಇಈಉಊಋಌಎಏಐಒಓಔ"
    + "അആഇഈഉഊഋഌഎഏഐഒഓഔ"
    + "අආඇඈඉඊඋූඑඒඓඔඕඖ"
    + "".join(chr(c) for c in [
        0x0E30,0x0E31,0x0E32,0x0E33,0x0E34,0x0E35,
        0x0E36,0x0E37,0x0E38,0x0E39,
        0x0E40,0x0E41,0x0E42,0x0E43,0x0E44,0x0E45,
    ])
    + "".join(chr(c) for c in [
        0x0EB0,0x0EB1,0x0EB2,0x0EB3,0x0EB4,0x0EB5,
        0x0EB6,0x0EB7,0x0EB8,0x0EB9,
        0x0EC0,0x0EC1,0x0EC2,0x0EC3,0x0EC4,
    ])
    + "".join(chr(c) for c in range(0x0F71, 0x0F7E))
    + "".join(chr(c) for c in range(0x1161, 0x1176))
    + "".join(chr(c) for c in range(0x314F, 0x3164))
    + "あいうえおアイウエオァィゥェォ\u30FC"
    + "አኣኤኢዑዕዓ"
)

# ── Script-level combining vowel sets ────────────────────────────────────
# These are NOT language-specific.  They define which characters act as
# combining vowels in each *writing system*.  Greek/Latin compound
# morphology uses -o- and -i- across ALL languages that borrow these roots.

COMBINING_VOWELS_LATIN = frozenset("oi")
COMBINING_VOWELS_CYRILLIC = frozenset("\u043e\u0438")  # о, и
COMBINING_VOWELS_GREEK = frozenset("\u03bf\u03b9")  # omicron, iota

ALL_VOWELS_LATIN = frozenset("aeiou")
ALL_VOWELS_CYRILLIC = frozenset("\u0430\u0435\u0451\u0438\u043e\u0443\u044b\u044d\u044e\u044f")
ALL_VOWELS_GREEK = frozenset("\u03b1\u03b5\u03b7\u03b9\u03bf\u03c5\u03c9")

# ══════════════════════════════════════════════════════════════════════════════
# §1  Unicode writing-system registry
# ══════════════════════════════════════════════════════════════════════════════
SCRIPT_REGISTRY: Dict[str, Tuple[bool, List[Tuple[int, int]]]] = {
    "Latin":     (True,  [(0x0000,0x007F),(0x0080,0x00FF),(0x0100,0x017F),
                          (0x0180,0x024F),(0x0250,0x02AF),(0x1E00,0x1EFF),
                          (0x2C60,0x2C7F),(0xA720,0xA7FF),(0xAB30,0xAB6F)]),
    "Greek":     (True,  [(0x0370,0x03FF),(0x1F00,0x1FFF)]),
    "Cyrillic":  (True,  [(0x0400,0x04FF),(0x0500,0x052F),(0x2DE0,0x2DFF),
                          (0xA640,0xA69F),(0x1C80,0x1C8F)]),
    "Armenian":  (True,  [(0x0530,0x058F),(0xFB00,0xFB06)]),
    "Georgian":  (True,  [(0x10A0,0x10FF),(0x2D00,0x2D2F),(0x1C90,0x1CBF)]),
    "Hebrew":    (False, [(0x0590,0x05FF),(0xFB1D,0xFB4F)]),
    "Arabic":    (False, [(0x0600,0x06FF),(0x0750,0x077F),
                          (0x08A0,0x08FF),(0xFB50,0xFDFF),(0xFE70,0xFEFF)]),
    "Syriac":    (False, [(0x0700,0x074F),(0x0860,0x086F)]),
    "Thaana":    (False, [(0x0780,0x07BF)]),
    "NKo":       (False, [(0x07C0,0x07FF)]),
    "Samaritan": (False, [(0x0800,0x083F)]),
    "Mandaic":   (False, [(0x0840,0x085F)]),
    "Devanagari":(False, [(0x0900,0x097F),(0xA8E0,0xA8FF)]),
    "Bengali":   (False, [(0x0980,0x09FF)]),
    "Gurmukhi":  (False, [(0x0A00,0x0A7F)]),
    "Gujarati":  (False, [(0x0A80,0x0AFF)]),
    "Oriya":     (False, [(0x0B00,0x0B7F)]),
    "Tamil":     (False, [(0x0B80,0x0BFF)]),
    "Telugu":    (False, [(0x0C00,0x0C7F)]),
    "Kannada":   (False, [(0x0C80,0x0CFF)]),
    "Malayalam": (False, [(0x0D00,0x0D7F)]),
    "Sinhala":   (False, [(0x0D80,0x0DFF),(0x111E0,0x111FF)]),
    "Thai":      (False, [(0x0E00,0x0E7F)]),
    "Lao":       (False, [(0x0E80,0x0EFF)]),
    "Tibetan":   (False, [(0x0F00,0x0FFF)]),
    "Myanmar":   (False, [(0x1000,0x109F),(0xA9E0,0xA9FF),(0xAA60,0xAA7F)]),
    "Ethiopic":  (False, [(0x1200,0x137F),(0x1380,0x139F),
                          (0x2D80,0x2DDF),(0xAB01,0xAB2F)]),
    "Cherokee":  (True,  [(0x13A0,0x13FF),(0xAB70,0xABBF)]),
    "UCAS":      (False, [(0x1400,0x167F),(0x18B0,0x18FF)]),
    "Ogham":     (False, [(0x1680,0x169F)]),
    "Runic":     (False, [(0x16A0,0x16FF)]),
    "Khmer":     (False, [(0x1780,0x17FF),(0x19E0,0x19FF)]),
    "Mongolian": (False, [(0x1800,0x18AF),(0x11660,0x1166F)]),
    "Hiragana":  (False, [(0x3040,0x309F)]),
    "Katakana":  (False, [(0x30A0,0x30FF),(0x31F0,0x31FF),(0xFF65,0xFF9F)]),
    "Bopomofo":  (False, [(0x02EA,0x02EB),(0x3100,0x312F),(0x31A0,0x31BF)]),
    "CJK":       (False, [(0x4E00,0x9FFF),(0x3400,0x4DBF),(0x20000,0x2A6DF),
                          (0x2A700,0x2B73F),(0x2B740,0x2B81F),(0x2B820,0x2CEAF),
                          (0x2CEB0,0x2EBEF),(0xF900,0xFAFF)]),
    "Hangul":    (False, [(0xAC00,0xD7AF),(0x1100,0x11FF),(0xA960,0xA97F),
                          (0xD7B0,0xD7FF),(0x3130,0x318F)]),
    "Yi":        (False, [(0xA000,0xA48F),(0xA490,0xA4CF)]),
    "Lisu":      (False, [(0xA4D0,0xA4FF)]),
    "Vai":       (False, [(0xA500,0xA63F)]),
    "Bamum":     (False, [(0xA6A0,0xA6FF),(0x16800,0x16A3F)]),
    "SylotiNagri":  (False,[(0xA800,0xA82F)]),
    "PhagsPa":      (False,[(0xA840,0xA87F)]),
    "Saurashtra":   (False,[(0xA880,0xA8DF)]),
    "KayahLi":      (False,[(0xA900,0xA92F)]),
    "Rejang":       (False,[(0xA930,0xA95F)]),
    "Javanese":     (False,[(0xA980,0xA9DF)]),
    "Cham":         (False,[(0xAA00,0xAA5F)]),
    "TaiViet":      (False,[(0xAA80,0xAADF)]),
    "MeeteiMayek":  (False,[(0xABC0,0xABFF),(0xAAE0,0xAAFF)]),
    "TaiTham":      (False,[(0x1A20,0x1AAF)]),
    "TaiLe":        (False,[(0x1950,0x197F)]),
    "NewTaiLue":    (False,[(0x1980,0x19DF)]),
    "Batak":        (False,[(0x1BC0,0x1BFF)]),
    "Lepcha":       (False,[(0x1C00,0x1C4F)]),
    "OlChiki":      (False,[(0x1C50,0x1C7F)]),
    "Sundanese":    (False,[(0x1B80,0x1BBF),(0x1CC0,0x1CCF)]),
    "Balinese":     (False,[(0x1B00,0x1B7F)]),
    "Buginese":     (False,[(0x1A00,0x1A1F)]),
    "Tagalog":      (False,[(0x1700,0x171F)]),
    "Hanunoo":      (False,[(0x1720,0x173F)]),
    "Buhid":        (False,[(0x1740,0x175F)]),
    "Tagbanwa":     (False,[(0x1760,0x177F)]),
    "Limbu":        (False,[(0x1900,0x194F)]),
    "PahawhHmong":  (False,[(0x16B00,0x16B8F)]),
    "Miao":         (False,[(0x16F00,0x16F9F)]),
    "Coptic":       (True, [(0x2C80,0x2CFF)]),
    "Glagolitic":   (True, [(0x2C00,0x2C5F),(0x1E000,0x1E02F)]),
    "Gothic":       (False,[(0x10330,0x1034F)]),
    "OldItalic":    (False,[(0x10300,0x1032F)]),
    "Deseret":      (True, [(0x10400,0x1044F)]),
    "Shavian":      (False,[(0x10450,0x1047F)]),
    "Osage":        (True, [(0x10480,0x104AF)]),
    "Elbasan":      (False,[(0x10500,0x1052F)]),
    "CaucasianAlbanian": (False,[(0x10530,0x1056F)]),
    "LinearB":      (False,[(0x10000,0x1007F),(0x10080,0x100FF)]),
    "Lycian":       (False,[(0x10280,0x1029F)]),
    "Lydian":       (False,[(0x10920,0x1093F)]),
    "Carian":       (False,[(0x102A0,0x102DF)]),
    "OldPersian":   (False,[(0x103A0,0x103DF)]),
    "Ugaritic":     (False,[(0x10380,0x1039F)]),
    "Cuneiform":    (False,[(0x12000,0x123FF),(0x12400,0x1247F)]),
    "OldTurkic":    (False,[(0x10C00,0x10C4F)]),
    "OldHungarian": (False,[(0x10C80,0x10CFF)]),
    "Tifinagh":     (False,[(0x2D30,0x2D7F)]),
    "Adlam":        (True, [(0x1E900,0x1E95F)]),
    "HanifiRohingya":    (False,[(0x10D00,0x10D3F)]),
    "Sogdian":      (False,[(0x10F30,0x10F6F)]),
    "OldSogdian":   (False,[(0x10F00,0x10F2F)]),
    "Elymaic":      (False,[(0x10FE0,0x10FFF)]),
    "Nandinagari":  (False,[(0x119A0,0x119FF)]),
    "NyiakengPuachueHmong":(False,[(0x1E100,0x1E14F)]),
    "Wancho":       (False,[(0x1E2C0,0x1E2FF)]),
    "Makasar":      (False,[(0x11EE0,0x11EFF)]),
    "ZanabazarSquare":(False,[(0x11A00,0x11A4F)]),
    "Soyombo":      (False,[(0x11A50,0x11AAF)]),
    "Dogra":        (False,[(0x11800,0x1184F)]),
    "GunjalaGondi": (False,[(0x11D60,0x11DAF)]),
    "MasaramGondi": (False,[(0x11D00,0x11D5F)]),
    "Khojki":       (False,[(0x11200,0x1124F)]),
    "Khudawadi":    (False,[(0x112B0,0x112FF)]),
    "Ahom":         (False,[(0x11700,0x1173F)]),
    "Multani":      (False,[(0x11280,0x112AF)]),
    "Newa":         (False,[(0x11400,0x1147F)]),
    "Tirhuta":      (False,[(0x11480,0x114DF)]),
    "Siddham":      (False,[(0x11580,0x115FF)]),
    "Modi":         (False,[(0x11600,0x1165F)]),
    "Takri":        (False,[(0x11680,0x116CF)]),
    "Sharada":      (False,[(0x11180,0x111DF)]),
    "Brahmi":       (False,[(0x11000,0x1107F)]),
    "Kharoshthi":   (False,[(0x10A00,0x10A5F)]),
    "Grantha":      (False,[(0x11300,0x1137F)]),
    "Chakma":       (False,[(0x11100,0x1114F)]),
    "Mahajani":     (False,[(0x11150,0x1117F)]),
    "Kaithi":       (False,[(0x11080,0x110CF)]),
    "Hatran":       (False,[(0x108E0,0x108FF)]),
    "OldSouthArabian":   (False,[(0x10A60,0x10A9F)]),
    "OldNorthArabian":   (False,[(0x10AA0,0x10ACF)]),
    "InscriptionalParthian":(False,[(0x10B40,0x10B5F)]),
    "InscriptionalPahlavi": (False,[(0x10B60,0x10B7F)]),
    "PsalterPahlavi":       (False,[(0x10B80,0x10BAF)]),
    "Nabataean":    (False,[(0x10880,0x108AF)]),
    "Palmyrene":    (False,[(0x10860,0x1087F)]),
    "Phoenician":   (False,[(0x10900,0x1091F)]),
    "MeroiticHieroglyphs":(False,[(0x10980,0x1099F)]),
    "MeroiticCursive":    (False,[(0x109A0,0x109FF)]),
    "OldPermic":    (False,[(0x10350,0x1037F)]),
    "PauCinHau":    (False,[(0x11AC0,0x11AFF)]),
    "Duployan":     (False,[(0x1BC00,0x1BC9F)]),
}

# ── Pre-build bisect table ────────────────────────────────────────────────────
_RANGES_SORTED: List[Tuple[int, int, str, bool]] = sorted(
    (lo, hi, s, hc)
    for s, (hc, ranges) in SCRIPT_REGISTRY.items()
    for lo, hi in ranges
)
_RANGE_LOS: List[int] = [r[0] for r in _RANGES_SORTED]


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

# ══════════════════════════════════════════════════════════════════════════════
# §5  Corpus profile  (Stage 0)
# ══════════════════════════════════════════════════════════════════════════════

class CorpusProfile:
    __slots__ = (
        "primary_script", "has_case",
        "corpus_chars", "script_chars", "corpus_vowels",
    )

    def __init__(
        self,
        primary_script: str,
        has_case: bool,
        corpus_chars: FrozenSet[str],
        script_chars: FrozenSet[str],
        corpus_vowels: FrozenSet[str],
    ) -> None:
        self.primary_script = primary_script
        self.has_case       = has_case
        self.corpus_chars   = corpus_chars
        self.script_chars   = script_chars
        self.corpus_vowels  = corpus_vowels


def _codepoint_script(cp: int) -> Optional[Tuple[str, bool]]:
    """O(log S) bisect lookup → (script_name, has_case) or None."""
    idx = bisect.bisect_right(_RANGE_LOS, cp) - 1
    if idx >= 0:
        lo, hi, script, has_case = _RANGES_SORTED[idx]
        if lo <= cp <= hi:
            return script, has_case
    return None

def _build_corpus_profile(corpus: Counter, top_n: int = 1000) -> CorpusProfile:
    script_weight: Counter = Counter()
    has_case_map:  Dict[str, bool] = {}
    corpus_chars_set: Set[str] = set()
    profile_characters: Counter = Counter()

    count = 0
    for word, freq in corpus.most_common(top_n + 500):
        if not isinstance(word, str) or not word or not unicodedata.category(word[0]).startswith("L"):
            continue
        profile_characters.update(word.lower())
        count += 1
        if count >= top_n:
            break

    word_scripts: Counter = Counter()
    for ch, freq in profile_characters.items():
        if not unicodedata.category(ch).startswith("L"):
            continue
        corpus_chars_set.add(ch)
        result = _codepoint_script(ord(ch))
        if result:
            s, hc = result
            word_scripts[s] += 1
            has_case_map[s]  = hc
        if word_scripts:
            top_s = word_scripts.most_common(1)[0][0]
            script_weight[top_s] += freq

    if not script_weight:
        primary_script, has_case = "Latin", True
    else:
        primary_script = script_weight.most_common(1)[0][0]
        has_case       = has_case_map[primary_script]

    corpus_chars: FrozenSet[str]  = frozenset(corpus_chars_set)
    script_chars: FrozenSet[str]  = frozenset(
        ch for ch in corpus_chars
        if (r := _codepoint_script(ord(ch))) is not None and r[0] == primary_script
    )
    corpus_vowels: FrozenSet[str] = _VOWELS_ALL & corpus_chars

    return CorpusProfile(
        primary_script=primary_script,
        has_case=has_case,
        corpus_chars=corpus_chars,
        script_chars=script_chars,
        corpus_vowels=corpus_vowels,
    )

# ── Per-script combining-vowel sets ────────────────────────────────────────
# Neoclassical compounding uses a *connecting* vowel (most often 'o' or 'i')
# that is phonologically inserted between two bound roots.  The sets below
# are writing-system-level constants, not language-specific: Greek/Latin roots
# borrowed into Cyrillic-script languages still use the same -o-/-i- linkers.
#
# For scripts that do not have an established neoclassical compounding
# tradition we return the Latin set as a safe fallback, because:
#   • The pipeline only *removes* connecting vowels it recognises, so a
#     conservative fallback never introduces errors.
#   • Most scientific/technical terminology worldwide is built from
#     Greco-Latin roots even in non-Latin scripts, so 'o' and 'i' remain
#     relevant connecting phonemes.

# Abugida / Indic scripts that mark vowels as diacritics attached to
# consonants.  The canonical combining vowels for compound analysis are
# rendered as *matra* (vowel signs) rather than independent vowel letters.
COMBINING_VOWELS_DEVANAGARI = frozenset("\u093E\u093F")   # ā, i (matra forms)
COMBINING_VOWELS_BENGALI    = frozenset("\u09BE\u09BF")    # ā, i (matra forms)
COMBINING_VOWELS_GURMUKHI   = frozenset("\u0A3E\u0A3F")    # ā, i
COMBINING_VOWELS_GUJARATI   = frozenset("\u0ABE\u0ABF")    # ā, i
COMBINING_VOWELS_ORIYA      = frozenset("\u0B3E\u0B3F")    # ā, i
COMBINING_VOWELS_TAMIL      = frozenset("\u0BBE\u0BBF")    # ā, i
COMBINING_VOWELS_TELUGU     = frozenset("\u0C3E\u0C3F")    # ā, i
COMBINING_VOWELS_KANNADA    = frozenset("\u0CBE\u0CBF")    # ā, i
COMBINING_VOWELS_MALAYALAM  = frozenset("\u0D3E\u0D3F")    # ā, i

# Semitic abjads: vowels are normally omitted; when written they appear as
# *matres lectionis* (consonant letters repurposed as long-vowel markers) or
# as diacritical *nikud/harakat* (rarely present in plain text).  We supply
# the most common matres lectionis only.
COMBINING_VOWELS_ARABIC  = frozenset("\u0627\u0648\u064A")  # alef, waw, ya
COMBINING_VOWELS_HEBREW  = frozenset("\u05D0\u05D5\u05D9")  # alef, vav, yod

# South-East Asian syllabic scripts: vowels are written as free-standing
# letters or leading/trailing marks.  We include the short-/a/ nucleus and
# the high-front /i/ since these are the most common phoneme-level linking
# vowels in compounding constructions.
COMBINING_VOWELS_THAI    = frozenset(
    "\u0E30\u0E31\u0E32\u0E34\u0E40"   # sara a, mai han akat, sara aa, sara i, sara e
)
COMBINING_VOWELS_LAO     = frozenset(
    "\u0EB0\u0EB1\u0EB2\u0EB4\u0EC0"   # sara a, mai kan, sara aa, sara i, sara e
)
COMBINING_VOWELS_TIBETAN = frozenset(
    "\u0F71\u0F72\u0F74"               # aa, i, u (vowel signs)
)
COMBINING_VOWELS_MYANMAR = frozenset(
    "\u1031\u102B\u102C\u102D"          # e, aa, aa (asat), i
)
COMBINING_VOWELS_KHMER   = frozenset(
    "\u17B6\u17B7\u17B8"               # aa, i, ii
)

# Alphabetic scripts for which vowels are fully written.
COMBINING_VOWELS_ARMENIAN = frozenset("\u0565\u056B")       # ե e, ի i
COMBINING_VOWELS_GEORGIAN = frozenset("\u10D0\u10D8")       # ა a, ი i

# Hangul: vowels are Jamo letters embedded in syllable blocks.
# The combining jamo for /a/ (ᅡ) and /i/ (ᅵ) are the most neutral linkers.
COMBINING_VOWELS_HANGUL   = frozenset("\u1161\u1175")       # ᅡ a, ᅵ i

# Japanese: both Hiragana あ・い and Katakana ア・イ serve as linking vowels
# in compound words (e.g. 雨合羽 uses intermediate mora).
COMBINING_VOWELS_HIRAGANA = frozenset("\u3042\u3044")       # あ a, い i
COMBINING_VOWELS_KATAKANA = frozenset("\u30A2\u30A4")       # ア A, イ I

# Ethiopic (Ge'ez): the 1st and 5th orders of the syllabary are the
# phonologically 'neutral' vowel grades used in compounding.
COMBINING_VOWELS_ETHIOPIC = frozenset("\u12A0\u12A2")       # አ a, ኢ i

# CJK / Bopomofo: logographic scripts do not use explicit connecting vowels
# between morphemes; an empty frozenset is the linguistically correct answer.
COMBINING_VOWELS_CJK      = frozenset()
COMBINING_VOWELS_BOPOMOFO = frozenset()


# Master dispatch table — keys must match SCRIPT_REGISTRY names (exact case).
_COMBINING_VOWELS_BY_SCRIPT: Dict[str, frozenset] = {
    # Well-studied European scripts
    "Latin":      COMBINING_VOWELS_LATIN,
    "Cyrillic":   COMBINING_VOWELS_CYRILLIC,
    "Greek":      COMBINING_VOWELS_GREEK,
    "Armenian":   COMBINING_VOWELS_ARMENIAN,
    "Georgian":   COMBINING_VOWELS_GEORGIAN,
    "Coptic":     COMBINING_VOWELS_GREEK,     # Coptic alphabet derived from Greek
    "Glagolitic": COMBINING_VOWELS_CYRILLIC,  # Old Slavonic — same phonology
    "Deseret":    COMBINING_VOWELS_LATIN,     # English phonemic alphabet
    "Osage":      COMBINING_VOWELS_LATIN,     # Latin-derived
    "Adlam":      COMBINING_VOWELS_LATIN,     # Modern Latin-influenced
    # Semitic / abjad
    "Arabic":     COMBINING_VOWELS_ARABIC,
    "Hebrew":     COMBINING_VOWELS_HEBREW,
    "Syriac":     COMBINING_VOWELS_ARABIC,
    "Thaana":     COMBINING_VOWELS_ARABIC,
    "NKo":        COMBINING_VOWELS_ARABIC,
    "Samaritan":  COMBINING_VOWELS_HEBREW,
    "Mandaic":    COMBINING_VOWELS_ARABIC,
    # Indic abugidas
    "Devanagari": COMBINING_VOWELS_DEVANAGARI,
    "Bengali":    COMBINING_VOWELS_BENGALI,
    "Gurmukhi":   COMBINING_VOWELS_GURMUKHI,
    "Gujarati":   COMBINING_VOWELS_GUJARATI,
    "Oriya":      COMBINING_VOWELS_ORIYA,
    "Tamil":      COMBINING_VOWELS_TAMIL,
    "Telugu":     COMBINING_VOWELS_TELUGU,
    "Kannada":    COMBINING_VOWELS_KANNADA,
    "Malayalam":  COMBINING_VOWELS_MALAYALAM,
    "Sinhala":    frozenset("\u0DCF\u0DD2"),  # ā, i vowel signs
    "Tibetan":    COMBINING_VOWELS_TIBETAN,
    "Myanmar":    COMBINING_VOWELS_MYANMAR,
    "Khmer":      COMBINING_VOWELS_KHMER,
    "Thai":       COMBINING_VOWELS_THAI,
    "Lao":        COMBINING_VOWELS_LAO,
    # East Asian
    "Hiragana":   COMBINING_VOWELS_HIRAGANA,
    "Katakana":   COMBINING_VOWELS_KATAKANA,
    "Bopomofo":   COMBINING_VOWELS_BOPOMOFO,
    "CJK":        COMBINING_VOWELS_CJK,
    "Hangul":     COMBINING_VOWELS_HANGUL,
    # African
    "Ethiopic":   COMBINING_VOWELS_ETHIOPIC,
    "Cherokee":   COMBINING_VOWELS_LATIN,     # Latin-script phoneme inventory
    "Vai":        frozenset("\u00E1\u00E9"),   # Vai uses a, e as linking vowels
    "Bamum":      frozenset(),
    "Tifinagh":   frozenset("\u2D30\u2D49"),   # Tamazight a, i
    "Lisu":       COMBINING_VOWELS_LATIN,     # Fraser script — Latin-based phonology
    # Scripts with no established compounding tradition — empty sets
    "Mongolian":  frozenset("\u1820\u1822"),   # а, и  (Mongolian letters)
    "Yi":         frozenset(),
    "UCAS":       frozenset(),                 # Unified Canadian Aboriginal Syllabics
    "Ogham":      frozenset(),
    "Runic":      frozenset(),
    # Ancient / rare scripts — Latin fallback
    "LinearB":    COMBINING_VOWELS_LATIN,
    "Lycian":     COMBINING_VOWELS_LATIN,
    "Lydian":     COMBINING_VOWELS_LATIN,
    "Carian":     COMBINING_VOWELS_LATIN,
    "OldItalic":  COMBINING_VOWELS_LATIN,
    "Gothic":     COMBINING_VOWELS_LATIN,
    "OldPersian": frozenset(),
    "Ugaritic":   frozenset(),
    "Cuneiform":  frozenset(),
    "OldTurkic":  frozenset(),
    "OldHungarian": COMBINING_VOWELS_LATIN,
    "OldPermic":  COMBINING_VOWELS_CYRILLIC,
}

# ── Per-script full vowel sets ─────────────────────────────────────────────
# These contain ALL vowel characters (including long-vowel marks and
# diacritical vowel signs) that the pipeline should treat as vowel positions
# when applying CV-structure heuristics.  Derived from _VOWELS_ALL partitioned
# by script, augmented with script-specific vowel marks not covered there.

ALL_VOWELS_DEVANAGARI = frozenset(
    "अआइईउऊएऐओऔऋॠऌॡ"
    + "\u093E\u093F\u0940\u0941\u0942\u0943\u0947\u0948\u094B\u094C\u0945\u0946\u0949\u094A"
)
ALL_VOWELS_BENGALI = frozenset(
    "অআইঈউঊএঐওঔঋৠঌৡ"
    + "\u09BE\u09BF\u09C0\u09C1\u09C2\u09C3\u09C7\u09C8\u09CB\u09CC"
)
ALL_VOWELS_GURMUKHI  = frozenset("ਅਆਇਈਉਊਏਐਓਔ\u0A3E\u0A3F\u0A40\u0A41\u0A42\u0A47\u0A48\u0A4B\u0A4C")
ALL_VOWELS_GUJARATI  = frozenset("અઆઇઈઉઊઋઌઍએઐઑઓઔ\u0ABE\u0ABF\u0AC0\u0AC1\u0AC2\u0AC3\u0AC7\u0AC8\u0ACB\u0ACC")
ALL_VOWELS_ORIYA     = frozenset("ଅଆଇଈଉଊଋଌଏଐଓଔ\u0B3E\u0B3F\u0B40\u0B41\u0B42\u0B43\u0B47\u0B48\u0B4B\u0B4C")
ALL_VOWELS_TAMIL     = frozenset("அஆஇஈஉஊஏஐஒஓஔ\u0BBE\u0BBF\u0BC0\u0BC1\u0BC2\u0BC6\u0BC7\u0BC8\u0BCA\u0BCB\u0BCC")
ALL_VOWELS_TELUGU    = frozenset("అఆఇఈఉఊఋఌఎఏఐఒఓఔ\u0C3E\u0C3F\u0C40\u0C41\u0C42\u0C43\u0C46\u0C47\u0C48\u0C4A\u0C4B\u0C4C")
ALL_VOWELS_KANNADA   = frozenset("ಅಆಇಈಉಊಋಌಎಏಐಒಓಔ\u0CBE\u0CBF\u0CC0\u0CC1\u0CC2\u0CC3\u0CC6\u0CC7\u0CC8\u0CCA\u0CCB\u0CCC")
ALL_VOWELS_MALAYALAM = frozenset("അആഇഈഉഊഋഌഎഏഐഒഓഔ\u0D3E\u0D3F\u0D40\u0D41\u0D42\u0D43\u0D46\u0D47\u0D48\u0D4A\u0D4B\u0D4C")
ALL_VOWELS_SINHALA   = frozenset("අආඇඈඉඊඋූඑඒඓඔඕඖ\u0DCF\u0DD0\u0DD1\u0DD2\u0DD3\u0DD4\u0DD6\u0DD8\u0DDA\u0DDC\u0DDD\u0DDE")
ALL_VOWELS_THAI      = frozenset("".join(chr(c) for c in [
    0x0E30,0x0E31,0x0E32,0x0E33,0x0E34,0x0E35,0x0E36,0x0E37,0x0E38,0x0E39,
    0x0E40,0x0E41,0x0E42,0x0E43,0x0E44,0x0E45,
]))
ALL_VOWELS_LAO       = frozenset("".join(chr(c) for c in [
    0x0EB0,0x0EB1,0x0EB2,0x0EB3,0x0EB4,0x0EB5,0x0EB6,0x0EB7,0x0EB8,0x0EB9,
    0x0EC0,0x0EC1,0x0EC2,0x0EC3,0x0EC4,
]))
ALL_VOWELS_TIBETAN   = frozenset("".join(chr(c) for c in range(0x0F71, 0x0F7E)))
ALL_VOWELS_KHMER     = frozenset("".join(chr(c) for c in [
    0x17B6,0x17B7,0x17B8,0x17B9,0x17BA,0x17BB,0x17BC,0x17BD,0x17BE,0x17BF,
    0x17C0,0x17C1,0x17C2,0x17C3,0x17C4,0x17C5,
]))
ALL_VOWELS_ARABIC    = frozenset(
    "\u0627\u0622\u0623\u0625\u0671\u0648\u064A\u0649"
    + "".join(chr(c) for c in range(0x064B, 0x0651))  # harakat diacritics
)
ALL_VOWELS_HEBREW    = frozenset(
    "\u05D0\u05D4\u05D5\u05D9\u05E2"
    + "".join(chr(c) for c in range(0x05B0, 0x05BC))  # nikud vowel points
)
ALL_VOWELS_ARMENIAN  = frozenset("եԵէԷըԸիԻոՈ\u0531\u0561")
ALL_VOWELS_GEORGIAN  = frozenset("აეიოუ")
ALL_VOWELS_ETHIOPIC  = frozenset("አኣኤኢዑዕዓ")
ALL_VOWELS_HANGUL    = frozenset("".join(chr(c) for c in range(0x1161, 0x1176))
                                 + "".join(chr(c) for c in range(0x314F, 0x3164)))
ALL_VOWELS_HIRAGANA  = frozenset("あいうえおァィゥェォ")
ALL_VOWELS_KATAKANA  = frozenset("アイウエオァィゥェォ")

_ALL_VOWELS_BY_SCRIPT: Dict[str, frozenset] = {
    "Latin":      ALL_VOWELS_LATIN,
    "Cyrillic":   ALL_VOWELS_CYRILLIC,
    "Greek":      ALL_VOWELS_GREEK,
    "Armenian":   ALL_VOWELS_ARMENIAN,
    "Georgian":   ALL_VOWELS_GEORGIAN,
    "Coptic":     ALL_VOWELS_GREEK,
    "Glagolitic": ALL_VOWELS_CYRILLIC,
    "Deseret":    ALL_VOWELS_LATIN,
    "Osage":      ALL_VOWELS_LATIN,
    "Adlam":      ALL_VOWELS_LATIN,
    "Arabic":     ALL_VOWELS_ARABIC,
    "Hebrew":     ALL_VOWELS_HEBREW,
    "Syriac":     ALL_VOWELS_ARABIC,
    "Thaana":     ALL_VOWELS_ARABIC,
    "NKo":        ALL_VOWELS_ARABIC,
    "Samaritan":  ALL_VOWELS_HEBREW,
    "Mandaic":    ALL_VOWELS_ARABIC,
    "Devanagari": ALL_VOWELS_DEVANAGARI,
    "Bengali":    ALL_VOWELS_BENGALI,
    "Gurmukhi":   ALL_VOWELS_GURMUKHI,
    "Gujarati":   ALL_VOWELS_GUJARATI,
    "Oriya":      ALL_VOWELS_ORIYA,
    "Tamil":      ALL_VOWELS_TAMIL,
    "Telugu":     ALL_VOWELS_TELUGU,
    "Kannada":    ALL_VOWELS_KANNADA,
    "Malayalam":  ALL_VOWELS_MALAYALAM,
    "Sinhala":    ALL_VOWELS_SINHALA,
    "Thai":       ALL_VOWELS_THAI,
    "Lao":        ALL_VOWELS_LAO,
    "Tibetan":    ALL_VOWELS_TIBETAN,
    "Khmer":      ALL_VOWELS_KHMER,
    "Myanmar":    frozenset("\u1031\u102B\u102C\u102D\u102E\u1032\u1036"),
    "Hiragana":   ALL_VOWELS_HIRAGANA,
    "Katakana":   ALL_VOWELS_KATAKANA,
    "Hangul":     ALL_VOWELS_HANGUL,
    "Ethiopic":   ALL_VOWELS_ETHIOPIC,
    "Cherokee":   ALL_VOWELS_LATIN,
    "Tifinagh":   frozenset("\u2D30\u2D49\u2D53"),   # a, i, u
    "Lisu":       ALL_VOWELS_LATIN,
    "OldHungarian": ALL_VOWELS_LATIN,
    "OldItalic":  ALL_VOWELS_LATIN,
    "Gothic":     ALL_VOWELS_LATIN,
    "LinearB":    ALL_VOWELS_LATIN,
    "Lycian":     ALL_VOWELS_LATIN,
    "Lydian":     ALL_VOWELS_LATIN,
    "Carian":     ALL_VOWELS_LATIN,
    "Mongolian":  frozenset("\u1820\u1821\u1822\u1823\u1824\u1825"),
    # Logographic / syllabic scripts where vowels are not easily separable
    "CJK":        frozenset(),
    "Bopomofo":   frozenset(),
    "Yi":         frozenset(),
    "UCAS":       frozenset(),
    "Ogham":      frozenset(),
    "Runic":      frozenset(),
    "OldPersian": frozenset(),
    "Ugaritic":   frozenset(),
    "Cuneiform":  frozenset(),
}

MIN_FORM_LEN = 3
MAX_FORM_LEN = 10


def get_combining_vowels(script: str) -> frozenset:
    """Return the combining-vowel set for *script*.

    Combining vowels are the phonemes that can appear as *linkers* between
    two bound roots in compounding morphology (e.g. the '-o-' in 'therm-o-
    meter').  The returned set is used by the morpheme segmenter to:

      * Strip a final combining vowel from a root before testing adjacency.
      * Distinguish true compounds from accidental substring matches.

    Lookup is case-insensitive and falls back to the Latin set when the
    script is unrecognised, because:

      1. Most technical vocabulary worldwide borrows Greco-Latin roots
         regardless of the target script.
      2. An over-inclusive fallback never introduces false splits — it can
         only fail to *detect* a compounding boundary, which is the
         conservative error.

    Parameters
    ----------
    script : str
        Writing-system name.  Should match a key in ``SCRIPT_REGISTRY``
        (e.g. ``"Latin"``, ``"Cyrillic"``, ``"Devanagari"``), but any
        case variant is accepted.

    Returns
    -------
    frozenset
        Immutable set of Unicode character strings that act as combining
        vowels for *script*.  May be empty for logographic scripts (CJK,
        Yi, etc.) where the concept does not apply.

    Examples
    --------
    >>> get_combining_vowels("latin")
    frozenset({'o', 'i'})
    >>> get_combining_vowels("Greek")
    frozenset({'ο', 'ι'})
    >>> get_combining_vowels("Devanagari")       # ā and i matra
    frozenset({'ा', 'ि'})
    >>> get_combining_vowels("CJK")              # no connecting vowels
    frozenset()
    >>> get_combining_vowels("unknown_script")   # graceful fallback
    frozenset({'o', 'i'})
    """
    # Normalise input: strip whitespace
    key = script.strip().casefold()
    # Case-insensitive scan
    for k, v in _COMBINING_VOWELS_BY_SCRIPT.items():
        if k.casefold() == key:
            return v
    # Unknown script — Latin fallback (conservative choice)
    return COMBINING_VOWELS_LATIN


def get_all_vowels(script: str) -> frozenset:
    """Return the full vowel set for *script*.

    The full vowel set contains every character that should be treated as a
    *vowel position* when the pipeline applies CV-structure (consonant-vowel
    alternation) heuristics during morpheme boundary detection and root
    normalisation.  This is a superset of :func:`get_combining_vowels` and
    includes long-vowel marks, diacritical vowel signs, and matres lectionis.

    Lookup and fallback semantics are identical to :func:`get_combining_vowels`.

    Parameters
    ----------
    script : str
        Writing-system name (case-insensitive).

    Returns
    -------
    frozenset
        Immutable set of Unicode character strings that represent vowel
        positions for *script*.  Empty for logographic/syllabic scripts
        where the vowel/consonant distinction is not encoded at the
        character level (e.g. CJK, Yi, UCAS).

    Examples
    --------
    >>> 'a' in get_all_vowels("Latin")
    True
    >>> 'е' in get_all_vowels("Cyrillic")
    True
    >>> get_all_vowels("CJK")
    frozenset()
    """
    # Normalise input: strip whitespace
    key = script.strip().casefold()
    # Case-insensitive scan
    for k, v in _ALL_VOWELS_BY_SCRIPT.items():
        if k.casefold() == key:
            return v
    return ALL_VOWELS_LATIN


class SuffixTrie:
    def __init__(self):
        self.root = {}
        self.end_symbol = "__end__"

    def insert(self, fragment, replacement):
        # We reverse the fragment because we are looking for suffixes
        node = self.root
        for char in reversed(fragment):
            node = node.setdefault(char, {})
        # Store the replacement value at the leaf
        node[self.end_symbol] = replacement

    def find_and_replace(self, word):
        node = self.root
        match_len = 0
        replacement = None

        # Traverse the word from back to front
        for i, char in enumerate(reversed(word)):
            if char in node:
                node = node[char]
                if self.end_symbol in node:
                    # We found a valid fragment!
                    # Store the longest match found so far
                    match_len = i + 1
                    replacement = node[self.end_symbol]
            else:
                break

        # If we found a match at the very end of the word
        if replacement:
            return word[:-match_len] + replacement
        return word


class CompoundLineageTrie:
    """Efficient forward/reversed trie for morpheme boundary scanning.

    Uses ``_END = None`` as a sentinel (cannot collide with character keys).
    String construction is deferred — ``"".join()`` is only called when a
    terminal node is reached, not on every character step.
    """
    __slots__ = ('trie', 'is_suffix_mode')
    _END = None  # sentinel: end-of-morpheme marker

    def __init__(self, is_suffix_mode: bool = False):
        self.trie: Dict = {}
        self.is_suffix_mode = is_suffix_mode

    def insert(self, morpheme: str) -> None:
        node = self.trie
        chars = reversed(morpheme) if self.is_suffix_mode else morpheme
        for ch in chars:
            node = node.setdefault(ch, {})
        node[self._END] = True

    def insert_all(self, morphemes, min_len: int = 0, max_len: int = 999) -> None:
        for m in morphemes:
            if min_len <= len(m) <= max_len:
                self.insert(m)

    def find_all_prefix_matches(self, word: str, start: int) -> List[str]:
        """All morphemes that start at ``word[start]`` (prefix mode only)."""
        matches: List[str] = []
        node = self.trie
        buf: List[str] = []
        for i in range(start, len(word)):
            ch = word[i]
            if ch not in node:
                break
            node = node[ch]
            buf.append(ch)
            if self._END in node:
                matches.append("".join(buf))
        return matches

    def find_exact_prefix(self, word: str, start: int, end: int) -> bool:
        """Return True iff ``word[start:end]`` is exactly a stored morpheme."""
        node = self.trie
        for i in range(start, end):
            ch = word[i]
            if ch not in node:
                return False
            node = node[ch]
        return self._END in node

    def find_all_suffix_matches_reversed(self, word: str, rstart: int) -> List[str]:
        """All morphemes whose reversed chars start at ``rstart`` in the
        reversed word (suffix mode — trie was built with reversed strings)."""
        matches: List[str] = []
        node = self.trie
        buf: List[str] = []
        wlen = len(word)
        for i in range(rstart, wlen):
            ch = word[wlen - 1 - i]  # walk word right-to-left
            if ch not in node:
                break
            node = node[ch]
            buf.append(ch)
            if self._END in node:
                # buf holds reversed chars; reverse to get the actual morpheme
                matches.append("".join(reversed(buf)))
        return matches


MIN_BOUND_ROOT_FREQ: int = 4

@log_time
def normalize_combining_forms(
    combining_forms: Set[str],
    morpheme_freq: Counter,
    script: str,
    min_freq: int = MIN_BOUND_ROOT_FREQ,
) -> Set[str]:
    """
    Extend the combining forms set by adding/removing connecting vowels.

    Greek/Latin compounds use connecting vowels (typically 'o' or 'i')
    between morphemes.  Some combining forms are stored WITH the vowel
    (e.g. 'chemio', 'cardio') and some WITHOUT (e.g. 'chem', 'electr').

    This function ensures BOTH variants exist when supported by
    morpheme frequency evidence:

    - For forms ending in a connecting vowel: add the stripped version
      e.g. 'chemio' → also add 'chemi'  (strip 'o')
      e.g. 'cardio' → also add 'cardi'  (strip 'o')

    - For forms NOT ending in a connecting vowel: add vowel-appended versions
      e.g. 'chem'  → also add 'chemi'   (append 'i')
      e.g. 'electr' → also add 'electri' (append 'i')

    Each new form is only added if its morpheme frequency ≥ min_freq,
    preventing noise from rare substrings.
    """
    cv_set = get_combining_vowels(script)
    extended = set(combining_forms)
    added = 0

    for cform in list(combining_forms):
        if len(cform) < 2:
            continue

        if cform[-1] in cv_set:
            # Has connecting vowel → add stripped version
            stripped = cform[:-1]
            if stripped not in extended and len(stripped) >= 2:
                freq = morpheme_freq.get(stripped, 0)
                if freq >= min_freq:
                    extended.add(stripped)
                    added += 1
        else:
            # No connecting vowel → add vowel-appended versions
            for cv in cv_set:
                appended = cform + cv
                if appended not in extended and len(appended) >= 3:
                    freq = morpheme_freq.get(appended, 0)
                    if freq >= min_freq:
                        extended.add(appended)
                        added += 1

    if added > 0:
        print(f"Normalized combining forms: +{added} variants ({len(combining_forms)} → {len(extended)})")
    return extended


def build_morpheme_freq(corpus: List[str]) -> Counter:
    """
    Build a morpheme frequency Counter from a word corpus.

    Counts how many words in *corpus* start with each possible prefix,
    giving a language-agnostic proxy for morpheme productivity.  This
    is the natural input for ``normalize_combining_forms``.

    Example
    -------
    >>> freq = build_morpheme_freq(["effuse", "effusion", "effusive"])
    >>> freq["effus"]
    3
    """
    freq: Counter = Counter()
    for w in corpus:
        w = w.lower().strip()
        for i in range(2, len(w) + 1):
            freq[w[:i]] += 1
    return freq


def decompose_compound_forms(
        combining_forms: Set[str],
        min_component_freq: int = 2,
        dictionary: Optional[Set[str]] = None,
) -> Set[str]:
    """
    Decompose compound combining forms into individual components and
    restore truncated forms to their full dictionary spellings.

    Many combining forms in the extracted set are themselves compounds of
    two or more neoclassical roots that were fused during suffix-stripping.
    For example, "adenovirus" was stripped to "adenoviru" (removing "-s"),
    then stored as a single combining form.  This function:

    1. **Splits** compound forms at known-form boundaries:
           "adenoviru"  → "adeno" + "viru"
           "gyroscop"   → "gyro" + "scop"

    2. **Restores** truncated components to full dictionary words
       by checking ``form + c`` for every character ``c`` that appears
       word-finally in the dictionary.  This is fully language-agnostic:
       no hardcoded endings are needed.
           "viru"      + "s" → "virus"      (dictionary word ✓)
           "abbreviat" + "e" → "abbreviate" (dictionary word ✓)

    Both the truncated form AND the restored full form are kept.

    Only LEFT-anchored decomposition is used: if the LEFT part is a
    known form, the RIGHT part is a candidate.  This prevents junk
    like "nim" from "unanim" = "una" + "nim" (real: "un" + "anim").

    Each candidate requires evidence from >= ``min_component_freq``
    DISTINCT left-side anchors.

    Parameters
    ----------
    combining_forms :
        Current set of combining forms.
    min_component_freq :
        Minimum number of compound forms (with distinct left anchors)
        a component must appear in to be promoted.
    dictionary :
        Optional set of dictionary words.  When provided, truncated
        components are restored to full dictionary spellings.

    Returns
    -------
    Expanded set of combining forms including newly discovered components
    and restored full dictionary forms.
    """
    expanded = set(combining_forms)
    sorted_forms = sorted(combining_forms, key=len)
    dict_set = dictionary if dictionary else set()
    max_rounds = 3

    for _round in range(max_rounds):
        component_freq: Dict[str, int] = defaultdict(int)
        component_anchors: Dict[str, Set[str]] = defaultdict(set)

        by_len: Dict[int, Set[str]] = defaultdict(set)
        for f in sorted_forms:
            by_len[len(f)].add(f)

        for form in sorted_forms:  # Process short forms first.
            flen = len(form)
            if flen < 6: continue
            for i in range(3, flen - 2):
                left = form[:i]
                right = form[i:]
                if len(right) >= 3 and left in by_len.get(len(left), set()):
                    component_freq[right] += 1
                    component_anchors[right].add(left)

                elif len(left) >= 3 and right in by_len.get(len(right), set()):
                    component_freq[left] += 1
                    component_anchors[left].add(right)

        new_components = {
            comp for comp, freq in component_freq.items()
            if freq >= min_component_freq
            and len(component_anchors[comp]) >= min_component_freq
            and comp not in expanded
            and 3 <= len(comp) <= 12
        }

        if not new_components:
            break
        expanded |= new_components

    # ── Language-agnostic restoration ──────────────────────────────
    # Discover common word-final characters from the dictionary itself
    # (covers Latin -s/-e/-a/-us, Greek -ν/-ς, Cyrillic -ь/-а, etc.)
    # then try form + each final char.  No hardcoded endings.
    if dict_set:
        final_chars: Counter = Counter()
        for w in dict_set:
            if w:
                final_chars[w[-1]] += 1

        # Use the top final characters that collectively cover ≥ 80%
        total = sum(final_chars.values())
        restore_chars: List[str] = []
        cumulative = 0
        for ch, cnt in final_chars.most_common():
            restore_chars.append(ch)
            cumulative += cnt
            if cumulative >= total * 0.80:
                break

        restored = {}
        for form in expanded:
            if form in dict_set:
                continue
            for ch in restore_chars:
                full = form + ch
                if full in dict_set and full not in expanded:
                    restored[form] = full
                    break

        # Restore all fragmented forms
        # 1. Build the Trie
        trie = SuffixTrie()
        for frag, repl in restored.items():
            trie.insert(frag, repl)

        # 2. Process all restored words
        # Using a set comprehension for the final output
        expanded = {trie.find_and_replace(word) for word in expanded}

    return expanded


@log_time
def extract_latin__greek_roots(
        words: Set[str],
        suffixes: Set[str],
        suffix_freq: CounterType,
        prefix_freq: CounterType,
        lang_code: str,
        script: str,
        *,
        min_root_len: int = 3,
        min_suffix_len: int = 2,
        min_word_len: int = 7,
        max_root_len: int = 15,
        # Pattern A
        pattern_a_min_cv_ratio: float = 0.40,
        pattern_a_min_cv_hits: int = 3,
        pattern_a_min_diversity: int = 3,
        pattern_a_min_suffix_len: int = 5,
        # Pattern B
        pattern_b_min_rights: int = 5,
        pattern_b_min_standalone_ratio: float = 0.50,
        # Pattern C
        pattern_c_min_hits: int = 2,
        # Pattern D (Participial-T)
        pattern_d_min_continuations: int = 3,
        # Fragment filter (pre-Pattern-C)
        fragment_min_surviving_examples: int = 2,
) -> Tuple[Dict[str, Dict[str, Any]], Set[str]]:
    """
    Extract Greek/Latin combining-form roots from *words*.

    Uses purely statistical methods — no hardcoded root or morpheme lists.

    Returns
    -------
    (roots, source_words) :
        roots — dict mapping each root to its statistics.
        source_words — set of all dictionary words from which roots
        were extracted (i.e. the neoclassical compound vocabulary).

    Parameters
    ----------
    words :
        Full dictionary for the target language, (lowercased internally).
    suffixes :
        Cleaned suffix sets from the target language.
    suffix_freq :
        Counter object with lookup_dict word ending counts.
    lang_code :
        The ISO 639-3 code of the target language.
    script :
        The written language script of the target language. e.g. 'latin'.
    min_root_len :
        Minimum root length.
    min_word_len :
        Ignore words shorter than this.
    min_suffix_len :
        Min suffix length to count as valid (root + suffix) pattern e.g. 'geo-nom-ic'.
        * Root:   geo (from gaîa)
        * Root:   nom (from nomos)
        * Suffix: -ic (from -ikos)
    max_root_len :
        Maximum root prefix length.
    pattern_a_min_cv_ratio :
        Min fraction of occurrences with valid CV + continuation.
    pattern_a_min_cv_hits :
        Min absolute combining-vowel hits.
    pattern_a_min_diversity :
        Min distinct first-characters among continuations.
        Blocks inflection families (worsh → all start with 'p').
    pattern_a_min_suffix_len :
        Min suffix length to count as valid continuation.
    pattern_b_min_rights :
        Min unique right-side partners.
    pattern_b_min_standalone_ratio :
        Min fraction of right partners that are standalone words.
    pattern_c_min_hits :
        Min number of words a combining-form component must appear in
        (via Pattern C) to be promoted.
    script :
        'latin', 'cyrillic', or 'greek'.  Auto-detected if None.
    pattern_d_min_continuations :
        Min distinct inflectional continuations a Participial-T stem must
        exhibit to be promoted by Pattern D.
    fragment_min_surviving_examples :
        Min number of examples that must survive the overlapping-compound
        fragment filter before Pattern C for a Pattern A root to be kept.
    """
    # 1. Check memory cache
    if lang_code in LATIN_GREEK_ROOTS_CACHE:
        return LATIN_GREEK_ROOTS_CACHE[lang_code]

    cv_set = get_combining_vowels(script)
    word_set: Set[str] = {w for w in words if len(w) >= min_root_len * 2}
    strong_cf: Set[str] = set()  # forms that combine with at least 2 or more other productive CF. e.g. 'gram'
    weak_cf: Set[str] = set()  # rare forms that only appear with a few other forms like 'sklero' or 'pelite'
    # forms need to be found attach to at least 1 form from strong_cf and 1 form from weak_cf to be
    # added as a weak_cf, and also have 1 > frequency < 10 like 'pelite'

    # 2. Extract productive & 'Amphicombining Forms' or 'Double-Ended' combining forms
    candidates: Set[str] = {k for k, v in prefix_freq.items() if len(k) >= min_root_len and v > 1}
    candidates.update([k for k, v in suffix_freq.items() if len(k) >= min_root_len and v > 1])
    cf_candidates: List[str] = []  # Extract initial candidates
    for c in candidates:
        p_freq = prefix_freq.get(c, 0)
        s_freq = suffix_freq.get(c, 0)
        if (p_freq > 1 and s_freq >= 10) or (p_freq >= 10 and s_freq > 1):
            cf_candidates.append(c)

    # Add initial strong combining forms (cand + cv + cand)
    p1_evidence: Dict[str, set] = defaultdict(set)
    p1_combining_forms = defaultdict(set)
    candidates_by_len = sorted(get_affix_by_length(cf_candidates).items(), key=lambda x: x[0])
    for w in word_set:
        if len(w) >= min_root_len * 2:
            for length, cand_set in candidates_by_len:
                if length < min_root_len:
                    continue

                if length >= len(w) - min_root_len:
                    break

                if w[length] in cv_set:
                    if w[length+1] in cv_set:  # Check if next char is also a cv
                        if len(w) - length + 1 < min_root_len:
                            break  # most likely a suffix ending like "ing" instead of a valid neo compound
                        else:
                            # Split double vowel equally between each form. e.g. 'effusi-ometer'
                            part_a = w[:length + 1]
                            p1_evidence[part_a].add(w)
                            p1_combining_forms[part_a].add(w[length + 1:])
                    else:
                        # Add both forms. e.g. 'term' & 'termo'.
                        part_a = w[:length]
                        part_b = w[:length + 1]
                        p1_evidence[part_a].add(w)
                        p1_combining_forms[part_a].add(w[length:])
                        p1_combining_forms[part_b].add(w[length + 1:])

    pattern_1_roots: Dict[str, Dict[str, Any]] = {}
    evidence_stats: Dict[str, Any] = {}
    total_evidence: Dict[str, int] = {}
    root_entropy: Dict[str, float] = {}
    combined_entropy: Dict[str, float] = defaultdict(float)
    shared_forms: Dict[str, Set[str]] = defaultdict(set)
    for root, evidence in p1_combining_forms.items():
        if len(evidence) > 1:
            total_evidence[root] = len(evidence)
            # Need to use combined entropy since NF like 'hemo' also uses non cv as 'hema', 'hemi', etc.
            # in neoclassical compounds like 'hemachrome', 'hemerobius', 'hemocytoblast', etc.
            # total_entropy will need to account for the internal morphology of the Greek root itself.
            parent = root[:-1]
            parent_freq = prefix_freq.get(parent, 1)
            entropy = prefix_freq.get(root, 1) / parent_freq
            combined_entropy[parent] += entropy
            root_entropy[root] = entropy
            shared_forms[parent].add(root[-1])

    for root, evidence in p1_combining_forms.items():
        candidate_total = []
        for x in evidence:
            total = total_evidence.get(x, 0)
            if total > 0:
                candidate_total.append((x, total))
        evidence_stats[root] = sorted(candidate_total, key=lambda x: -x[1])

    cf_forms_cand = [(x, e) for x, e in evidence_stats.items() if e]
    cf_forms_cand.sort(key=lambda x: -len(x[1]))
    combining_forms = [x for x, e in cf_forms_cand]
    valid_cf_candidates: Dict[str, Set[str]] = {root: set() for root, e in cf_forms_cand}
    low_entropy_fragments = {}

    for root in list(valid_cf_candidates.keys()):
        total_entropy = combined_entropy.get(root[:-1], 0)
        if total_entropy < 0.5:
            low_entropy_fragments[root] = total_entropy

    for root, evidence in cf_forms_cand:
        for r, t in evidence:
            if root == r:
                continue
            if r in valid_cf_candidates:  # Valid combining forms can validate each other
                valid_cf_candidates[r].add(root)
                valid_cf_candidates[root].add(r)

    productive_cf = [(x, c) for x, c in valid_cf_candidates.items() if len(c) > 1 and x not in low_entropy_fragments]
    productive_cf.sort(key=lambda x: -len(x[1]))
    strong_cf = {x for x, e in productive_cf}

    print(f"Extracted {len(strong_cf)} strong cf forms {len(weak_cf)} cf forms. combining_forms={combining_forms}")
    # prod_suffixes = extract_productive_affixes(
    #     suffixes, suffix_freq, is_suffix=True, dict_words=words, min_len=min_suffix_len, top_n=250
    # )


    productive_suffixes = frozenset(s for s in suffixes if len(s) >= pattern_a_min_suffix_len)
    sorted_prod_suffixes = sorted(productive_suffixes, key=len, reverse=True)
    suffix_set_lower: Set[str] = {s.lower() for s in suffixes}

    # TODO - Extract each pattern into a separate testable function. e.g. 'Pattern A: root + CV + suffix'

    # ── Pattern A: root + CV + suffix ────────────────────────────
    # The canonical Greek combining vowel is -o- (therm-o-meter) and the
    # secondary is -i- (card-i-ology), but many neoclassical roots surface
    # with -a-, -e-, or -u- at the morpheme boundary:
    #   chrom-a-tocracy, sperm-a-theca, pneum-a-togram    (-a-)
    #   synth-e-sis, synth-e-tize                         (-e-)
    #   vasc-u-lar, vasc-u-lature                         (-u-)
    # Restricting to {o, i} caused false negatives for neur, gastr, chrom,
    # glyc, immun, kinet, ophthalm, orth, path, rhin, sperm, tox, vertebr.
    # We use the full vowel set of the script; the diversity and ratio
    # thresholds (min_diversity=3, min_ratio=0.40) keep the FP rate low.
    pa_cv_set = get_all_vowels(script)

    pa_freq: Dict[str, int] = defaultdict(int)
    pa_cv_hits: Dict[str, int] = defaultdict(int)
    pa_cont_sigs: Dict[str, set] = defaultdict(set)
    pa_examples: Dict[str, set] = defaultdict(set)
    pa_source_words: Dict[str, set] = defaultdict(set)

    for w in list(word_set):
        wlen = len(w)
        upper = min(wlen - 4, max_root_len)
        for p in range(min_root_len, upper + 1):
            ch = w[p]
            if ch not in pa_cv_set:
                continue
            root = w[:p]
            after_cv = w[p + 1:]
            if len(after_cv) < 4:
                continue

            pa_freq[root] += 1

            matched = False
            for suf in sorted_prod_suffixes:
                if after_cv.startswith(suf):
                    matched = True
                    break
            if not matched and len(after_cv) >= 4 and after_cv in word_set:
                matched = True

            if matched:
                pa_cv_hits[root] += 1
                pa_cont_sigs[root].add(after_cv[:5])
                pa_source_words[root].add(w)
                if len(pa_examples[root]) < 6:
                    pa_examples[root].add(w)

    pattern_a_roots: Dict[str, Dict[str, Any]] = {}
    for root in pa_freq:
        freq = pa_freq[root]
        cv_hits = pa_cv_hits.get(root, 0)
        if freq < pattern_a_min_cv_hits or cv_hits < pattern_a_min_cv_hits:
            continue
        ratio = cv_hits / freq
        if ratio < pattern_a_min_cv_ratio:
            continue
        sigs = pa_cont_sigs.get(root, set())
        first_chars = {s[0] for s in sigs if s}
        if len(first_chars) < pattern_a_min_diversity:
            continue

        # Tighter guard for short roots that are also common dictionary words.
        # Expanding cv_set to all vowels raises the risk that inflection families
        # of ordinary English words (ring→ring[i]ness, sing→sing[i]ng) pass the
        # base thresholds.  For a root of length ≤ 5 that IS in the dictionary,
        # we require ≥ 8 of its CV hits to come from words of ≥ 10 characters
        # (i.e., genuine long compounds, not short derivations like -ness/-ful).
        # Greek/Latin combining forms appear heavily in long technical words;
        # common English words do not (ring: 2 long-word hits, acid: 19, derm: 53).
        if len(root) <= 5 and root in word_set:
            long_hits = sum(1 for w in pa_source_words.get(root, set()) if len(w) >= 10)
            if long_hits < 8:
                continue

        pattern_a_roots[root] = {
            "frequency": freq,
            "combining_vowel_count": cv_hits,
            "combining_vowel_percentage": ratio,
            "continuation_diversity": len(first_chars),
            "pattern": "A",
            "is_root": True,
            "examples": sorted(pa_examples.get(root, set()))[:25],
        }

    # ── Pattern B: combining-form + morpheme ─────────────────────
    left_to_rights: Dict[str, set] = defaultdict(set)
    pb_source_words: Dict[str, set] = defaultdict(set)

    for w in list(word_set):
        wlen = len(w)
        upper = min(wlen - 3, max_root_len)
        for p in range(min_root_len, upper + 1):
            if w[p - 1] in cv_set:
                left = w[:p]
                right = w[p:]
                if len(right) >= 3:
                    left_to_rights[left].add(right)
                    pb_source_words[left].add(w)

    pattern_b_roots: Dict[str, Dict[str, Any]] = {}
    for left, rights in left_to_rights.items():
        if len(left) < min_root_len:
            continue
        n_rights = len(rights)
        if n_rights < pattern_b_min_rights:
            continue
        standalone_count = sum(1 for r in rights if r in word_set)
        ratio = standalone_count / n_rights
        if ratio < pattern_b_min_standalone_ratio:
            continue

        pattern_b_roots[left] = {
            "frequency": n_rights,
            "combining_vowel_count": standalone_count,
            "combining_vowel_percentage": ratio,
            "continuation_diversity": len({r[0] for r in rights if r}),
            "pattern": "B",
            "is_root": True,
            "examples": sorted(rights)[:25],
        }

    # ── Pattern B-strip: bare roots from combining forms ─────────
    # Many canonical Greek/Latin roots appear in the dictionary ONLY with
    # a terminal combining vowel: neuro-, gastro-, patho-, chromo-.
    # Pattern B captures the -o/-i form; stripping that final vowel recovers
    # the morphologically pure root (neur, gastr, path, chrom).
    #
    # Guards (all must hold):
    #   1. B root must end in a combining vowel (cv_set = {o, i}).
    #   2. Stripped form must be ≥ min_root_len characters.
    #   3. Stripped form must NOT itself be a productive suffix.
    #   4. B root must have standalone_ratio ≥ 0.60 and frequency ≥ 15.
    #   5. If the stripped form IS a dictionary word, require B-root
    #      frequency ≥ 50 (prevents 'home' from 'homeo' etc.).
    #   6. The B root's source words must show genuine neoclassical diversity:
    #      ≥ 10 distinct 3-char continuations *after* the B root across all
    #      source words.  Inflection families (e.g. recoi→ recoiled/recoiling/
    #      recoils — all continuations of the same stem 'recoil') collapse to
    #      far fewer distinct post-root trigrams than true combining forms
    #      (neuro→ neuroanatomy/neurobiology/neurocardiac → 124 distinct).
    _B_STRIP_MIN_POST_FIRST_CHARS = 5   # minimum distinct post-root first chars
    pattern_b_stripped_roots: Dict[str, Dict[str, Any]] = {}
    pb_stripped_source_words: Dict[str, Set[str]] = defaultdict(set)

    for left, stats in pattern_b_roots.items():
        if left[-1] not in cv_set:
            continue
        stripped = left[:-1]
        if len(stripped) < min_root_len:
            continue
        if stripped in suffix_set_lower:
            continue
        freq = stats["frequency"]
        ratio = stats["combining_vowel_percentage"]
        if freq < 15 or ratio < 0.60:
            continue
        if stripped in word_set and freq < 50:
            continue
        # Guard 6: require genuine neoclassical diversity in source words.
        # Measure the number of distinct characters that immediately follow
        # the B root in its source words.  An inflection family (e.g. recoi→
        # recoiled/recoiling/recoilless/recoinage — all continuations of the
        # single stem 'recoil') shows only 2 distinct post-root first chars
        # (l, n), while a true combining form (neuro→ neuroanatomy/neurobiology/
        # neurocardiac/…) shows 10–20.  Threshold 5 cleanly separates them.
        src_words = pb_source_words.get(left, set())
        post_first_chars: Set[str] = set()
        left_len = len(left)
        for sw in src_words:
            if len(sw) > left_len:
                post_first_chars.add(sw[left_len])
        if len(post_first_chars) < 5:
            continue
        # Inherit source words from the B root
        pb_stripped_source_words[stripped].update(src_words)
        pattern_b_stripped_roots[stripped] = {
            "frequency": freq,
            "combining_vowel_count": stats["combining_vowel_count"],
            "combining_vowel_percentage": ratio,
            "continuation_diversity": stats["continuation_diversity"],
            "pattern": "B",
            "is_root": True,
            "examples": stats["examples"],
        }

    # ── Pattern D: Participial-T (language-agnostic) ──────────────
    # Latin first/second/fourth conjugation perfect passive participles end
    # in [vowel]+[t/d].  This "DNA" survives in Romance and borrowed
    # scholarly vocabulary across all ISO 639-3 languages.
    # Rule: a stem that ends in [vowel]+[t] and has ≥ pattern_d_min_continuations
    # distinct inflectional endings (e.g. -ion, -e, -ing, -ed, -or, -ure,
    # -able, -ory, -al, -ive, -eth) is a genuine Latin participial root.
    _PARTICIPIAL_SUFFIXES: Tuple[str, ...] = (
        "ion", "ions", "ive", "ives", "or", "ors", "ory",
        "ed", "ing", "e", "es", "ely", "eness",
        "able", "ables", "ure", "ures", "al", "als", "eth",
    )
    _ALL_VOWELS_LATIN_STR = "aeiou"

    pd_stem_conts: Dict[str, Set[str]] = defaultdict(set)
    pd_source_words: Dict[str, Set[str]] = defaultdict(set)

    for w in list(word_set):
        wlen = len(w)
        upper = min(wlen - 2, max_root_len)  # cap: stem w[:p+1] has len p+1 ≤ max_root_len
        for p in range(4, upper + 1):
            if w[p] == 't' and w[p - 1] in _ALL_VOWELS_LATIN_STR:
                stem = w[:p + 1]          # includes the terminal 't'
                rest = w[p + 1:]
                for suf in _PARTICIPIAL_SUFFIXES:
                    if rest.startswith(suf):
                        pd_stem_conts[stem].add(suf)
                        pd_source_words[stem].add(w)
                        break

    pattern_d_roots: Dict[str, Dict[str, Any]] = {}
    for stem, conts in pd_stem_conts.items():
        if len(conts) < pattern_d_min_continuations:
            continue
        pattern_d_roots[stem] = {
            "frequency": len(pd_source_words[stem]),
            "combining_vowel_count": len(conts),
            "combining_vowel_percentage": 1.0,
            "continuation_diversity": len(conts),
            "pattern": "D",
            "is_root": True,
            "examples": sorted(pd_source_words[stem])[:25],
        }

    # ── Pattern E: double-consonant cluster fingerprint ──────────
    # Latin/Greek words are uniquely identified by specific consonant
    # clusters that are nearly absent from native Germanic/Slavic roots:
    #   -ct-  (factum, dictum),  -pt-  (scriptum, adoptum)
    #   -gn-  (signum, magnum),  -x-   (fixus, nexus)
    # A word containing ≥ 2 such markers, or a stem bearing one marker
    # that appears across ≥ effective_min_hits compound words, is a strong
    # candidate for a Latin/Greek root.  We collect source words here;
    # the stems are emitted as Pattern-E roots only if they are new
    # (not already captured by A/B/D).
    _CLUSTER_PATTERNS: Tuple[str, ...] = ("ct", "pt", "gn", "nct", "mpt", "rct")

    pe_source_words: Set[str] = set()
    for w in list(word_set):
        inner = w[2:-2]  # avoid word-boundary clusters like "backed"
        cluster_hits = sum(1 for cp in _CLUSTER_PATTERNS if cp in inner)
        if cluster_hits >= 1:
            pe_source_words.add(w)

    # ── Pattern F: abstract-suffix map (cross-linguistic) ────────
    # The morphological slot for nouns of action, state, means, and fitness
    # is stable across Romance/Germanic borrowings from Latin.  Words ending
    # in these high-density suffixes that ALSO bear a Latin/Greek root
    # (by Patterns A–E) are promoted as source words for Pattern C.
    _ABSTRACT_SUFFIXES: Tuple[str, ...] = (
        "tion", "sion", "cion",                        # -tionem (Noun of Action)
        "ity", "idad", "ità", "ité",                   # -itatem (State of Being)
        "ment", "miento", "mento",                     # -mentum (Means/Instrument)
        "able", "ible", "evole", "ável",               # -abilem (Fitness)
    )

    pf_source_words: Set[str] = set()
    for w in list(word_set):
        for suf in _ABSTRACT_SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= min_root_len:
                pf_source_words.add(w)
                break

    # ── Pattern G: scholarly-prefix check (ISV markers) ──────────
    # Latin/Greek prefixes survived largely unchanged into the International
    # Scientific Vocabulary and appear in all ISO 639-3 borrowing languages.
    # Words bearing these prefixes are strong evidence of Latin/Greek roots.
    # Assimilation variants (ad-→ac-,af-,ag-,ap-,as-,at-) are included.
    _SCHOLARLY_PREFIXES: Tuple[str, ...] = (
        "trans", "inter", "sub",
        "pre", "pro",
        # ad- assimilation family (mirrors filter_affixes logic)
        "ac", "af", "ag", "al", "an", "ap", "ar", "as", "at",
    )

    pg_source_words: Set[str] = set()
    for w in list(word_set):
        for pfx in _SCHOLARLY_PREFIXES:
            if w.startswith(pfx) and len(w) - len(pfx) >= min_root_len:
                pg_source_words.add(w)
                break

    # ── Pre-Pattern-C: overlapping-compound fragment filter ───────
    # Some Pattern A candidates are not genuine roots but boundary
    # fragments of prefix+word compounds.  Example: 'afterc' arises
    # from aftercomers / aftercoming / aftercooler / aftercourse.
    #
    # Detection algorithm (applied per candidate root R with examples E):
    #   For each example word W in E:
    #     Try split positions ±1 char relative to the root boundary
    #     (i.e. split at len(R)-1 and len(R)+1).
    #     If the right part at that split is:
    #       (a) a dictionary word  AND
    #       (b) NOT a productive suffix
    #     then W is evidence of an overlapping compound, not of R as a
    #     genuine root.  Remove W from E.
    #   After pruning, if |E| < fragment_min_surviving_examples,
    #   discard R entirely from pattern_a_roots.

    def _is_fragment_example(
        root: str,
        example: str,
        word_set_: Set[str],
        suf_set: Set[str],
    ) -> bool:
        """Return True if *example* is better explained by a compound split
        that shifts the root boundary by ±1 character."""
        rlen = len(root)
        for delta in (-1, +1):
            split = rlen + delta
            if split < 2 or split >= len(example) - 1:
                continue
            right = example[split:]
            if len(right) < 3:
                continue
            if right in word_set_ and right not in suf_set:
                return True
        return False

    filtered_pattern_a: Dict[str, Dict[str, Any]] = {}

    for root, stats in pattern_a_roots.items():
        examples = stats["examples"]
        surviving = [
            ex for ex in examples
            if not _is_fragment_example(root, ex, word_set, suffix_set_lower)
        ]
        if len(surviving) >= fragment_min_surviving_examples:
            stats["examples"] = surviving
            filtered_pattern_a[root] = stats

    pattern_a_roots = filtered_pattern_a

    # ── Pattern C: combining-form anchored discovery ────────────
    # Use the roots discovered by Patterns A, B, and D as anchors.
    # 1. Decompose compound roots into individual components
    #    (e.g. "adenovirus" → "adeno" + "virus").
    # 2. Scan ALL words for those containing 2+ known forms.
    #    Single-form matches are NOT counted — they produce too many
    #    false positives (e.g. "freakdom" matching "dom" alone).
    # 3. Promote components that appear in enough compounds.
    # Patterns E, F, G contribute source words that enrich the scan pool.
    pattern_c_roots: Dict[str, Dict[str, Any]] = {}
    pc_source_words: Dict[str, set] = defaultdict(set)

    # Combine A+B+B-stripped+D roots as anchors for Pattern C
    all_cf_set = (
        set(pattern_a_roots.keys())
        | set(pattern_b_roots.keys())
        | set(pattern_b_stripped_roots.keys())
        | set(pattern_d_roots.keys())
    )
    # Scale min_hits with dictionary size to avoid false positives
    # on large dictionaries while still discovering on small ones.
    effective_min_hits = max(
        pattern_c_min_hits,
        int(math.log10(max(len(word_set), 100)))  # 100→2, 10K→4, 500K→5
    )

    # Decompose compound forms to discover hidden components
    # and restore truncated forms (e.g. "viru" → "virus")
    # TODO - update the code to only strip affixes if the resulting stem
    #  is a valid dictionary word or a stem the can be restored using orth rules,
    #  thus avoiding the need to restore fragment that were incorrectly stripped
    all_cf_set = decompose_compound_forms(
        all_cf_set, min_component_freq=effective_min_hits,
        dictionary=word_set,
    )

    # Build a length-bucketed lookup for efficient greedy scanning
    cf_by_length: Dict[int, Set[str]] = defaultdict(set)
    for cf in all_cf_set:
        cf_by_length[len(cf)].add(cf)
    cf_lengths_desc = sorted(cf_by_length.keys(), reverse=True)

    # Scan ALL words for neoclassical compounds — REQUIRE 2+ combining forms.
    cf_hit_count: Dict[str, int] = defaultdict(int)
    all_pattern_c_source: Set[str] = set()

    for w in word_set:
        wlen = len(w)
        if wlen < 4:
            continue

        # Greedy left-to-right scan
        pos = 0
        found_forms: List[str] = []
        while pos < wlen:
            matched = False
            for fl in cf_lengths_desc:
                end = pos + fl
                if end <= wlen:
                    chunk = w[pos:end]
                    if chunk in cf_by_length[fl]:
                        found_forms.append(chunk)
                        pos = end
                        matched = True
                        break
            if not matched:
                pos += 1

        # ONLY 2+ combining forms qualifies as neoclassical.
        if len(found_forms) >= 2:
            all_pattern_c_source.add(w)
            for form in found_forms:
                cf_hit_count[form] += 1
                pc_source_words[form].add(w)

    # Promote forms that appear in enough compounds and are new
    existing_roots = (
        set(pattern_a_roots.keys())
        | set(pattern_b_roots.keys())
        | set(pattern_b_stripped_roots.keys())
        | set(pattern_d_roots.keys())
    )
    for form, hits in cf_hit_count.items():
        if hits < effective_min_hits:
            continue
        if form in existing_roots:
            continue

        diversity = set()
        for w in pc_source_words[form]:
            idx = w.find(form)
            nxt = idx + len(form)
            if nxt < len(w):
                diversity.add(w[nxt])
        pattern_c_roots[form] = {
            "frequency": hits,
            "combining_vowel_count": hits,
            "combining_vowel_percentage": 1.0,
            "continuation_diversity": len(diversity),
            "pattern": "C",
            "is_root": True,
            "examples": sorted(pc_source_words[form])[:5],
        }

    # ── Merge ────────────────────────────────────────────────────
    all_roots: Dict[str, Dict[str, Any]] = {}
    all_source_words: Set[str] = set()

    for root, stats in pattern_a_roots.items():
        all_roots[root] = stats
        all_source_words.update(pa_source_words.get(root, set()))

    for root, stats in pattern_b_roots.items():
        if root in all_roots:
            existing = all_roots[root]
            if stats["frequency"] > existing["frequency"]:
                stats["pattern"] = "A+B"
                stats["pattern_a_cv_hits"] = existing["combining_vowel_count"]
                all_roots[root] = stats
            else:
                existing["pattern"] = "A+B"
                existing["pattern_b_standalone"] = stats["combining_vowel_count"]
        else:
            all_roots[root] = stats
        all_source_words.update(pb_source_words.get(root, set()))

    # Merge Pattern B-stripped — bare roots derived from high-confidence B forms
    # (e.g. neuro→neur, gastro→gastr).  Only add if not already captured by A.
    for root, stats in pattern_b_stripped_roots.items():
        if root not in all_roots:
            all_roots[root] = stats
        all_source_words.update(pb_stripped_source_words.get(root, set()))

    # Merge Pattern D (Participial-T) — only add roots not already found by A/B
    for root, stats in pattern_d_roots.items():
        if root not in all_roots:
            all_roots[root] = stats
        all_source_words.update(pd_source_words.get(root, set()))

    # Merge Pattern C — only add roots not already discovered by A, B, or D
    for root, stats in pattern_c_roots.items():
        if root not in all_roots:
            all_roots[root] = stats
        all_source_words.update(pc_source_words.get(root, set()))

    # Patterns E, F, G contribute source words (not new roots) to the pool
    all_source_words.update(pe_source_words)
    all_source_words.update(pf_source_words)
    all_source_words.update(pg_source_words)
    # Keep only source words that are actual dictionary entries
    all_source_words = {w for w in all_source_words if w in word_set}

    return all_roots, all_source_words

def filter_non_maximal_forms(
        candidates: Set[str],
        word_set: Set[str],
        prefix_freq: dict,
        suffix_freq: dict,
        total_freq: dict,
        threshold: float = 0.50,
        vowels: str = "aeiouy",
) -> Set[str]:
    """
    Filter word-boundary fragments from ``extract_neoclassical_forms`` results.

    ``extract_neoclassical_forms`` uses bilateral entropy to detect neoclassical
    combining forms, but also picks up **boundary fragments** — substrings
    formed when a productive English affix abuts a root's adjacent consonant:

    Core signal — word-final dominance ratio
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For a suffix fragment ``form``, its suf_parent ``form[1:]`` (e.g. ``king``
    for ``cking``) appears at word-end in many words — but ``form`` itself
    covers only a **small fraction** of those word-final occurrences.  The
    ratio::

        wf_ratio = suffix_freq[form] / suffix_freq[suf_parent]

    where ``suffix_freq[x]`` is the count of dictionary words *ending* with
    ``x``, cleanly separates fragments from valid neoclassical forms:

    ============  =====================  =========  ==========
    form          suf_parent             wf_ratio   verdict
    ============  =====================  =========  ==========
    cking         king (wf=1194)          0.235      fragment
    rship         ship (wf=1450)          0.335      fragment
    ssion         sion (wf=815)           0.310      fragment
    graph         raph (wf=484)           0.988      valid
    logy          ogy  (wf=1116)          0.987      valid
    meter         eter (wf=984)           0.866      valid
    gram          ram  (wf=351)           0.712      valid
    ============  =====================  =========  ==========

    A valid neoclassical form dominates its suf_parent's word-final slot
    (wf_ratio >= 0.50).  A fragment does not (wf_ratio < 0.50).

    Secondary signal — prefix-boundary (Signal C)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For prefix-boundary fragments (``counterc``, ``afterb``): the form ends
    in 2+ consecutive consonants AND is a tiny fraction of its
    prefix-parent ``form[:-1]`` in ``prefix_freq``.

    Guards
    ~~~~~~
    * Candidate is a standalone dictionary word -> always kept.
    * Candidate length < 4 -> skipped (too short to diagnose safely).

    Parameters
    ----------
    candidates : set of str
        Candidates from ``extract_neoclassical_forms``.  Mutated in place.
    word_set : set of str
        Full dictionary.  Used to protect standalone-word candidates and to
        compute word-final counts when ``suffix_freq`` is absent.
    prefix_freq : dict
        ``{form: count}`` where count = number of dictionary words *starting*
        with ``form``.  Used for Signal C (prefix-boundary fragments).
    suffix_freq : dict
        ``{form: count}`` where count = number of dictionary words *ending*
        with ``form``.  From ``_extract_affix_frequencies`` or pre-built
        JSON files.  This is the primary input for the wf_ratio signal.
    total_freq : dict
        Raw ``t_map`` from ``extract_neoclassical_forms`` — total substring
        occurrence counts.  Used as denominator with ``suffix_freq``
    threshold : float
        ``wf_ratio`` ceiling for the fragment signal.  Default 0.50.
    vowels : str
        Characters treated as vowels.  Default ``"aeiouy"``.

    Returns
    -------
    set of str
        The filtered set (same object as ``candidates``).
    """
    if not candidates:
        return candidates

    vowels_set: FrozenSet[str] = frozenset(vowels)

    # ── Minimum word-final count for the suf_parent to be a productive suffix
    # Parents with very few word-final occurrences are niche sequences, not
    # real English suffixes, so the wf_ratio signal is unreliable for them.
    _MIN_PARENT_WF    = 30     # suf_parent must end >= this many words
    _WF_RATIO_THRESH  = threshold        # fragment wf_ratio ceiling (default 0.50)
    _PRE_RATIO_CEIL   = 0.10   # Signal C: candidate is tiny slice of prefix-parent
    _MIN_PRE_FREQ     = 200    # prefix-parent must start >= this many words
    _MIN_CAND_LEN     = 4      # don't filter very short candidates

    # ── Main filter loop ─────────────────────────────────────────────────────
    for cand in list(candidates):
        if len(cand) < _MIN_CAND_LEN:
            continue

        # Guard: standalone dictionary word -> always keep
        if word_set and cand in word_set:
            continue

        suf_parent = cand[1:]    # strip leading char  (suffix-boundary parent)
        pre_parent = cand[:-1]   # strip trailing char (prefix-boundary parent)

        # ── Signal A: word-final dominance ratio ─────────────────────────────
        # How much of the suf_parent's word-final slot does this form own?
        # Fragments own very little; valid neoclassical forms dominate.
        sf_parent = suffix_freq.get(suf_parent, 0)
        if sf_parent >= _MIN_PARENT_WF:
            sf_cand  = suffix_freq.get(cand, 0)
            wf_ratio = sf_cand / sf_parent
            if wf_ratio < _WF_RATIO_THRESH:
                candidates.discard(cand)
                continue

        # ── Signal B: deep suffix chain ───────────────────────────────────────
        # When t_map[form] / t_map[suf_parent] >= 0.80 the suf_parent barely
        # exists outside this form — it is itself a junk substring (e.g. therg
        # captures 80 % of herg).
        f_cand = total_freq.get(cand, 0)
        f_suf  = total_freq.get(suf_parent, 0)
        if f_suf > 0 and f_cand / f_suf >= 0.80:
            candidates.discard(cand)
            continue

        # ── Signal C: prefix-boundary fragment ───────────────────────────────
        # Form ends in 2+ consecutive consonants AND covers only a tiny slice
        # of its prefix-parent's word-initial occurrences (counterc, afterb…).
        f_pre = prefix_freq.get(pre_parent, 0)
        if f_pre >= _MIN_PRE_FREQ:
            trailing_cons = 0
            for ch in reversed(cand):
                if ch not in vowels_set:
                    trailing_cons += 1
                else:
                    break
            if trailing_cons >= 2:
                f_cand_pre = prefix_freq.get(cand) or total_freq.get(cand, 0)
                if f_cand_pre / f_pre <= _PRE_RATIO_CEIL:
                    candidates.discard(cand)
                    continue

    return candidates

def _calculate_entropy(counts: list[int], total: int) -> float:
    # Small optimization: use a generator expression for memory efficiency
    return -sum((c / total) * math.log2(c / total) for c in counts)

@log_time
def extract_neoclassical_forms(
        word_set: set[str],
        script: str,
        min_len: int = 3,
        max_len: int = 10,
        min_freq: int = 12,
        desired_freq: int = 50
) -> Tuple[set, set, dict, dict]:
    """Extract candidate neoclassical combining forms from *word_set*.

    Returns
    -------
    (results, t_map) :
        results — dict mapping each passing substring to its entropy/frequency
        statistics (same format as before).
        t_map — total substring → word-count mapping for ALL substrings seen,
        regardless of whether they passed the entropy filter.  Pass this as
        ``total_freq`` to :func:`filter_non_maximal_forms` so that parent-child
        frequency ratios can be computed even when the parent is not in *results*.
    """
    # Check memory cache
    sorted_words = sorted(word_set)
    cache_key = get_hash_key(sorted_words)
    if cache_key in LATIN_GREEK_ROOTS_CACHE:
        return LATIN_GREEK_ROOTS_CACHE[cache_key]

    v_set = get_all_vowels(script)
    f_map = defaultdict(int)
    b_map = defaultdict(int)
    t_map = defaultdict(int)
    cv_map = defaultdict(int)

    # TODO - remove invalid combining forms like: "incombusti", "incommo", "incommod", "incommodat", "incommodi",
    #  "incommunicat", "incompati", "incompl", "incomplet", "incompli", "incomprehensi". While they are Latin setm
    #  "forms" of words, they do not meet the strict morphological definition of a "combining form" used to bridge
    #  two distinct word parts.

    # TODO - update the code to ensure longer combining forms like 'metallurg' is also extracted from words like:
    #  metallurgical/metallurgist/micrometallurgy/hydrometallurgical/etc.

    # TODO - Remove compound words to avoid creating fragments like: 'counterc' or 'sunb' from 'sunbathed'
    productive_prefix_roots: set[str] = set()
    productive_suffix_roots: set[str] = set()
    for word in sorted_words:
        w_len = len(word)
        break

    # 1. OPTIMIZED SINGLE PASS
    for word in sorted_words:
        w_len = len(word)
        if w_len < min_len:
            continue

        for n in range(min_len, max_len + 1):
            for i in range(w_len - n + 1):

                sub = word[i: i + n]
                l_char = word[i - 1] if i > 0 else '^'
                r_char = word[i + n] if i + n < w_len else '$'

                t_map[sub] += 1
                f_map[(sub, r_char)] += 1
                b_map[(sub, l_char)] += 1

                if r_char in v_set or l_char in v_set:
                    cv_map[sub] += 1

    # 2. VECTORIZED ENTROPY CALCULATION
    f_entropy_data = defaultdict(list)
    b_entropy_data = defaultdict(list)

    # Pre-group counts for entropy to avoid repeated lookups
    for (sub, _), count in f_map.items():
        if sub in t_map:  # Only process if not pruned
            f_entropy_data[sub].append(count)

    for (sub, _), count in b_map.items():
        if sub in t_map:
            b_entropy_data[sub].append(count)

    results = {}
    for sub, total in t_map.items():
        if total < min_freq:
            continue

        cv_ratio = cv_map[sub] / total
        if cv_ratio < 0.7 and total < desired_freq:
            continue

        h_forward = _calculate_entropy(f_entropy_data[sub], total)
        h_backward = _calculate_entropy(b_entropy_data[sub], total)
        t_entropy = h_forward + h_backward

        # Neoclassical Pivot Logic
        if t_entropy > 3.5 or (cv_ratio >= 0.8 and t_entropy > 0.1):
            results[sub] = {
                "f_entropy": round(h_forward, 2),
                "b_entropy": round(h_backward, 2),
                "t_entropy": round(t_entropy, 2),
                "cv_ratio": round(cv_ratio, 2),
                "frequency": total
            }

    expanded_forms = discover_anchor_based_forms(word_set, results, script=script)
    # results.update(new_forms)
    neo_forms = sorted(results.keys())
    neo_bound_roots = [x for x in neo_forms if x not in word_set]
    print(f"\nTotal {len(neo_forms)} combining form.")
    print(f"Extracted {len(neo_bound_roots)} bound roots combining forms.")
    save_json_file(STORAGE_DIR, "eng_combining_forms_raw.json", neo_forms)
    save_json_file(STORAGE_DIR, "eng_expanded_forms.json", sorted(expanded_forms))
    save_json_file(STORAGE_DIR, "eng_neo_bound_roots.json", sorted(neo_bound_roots))

    total_freq = dict(t_map)
    LATIN_GREEK_ROOTS_CACHE[cache_key] = (results, total_freq)
    # cache_file = f"{cache_key}_greek_latin_roots_cache.json"
    # print(f"Saving neo forms & t_map freq {len(t_map)} -> {len(total_freq)} in cache: {cache_file}...")
    # save_json_file(CACHE_DIR, cache_file, {"neoclassical_forms": results, "total_freq": total_freq})

    # TODO - use 'cache_key' to ensure prefix/suffix freq was extracted from the same sorted_words
    #  e.g. f'{cache_key}_prefix_freq.json'
    prefix_freq = load_json_file("eng_prefix_freq.json")
    suffix_freq = load_json_file("eng_suffix_freq.json")

    combining_forms = filter_non_maximal_forms(
        set(results.keys()), word_set, prefix_freq, suffix_freq, total_freq,
        vowels="".join(v_set),
    )

    return combining_forms, expanded_forms, results, total_freq


@log_time
def discover_anchor_based_forms(
        word_set: Set[str],
        neo_results: Dict[str, Any],
        script: str,
        *,
        min_anchor_freq: int = 15,
        min_partners: int = 3,
        expanded_forms_threshold: int = 8,
) -> Set[str]: # Dict[str, Any],
    """
    Discover prefix-only and suffix-only neoclassical combining forms missed
    by ``extract_neoclassical_forms`` due to insufficient bilateral entropy.

    ``extract_neoclassical_forms`` requires a substring to have high entropy
    on **both** sides.  Rare prefix-only forms like ``picro-``, ``nepho-``,
    ``hadro-`` appear almost exclusively at word-start, so their backward
    entropy is near-zero and they fail the bilateral threshold.

    Strategy — inverted index anchor scan
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1.  **Build inverted indexes** (one O(words × max_len) pass):
        - ``prefix_words[form]`` = set of words that *start* with ``form``
        - ``suffix_words[form]`` = set of words that *end*   with ``form``

        Only forms already confirmed in ``neo_results`` with frequency ≥
        *min_anchor_freq* are indexed — they become **anchors**.

    2.  **Suffix anchors → discover prefix-only forms**:
        For every word ending with a confirmed suffix anchor, the portion
        before the anchor is a candidate prefix-only combining form.
        A candidate is promoted when it pairs with ≥ *min_partners* distinct
        suffix anchors.

    3.  **Prefix anchors → discover suffix-only forms**:
        Symmetrically, for every word starting with a confirmed prefix anchor,
        the portion after the anchor is a candidate suffix-only combining form.

    4.  **Cross-directional**: prefix-only anchors (high prefix_words count,
        low suffix_words count) also drive suffix-only discovery.

    Combining-vowel stripping
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    At morpheme junctions a connecting vowel (``o`` or ``i`` for Latin/Greek)
    may be present.  Both the raw remainder and the vowel-stripped version are
    tested as candidates so that ``picrolite`` yields both ``picro`` (raw) and
    ``picr`` (stripped — less useful but harmless).

    Expanded forms
    ~~~~~~~~~~~~~~
    A discovered form that pairs with ≥ *expanded_forms_threshold* distinct
    productive anchors is placed in ``expanded_forms`` rather than
    ``new_forms``.  These are highly productive hub roots (``-logy``,
    ``-meter``) that should not themselves become anchors in further passes
    to avoid combinatorial noise.  Expanded forms are already well-represented
    in ``neo_results`` and are returned separately for bookkeeping only.

    Parameters
    ----------
    word_set :
        Full dictionary (lowercased).
    neo_results :
        Output of ``extract_neoclassical_forms`` — ``{form: stats_dict}``.
    script :
        Writing-system name for combining-vowel lookup.
    min_anchor_freq :
        Minimum ``frequency`` in *neo_results* for a form to serve as anchor.
    min_partners :
        Minimum number of distinct anchor partners required to promote a
        candidate.
    expanded_forms_threshold :
        Candidates with this many or more distinct anchor partners go to
        ``expanded_forms`` rather than ``new_forms``.

    Returns
    -------
    (new_forms, expanded_forms) :
        new_forms — ``{form: stats_dict}`` of newly discovered combining forms
        not already in *neo_results*, with ``"pattern": "anchor"``.
        expanded_forms — set of highly productive forms paired with many
        anchors; these should not be re-used as anchors.
    """
    if not neo_results or not word_set:
        return set()

    cv_set = get_combining_vowels(script)

    # ── Step 1: select anchors ────────────────────────────────────────────────
    # Only confirmed neo forms with high enough frequency become anchors.
    anchor_candidates = {
        f for f, s in neo_results.items()
        if s.get('frequency', 0) >= min_anchor_freq
        and MIN_FORM_LEN <= len(f) <= MAX_FORM_LEN
    }

    # ── Step 2: build inverted indexes in one O(words × max_len) pass ────────
    # prefix_words[anchor] = words starting with anchor
    # suffix_words[anchor] = words ending with anchor
    # Using pre-built freq dicts when available avoids re-scanning word_set.
    prefix_words: Dict[str, Set[str]] = defaultdict(set)
    suffix_words: Dict[str, Set[str]] = defaultdict(set)

    for word in word_set:
        wlen = len(word)
        for n in range(MIN_FORM_LEN, min(MAX_FORM_LEN + 1, wlen)):
            pref = word[:n]
            if pref in anchor_candidates:
                prefix_words[pref].add(word)
            suf = word[-n:]
            if suf in anchor_candidates:
                suffix_words[suf].add(word)

    # Anchors must actually appear on their respective side in >= min_anchor_freq words
    prefix_anchors: Set[str] = {f for f, ws in prefix_words.items() if len(ws) >= min_anchor_freq}
    suffix_anchors: Set[str] = {f for f, ws in suffix_words.items() if len(ws) >= min_anchor_freq}

    # Prefix-only anchors: appear frequently at word-start but NOT at word-end
    prefix_only_anchors = prefix_anchors - suffix_anchors

    # ── Step 3: suffix anchors → discover prefix-only candidates ─────────────
    # For each word ending with a suffix anchor, collect the left portion.
    cand_pre_partners: Dict[str, Set[str]] = defaultdict(set)  # cand -> anchors
    cand_pre_words:    Dict[str, Set[str]] = defaultdict(set)  # cand -> source words

    for anchor in suffix_anchors:
        alen = len(anchor)
        for word in suffix_words[anchor]:
            left = word[:-alen]
            llen = len(left)
            if llen < MIN_FORM_LEN or llen > MAX_FORM_LEN:
                continue
            # Try raw left and combining-vowel-stripped left
            variants = {left}
            if left[-1] in cv_set and llen > MIN_FORM_LEN:
                variants.add(left[:-1])
            for cand in variants:
                if MIN_FORM_LEN <= len(cand) <= MAX_FORM_LEN and cand not in neo_results:
                    cand_pre_partners[cand].add(anchor)
                    cand_pre_words[cand].add(word)

    # ── Step 4: prefix anchors → discover suffix-only candidates ─────────────
    cand_suf_partners: Dict[str, Set[str]] = defaultdict(set)
    cand_suf_words:    Dict[str, Set[str]] = defaultdict(set)

    for anchor, p_words_set in prefix_words.items():
        if anchor not in prefix_anchors:
            continue

        alen = len(anchor)
        for word in p_words_set:
            rlen  = len(word) - alen
            if rlen < MIN_FORM_LEN or rlen > MAX_FORM_LEN:
                continue
            right = word[alen:]
            variants = {right}
            if right[0] in cv_set and rlen > MIN_FORM_LEN:
                variants.add(right[1:])
            for cand in variants:
                if MIN_FORM_LEN <= len(cand) <= MAX_FORM_LEN and cand not in neo_results:
                    cand_suf_partners[cand].add(anchor)
                    cand_suf_words[cand].add(word)

    # ── Step 5: merge, count, classify ───────────────────────────────────────
    # Frequency and example words come directly from the word sets already
    # collected during the anchor scan — no second pass over word_set needed.
    new_forms:      Dict[str, Any] = {}
    expanded_forms: Set[str]       = set()

    all_cand_pre = {c for c, p in cand_pre_partners.items() if len(p) >= min_partners}
    all_cand_suf = {c for c, p in cand_suf_partners.items() if len(p) >= min_partners}
    all_cands    = all_cand_pre | all_cand_suf

    for cand in all_cands:
        pre_p = cand_pre_partners.get(cand, set())
        suf_p = cand_suf_partners.get(cand, set())
        total_partners = pre_p | suf_p
        n_partners     = len(total_partners)

        # Source words already collected: union of prefix-scan and suffix-scan hits
        source_words = (
            cand_pre_words.get(cand, set()) | cand_suf_words.get(cand, set())
        )
        freq = len(source_words)
        if freq == 0:
            continue

        if n_partners >= expanded_forms_threshold:
            expanded_forms.add(cand)
        # else:
        #     example_words = sorted(source_words)[:10]
        #     new_forms[cand] = {
        #         'frequency':       freq,
        #         'anchor_partners': sorted(total_partners),
        #         'n_partners':      n_partners,
        #         'examples':        example_words,
        #         'pattern':         'anchor',
        #         'is_root':         True,
        #     }

    print(
        f"Anchor discovery: {len(prefix_anchors)} prefix anchors, "
        f"{len(suffix_anchors)} suffix anchors, "
        f"{len(prefix_only_anchors)} prefix-only anchors -> "
        f"{len(expanded_forms)} expanded forms."
    )
    return expanded_forms


if check_cache_validity(extract_latin__greek_roots, exclude_globals={"LATIN_GREEK_ROOTS_CACHE"}):
    global LATIN_GREEK_ROOTS_CACHE
    # Check disk cache
    cache_files = find_files_by_pattern(CACHE_DIR, "_greek_latin_roots_cache.json")
    file_count = 0
    for cache_file in cache_files:
        cache_path = os.path.join(CACHE_DIR, cache_file)
        try:
            root_data = load_json_file(cache_path)
            all_roots = root_data['all_roots']
            all_source_words = set(root_data['all_source_words'])
            if all_roots and all_source_words:
                lang_code = cache_file.split("_")[0]
                LATIN_GREEK_ROOTS_CACHE[lang_code] = (all_roots, all_source_words)
                file_count += 1
        except (FileNotFoundError, KeyError):
            pass
    print(f"Loaded greek & latin roots from cache for {file_count} language(s).")
else:
    # Reset disk cache
    print("Resetting cache for greek & latin roots...")
    cache_files = find_files_by_pattern(CACHE_DIR, "_greek_latin_roots_cache.json")
    if cache_files:
        delete_files_from_list(CACHE_DIR, cache_files)


# TODO - Pending removal
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

# TODO - Pending removal
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


# TODO - Pending removal
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


if __name__ == "__main__":
    print(f"SCRIPT_REGISTRY Size: {len(SCRIPT_REGISTRY)}")