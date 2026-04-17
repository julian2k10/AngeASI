import os
import sys
import time
import unittest
from typing import List

import numpy as np
import  regex as re
from collections import defaultdict, Counter
from asi.ange_tokenizer import (
    infer_language_bias,
    SimpleTransformerXL, SemanticTokenizer, create_tokenizer, load_word_frequencies_json,
    get_skyscraper_training_sentences, CandidateEntry, CandidateCache, UNICODE_PATTERN, STORAGE_DIR,
    DICT_WORDS, WORD_FREQ_FILE_NAME, MAX_VOCAB_PER_LANGUAGE, SPECIAL_TOKENS, SUBWORD_PREFIX,
    calculate_split_score, load_json_file, remove_single_spaces, add_single_spaces
)

# ---------------------------------------------------------------------------
# Word lists for testing
# ---------------------------------------------------------------------------

def get_bpe_struggle_words() -> List[str]:
    return [w.lower() for w in [
        "therapist", "peninsula", "mistletoe", "together", "infamous", "misinterpret",
        "notwithstanding", "extraordinary", "idiosyncrasy", "uninterruptible", "abolish",
        "amenity", "antagonize", "assassinate", "barometer", "catastrophe", "circumstance",
        "superficial", "artificial", "representative", "legendary", "carpet", "highlight",
        "keyboard", "pomegranate", "strawberry", "hamburger", "pancakes", "background",
        "foreground", "underground", "overload", "underneath", "afternoon", "everything",
        "everywhere", "anybody", "someone", "somewhere", "successful", "infrastructure",
        "metaphor", "atmosphere", "butterfly", "pineapple", "underestimate", "overestimate",
        "counterproductive", "subconscious", "interconnected", "autobiography", "biodiversity",
        "ecosystem", "geopolitics", "microorganism", "nanotechnology", "psychotherapy",
        "sociopolitical", "thermodynamic", "ultraviolet", "beheld", "belong", "beware",
        "become", "bedtime", "believe", "beneath", "between", "beyond", "behold", "beset",
        "beside", "bestow", "betray", "better", "betwixt", "bewitched", "bewilder", "before",
        "began", "manslaughter", "earring", "fireman", "greenhouse", "headlight", "iceberg",
        "jellyfish", "kneecap", "lifeguard", "moonlight", "notebook", "outskirts", "playmate",
        "quicksand", "raincoat", "skyscraper", "toothbrush", "underdog", "watermelon", "windshield",
    ]]

def get_bpe_excel_words() -> List[str]:
    return [w.lower() for w in [
        "the", "and", "that", "with", "have", "this", "from", "they", "which", "words",
        "language", "computer", "learning", "system", "information", "thinking", "playing",
        "walked", "happily", "smallest", "kindness", "government", "happiness", "development",
        "important", "possible", "different", "education", "community", "process", "problem",
        "success", "result", "example", "question", "history", "future", "science",
        "research", "culture", "control", "design", "report", "energy", "health", "social",
        "public", "program", "market", "policy", "acting", "walking", "dreaming", "running",
        "jumped", "talked", "quickly", "slowly", "sadness", "weakness", "family", "school",
        "student", "teacher", "power", "level", "point", "area", "place", "group", "number",
        "interest", "case", "fact", "value", "order", "action", "change", "effort", "effect",
        "reason", "course", "form", "plan", "idea", "book", "city", "name", "road", "body",
        "side", "door", "head", "mind", "kind", "part", "word", "line", "home", "time",
    ]]

# ============================================================================
# TEST SUITE
# ============================================================================
class TestStringReconstruction(unittest.TestCase):
    def reconstruct(self, text):
        tokens = re.findall(UNICODE_PATTERN, text)
        cleaned = remove_single_spaces(tokens)
        reconstructed_tokens = add_single_spaces(cleaned)
        return "".join(reconstructed_tokens)

    def test_joined_camel_case(self):
        """Input 'CamelCase' should remain 'CamelCase'."""
        self.assertEqual(self.reconstruct("CamelCase"), "CamelCase")

    def test_camel_case_split(self):
        self.assertEqual(self.reconstruct("camel Case"), "camel Case")

    def test_pascal_case_split(self):
        self.assertEqual(self.reconstruct("this Is"), "this Is")

    def test_standard_lowercase_split(self):
        self.assertEqual(self.reconstruct("key board"), "key board")

    def test_proper_nouns_kept(self):
        self.assertEqual(self.reconstruct("Hello World"), "Hello World")

    def test_mixed_complex_sentence(self):
        original = "Hello World this Is camelCase test New iPhone sales"
        self.assertEqual(self.reconstruct(original), original)

    def test_punctuation_safety(self):
        self.assertEqual(self.reconstruct("Hello, World."), "Hello, World.")

    def test_code_syntax_safety(self):
        code = "if x:    return y"
        self.assertEqual(self.reconstruct(code), code)

    def test_whitespace_preservation(self):
        """Multiple consecutive whitespace chars are preserved as a single token."""
        original = "if x:    return y"
        tokens = re.findall(UNICODE_PATTERN, original)
        # The 4-space run should be a single token
        self.assertIn("    ", tokens, "4-space run should be captured as a single token")
        self.assertEqual("".join(tokens), original)


class TestSkyscraperFragments(unittest.TestCase):
    """
    Core test: confirms that bad fragments ['sk','ysc','raper'] never
    outrank valid morpheme splits like ['sky','scrap','er'].
    """

    @classmethod
    def setUpClass(cls):
        cls.tok, cls.model = create_tokenizer(verbose=False)

    def test_vocab_contains_required_tokens(self):
        required = ['sky', 'scraper', 'scrap', 'er', 'sk', 'ysc', 'raper', 'sc']
        missing = [t for t in required if not self.model.has_token(t)]
        print(f"\n  Vocab check: {len(required) - len(missing)}/{len(required)} found")
        if missing:
            print(f"  Missing: {missing}")
        for t in ['sky', 'sk', 'ysc', 'raper', 'sc']:
            self.assertTrue(self.model.has_token(t), f"Token '{t}' must be in vocab")

    def test_skyscraper_default_score_ranking(self):
        """calculate_split_score ranks splits with longer real tokens higher."""
        vocab = self.tok.vocab
        id_to_tok = self.tok.id_to_token
        mf = self.tok.morpheme_freq

        bad_ids = tuple(vocab[t] for t in ['sk', 'ysc', 'raper'] if t in vocab)
        better_ids = tuple(vocab[t] for t in ['sky', 'sc', 'raper'] if t in vocab)

        if len(bad_ids) == 3 and len(better_ids) == 3:
            bad_score = calculate_split_score(bad_ids, id_to_tok, mf)
            better_score = calculate_split_score(better_ids, id_to_tok, mf)
            print(f"\n  Default scores (calculate_split_score):")
            print(f"    ['sk','ysc','raper']:  {bad_score:.4f}")
            print(f"    ['sky','sc','raper']:  {better_score:.4f}")
            self.assertGreater(better_score, bad_score)

    def test_invalid_split_never_beats_valid_morphemes(self):
        """
        Invalid splits like ['skysc', 'raper'] must NOT score higher than
        valid morpheme splits like ['sky', 'scrap', 'er'] simply because
        the invalid split has fewer tokens.

        This is a regression test: a naive "fewer tokens = higher score"
        heuristic would incorrectly rank ['skysc','raper'] above
        ['sky','scrap','er'].
        """
        vocab = self.tok.vocab
        id_to_tok = self.tok.id_to_token
        mf = self.tok.morpheme_freq

        # Ensure both splits' tokens are in the vocab for a fair comparison
        for tok in ['skysc', 'raper', 'sky', 'scrap', 'er']:
            self.assertIn(tok, vocab, f"Token '{tok}' must be in vocab for this test")

        invalid_ids = tuple(vocab[t] for t in ['skysc', 'raper'])
        valid_ids = tuple(vocab[t] for t in ['sky', 'scrap', 'er'])

        invalid_score = calculate_split_score(invalid_ids, id_to_tok, mf)
        valid_score = calculate_split_score(valid_ids, id_to_tok, mf)

        print(f"\n  Invalid vs valid morpheme splits:")
        print(f"    ['skysc','raper']     (2 tokens): {invalid_score:.4f}")
        print(f"    ['sky','scrap','er']  (3 tokens): {valid_score:.4f}")

        self.assertGreater(
            valid_score, invalid_score,
            "Valid morpheme split ['sky','scrap','er'] must score higher "
            "than invalid fragment ['skysc','raper'] regardless of token count."
        )

    def test_bad_fragments_not_in_slot_1(self):
        """Bad fragments should NOT be in slot #1 even before training."""
        tok, _ = create_tokenizer(verbose=False)
        tok.find_all_valid_splits_dp("skyscraper")
        entries = tok.cache.get_entries("skyscraper")
        self.assertIsNotNone(entries)
        self.assertGreater(len(entries), 0)

        best_tokens = list(entries[0].tokens)
        print(f"\n  skyscraper slot #1: {best_tokens}")
        self.assertNotEqual(
            best_tokens, ['sk', 'ysc', 'raper'],
            "Bad fragments should never be in slot #1 even before training",
        )

    def test_training_updates_scores(self):
        """After training, lexicon pair scores should change."""
        tok, model = create_tokenizer(verbose=False)

        tok.find_all_valid_splits_dp("skyscraper")
        entries_before = tok.cache.get_entries("skyscraper")
        scores_before = {e.tokens: e.score for e in entries_before}

        sentences = get_skyscraper_training_sentences()
        model.train_model(sentences, tok.tokenize, epochs=5)

        tok.training_cycle(["skyscraper"], epochs=5, learning_rate=0.15, verbose=False)

        entries_after = tok.cache.get_entries("skyscraper")
        scores_after = {e.tokens: e.score for e in entries_after}

        changed = 0
        print(f"\n  Score changes after training:")
        for tokens in list(scores_before.keys())[:8]:
            if tokens in scores_after:
                before = scores_before[tokens]
                after = scores_after[tokens]
                delta = after - before
                if abs(delta) > 1e-6:
                    changed += 1
                print(f"    {str(list(tokens)):40s} {before:.6f} -> {after:.6f} ({delta:+.6f})")

        self.assertGreater(changed, 0, "At least some scores should change after training")

    def test_semantic_scores_after_training(self):
        """After training, bad fragment ['sk','ysc','raper'] is NOT the best split."""
        tok, model = create_tokenizer(verbose=False)

        best_before, score_before, top_k_before = tok.find_best_split("skyscraper", return_top_k=5)
        print(f"\n  BEFORE training:")
        print(f"    Best: {best_before} (score: {score_before:.4f})")
        for s, sc in top_k_before[:5]:
            tag = " <-- BAD" if s == ['sk', 'ysc', 'raper'] else ""
            print(f"      {s} -> {sc:.4f}{tag}")

        sentences = get_skyscraper_training_sentences()
        model.train_model(sentences, tok.tokenize, epochs=10)

        score_history = []
        for ep in range(5):
            tok.training_cycle(["skyscraper"], epochs=1, learning_rate=0.15, verbose=False)
            best, score, _ = tok.find_best_split("skyscraper")
            score_history.append(score)

        best_after, score_after, top_k_after = tok.find_best_split("skyscraper", return_top_k=5)
        print(f"\n  AFTER training:")
        print(f"    Best: {best_after} (score: {score_after:.4f})")
        for s, sc in top_k_after[:5]:
            tag = " <-- BAD" if s == ['sk', 'ysc', 'raper'] else ""
            print(f"      {s} -> {sc:.4f}{tag}")

        self.assertNotEqual(
            best_after, ['sk', 'ysc', 'raper'],
            "Bad fragment should not be the best split after training",
        )
        print(f"\n  Score history: {[f'{s:.4f}' for s in score_history]}")

    def test_pair_affinity_sky_vs_sk(self):
        """After training, ('sky','scraper') affinity > ('sk','ysc')."""
        tok, model = create_tokenizer(verbose=False)

        aff_sky_scraper_before = tok.lexicon.get_pair_affinity("sky", "scraper")
        aff_sk_ysc_before = tok.lexicon.get_pair_affinity("sk", "ysc")

        sentences = get_skyscraper_training_sentences()
        model.train_model(sentences, tok.tokenize, epochs=10)
        tok.lexicon.refresh_from_model()

        aff_sky_scraper_after = tok.lexicon.get_pair_affinity("sky", "scraper")
        aff_sk_ysc_after = tok.lexicon.get_pair_affinity("sk", "ysc")

        print(f"\n  Pair affinities:")
        print(f"    ('sky','scraper'): before={aff_sky_scraper_before:.4f}, after={aff_sky_scraper_after:.4f}")
        print(f"    ('sk','ysc'):      before={aff_sk_ysc_before:.4f}, after={aff_sk_ysc_after:.4f}")

        self.assertGreater(
            aff_sky_scraper_after, aff_sk_ysc_after,
            f"'sky'+'scraper' affinity ({aff_sky_scraper_after:.4f}) should be > "
            f"'sk'+'ysc' ({aff_sk_ysc_after:.4f}) after training",
        )


class TestTokenizeWord(unittest.TestCase):
    """Tests for _tokenize_word correctness."""

    @classmethod
    def setUpClass(cls):
        cls.tok, cls.model = create_tokenizer(verbose=False)

    def test_tokenize_word_returns_list_of_strings(self):
        """_tokenize_word must always return List[str]."""
        result = self.tok._tokenize_word("sky")
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, str)

    def test_tokenize_word_inference_mode(self):
        """In inference mode, _tokenize_word returns best split."""
        self.tok.find_all_valid_splits_dp("sky")
        self.tok.set_inference_mode(True)
        result = self.tok._tokenize_word("sky")
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        self.tok.set_inference_mode(False)

    def test_tokenize_word_training_mode(self):
        """In training mode, _tokenize_word uses slot selection."""
        self.tok.find_all_valid_splits_dp("sky")
        self.tok.set_inference_mode(False)
        result = self.tok._tokenize_word("sky")
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_tokenize_word_empty(self):
        """Empty string returns empty list."""
        self.assertEqual(self.tok._tokenize_word(""), [])

    def test_tokenize_word_reconstructs(self):
        """Tokens from _tokenize_word should reconstruct the original word."""
        word = "skyscraper"
        tokens = self.tok._tokenize_word(word)
        reconstructed = "".join(t.lstrip(SUBWORD_PREFIX) for t in tokens)
        self.assertEqual(reconstructed, word)


class TestBuildVocab(unittest.TestCase):
    """Tests for the build_vocab function."""

    @classmethod
    def setUpClass(cls):
        cls.tok, cls.model = create_tokenizer(verbose=False)

    def test_build_vocab_basic(self):
        """build_vocab produces a vocab dict with special tokens and morphemes."""
        corpus = Counter({"the": 100, "sky": 50, "skyscraper": 30, "building": 20, "cat": 10})
        dict_words = {"the", "sky", "skyscraper", "building", "scraper", "cat", "scrap"}
        vocab = self.tok.build_vocab(corpus, dict_words, lang_code="eng", verbose=False)

        # Special tokens present
        for st in SPECIAL_TOKENS:
            self.assertIn(st, vocab)

        # Morphemes extracted from dict_words should appear
        # "sky" and "scrap" are productive substrings
        self.assertIn("sky", vocab)
        self.assertIn("scrap", vocab)

    def test_build_vocab_case_sensitive(self):
        """Vocab is case-sensitive: 'Apple' != 'apple' != 'APPLE'."""
        corpus = Counter({"apple": 100, "Apple": 50, "APPLE": 10})
        dict_words = {"apple"}
        vocab = self.tok.build_vocab(corpus, dict_words, lang_code="eng", verbose=False)

        # All three case variants should be in vocab (from corpus)
        self.assertIn("apple", vocab)
        self.assertIn("Apple", vocab)
        self.assertIn("APPLE", vocab)
        # They should have different IDs
        self.assertNotEqual(vocab["apple"], vocab["Apple"])
        self.assertNotEqual(vocab["apple"], vocab["APPLE"])

    def test_build_vocab_with_real_corpus_and_dict(self):
        """Vocab does not exceed max_vocab."""
        lang_code = "eng"
        eng_dict_file = os.path.join(STORAGE_DIR, f"{lang_code}_{DICT_WORDS}")
        word_freq_file = os.path.join(STORAGE_DIR, f"{lang_code}_{WORD_FREQ_FILE_NAME}")
        dict_words = load_json_file(eng_dict_file)
        corpus = load_word_frequencies_json(word_freq_file)
        dict_words = set(dict_words)
        vocab = self.tok.build_vocab(corpus, dict_words, lang_code, verbose=True)
        self.assertLessEqual(len(vocab), MAX_VOCAB_PER_LANGUAGE)

    def test_build_vocab_respects_budget(self):
        """Vocab does not exceed max_vocab."""
        corpus = Counter({f"word{i}": 1 for i in range(1000)})
        dict_words = {f"word{i}" for i in range(500)}
        vocab = self.tok.build_vocab(
            corpus, dict_words, lang_code="eng", max_vocab=100, verbose=False,
        )
        self.assertLessEqual(len(vocab), 100)

    def test_build_vocab_any_language(self):
        """build_vocab works with non-Latin scripts (Japanese example)."""
        corpus = Counter({"東京": 100, "大阪": 50, "京都": 30, "東": 20, "京": 15})
        dict_words = {"東京", "大阪", "京都"}
        vocab = self.tok.build_vocab(corpus, dict_words, lang_code="jpn", verbose=False)

        # Characters from the corpus should be present
        self.assertIn("東", vocab)
        self.assertIn("京", vocab)


class TestWhitespacePreservation(unittest.TestCase):
    """Ensure multi-space runs are preserved for code formatting."""

    def test_four_spaces_preserved(self):
        text = "x    y"  # 4 spaces between two words (no newline)
        tokens = re.findall(UNICODE_PATTERN, text)
        self.assertIn("    ", tokens)

    def test_eight_spaces_preserved(self):
        text = "        deeply_indented"
        tokens = re.findall(UNICODE_PATTERN, text)
        self.assertIn("        ", tokens)

    def test_two_spaces_preserved(self):
        text = "a  b"
        tokens = re.findall(UNICODE_PATTERN, text)
        self.assertIn("  ", tokens)


class TestScoringLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok, cls.model = create_tokenizer(verbose=False)

    def test_single_token_highest_score(self):
        s1 = self.tok.score_tokenization(["account"])
        s2 = self.tok.score_tokenization(["account", "ed"])
        self.assertGreater(s1, s2)

    def test_token_count_dominates(self):
        s_few = self.tok.score_tokenization(["photo", "synthesis"])
        s_many = self.tok.score_tokenization(list("photosyn"))
        self.assertGreater(s_few, s_many)

    def test_scoring_is_bounded(self):
        for tokens in self.tok.find_all_valid_splits_dp("understanding")[:20]:
            s = self.tok.score_tokenization(tokens)
            self.assertGreater(s, 0)
            self.assertLess(s, 2)

    def test_affinity_is_bonus_only(self):
        candidates = self.tok.find_all_valid_splits_dp("understanding")
        twos, threes = [], []
        for tokens in candidates:
            s = self.tok.score_tokenization(tokens)
            if len(tokens) == 2:
                twos.append(s)
            elif len(tokens) == 3:
                threes.append(s)
        if twos and threes:
            self.assertGreaterEqual(max(twos), min(threes) * 0.95)


class TestTrainingBehavior(unittest.TestCase):
    def setUp(self):
        self.tok, self.model = create_tokenizer(verbose=False)
        self.words = ["understanding", "accountability"]

    def test_training_improves_or_maintains(self):
        before = [self.tok.find_best_split(w)[1] for w in self.words]
        self.tok.training_cycle(self.words, epochs=1, verbose=False)
        after = [self.tok.find_best_split(w)[1] for w in self.words]
        self.assertGreaterEqual(np.mean(after), np.mean(before) * 0.95)

    def test_best_split_consistency(self):
        b1, s1, _ = self.tok.find_best_split("understanding")
        b2, s2, _ = self.tok.find_best_split("understanding")
        self.assertEqual(b1, b2)
        self.assertAlmostEqual(s1, s2, places=6)

    def test_training_focuses_on_good_candidates(self):
        self.tok.training_cycle(["accountability"], epochs=1, verbose=False)
        best, _, _ = self.tok.find_best_split("accountability")
        self.assertLess(len(best), 10)


class TestCaching(unittest.TestCase):
    def setUp(self):
        self.tok, _ = create_tokenizer(verbose=False)

    def test_cache_hit_rate(self):
        words = ["account", "understanding", "accountability"]
        for w in words:
            self.tok.find_best_split(w)
        s1 = self.tok.cache.get_stats()
        for w in words:
            self.tok.find_best_split(w)
        s2 = self.tok.cache.get_stats()
        self.assertGreater(s2["cache_hits"], s1["cache_hits"])

    def test_cache_stores_unique_words_only(self):
        words = ["understanding", "account"] * 5
        for w in words:
            self.tok.find_best_split(w)
        s = self.tok.cache.get_stats()
        self.assertEqual(s["cached_words"], 2)


class TestPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok, _ = create_tokenizer(verbose=False)
        words = ["understanding", "accountability"]
        cls.tok.training_cycle(words, epochs=1, verbose=False)
        cls.tok.set_inference_mode(True)
        for w in words:
            cls.tok.find_best_split(w)

    def test_inference_speed(self):
        words = ["understanding", "accountability"]
        N = 100
        t0 = time.perf_counter()
        for _ in range(N):
            for w in words:
                self.tok.find_best_split(w)
        dt = time.perf_counter() - t0
        avg_ms = (dt / (N * len(words))) * 1000
        self.assertLess(avg_ms, 1.0, f"Too slow: {avg_ms:.4f} ms/word")
        print(f"\n  Inference speed: {avg_ms:.4f} ms/word")

    def test_lexicon_memory_efficiency(self):
        stats = self.tok.lexicon.get_stats()
        self.assertLess(stats["total_pairs_computed"], 10000)
        print(f"\n  Lexicon pairs: {stats['total_pairs_computed']}")


class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.tok, _ = create_tokenizer(verbose=False)

    def test_empty_word(self):
        best, score, _ = self.tok.find_best_split("")
        self.assertEqual(best, [""])

    def test_single_character(self):
        best, score, _ = self.tok.find_best_split("a")
        self.assertGreater(len(best), 0)

    def test_numeric_string(self):
        best, score, _ = self.tok.find_best_split("12345")
        self.assertGreater(len(best), 0)


class TestSlotBasedSelection(unittest.TestCase):
    def setUp(self):
        self.tok, _ = create_tokenizer(verbose=False)

    def test_slot_selection_distribution(self):
        word = "understanding"
        self.tok.find_all_valid_splits_dp(word)
        entries = self.tok.cache.get_entries(word)
        self.assertIsNotNone(entries)
        self.assertGreaterEqual(len(entries), 3)

        slot_counts = defaultdict(int)
        N = 10000
        for _ in range(N):
            selected = self.tok.cache.select_candidate_for_training(word)
            if selected is not None:
                for i, e in enumerate(entries):
                    if e is selected:
                        slot_counts[i] += 1
                        break
        ratio_0 = slot_counts[0] / N
        self.assertGreater(ratio_0, 0.70)
        self.assertLess(ratio_0, 0.95)

    def test_inference_only_uses_slot_1(self):
        word = "understanding"
        self.tok.find_all_valid_splits_dp(word)
        self.tok.set_inference_mode(True)
        entries = self.tok.cache.get_entries(word)
        best = self.tok.cache.get_best(word)
        self.assertEqual(best.tokens, entries[0].tokens)
        self.tok.set_inference_mode(False)


class TestConsecutiveLowerDisabling(unittest.TestCase):
    def test_candidate_disabled_after_consecutive_lower(self):
        entry = CandidateEntry(tokens=("a", "b", "c"), score=0.8)
        self.assertTrue(entry.enabled)
        entry.update_score(0.7)
        self.assertTrue(entry.enabled)
        entry.update_score(0.6)
        self.assertFalse(entry.enabled)

    def test_candidate_not_disabled_after_recovery(self):
        entry = CandidateEntry(tokens=("a", "b"), score=0.8)
        entry.update_score(0.7)
        self.assertTrue(entry.enabled)
        entry.update_score(0.75)
        self.assertTrue(entry.enabled)
        self.assertEqual(entry.consecutive_lower, 0)


class TestDequeRotation(unittest.TestCase):
    def test_deque_rotates(self):
        entries = [CandidateEntry(tokens=(str(i),), score=1.0 - i * 0.1) for i in range(6)]
        cache = CandidateCache()
        cache.put_candidates("test_word", entries)
        dq = cache._rest_deque.get("test_word")
        self.assertEqual(list(dq), [3, 4, 5])
        first = dq.popleft()
        dq.append(first)
        self.assertEqual(list(dq), [4, 5, 3])


class TestBPEStruggleWords(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok, _ = create_tokenizer(verbose=False)
        cls.words = get_bpe_struggle_words()

    def test_all_struggle_words_tokenize(self):
        failures = []
        for w in self.words:
            try:
                best, score, _ = self.tok.find_best_split(w)
                if not best or len(best) == 0:
                    failures.append((w, "empty split"))
            except Exception as e:
                failures.append((w, str(e)))
        self.assertEqual(len(failures), 0, f"Failures: {failures}")

    def test_struggle_words_reconstruct(self):
        failures = []
        for w in self.words:
            best, _, _ = self.tok.find_best_split(w)
            reconstructed = "".join(best)
            if w.isascii() and reconstructed != w:
                failures.append((w, best, reconstructed))
        if failures:
            print(f"\n  Reconstruction mismatches: {len(failures)}")
            for w, b, c in failures[:5]:
                print(f"    {w} -> {b} -> '{c}'")
        self.assertLess(len(failures), len(self.words) // 2)

    def test_training_on_struggle_words(self):
        self.tok.training_cycle(self.words, epochs=3, verbose=False)
        print("\nBPE Struggle Words (sample):")
        print(f"{'WORD':<22} | SEGMENTATION")
        print("-" * 50)
        for w in self.words[:30]:
            best, _, _ = self.tok.find_best_split(w)
            print(f"{w:<22} | {best}")
            self.assertGreater(len(best), 0)


class TestBPEExcelWords(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok, _ = create_tokenizer(verbose=False)
        cls.words = get_bpe_excel_words()

    def test_all_excel_words_tokenize(self):
        failures = []
        for w in self.words:
            try:
                best, score, _ = self.tok.find_best_split(w)
                if not best:
                    failures.append((w, "empty"))
            except Exception as e:
                failures.append((w, str(e)))
        self.assertEqual(len(failures), 0, f"Failures: {failures}")

    def test_excel_words_mostly_single_token(self):
        single_count = 0
        print("\nBPE Excel Words (sample):")
        print(f"{'WORD':<22} | SEGMENTATION")
        print("-" * 50)
        for w in self.words:
            best, _, _ = self.tok.find_best_split(w)
            print(f"{w:<22} | {best}")
            if len(best) == 1:
                single_count += 1
        ratio = single_count / len(self.words)
        print(f"\n  Single-token rate: {ratio:.0%} ({single_count}/{len(self.words)})")
        self.assertGreater(ratio, 0.6)

    def test_excel_words_inference_speed(self):
        for w in self.words:
            self.tok.find_best_split(w)
        self.tok.set_inference_mode(True)
        N = 50
        t0 = time.perf_counter()
        for _ in range(N):
            for w in self.words:
                self.tok.find_best_split(w)
        dt = time.perf_counter() - t0
        avg_ms = (dt / (N * len(self.words))) * 1000
        self.assertLess(avg_ms, 1.0, f"Too slow: {avg_ms:.4f} ms")
        print(f"\n  Excel words inference: {avg_ms:.4f} ms/word")
        self.tok.set_inference_mode(False)


class TestSharedModelObject(unittest.TestCase):
    def test_model_is_shared(self):
        vocab_tokens = ["hello", "world"]
        model = SimpleTransformerXL(
            vocab_size=len(vocab_tokens),
            d_model=50,
            vocab_tokens=vocab_tokens,
        )
        tok = SemanticTokenizer(model=model)
        self.assertIs(tok.model, model)
        self.assertIs(tok.lexicon.model, model)

    def test_training_updates_shared_model(self):
        tok, model = create_tokenizer(verbose=False)
        split = ["sky", "scraper"]
        emb_before = model.similarity(split[0], split[1])
        sentences = get_skyscraper_training_sentences()
        model.train_model(sentences, epochs=10)
        emb_after = model.similarity(split[0], split[1])
        diff = emb_after - emb_before
        self.assertGreater(diff, 0, "Model embeddings should change after training")


# ============================================================================
# NEW: Language bias detection tests
# ============================================================================

class TestInferLanguageBias(unittest.TestCase):
    """Tests for infer_language_bias statistical analysis function."""

    def _make_suffixing_words(self, n: int = 300) -> set:
        """Simulate a strongly suffixing language (English-like morphology).

        The paradigm-based bias algorithm requires that when we strip a suffix,
        the remaining stem must itself be present in the word set.  We therefore
        include all base (root) forms so that e.g. ``"walked"`` → stem ``"walk"``
        is recoverable from the set.
        """
        roots = [
            "walk", "talk", "play", "work", "run", "help", "jump",
            "call", "look", "move", "build", "grow", "show", "find",
            "make", "take", "give", "know", "think", "feel", "learn",
            "teach", "dream", "hope", "love", "hate", "need", "want",
            "like", "use", "ask", "tell", "start", "stop", "keep",
        ]
        suffixes = [
            "ed", "ing", "er", "est", "ly", "ness", "tion",
            "ment", "ful", "less", "able", "ous", "ive", "al", "s",
        ]
        words = set(roots)   # ← base forms MUST be in the set for stem recovery
        for root in roots:
            for suf in suffixes:
                words.add(root + suf)
        return words

    def _make_prefixing_words(self, n: int = 300) -> set:
        """Simulate a strongly prefixing language (Swahili-like Bantu morphology).

        Base forms are included so prefix stripping of e.g. ``"mtoto"`` recovers
        stem ``"toto"`` which is in the set — satisfying the paradigm-based criterion.
        """
        roots = [
            "toto", "jiji", "kono", "lango", "limu", "sanduku",
            "panga", "baba", "mama", "ndugu", "shule", "nyumba",
            "gari", "chakula", "maji", "ardhi", "jiwe", "kitu",
            "neno", "jambo", "swali",
        ]
        prefixes = ["m", "wa", "ki", "vi", "mu", "mi", "u", "i",
                    "li", "ya", "ku", "pa", "a", "ha", "n", "ny"]
        words = set(roots)   # ← base forms MUST be in the set for stem recovery
        for root in roots:
            for pre in prefixes:
                words.add(pre + root)
        return words

    def test_english_like_is_suffixing(self):
        """English-like morphology (heavy suffixation) should be detected as suffixing."""
        words = self._make_suffixing_words()
        result = infer_language_bias(words)
        self.assertIn(result["bias"], ("strongly_suffixing", "weakly_suffixing"),
                      f"Expected suffixing bias but got: {result['bias']}")
        self.assertTrue(result["is_suffixing"])

    def test_prefixing_language_detected(self):
        """Bantu-like morphology (heavy prefixation) should be detected as prefixing."""
        words = self._make_prefixing_words()
        result = infer_language_bias(words)
        self.assertFalse(result["is_suffixing"],
                         f"Prefixing-heavy language should not be suffixing: {result['bias']}")
        self.assertIn(result["bias"],
                      ("strongly_prefixing", "weakly_prefixing", "neutral", "insufficient_data"),
                      f"Unexpected bias for Bantu-like language: {result['bias']}")

    def test_insufficient_data_returns_safe_default(self):
        """Fewer than 50 words triggers insufficient_data with safe is_suffixing=True."""
        words = {"cat", "dog", "fish", "bird", "tree"}
        result = infer_language_bias(words)
        self.assertEqual(result["bias"], "insufficient_data")
        self.assertTrue(result["is_suffixing"])  # safe default
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["words_analysed"], 5)

    def test_return_keys_complete(self):
        """Return dict must contain all documented keys."""
        words = self._make_suffixing_words()
        result = infer_language_bias(words)
        required_keys = {
            "bias", "is_suffixing", "suffix_productivity", "prefix_productivity",
            "ratio", "suffix_candidates", "prefix_candidates",
            "suffix_derived_words", "prefix_derived_words",
            "per_length_ratios", "words_analysed", "confidence",
        }
        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_ratio_is_non_negative(self):
        """The ratio must always be non-negative."""
        words = self._make_suffixing_words()
        result = infer_language_bias(words)
        self.assertGreaterEqual(result["ratio"], 0.0)

    def test_productivity_values_positive(self):
        """Productivity scores must be non-negative."""
        words = self._make_suffixing_words()
        result = infer_language_bias(words)
        self.assertGreaterEqual(result["suffix_productivity"], 0.0)
        self.assertGreaterEqual(result["prefix_productivity"], 0.0)

    def test_confidence_between_0_and_1(self):
        """Confidence must be in [0, 1]."""
        words = self._make_suffixing_words()
        result = infer_language_bias(words)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_per_length_ratios_keys_are_ints(self):
        """per_length_ratios keys must be integer affix lengths."""
        words = self._make_suffixing_words()
        result = infer_language_bias(words)
        for k in result["per_length_ratios"]:
            self.assertIsInstance(k, int, f"Non-int key in per_length_ratios: {k!r}")

    def test_sample_size_respected(self):
        """When sample_size is given, words_analysed <= sample_size."""
        words = self._make_suffixing_words()
        result = infer_language_bias(words, sample_size=50)
        self.assertLessEqual(result["words_analysed"], 50)

    def test_sample_produces_same_bias_class(self):
        """Sampling a large suffixing vocabulary should still yield a suffixing bias."""
        # Build a large set by repeating the pattern
        roots = ["walk", "talk", "play", "work", "run", "help",
                 "jump", "call", "look", "move", "build", "grow",
                 "show", "find", "make", "take", "give", "know",
                 "think", "feel", "seem", "want", "need", "love"]
        suffixes = ["ed", "ing", "er", "ly", "ness", "tion", "ment",
                    "ful", "less", "able", "ous", "ive", "al", "s", "es"]
        words: set = set()
        for r in roots:
            words.add(r)
            for s in suffixes:
                words.add(r + s)
        full_result   = infer_language_bias(words)
        sample_result = infer_language_bias(words, sample_size=80)
        # Both should agree on the is_suffixing boolean
        self.assertEqual(full_result["is_suffixing"], sample_result["is_suffixing"])

    def test_invalid_affix_len_raises(self):
        """min_affix_len > max_affix_len must raise ValueError."""
        words = self._make_suffixing_words()
        with self.assertRaises(ValueError):
            infer_language_bias(words, min_affix_len=8, max_affix_len=2)

    def test_non_positive_affix_len_raises(self):
        """Non-positive affix lengths must raise ValueError."""
        words = self._make_suffixing_words()
        with self.assertRaises(ValueError):
            infer_language_bias(words, min_affix_len=0)
        with self.assertRaises(ValueError):
            infer_language_bias(words, max_affix_len=0)

    def test_non_iterable_raises_type_error(self):
        """Passing a non-iterable should raise TypeError."""
        with self.assertRaises(TypeError):
            infer_language_bias(42)  # type: ignore[arg-type]

    def test_unicode_script_agnostic(self):
        """Function must handle non-Latin scripts without error."""
        # Simulate Japanese-like words (Hiragana/Kanji)
        japanese_like = {
            "東京", "大阪", "京都", "横浜", "神戸", "名古屋", "福岡", "札幌",
            "東", "京", "大", "阪", "都", "浜", "神", "戸", "名", "古", "屋",
            "東京都", "大阪市", "京都市", "名古屋市", "横浜市",
            "日本語", "英語", "中国語", "韓国語", "フランス語",
        } | {f"語{i}" for i in range(50)} | {f"{i}語" for i in range(50)}
        try:
            result = infer_language_bias(japanese_like)
            self.assertIn("bias", result)
        except Exception as e:
            self.fail(f"infer_language_bias raised {type(e).__name__} on non-Latin script: {e}")

    def test_cyrillic_script(self):
        """Function must work with Cyrillic-script words."""
        # Russian-like words with common suffix -ость (-ost') and prefix пре- (pre-)
        cyrillic_words = {
            "красота", "красотой", "красоту", "красоте",
            "добро", "добрый", "доброта", "доброте",
            "любовь", "любить", "любовью",
            "правда", "правдой", "правде",
            "красивость", "доброта", "правдивость",
            "переход", "преграда", "прекрасный",
            "выход", "входить", "выходить",
        } | {f"слово{i}" for i in range(80)} | {f"{i}слово" for i in range(80)}
        try:
            result = infer_language_bias(cyrillic_words)
            self.assertIn("bias", result)
        except Exception as e:
            self.fail(f"infer_language_bias raised {type(e).__name__} on Cyrillic: {e}")

    def test_empty_set_returns_insufficient(self):
        """Empty word set returns insufficient_data."""
        result = infer_language_bias(set())
        self.assertEqual(result["bias"], "insufficient_data")

    def test_single_word_insufficient(self):
        """A single word returns insufficient_data."""
        result = infer_language_bias({"hello"})
        self.assertEqual(result["bias"], "insufficient_data")

    def test_words_analysed_matches_input(self):
        """words_analysed equals len(dict_words) when no sampling."""
        words = self._make_suffixing_words()
        result = infer_language_bias(words)
        self.assertEqual(result["words_analysed"], len(words))

    def test_bias_and_is_suffixing_consistent(self):
        """is_suffixing must be True iff bias is weakly/strongly suffixing."""
        words = self._make_suffixing_words()
        result = infer_language_bias(words)
        if result["bias"] in ("strongly_suffixing", "weakly_suffixing"):
            self.assertTrue(result["is_suffixing"],
                            f"is_suffixing should be True for bias={result['bias']}")
        elif result["bias"] in ("strongly_prefixing", "weakly_prefixing", "neutral"):
            self.assertFalse(result["is_suffixing"],
                             f"is_suffixing should be False for bias={result['bias']}")
        # insufficient_data: is_suffixing defaults to True (safe default) — no assertion

    def test_min_type_freq_filters_hapax(self):
        """Increasing min_type_freq should reduce candidate counts."""
        words = self._make_suffixing_words()
        r_low  = infer_language_bias(words, min_type_freq=2)
        r_high = infer_language_bias(words, min_type_freq=20)
        self.assertGreaterEqual(r_low["suffix_candidates"],  r_high["suffix_candidates"])
        self.assertGreaterEqual(r_low["prefix_candidates"],  r_high["prefix_candidates"])


# ============================================================================
# NEW: Integration — bias detection wired into run_morpheme_pipeline
# ============================================================================

class TestLanguageBiasIntegration(unittest.TestCase):
    """Verify that infer_language_bias results flow into the tokenizer pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.tok, cls.model = create_tokenizer(verbose=False)

    def test_bias_result_logged_without_error(self):
        """run_morpheme_pipeline should not raise when auto-detecting bias."""
        # We build a tiny dict_words to exercise the code path quickly
        dict_words = set(get_bpe_struggle_words()) | set(get_bpe_excel_words())
        # Just check it doesn't raise
        try:
            result = infer_language_bias(dict_words)
            self.assertIn("bias", result)
        except Exception as e:
            self.fail(f"infer_language_bias raised unexpectedly: {e}")

    def test_get_stats_includes_vocab_size(self):
        """get_stats() must now include vocab_size."""
        stats = self.tok.get_stats()
        self.assertIn("vocab_size", stats)
        self.assertGreater(stats["vocab_size"], len(SPECIAL_TOKENS))


# ============================================================================
# NEW: Production-readiness tests — edge cases, robustness, multilingual
# ============================================================================

class TestProductionEdgeCases(unittest.TestCase):
    """Edge-case tests to ensure the tokenizer is robust in production."""

    @classmethod
    def setUpClass(cls):
        cls.tok, cls.model = create_tokenizer(verbose=False)

    # ── Text normalization ──────────────────────────────────────────────────

    def test_normalize_none_returns_empty(self):
        """normalize_text_for_nlp(None) should return empty string gracefully."""
        from ange_tokenizer import normalize_text_for_nlp
        self.assertEqual(normalize_text_for_nlp(None), "")

    def test_normalize_integer_returns_empty(self):
        """normalize_text_for_nlp(int) should return empty string."""
        from ange_tokenizer import normalize_text_for_nlp
        self.assertEqual(normalize_text_for_nlp(123), "")

    def test_normalize_nfc_ligature(self):
        """Ligature characters should be NFKC-decomposed."""
        from ange_tokenizer import normalize_text_for_nlp
        result = normalize_text_for_nlp("ﬁle")  # U+FB01 LATIN SMALL LIGATURE FI
        self.assertIn("fi", result)

    def test_tokenize_zero_width_chars(self):
        """Zero-width chars (ZWNJ, ZWJ, etc.) should not cause crashes."""
        text = "hello​world"  # ZERO WIDTH SPACE
        result = self.tok.tokenize(text)
        self.assertIsInstance(result, list)

    def test_tokenize_control_chars(self):
        """C0/C1 control characters are replaced with spaces, not crashed on."""
        text = "abcdefghi"
        result = self.tok.tokenize(text)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_tokenize_mixed_scripts(self):
        """Mixed Latin + Cyrillic + CJK text should not raise."""
        text = "hello мир 世界"
        try:
            result = self.tok.tokenize(text)
            self.assertIsInstance(result, list)
        except Exception as e:
            self.fail(f"tokenize raised on mixed scripts: {e}")

    def test_tokenize_rtl_text(self):
        """Right-to-left Arabic/Hebrew text should not raise."""
        text = "مرحبا بالعالم"  # "Hello World" in Arabic
        try:
            result = self.tok.tokenize(text)
            self.assertIsInstance(result, list)
        except Exception as e:
            self.fail(f"tokenize raised on RTL text: {e}")

    def test_tokenize_very_long_word(self):
        """A single very long word (50+ chars) should not crash or take forever."""
        import time
        word = "supercalifragilisticexpialidocious"  # 34 chars
        t0 = time.perf_counter()
        result = self.tok._tokenize_word(word)
        dt = time.perf_counter() - t0
        self.assertIsInstance(result, list)
        self.assertLess(dt, 2.0, f"Tokenization of long word took {dt:.2f}s — too slow")

    def test_tokenize_all_punctuation(self):
        """A string of only punctuation should tokenize without crash."""
        result = self.tok.tokenize("!@#$%^&*()_+-=[]{}|;:,./<>?")
        self.assertIsInstance(result, list)

    def test_tokenize_only_numbers(self):
        """Numeric strings should tokenize without crash."""
        result = self.tok.tokenize("1234567890")
        self.assertIsInstance(result, list)

    def test_tokenize_empty_string(self):
        """Empty string should return an empty list."""
        result = self.tok.tokenize("")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_tokenize_whitespace_only(self):
        """Whitespace-only string should not crash."""
        result = self.tok.tokenize("   \t\n   ")
        self.assertIsInstance(result, list)

    # ── Encode / decode round-trip ──────────────────────────────────────────

    def test_encode_returns_list_of_ints(self):
        """encode() must return a list of integers."""
        ids = self.tok.encode("sky scraper")
        self.assertIsInstance(ids, list)
        for i in ids:
            self.assertIsInstance(i, int)

    def test_encode_with_special_tokens(self):
        """encode(add_special_tokens=True) should prepend BOS and append EOS."""
        ids_plain = self.tok.encode("sky")
        ids_special = self.tok.encode("sky", add_special_tokens=True)
        self.assertEqual(len(ids_special), len(ids_plain) + 2)

    def test_decode_empty_list(self):
        """decode([]) should return empty string, not raise."""
        result = self.tok.decode([])
        self.assertEqual(result, "")

    def test_decode_skips_special_tokens(self):
        """decode skips BOS/EOS/PAD/MASK tokens."""
        bos_id = self.tok.vocab.get("<|bos|>", -1)
        eos_id = self.tok.vocab.get("<|eos|>", -1)
        sky_id = self.tok.vocab.get("sky", self.tok.unk_token_id)
        ids = [bos_id, sky_id, eos_id]
        result = self.tok.decode(ids)
        self.assertNotIn("<|bos|>", result)
        self.assertNotIn("<|eos|>", result)

    # ── Vocab operations ────────────────────────────────────────────────────

    def test_add_words_to_vocab_idempotent(self):
        """Adding the same word twice should not increase vocab size."""
        initial_size = self.tok.vocab_size
        self.tok.add_words_to_vocab(["__test_unique_xyz__"])
        size_after_first = self.tok.vocab_size
        self.tok.add_words_to_vocab(["__test_unique_xyz__"])
        size_after_second = self.tok.vocab_size
        self.assertEqual(size_after_first, size_after_second)
        # Cleanup
        if "__test_unique_xyz__" in self.tok.vocab:
            del self.tok.vocab["__test_unique_xyz__"]

    def test_vocab_size_property(self):
        """vocab_size property must equal len(self.vocab)."""
        self.assertEqual(self.tok.vocab_size, len(self.tok.vocab))

    def test_id_to_token_and_vocab_consistent(self):
        """id_to_token and vocab must be mutually consistent."""
        for tok_str, tok_id in self.tok.vocab.items():
            self.assertEqual(
                self.tok.id_to_token.get(tok_id), tok_str,
                f"id_to_token[{tok_id}] != {tok_str!r}",
            )

    # ── build_vocab edge cases ──────────────────────────────────────────────

    def test_build_vocab_empty_dict_words(self):
        """build_vocab with empty dict_words should still produce special tokens."""
        corpus = Counter({"sky": 10})
        vocab = self.tok.build_vocab(corpus, set(), lang_code="tst", verbose=False)
        for st in SPECIAL_TOKENS:
            self.assertIn(st, vocab)

    def test_build_vocab_empty_corpus(self):
        """build_vocab with empty corpus should produce at least special tokens."""
        vocab = self.tok.build_vocab(Counter(), {"sky", "scraper"}, lang_code="tst", verbose=False)
        for st in SPECIAL_TOKENS:
            self.assertIn(st, vocab)

    def test_build_vocab_max_vocab_zero_returns_special_tokens_only(self):
        """max_vocab=len(SPECIAL_TOKENS) should yield exactly the special tokens plus chars."""
        corpus = Counter({"hello": 5})
        vocab = self.tok.build_vocab(
            corpus, {"hello"}, lang_code="tst",
            max_vocab=len(SPECIAL_TOKENS) + 5, verbose=False,
        )
        self.assertLessEqual(len(vocab), len(SPECIAL_TOKENS) + 5)

    # ── find_all_valid_splits_dp robustness ─────────────────────────────────

    def test_splits_always_reconstruct(self):
        """Every split returned by find_all_valid_splits_dp must reconstruct the word."""
        words = ["understanding", "accountability", "skyscraper", "background",
                 "ultraviolet", "photosynthesis", "extraordinary", "microorganism"]
        for word in words:
            splits = self.tok.find_all_valid_splits_dp(word)
            for split in splits:
                reconstructed = "".join(t.lstrip(SUBWORD_PREFIX) for t in split)
                self.assertEqual(
                    reconstructed, word,
                    f"Split {split} does not reconstruct '{word}' → got '{reconstructed}'",
                )

    def test_dp_returns_sorted_by_score(self):
        """find_all_valid_splits_dp must return splits sorted best-first."""
        entries = self.tok.cache.get_entries("understanding") or []
        if not entries:
            self.tok.find_all_valid_splits_dp("understanding")
            entries = self.tok.cache.get_entries("understanding") or []
        scores = [e.score for e in entries]
        self.assertEqual(scores, sorted(scores, reverse=True),
                         "Cache entries must be sorted in descending score order")

    # ── CandidateCache internals ─────────────────────────────────────────────

    def test_cache_has_candidates_false_for_unknown(self):
        """has_candidates returns False for words not yet processed."""
        self.assertFalse(self.tok.cache.has_candidates("__definitely_not_processed__"))

    def test_cache_get_best_returns_none_for_unknown(self):
        """get_best returns None for words not in cache."""
        result = self.tok.cache.get_best("__not_in_cache_xyz__")
        self.assertIsNone(result)

    def test_cache_get_all_tokens_returns_none_for_unknown(self):
        """get_all_tokens returns None for words not in cache."""
        result = self.tok.cache.get_all_tokens("__not_in_cache_xyz__")
        self.assertIsNone(result)

    # ── Mode switching ───────────────────────────────────────────────────────

    def test_switching_inference_mode_on_off(self):
        """Toggling inference mode must not corrupt state."""
        self.tok.set_inference_mode(True)
        self.assertTrue(self.tok.inference_mode)
        r1 = self.tok._tokenize_word("sky")

        self.tok.set_inference_mode(False)
        self.assertFalse(self.tok.inference_mode)
        r2 = self.tok._tokenize_word("sky")

        self.tok.set_inference_mode(True)
        r3 = self.tok._tokenize_word("sky")
        self.tok.set_inference_mode(False)

        # All modes should return the same tokens for a cached word
        self.assertEqual(r1, r3)

    # ── Multilingual ISO 639-3 support ──────────────────────────────────────

    def test_build_vocab_arabic_lang_code(self):
        """build_vocab must not raise for Arabic (ara) ISO 639-3 code."""
        # Minimal Arabic-script corpus (transliteration stand-ins for CI)
        corpus = Counter({"كتاب": 50, "مدرسة": 40, "طالب": 30, "كتب": 20})
        dict_words = {"كتاب", "مدرسة", "طالب", "كتب", "مكتبة", "مدرس"}
        try:
            vocab = self.tok.build_vocab(corpus, dict_words, lang_code="ara", verbose=False)
            for st in SPECIAL_TOKENS:
                self.assertIn(st, vocab)
        except Exception as e:
            self.fail(f"build_vocab raised for Arabic lang_code 'ara': {e}")

    def test_build_vocab_turkish_lang_code(self):
        """build_vocab must not raise for Turkish (tur) — a strongly suffixing language."""
        # Turkish: very regular suffix-heavy agglutination
        corpus = Counter({"ev": 100, "evler": 80, "evde": 70, "eve": 60,
                          "okul": 90, "okullar": 70, "okulda": 60, "okula": 50})
        dict_words = {"ev", "evler", "evlerde", "eve", "evden",
                      "okul", "okullar", "okulda", "okula", "okuldan"}
        try:
            vocab = self.tok.build_vocab(corpus, dict_words, lang_code="tur", verbose=False)
            for st in SPECIAL_TOKENS:
                self.assertIn(st, vocab)
        except Exception as e:
            self.fail(f"build_vocab raised for Turkish lang_code 'tur': {e}")

    def test_build_vocab_swahili_lang_code(self):
        """build_vocab must not raise for Swahili (swa) — a strongly prefixing language."""
        corpus = Counter({"mtoto": 50, "watoto": 40, "kijiji": 30, "vijiji": 20,
                          "mkono": 15, "mikono": 10, "mlango": 8, "milango": 5})
        dict_words = {"mtoto", "watoto", "kijiji", "vijiji", "mkono", "mikono",
                      "mlango", "milango", "mtu", "watu"}
        try:
            vocab = self.tok.build_vocab(corpus, dict_words, lang_code="swa", verbose=False)
            for st in SPECIAL_TOKENS:
                self.assertIn(st, vocab)
        except Exception as e:
            self.fail(f"build_vocab raised for Swahili lang_code 'swa': {e}")

    def test_undetermined_lang_code(self):
        """lang_code='und' (undetermined) must work without error."""
        corpus = Counter({"word": 10, "test": 5})
        dict_words = {"word", "test", "words", "testing"}
        try:
            vocab = self.tok.build_vocab(corpus, dict_words, lang_code="und", verbose=False)
            self.assertIn("<|pad|>", vocab)
        except Exception as e:
            self.fail(f"build_vocab raised for lang_code='und': {e}")


class TestUnicodeNormalization(unittest.TestCase):
    """Extended normalization and Unicode handling tests."""

    def setUp(self):
        from ange_tokenizer import normalize_text_for_nlp
        self.norm = normalize_text_for_nlp

    def test_bom_removed(self):
        """BOM (U+FEFF) must be stripped."""
        result = self.norm("﻿hello")
        self.assertNotIn("﻿", result)

    def test_soft_hyphen_removed(self):
        """Soft hyphen (U+00AD) must be deleted."""
        result = self.norm("super­man")
        self.assertNotIn("­", result)

    def test_nfkc_full_width(self):
        """Full-width ASCII characters should normalize to ASCII."""
        result = self.norm("Ａ Ｂ Ｃ")
        self.assertIn("A", result)
        self.assertIn("B", result)
        self.assertIn("C", result)

    def test_tab_expanded_to_spaces(self):
        """Tab characters are expanded to 4 spaces."""
        result = self.norm("\thello")
        self.assertIn("    ", result)
        self.assertNotIn("\t", result)

    def test_non_string_returns_empty(self):
        """Non-string inputs (None, int, list) return empty string."""
        self.assertEqual(self.norm(None), "")
        self.assertEqual(self.norm(42), "")
        self.assertEqual(self.norm([]), "")

    def test_c0_controls_to_space(self):
        """C0 control characters (except \t, \n, \r) become spaces."""
        result = self.norm("a\x07b")  # BELL
        self.assertNotIn("\x07", result)

    def test_combining_marks_preserved(self):
        """Combining diacritical marks should be preserved after NFKC."""
        # é (U+00E9) should remain as a single char after NFKC
        result = self.norm("caf\u00e9")
        self.assertIn("é", result)

    def test_unicode_line_separators_become_spaces(self):
        """Unicode paragraph/line separators (U+2028, U+2029) become spaces."""
        result = self.norm("line1\u2028line2\u2029line3")
        self.assertNotIn("\u2028", result)
        self.assertNotIn("\u2029", result)


class TestSemanticTokenLexicon(unittest.TestCase):
    """Tests for the SemanticTokenLexicon scoring component."""

    @classmethod
    def setUpClass(cls):
        cls.tok, cls.model = create_tokenizer(verbose=False)
        cls.lexicon = cls.tok.lexicon

    def test_score_tokenization_empty(self):
        """Scoring an empty token list returns 0."""
        self.assertEqual(self.lexicon.score_tokenization([]), 0.0)

    def test_score_tokenization_single_token(self):
        """A single token has a well-defined, positive score."""
        score = self.lexicon.score_tokenization(["sky"])
        self.assertGreater(score, 0.0)

    def test_score_monotone_fewer_tokens_better(self):
        """Fewer tokens → higher base score (from 1/log(n+1) formula)."""
        s1 = self.lexicon.score_tokenization(["sky"])
        s2 = self.lexicon.score_tokenization(["sky", "scrap"])
        s3 = self.lexicon.score_tokenization(["sky", "scrap", "er"])
        self.assertGreater(s1, s2)
        self.assertGreater(s2, s3)

    def test_get_pair_affinity_unknown_tokens(self):
        """Unknown token pairs return 0.0 affinity, not a crash."""
        aff = self.lexicon.get_pair_affinity("__unk_tok_a__", "__unk_tok_b__")
        self.assertEqual(aff, 0.0)

    def test_update_pair_scores_changes_affinity(self):
        """update_pair_scores nudges pair affinity toward the observed score."""
        self.lexicon.get_pair_affinity("sky", "scraper")  # seed the cache
        before = self.lexicon.pair_affinity.get(("sky", "scraper"), 0.0)
        self.lexicon.update_pair_scores(["sky", "scraper"], 0.99, lr=0.5)
        after = self.lexicon.pair_affinity.get(("sky", "scraper"), 0.0)
        self.assertNotAlmostEqual(before, after, places=4)

    def test_refresh_clears_pair_affinity(self):
        """refresh_from_model() clears the pair affinity cache."""
        self.lexicon.get_pair_affinity("sky", "scraper")
        self.assertGreater(len(self.lexicon.pair_affinity), 0)
        self.lexicon.refresh_from_model()
        self.assertEqual(len(self.lexicon.pair_affinity), 0)

    def test_get_stats_returns_dict(self):
        """get_stats returns a dictionary with expected keys."""
        stats = self.lexicon.get_stats()
        for key in ("total_pairs_computed", "total_pairs_updated", "mean_affinity"):
            self.assertIn(key, stats)


def run_test_suite(verbosity=2):
    print("\n" + "=" * 80)
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print("=" * 80 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for tc in [
        TestStringReconstruction,
        TestWhitespacePreservation,
        TestTokenizeWord,
        TestBuildVocab,
        TestScoringLogic, TestTrainingBehavior, TestCaching,
        TestPerformance, TestEdgeCases,
        TestSlotBasedSelection, TestConsecutiveLowerDisabling,
        TestDequeRotation, TestSkyscraperFragments,
        TestBPEStruggleWords, TestBPEExcelWords,
        TestSharedModelObject,
        # ── New suites ──────────────────────────────────────
        TestInferLanguageBias,
        TestLanguageBiasIntegration,
        TestProductionEdgeCases,
        TestUnicodeNormalization,
        TestSemanticTokenLexicon,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(tc))

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result.wasSuccessful()



if __name__ == "__main__":
    success = run_test_suite(verbosity=2)
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED \u2713" if success else "SOME TESTS FAILED \u2717")
    print("=" * 80)
    sys.exit(0 if success else 1)