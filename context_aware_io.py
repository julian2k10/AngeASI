"""
Test-Aware File I/O Wrapper

Automatically detects if code is running from a test context and redirects
all file operations to TEST_FILES_DIR instead of mixing test files with live data.

Usage:
    @test_aware_io
    def load_json_file(file_path):
        # Your existing implementation
        ...

    @test_aware_io
    def save_json_file(save_directory, file_name, json_data):
        # Your existing implementation
        ...
"""

import os
import json
import inspect
import functools
from collections import defaultdict
from typing import Callable, Any, Dict, Union, Iterable, Set
from pathlib import Path
import blake3
import ujson


# ============================================================================
# Configuration
# ============================================================================

try:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT_DIR = os.getcwd()

LOG_DIR = os.path.join(ROOT_DIR, "logs")
TEST_DIR = os.path.join(ROOT_DIR, "tests")
TEST_FILES_DIR = os.path.join(TEST_DIR, "files")
STORAGE_DIR = os.path.join(ROOT_DIR, "vocab_files")

# File names
DICT_WORDS = "words.json"
WORD_FREQ_FILE_NAME = 'word_frequencies.json'
MORPHEME_FREQ_FILE_NAME = 'morpheme_frequencies.json'
VOCAB_FILE_NAME = 'vocab.json'

# Full paths
VOCAB_FILE = os.path.join(STORAGE_DIR, VOCAB_FILE_NAME)
MORPHEME_FILE = os.path.join(STORAGE_DIR, 'morphemes.json')
FILTERED_WORD_FREQ_FILE = os.path.join(STORAGE_DIR, 'dict_word_frequencies.json')


# ============================================================================
# Test Detection & Path Resolution
# ============================================================================

class TestContextDetector:
    """
    Detects if the current execution is happening from a test file.

    A file is considered a "test file" if:
    1. Its filename starts with "test_"
    2. Its filename ends with "_test.py"
    3. It's located in a "tests" directory
    """

    _cache: Dict[str, bool] = {}  # Cache results to avoid repeated stack inspection

    @staticmethod
    def is_test_context() -> bool:
        """
        Check if we're running in a test context by inspecting the call stack.

        Returns:
            bool: True if any caller in the stack is a test file
        """
        stack = inspect.stack()

        for frame_info in stack:
            file_path = frame_info.filename

            # Check cache first
            if file_path in TestContextDetector._cache:
                if TestContextDetector._cache[file_path]:
                    return True
                continue

            is_test = TestContextDetector._is_test_file(file_path)
            TestContextDetector._cache[file_path] = is_test

            if is_test:
                return True

        return False

    @staticmethod
    def _is_test_file(file_path: str) -> bool:
        """
        Determine if a file is a test file.

        Args:
            file_path: Path to the file to check

        Returns:
            bool: True if the file is considered a test file
        """
        # Normalize path
        path = Path(file_path).resolve()
        file_name = path.name

        # Check filename patterns
        if file_name.startswith("test_") or file_name.endswith("_test.py"):
            return True

        # Check if file is in a "tests" directory
        if "tests" in path.parts or "test" in path.parts:
            return True

        return False

    @staticmethod
    def clear_cache():
        """Clear the detection cache (useful for testing the detector itself)."""
        TestContextDetector._cache.clear()


class PathResolver:
    """
    Resolves file paths based on execution context.

    If running in a test context, all paths are remapped to TEST_FILES_DIR.
    Otherwise, original paths are preserved.

    Additionally, if a file doesn't exist in the test directory but exists in
    the original location, it's automatically copied to the test directory.
    """

    @staticmethod
    def resolve_directory(directory: str, is_test: bool) -> str:
        """
        Resolve a directory path based on test context.

        Args:
            directory: Original directory path
            is_test: Whether we're in a test context

        Returns:
            str: Resolved directory path
        """
        if not is_test:
            return directory

        # Ensure TEST_FILES_DIR exists
        os.makedirs(TEST_FILES_DIR, exist_ok=True)

        # In test context, redirect to TEST_FILES_DIR
        return TEST_FILES_DIR

    @staticmethod
    def resolve_file_path(file_path: str, is_test: bool) -> str:
        """
        Resolve a full file path based on test context.

        If the file doesn't exist in test directory but exists in the original
        location, copy it automatically.

        Args:
            file_path: Original file path
            is_test: Whether we're in a test context

        Returns:
            str: Resolved file path
        """
        if not is_test:
            return file_path

        # Ensure TEST_FILES_DIR exists
        os.makedirs(TEST_FILES_DIR, exist_ok=True)

        # Extract just the filename
        file_name = os.path.basename(file_path)

        # Resolve to TEST_FILES_DIR
        test_file_path = os.path.join(TEST_FILES_DIR, file_name)

        # If file doesn't exist in test dir but exists in original location, copy it
        if not os.path.exists(test_file_path) and os.path.exists(file_path):
            PathResolver._copy_file(file_path, test_file_path)

        return test_file_path

    @staticmethod
    def resolve_directory_and_filename(directory: str, file_name: str, is_test: bool) -> tuple:
        """
        Resolve directory and filename separately (useful for save operations).

        Args:
            directory: Original directory path
            file_name: Original filename
            is_test: Whether we're in a test context

        Returns:
            tuple: (resolved_directory, file_name)
        """
        if not is_test:
            return directory, file_name

        # Ensure TEST_FILES_DIR exists
        os.makedirs(TEST_FILES_DIR, exist_ok=True)

        # In test context, use TEST_FILES_DIR for directory
        return TEST_FILES_DIR, file_name

    @staticmethod
    def _copy_file(source: str, destination: str) -> None:
        """
        Copy a file from source to destination.

        Args:
            source: Source file path
            destination: Destination file path
        """
        import shutil
        try:
            shutil.copy2(source, destination)
            print(f"[TEST SETUP] Copied file: {source} → {destination}")
        except Exception as e:
            print(f"[TEST SETUP] Warning: Could not copy {source} to {destination}: {e}")


# ============================================================================
# Decorator: test_aware_io
# ============================================================================

def context_aware_io(func: Callable) -> Callable:
    """
    Decorator that makes a function test-aware.

    Automatically detects if the function is called from a test context
    and redirects file operations to TEST_FILES_DIR.

    Works with two function signatures:
    1. load_json_file(file_path)
    2. save_json_file(save_directory, file_name, json_data)

    Args:
        func: The function to wrap

    Returns:
        Callable: Wrapped function with test-aware I/O

    Example:
        @test_aware_io
        def load_json_file(file_path):
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {}

        @test_aware_io
        def save_json_file(save_directory, file_name, json_data):
            os.makedirs(save_directory, exist_ok=True)
            file_path = os.path.join(save_directory, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2)
    """

    # Detect function type
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())

    if func.__name__ == "load_json_file" or "file_path" in param_names:
        return _wrap_load_json(func)
    elif func.__name__ == "save_json_file" or "save_directory" in param_names:
        return _wrap_save_json(func)
    else:
        raise ValueError(
            f"test_aware_io decorator doesn't recognize function signature "
            f"for {func.__name__}. Expected either load_json_file(file_path) "
            f"or save_json_file(save_directory, file_name, json_data)"
        )


def _wrap_load_json(func: Callable) -> Callable:
    """Wrap a load_json_file function."""

    @functools.wraps(func)
    def wrapper(file_path: str, *args, **kwargs) -> Any:
        is_test = TestContextDetector.is_test_context()
        resolved_path = PathResolver.resolve_file_path(file_path, is_test)

        if is_test:
            _log_test_redirect("[TEST CONTEXT] Load Redirect:", file_path, resolved_path)

        return func(resolved_path, *args, **kwargs)

    return wrapper


def _wrap_save_json(func: Callable) -> Callable:
    """Wrap a save_json_file function."""

    @functools.wraps(func)
    def wrapper(save_directory: str, file_name: str, json_data: Any, *args, **kwargs) -> None:
        is_test = TestContextDetector.is_test_context()
        resolved_dir, resolved_name = PathResolver.resolve_directory_and_filename(
            save_directory, file_name, is_test
        )

        if is_test:
            _log_test_redirect("[TEST CONTEXT] Save Redirect:",
                               os.path.join(save_directory, file_name),
                               os.path.join(resolved_dir, resolved_name))

        return func(resolved_dir, resolved_name, json_data, *args, **kwargs)

    return wrapper


def _log_test_redirect(operation: str, original_path: str, test_path: str) -> None:
    """Log when a file operation is redirected to test directory."""
    print(operation)
    print(f"  Original: {original_path}")
    print(f"  Test:     {test_path}")


# ============================================================================
# Standalone Helper Functions (if decorators aren't used)
# ============================================================================

def get_file_path_in_context(file_path: str) -> str:
    """
    Get the actual file path to use, accounting for test context.

    Args:
        file_path: Original file path

    Returns:
        str: Resolved file path
    """
    is_test = TestContextDetector.is_test_context()
    return PathResolver.resolve_file_path(file_path, is_test)


def get_directory_in_context(directory: str) -> str:
    """
    Get the actual directory to use, accounting for test context.

    Args:
        directory: Original directory path

    Returns:
        str: Resolved directory path
    """
    is_test = TestContextDetector.is_test_context()
    return PathResolver.resolve_directory(directory, is_test)


# ============================================================================
# Example Usage (before and after decorator)
# ============================================================================

# OPTION 1: Use the decorator (recommended)
@context_aware_io
def load_json_file(file_path: str) -> Dict:
    """Load JSON file from disk."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@context_aware_io
def save_json_file(save_directory: str, file_name: str, json_data: Dict) -> None:
    """Save JSON file to disk."""
    os.makedirs(save_directory, exist_ok=True)
    file_path = os.path.join(save_directory, file_name)
    print(f"Saving data to {file_path}...")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)


# ============================================================================
# OPTION 2: Manual usage without decorator (if you can't modify function definitions)
# ============================================================================

def load_json_file_manual(file_path: str) -> Dict:
    """Load JSON file with manual test detection (without decorator)."""
    # Manually resolve the path
    resolved_path = get_file_path_in_context(file_path)

    if os.path.exists(resolved_path):
        with open(resolved_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_json_file_manual(save_directory: str, file_name: str, json_data: Dict) -> None:
    """Save JSON file with manual test detection (without decorator)."""
    # Manually resolve the directory
    resolved_dir = get_directory_in_context(save_directory)

    os.makedirs(resolved_dir, exist_ok=True)
    file_path = os.path.join(resolved_dir, file_name)
    print(f"Saving data to {file_path}...")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)


# ============================================================================
# Convenience Functions for Common Paths
# ============================================================================

def load_vocab_file() -> Dict:
    """Load vocabulary file, using TEST_FILES_DIR if in test context."""
    return load_json_file(VOCAB_FILE)


def save_vocab_file(vocab_data: Dict) -> None:
    """Save vocabulary file, using TEST_FILES_DIR if in test context."""
    save_json_file(STORAGE_DIR, VOCAB_FILE_NAME, vocab_data)


def load_morpheme_file() -> Dict:
    """Load morpheme file, using TEST_FILES_DIR if in test context."""
    return load_json_file(MORPHEME_FILE)


def save_morpheme_file(morpheme_data: Dict) -> None:
    """Save morpheme file, using TEST_FILES_DIR if in test context."""
    save_json_file(STORAGE_DIR, 'morphemes.json', morpheme_data)


def load_word_frequencies() -> Dict:
    """Load word frequency file, using TEST_FILES_DIR if in test context."""
    return load_json_file(FILTERED_WORD_FREQ_FILE)


def save_word_frequencies(freq_data: Dict) -> None:
    """Save word frequency file, using TEST_FILES_DIR if in test context."""
    save_json_file(STORAGE_DIR, WORD_FREQ_FILE_NAME, freq_data)


def get_affix_by_length(affixes: Iterable[str], reverse=False) -> Dict[int, Set[str]]:
    """Group affixes by character length."""
    result: Dict[int, Set[str]] = defaultdict(set)
    for affix in sorted(affixes, reverse=reverse):
        result[len(affix)].add(affix)
    return result


def get_hash_key(sorted_dict, meta_str=""):
    # 1. Serialize to a compact JSON string (no whitespace)
    # Since it's already sorted, we just dump it
    serialized = ujson.dumps(sorted_dict, escape_forward_slashes=False) + meta_str

    # 2. Hash and encode to hex
    return blake3.blake3(serialized.encode()).hexdigest()

# ============================================================================
# Testing the Test Detection
# ============================================================================

if __name__ == "__main__":
    print("Testing Test Context Detection")
    print("=" * 60)

    # Test 1: Direct call (not in test context)
    is_test = TestContextDetector.is_test_context()
    print(f"\n1. Direct call in main module:")
    print(f"   Is test context: {is_test}")
    print(f"   Expected: False")

    # Test 2: Test file simulation (you can run this file with pytest)
    print(f"\n2. Current script: {__file__}")
    print(f"   Filename starts with 'test_': {Path(__file__).name.startswith('test_')}")

    # Test 3: Show paths that would be used
    print(f"\n3. Path resolution examples:")
    print(f"   TEST_FILES_DIR: {TEST_FILES_DIR}")
    print(f"   VOCAB_FILE: {VOCAB_FILE}")
    print(f"   Resolved (live): {get_file_path_in_context(VOCAB_FILE)}")

    # Test 4: Create test files directory
    print(f"\n4. Creating test files directory...")
    os.makedirs(TEST_FILES_DIR, exist_ok=True)
    print(f"   Created: {TEST_FILES_DIR}")

    print("\n" + "=" * 60)
    print("Test detection is working! Run your test files with pytest to see")
    print("the decorator in action.")