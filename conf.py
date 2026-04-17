import regex as re
import logging
import os
import time
from calendar import timegm
from pathlib import Path

PC_NAME = os.environ['COMPUTERNAME']
PARENT_DIR = os.path.dirname(__file__)
ROOT_DIR = str(Path(PARENT_DIR).parent)
CHECKPOINTS = str(os.path.join(ROOT_DIR, "checkpoints"))
DATASET_DIR = os.path.join(PARENT_DIR, "dataset")
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_STR_FORMAT = '%(asctime)s - %(name)s.%(threadName)s %(levelname)s: %(message)s'
WORKSTATION = os.path.join(ROOT_DIR, f"{PC_NAME}_{timegm(time.gmtime())}")
LOGS_FOLDER = os.path.join(WORKSTATION, 'logs')
TESTS_FOLDER = os.path.join(WORKSTATION, 'tests')
console_formatter = logging.Formatter(LOG_STR_FORMAT)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_formatter)
null_handler = logging.NullHandler()
null_handler.setLevel(logging.DEBUG)
null_handler.setFormatter(logging.Formatter(LOG_STR_FORMAT))
# Case-insensitive regex to extract file sizes with units MB, GB, KB, TB
SPLIT_UNICODE_WORDS = r"\d|\p{L}+|\p{M}+|\p{N}+|\p{P}+|\s|."
VALID_UNICODE_WORD = r"[\p{L}\p{M}\p{N}\s\p{P}]" # inhabitants of the place or their language.
UNICODE_BOOK_NAME_PATTERN = r"^[=~+\]\[\p{L}+\p{M}\p{N}\s\p{P}]{20,}+$"
WIKI_SECTION_PATTERN = r"===+([^=]+)===+"
LANGUAGE_PATTERN = r'^[^=]*==([\p{L}\p{M}\p{N}\s\p{P}]*)==[^=]*$'
PRESERVE_ALSO_PATTERN = r'.*{?{also[\\|](.*?)}?}.*'
PRESERVE_OTHER_PATTERN = r'(.+)'
# Back reference: (?:\/\1)
TAG_AND_TEXT_PATTERN = r".*<(title|comment|text)[a-zA-Z0-9=:\"\'\s]*?\/?>([^<>]*)(?:<\/\1>)?.*"
LANG_CODE_PATTERN = r".*{{(?:wikipedia\|lang=|alter\||audio\||hyphenation\|)([a-z]+).*"
WP_LANG_CODE_PATTERN = r".*{{(?:syn|lb|l|lang=|head|hyph[enation]*|alter|audio|IPA)\|([a-z]{2,3})\|.*"
FILE_SIZE_PATTERN = r"(?i)\b\d+(\.\d+)?(?:MB|GB|KB|TB)\b"
WORD_PATTERN = r"\b\w+\b"
ANY_CHAR_PATTERN = r"[\u0000-\uFFFF]"
UNICODE_CHAR_RANGE = 0x110000
MAX_UNICODE_VALUE = 1114111
# Books
LIBRARY_PATHS = ['lgli', 'lgrs', 'nexusstc', 'zlib', 'lgrsnf']
VERIFIED = "verified"
NON_FICTION_BOOKS = "libgen_rs_non_fic"
BOOKS_COLLECTION = "Books"
LANG = 'lang'
BOOK_NAME = 'book_name'
ORIGINAL_BOOK_NAME = 'original_book_name'
ORIGINAL_DESCRIPTION = 'original_description'
DESCRIPTION = 'description'
AUTHORS = 'authors'
MD5_HASH = 'md5_hash'
FILE_TYPE = 'file_type'
SIZE = 'size'
