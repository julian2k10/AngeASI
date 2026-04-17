import importlib
import inspect
import json
import math
import os
import pickle
import queue
import re
import sys
import bz2
import time
import uuid
import iso639
import logging
import concurrent.futures
from collections import defaultdict, Counter
from collections.abc import Iterable
from copy import deepcopy
from queue import Queue
from threading import Thread
import pymongo
import keyring
import platform
import hashlib
import torch
import torch.nn as nn
import regex
import numpy as np
import tempfile
from bson import ObjectId
from lxml import etree
from keyrings.cryptfile.cryptfile import CryptFileKeyring
from pathlib import Path
from pymongo import InsertOne, UpdateOne
from pymongo.common import MAX_BSON_SIZE
from pymongo.errors import WriteError
from model.conf import (
    ROOT_DIR, TAG_AND_TEXT_PATTERN, PRESERVE_ALSO_PATTERN, LANGUAGE_PATTERN,
    PRESERVE_OTHER_PATTERN, LANG_CODE_PATTERN, VALID_UNICODE_WORD, WP_LANG_CODE_PATTERN, WIKI_SECTION_PATTERN
)

# Configure logging
LOG_STR_FORMAT = '%(name)s.%(threadName)s %(levelname)s: %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=LOG_STR_FORMAT)
logger = logging.getLogger('AngeASI.utility')
APP_NAME = ""
ENCRYPT_KEY = ""
SYSTEM_NAME = "AngeAI"
CACHE_DIR = os.path.join(os.getcwd(), 'cache')
CACHE_META_FILE = os.path.join(CACHE_DIR, "cache_meta.json")
LATIN_GREEK_ROOTS_CACHE = {}


def get_language_name_from_part3(lang_code):
   """
   Looks up the language name for a given ISO 639-3 code.
   Args:
       lang_code (str): The ISO 639-3 language code.
   Returns:
       str: The language name if found, otherwise an empty string.
   """
   try:
       lang = iso639.Language.from_part3(lang_code)
       return lang.name
   except KeyError:
       return ""

def get_language_name_from_part2b(lang_code):
   """
   Looks up the language name for a given ISO 639-2B code.
   Args:
       lang_code (str): The ISO 639-2B language code.
   Returns:
       str: The language name if found, otherwise an empty string.
   """
   try:
       lang = iso639.Language.from_part2b(lang_code)
       return lang.name
   except KeyError:
       return ""

def get_language_name_from_part2t(lang_code):
   """
   Looks up the language name for a given ISO 639-2T code.
   Args:
       lang_code (str): The ISO 639-2T language code.
   Returns:
       str: The language name if found, otherwise an empty string.
   """
   try:
       lang = iso639.Language.from_part2t(lang_code)
       return lang.name
   except KeyError:
       return ""

def get_language_name_from_part1(lang_code):
   """
   Looks up the language name for a given ISO 639-1 code.
   Args:
       lang_code (str): The ISO 639-1 language code.
   Returns:
       str: The language name if found, otherwise an empty string.
   """
   try:
       lang = iso639.Language.from_part1(lang_code)
       return lang.name
   except KeyError:
       return ""

def get_all_iso639_languages():
    languages = iso639.ALL_LANGUAGES

    return languages

def load_env_variables():
    try:
        print("Loading environment values...")
        with open(os.path.join(ROOT_DIR, ".env"), mode="+r") as file:
            for line in file.readlines():
                seperator = line.find("=")
                key = line[:seperator]
                value = line[seperator + 1:].strip("\n")
                print(f"Loading environment value for: {key}...")
                save_secret(SYSTEM_NAME, key, value)
    except Exception as e:
        logger.error(f"{e} - Cannot load env variables!")

def extract_bz2(input_file, output_file, chunk_size_mb=100):
    chunk_size = chunk_size_mb * 1024 * 1024  # Convert MB to bytes
    with bz2.open(input_file, 'rb') as bz2f, open(output_file, 'wb') as outfile:
        while True:
            chunk = bz2f.read(chunk_size)
            if not chunk:
                break  # End of file
            outfile.write(chunk)

def process_large_xml_lxml(filename):
    context = etree.iterparse(filename, events=("start", "end"))
    root = None
    for event, elem in context:
        if event == "start":
            if root is None:
                root = elem
        elif event == "end":
            if elem.tag == "your_section_tag":
                process_section_lxml(elem)
            root.clear()
    del context

def process_section_lxml(section_element):
    title = section_element.xpath("./title/text()")[0] if section_element.xpath("./title/text()") else None
    content = section_element.xpath("./content/text()")[0] if section_element.xpath("./content/text()") else None
    print(f"Processing (lxml) section with title: {title}")
    # ... further processing using lxml's features ...

def view_xml_by_lines(filename, lines_per_chunk=50):
    with open(filename, 'r', encoding='utf-8') as file:
        while True:
            chunk = [file.readline() for _ in range(lines_per_chunk)]
            if not any(chunk):  # Check if the chunk is empty (end of file)
                break
            print("".join(chunk))
            input("Press Enter to view the next chunk...")

def parse_bz2_xml_pages_to_db(file_name):
    from model.util import get_local_mongodb_manager
    database = get_local_mongodb_manager()
    try:
        database.connect()
        with bz2.open(file_name, 'r') as file:
            meta_data = {}
            pages = []
            page = []
            save = False
            xml_name = str(file_name.split('.')[0]).replace('-', '_')
            collection = database.get_collection(xml_name)
            # Create index for page number
            collection.create_index(
                [('page_number', pymongo.ASCENDING)],
                name="page_number_index",
                background=True
            )

            languages_collection = database.get_collection(f"{xml_name}_languages")
            # Create index for page number
            languages_collection.create_index(
                [('page_number', pymongo.ASCENDING)],
                name="page_number_index",
                background=True
            )
            languages = []
            line = ""
            lang_code = ""
            page_number = 1
            count = 0
            while True:
                prev_line = line
                line = file.readline().decode(encoding='utf-8')
                if not line:  # Check if the chunk is empty (end of file)
                    break

                line = line.strip()
                if line.startswith("<page"):
                    save = True
                    continue
                if save:
                    if "</page" in line:
                        save = False
                        insert_op = InsertOne({
                            "meta_data": meta_data,
                            "page": page,
                            'page_number': str(page_number)
                        })
                        pages.append(insert_op)
                        lang_code = ""
                        page = []
                        meta_data = {}
                        page_number += 1
                        count = 0
                        if len(pages) >= 1000:
                            # Save wiki pages
                            collection.bulk_write(pages, ordered=False)
                            pages = []
                    else:
                        if count < 20 and line.startswith('<'):
                            match = regex.match(TAG_AND_TEXT_PATTERN, line)
                            count += 1
                            if not match:
                                if "<title" in line:
                                    logger.error(f"Cannot match title. Using entire line: {line}")
                                    meta_data['title'] = str(line)
                                elif "<comment" in line:
                                    logger.error(f"Cannot match comment. Using entire line: {line}")
                                    meta_data['comment'] = str(line)
                                elif "<text" in line:
                                    logger.error(f"Cannot match text. Using entire line: {line}")
                                    meta_data['text'] = str(line)
                                continue

                            meta_data[match.group(1)] = str(match.group(2))
                            if match.group(1) == "text":
                                lang_match = regex.search(LANGUAGE_PATTERN, match.group(2))
                                if lang_match:
                                    lang = lang_match.group(2)
                                    meta_data['lang'] = str(lang)
                                also_match = regex.match(PRESERVE_ALSO_PATTERN, match.group(2))
                                if also_match:
                                    meta_data['preserve_also'] = str(also_match.group(1))
                        else:
                            page.append(str(line))
                            match = regex.search(WP_LANG_CODE_PATTERN, line)
                            lang_match = regex.match(LANGUAGE_PATTERN, prev_line)
                            if match and lang_match:
                                lang_name = lang_match.group(2)
                                lang_code = match.group(1)
                                query = {"lang_name": str(lang_name), "lang_code": str(lang_code)}
                                update = {
                                    "$set": {
                                        "lang_name": str(lang_name),
                                        "lang_code": str(lang_code),
                                    },
                                    "$addToSet": {'page_number': str(page_number)}
                                }
                                update_op = UpdateOne(query, update, upsert=True)
                                languages.append(update_op)

                            if meta_data.get('lang') and not lang_code:
                                match = regex.search(LANG_CODE_PATTERN, line)
                                if match:
                                    lang_code = match.group(1)
                                    meta_data['lang_code'] = str(lang_code)

                            if len(languages) > 1000:
                                languages_collection.bulk_write(languages, ordered=False)
                                languages = []
                else:
                    continue

            if len(languages) > 0:
                languages_collection.bulk_write(languages, ordered=False)

            if len(pages) > 0:
                # Save remaining wiki pages
                collection.bulk_write(pages, ordered=False)

    finally:
        database.disconnect()

def sort_by_length_and_alphabetical(word_list):

    """

    Sorts a list of strings first by length (shortest to longest)

    and then alphabetically within each length group.

    Args:

        word_list: A list of strings.

    Returns:

        A new list of strings sorted as described.

    """

    return sorted(word_list, key=lambda word: (len(word), word))

def view_xml_by_pages(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        page = []
        pages = []
        extra = []
        save = False
        count = 0
        metadata = []
        while True:
            line = file.readline()
            if not line:  # Check if the chunk is empty (end of file)
                break
            if "<page>" in line:
                save = True
            if save:
                page.append(line)
                if count < 20:
                    metadata.append(line)
                    count += 1
                if "</page>" in line:
                    save = False
                    #print("".join(page))
                    #print("#########################################################################################################\n\n")
                    #input("Press Enter to view the next page...")
                    pages.append("".join(metadata))
                    page = []
                    metadata = []
                    count = 0
            else:
                extra.append(line)
                continue

        print(f"Total Pages={len(pages)}:")
        with open("metadata_file.txt", 'a', encoding='utf-8') as metafile:
            metafile.writelines(pages)

        print(f"Extra Pages={len(extra)}:")
        with open("extra_lines.txt", 'a', encoding='utf-8') as extra_file:
            extra_file.writelines(extra)

def create_wiktionary_indexes():
    from model.util import get_local_mongodb_manager
    database = get_local_mongodb_manager()
    try:
        database.connect()
        wiktionary_latest_pages_articles = database.get_collection("wiktionary-latest-pages-articles")
        wiktionary_vocabulary_words = database.get_collection(f'wiktionary_vocabulary_words')
        wiktionary_other_articles = database.get_collection(f'wiktionary_other_articles')

        # wiktionary-latest-pages-articles index
        wiktionary_latest_pages_articles.create_index(
            [('page_number', pymongo.ASCENDING)],
            name="page_number_index",
            background=True
        )
        # wiktionary_vocabulary_words index
        key = 'lang'
        wiktionary_vocabulary_words.create_index(
            [(key, pymongo.ASCENDING)],
            name=f"{key}_index",
            background=True
        )
        wiktionary_vocabulary_words.create_index(
            [('page_number', pymongo.ASCENDING)],
            name="page_number_index",
            background=True
        )
        # wiktionary_other_articles index
        key = 'other_articles'
        wiktionary_other_articles.create_index(
            [(key, pymongo.ASCENDING)],
            name=f"{key}_index",
            background=True
        )
        wiktionary_other_articles.create_index(
            [('page_number', pymongo.ASCENDING)],
            name="page_number_index",
            background=True
        )

    except Exception as e:
        logger.exception(f"Cannot create wiktionary indexes - {e}")
    finally:
        database.disconnect()

def sort_ancestry_key(language_info):
    population = language_info.get("population", "0")
    return int(population)

def select_language_ancestor(ancestor, query) -> dict:
    if len(ancestor) > 1:
        # sort ancestry with high frequency languages at top of list
        ancestor = sorted(ancestor, key=sort_ancestry_key, reverse=True)
        logger.debug(f"Multiple ancestor found for: {query}. sorted ancestor={ancestor}")
        if len(ancestor) > 2:
            # Select ancestry with most family matches
            families = Counter()
            for _ancestor in ancestor:
                families.update([_ancestor["ancestry"][0]])
            family = families.most_common(1)[0][0]
            for _ancestor in ancestor:
                if _ancestor["ancestry"][0] == family:
                    ancestor = _ancestor
                    break
        else:
            # Select ancestry with most branches
            if ancestor[0]["ancestry"][0] == ancestor[1]["ancestry"][0]:
                ancestor = ancestor[0]
            else:
                ancestor = ancestor[1]
    else:
        ancestor = ancestor[0]

    return ancestor

def create_wiktionary_vocab_list():
    from model.util import get_local_mongodb_manager
    database = get_local_mongodb_manager()
    tokenizer_database = get_local_mongodb_manager("UnicodeTokenizer")
    iso_639_code_database = get_local_mongodb_manager("ISO-639_Language_Codes")
    try:
        database.connect()
        tokenizer_database.connect()
        iso_639_code_database.connect()
        collection_name = "enwiktionary_latest_pages_articles"
        wiktionary_vocabulary_list = database.get_collection('wiktionary_vocabulary_list')
        languages_collection = tokenizer_database.get_collection("languages")
        record_queue, total_records = database.get_all_records(collection_name)
        results = {}
        overflow = {}
        pattern = r"[\s\p{P}]"
        values_to_add = {}
        failed_updates = {}
        invalid_pages = []
        invalid_title = []
        record_ids = {}
        iso_639_codes = {}
        invalid_names = set()
        verified_names = dict()
        for _ in range(total_records):
            record = record_queue.get(timeout=300)
            page_number = record.get('page_number')
            meta_data = record.get('meta_data', {})
            code = meta_data.get('lang_code')
            lang = meta_data.get('lang')
            word = meta_data.get('title')
            if word.startswith('Wiktionary'):
                logger.debug(f"Skipping invalid Wiktionary page: {page_number}...")
                invalid_pages.append(str(page_number))
                continue
            code_name = f"{lang} [{code}]"
            lang_name = verified_names.get(code)
            verified = False
            if lang_name:
                lang = lang_name
                logger.debug(f"Using verified name={lang_name} for code={code}...")
            # Save iso 639 codes
            if lang and code and code_name not in invalid_names and code not in iso_639_codes:
                parts = regex.split(r"[\s\p{P}]", lang)
                name = "|".join(parts)
                ancestry_pattern = {"$in": parts}
                # {'ancestry': {'$elemMatch': {'$regex': 'Estonian'}}}
                query = {
                    "language_code": {"$regex": rf"^{code[:2]}[a-z]$"},
                    "ancestry": ancestry_pattern
                }
                ancestor = list(languages_collection.find(query))
                queries = [query]

                # Search with name only
                if len(ancestor) < 1:
                    logger.warning(f"Using name only query for: {name}...")
                    query = {
                        "ancestry": ancestry_pattern
                    }
                    queries.append(query)
                    ancestor = list(languages_collection.find(query))

                # Search with code only
                if len(ancestor) < 1:
                    logger.warning(f"Using code & partial name query for: {code}...")
                    query = {
                        "language_code": {"$regex": rf"^{code[:2]}[a-z]$"},
                        "language_name": {"$regex": rf"^{lang[:3]}.*+$"}
                    }
                    queries.append(query)
                    ancestor = list(languages_collection.find(query))
                if len(ancestor) > 0:
                    ancestor = select_language_ancestor(ancestor, query)
                    description = ancestor["description"]
                    language_name = ancestor["language_name"]
                    language_code = ancestor["language_code"]
                    ancestry = ancestor["ancestry"]
                    verified = True
                    verified_names[code] = str(name)
                    iso_639_codes[code] = {
                        "iso_639_1": str(code),
                        "iso_639_3": str(language_code),
                        "name": str(language_name),
                        "ancestry": list(ancestry),
                        "description": str(description),
                        "verified": bool(verified)
                    }
                else:
                    logger.error(f"no language ancestor found for {code_name} with queries: {queries}")
                    invalid_names.add(code_name)

            if not regex.search(VALID_UNICODE_WORD, word):
                logger.debug(f"Skipping invalid unicode word={word} on page={page_number}")
                invalid_title.append(str(page_number))
                continue
            try:
                words = regex.split(pattern, word.lower())
            except TypeError:
                logger.error(f"Cannot split word from title: {word} on page #: {page_number}")
                continue
            if lang and code:
                lang_name, count = overflow.get(lang, (lang, 0))
            else:
                logger.debug(f"Skipping invalid language page={page_number}")
                invalid_pages.append(str(page_number))
                continue

            if lang_name in failed_updates:
                failed_updates[lang_name].update(words)
                continue

            try:
                # Use ISO-639-3 code if available
                code = iso_639_codes.get(code, {"iso_639_3": code})["iso_639_3"]
                verified = iso_639_codes.get(code, {"verified": False})["verified"]
                results[lang_name]['words'].update(words)
                if len(results[lang_name]['words']) >= 1000:
                    if record_ids.get(code):
                        query = {"_id": record_ids[code]}
                        record = wiktionary_vocabulary_list.find_one(query)
                    else:
                        record = None
                        query = {}

                    values_to_add = {x for x in results[lang_name]['words'] if len(x) > 1}
                    results[lang_name]['words'] = set()
                    if record:
                        try:
                            values_to_add.update(set(record.get('words')))
                        except TypeError:
                            logger.error(f"Cannot load '{lang_name}' with query={query} and record={record}")
                        values_to_add = list(values_to_add)
                        if len(values_to_add) > 100000:
                            values_to_add_1 = values_to_add[:100000]
                            values_to_add_2 = values_to_add[100000:]
                            # Insert first 100k words
                            record = {'words': values_to_add_1, 'lang_name': lang_name, 'lang_code': code, 'verified': verified}
                            wiktionary_vocabulary_list.replace_one(query, record, upsert=True)
                            # Insert remaining words
                            record = {'words': values_to_add_2, 'lang_name': lang_name, 'lang_code': code, 'verified': verified}
                            result = wiktionary_vocabulary_list.insert_one(record)
                            record_ids[code] = result.inserted_id
                        else:
                            record = {'words': values_to_add, 'lang_name': lang_name, 'lang_code': code, 'verified': verified}
                            wiktionary_vocabulary_list.replace_one(query, record, upsert=True)
                    else:
                        record = {'words': list(values_to_add), 'lang_name': lang_name, 'lang_code': code, 'verified': verified}
                        result = wiktionary_vocabulary_list.insert_one(record)
                        record_ids[code] = result.inserted_id
            except WriteError:
                logger.error(f"Cannot save langauge: {lang_name}")
                failed_updates[lang_name] = set(values_to_add)
                results[lang_name] = {'words': set(), 'lang_code': code, 'verified': verified}
            except KeyError:
                results[lang_name] = {'words': set(words), 'lang_code': code, 'verified': verified}

        # Save final results
        for lang_name in results:
            if len(results[lang_name]['words']) < 1:
                continue
            try:
                code = results[lang_name]['lang_code']
                verified = results[lang_name]['verified']
                if record_ids.get(lang_name):
                    query = {"_id": record_ids[lang_name]}
                    record = wiktionary_vocabulary_list.find_one(query)
                else:
                    record = None
                    query = {}

                values_to_add = {x for x in results[lang_name]['words'] if len(x) > 1}
                results[lang_name]['words'] = set()
                if record is not None:
                    try:
                        values_to_add.update(set(record.get('words')))
                    except TypeError:
                        logger.error(f"Cannot load '{lang_name}' with query={query} and record={record}")
                    values_to_add = list(values_to_add)
                    if len(values_to_add) > 100000:
                        values_to_add_1 = values_to_add[:100000]
                        values_to_add_2 = values_to_add[100000:]
                        # Insert first 100k words
                        record = {'words': values_to_add_1, 'lang_name': lang_name, 'lang_code': code, 'verified': verified}
                        wiktionary_vocabulary_list.replace_one(query, record, upsert=True)
                        # Insert remaining words
                        record = {'words': values_to_add_2, 'lang_name': lang_name, 'lang_code': code, 'verified': verified}
                        result = wiktionary_vocabulary_list.insert_one(record)
                        record_ids[lang_name] = result.inserted_id
                    else:
                        record = {'words': values_to_add, 'lang_name': lang_name, 'lang_code': code, 'verified': verified}
                        wiktionary_vocabulary_list.replace_one(query, record, upsert=True)
                else:
                    record = {'words': list(values_to_add), 'lang_name': lang_name, 'lang_code': code, 'verified': verified}
                    result = wiktionary_vocabulary_list.insert_one(record)
                    record_ids[lang_name] = result.inserted_id
            except WriteError:
                logger.error(f"Cannot save langauge: {lang_name}")
                failed_updates[lang_name] = set(values_to_add)
                results[lang_name]['words'] = set()

        # Save iso_639_codes
        updates = []
        iso_639_1_collection = iso_639_code_database.get_collection("ISO-639-1")
        for code in iso_639_codes:
            query = {"iso_639_1": code}
            update = {"$set": iso_639_codes[code]}
            updates.append(UpdateOne(query, update, upsert=True))

        logger.info(f"Updating {len(updates)} ISO-639-1 codes")
        iso_639_1_collection.bulk_write(updates)

        # Save invalid title
        collection = database.get_collection(f"{collection_name}_invalid_data")
        if sys.getsizeof(invalid_title) > MAX_BSON_SIZE:
            values = []
            current_size = 0
            for value in invalid_title:
                size = sys.getsizeof(value)
                current_size += size
                if size > MAX_BSON_SIZE:
                    collection.insert_one({'invalid_title': values})
                    current_size = size
                    values = [value]
                else:
                    values.append(value)
        else:
            collection.insert_one({'invalid_title': invalid_title})

        # Save invalid pages
        if sys.getsizeof(invalid_pages) > MAX_BSON_SIZE:
            values = []
            current_size = 0
            for value in invalid_pages:
                size = sys.getsizeof(value)
                current_size += size
                if size > MAX_BSON_SIZE:
                    collection.insert_one({'invalid_pages': values})
                    current_size = size
                    values = [value]
                else:
                    values.append(value)
        else:
            collection.insert_one({'invalid_pages': invalid_pages})

        # Save failed writes to file
        for idx, lang_name in enumerate(failed_updates):
            with open(f"failed_updates_{idx}.txt", "w", encoding="utf-8") as file:
                file.writelines(failed_updates[lang_name])
    except Exception as e:
        logger.exception(f"Cannot create wiktionary indexes - {e}")
    finally:
        database.disconnect()
        tokenizer_database.disconnect()
        iso_639_code_database.disconnect()

def generate_sha256_hash(string_value):
    """
    Generates the SHA256 hash of a given string.

    Args:
        string_value: The string to hash.

    Returns:
        The SHA256 hash as a hexadecimal string.
    """
    hash_object = hashlib.sha256(string_value.encode('utf-8'))
    hex_digest = hash_object.hexdigest()

    return hex_digest

def generate_encryption_key():
    """
    Attempts to get a unique value related to the network domain.

    This function prioritizes getting the Active Directory objectGUID if available,
    otherwise, it falls back to a combination of hostname and a generated UUID.

    Note: Getting the objectGUID reliably requires Active Directory and appropriate permissions.
    """

    try:
        return generate_sha256_hash("-".join((uuid.NAMESPACE_DNS.hex, ENCRYPT_KEY)))
    except Exception as e:
        logger.exception(f"Error getting hostname or generating UUID: {e}")
        return None

def generate_unique_key():
    """Generates a unique key based on the current machine's hardware and OS."""
    machine_id = uuid.getnode()
    os_name = platform.system()
    combined_id = f"{machine_id}-{os_name}"
    return combined_id.encode('utf-8')

def get_keyring_path():
    """Generates a unique path for the keyring file based on the machine ID."""
    key_id = generate_unique_key().decode('utf-8')
    return os.path.join(os.path.expanduser("~"), f".{APP_NAME}_keyring_{key_id}.crypt")

def setup_keyring():
    """Sets up the CryptFileKeyring with a unique path."""
    secure_env = CryptFileKeyring()
    filename = get_keyring_path()
    secure_env.filename = filename
    secure_env.keyring_key = generate_encryption_key()

    return secure_env

def save_secret(service_name, username, password):
    """Saves a secret to the encrypted keyring."""
    try:
        keyring.set_password(service_name, username, password)
    except Exception as e:
        logger.exception(f"Cannot saving secret: {e}")

def load_secret(service_name, username):
    """Loads a secret from the encrypted keyring."""
    try:
        backend = setup_keyring()
        secret = keyring.get_password(service_name, username)
        if secret:
            logger.debug(f"Secret loaded from keyring: {backend.filename}")
            return secret
        else:
            logger.debug("Secret not found.")
            return None
    except Exception as e:
        logger.exception(f"Cannot loading secret: {e}")
        return None

def save_skill_keywords_to_mongodb(skill_keywords, mongo_uri="mongodb://localhost:27017/", db_name="asi", collection_name="skill_keywords"):
    """
    Saves a dictionary of skill keywords to a MongoDB collection, ensuring no duplicate values.

    Args:
        skill_keywords (dict): Dictionary where keys are skill categories and values are lists of keywords.
        mongo_uri (str): MongoDB connection URI.
        db_name (str): Name of the database.
        collection_name (str): Name of the collection.
    """
    try:
        client = pymongo.MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]

        for skill, keywords in skill_keywords.items():
            # Check if the skill already exists
            existing_doc = collection.find_one({"skill": skill})

            if existing_doc:
                # Update existing document, removing duplicates
                existing_keywords = list(existing_doc.get("keywords", []))
                unique_keywords = list(set(existing_keywords + list(keywords))) #remove duplicates
                collection.update_one({"skill": skill}, {"$set": {"keywords": unique_keywords}})
            else:
                # Insert new document
                collection.insert_one({"skill": skill, "keywords": list(set(keywords))}) #remove duplicates on insert.

        logger.debug(f"Skill keywords saved to MongoDB: {mongo_uri}, database: {db_name}, collection: {collection_name}")

    except pymongo.errors.ConnectionFailure as e:
        logger.debug(f"Error connecting to MongoDB: {e}")
    except pymongo.errors.OperationFailure as e:
        logger.debug(f"Error saving skill keywords to MongoDB: {e}")
    finally:
        if 'client' in locals(): #check if client was created.
           client.close()

def load_skill_keywords_from_mongodb(mongo_uri="mongodb://localhost:27017/", db_name="asi", collection_name="skill_keywords"):
    """
    Loads skill keywords from a MongoDB collection into a dictionary.

    Args:
        mongo_uri (str): MongoDB connection URI.
        db_name (str): Name of the database.
        collection_name (str): Name of the collection.

    Returns:
        dict: A dictionary containing the skill keywords.
    """
    skill_keywords = {}
    try:
        client = pymongo.MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        documents = collection.find()
        for document in documents:
            skill_keywords[document["skill"]] = tuple(document["keywords"])

        return skill_keywords

    except pymongo.errors.ConnectionFailure as e:
        logger.debug(f"Error connecting to MongoDB: {e}")
        return skill_keywords
    except pymongo.errors.OperationFailure as e:
        logger.debug(f"Error loading skill keywords from MongoDB: {e}")
        return skill_keywords
    finally:
        if 'client' in locals(): #check if client was created.
            client.close()

def reshape_training_data(model, train_data):
    """
    Reshapes training data to match the expected input shape of a Keras model.

    Args:
        model: A Keras model.
        train_data: The training data (NumPy array or TensorFlow tensor).

    Returns:
        The reshaped training data.
    """

    import tensorflow as tf
    logger.debug(f"Reshaping training data={train_data.shape} for model={model.input_shape}...")
    input_shape = model.input_shape[1:]  # Get input shape, excluding batch dimension
    if not input_shape:
      raise ValueError("Model input shape is not defined. Cannot reshape data.")

    if len(input_shape) == 1: #1D input
        expected_size = input_shape[0]

        if isinstance(train_data, np.ndarray):
            if len(train_data.shape) == 1:
              if train_data.shape[0] != expected_size:
                if train_data.shape[0] < expected_size:
                  padded_data = tf.pad(tf.expand_dims(train_data, axis=0), [[0, 0], [0, expected_size - train_data.shape[0]]]).numpy()[0]
                  return padded_data
                else:
                  return train_data[:expected_size]
              return train_data #already correct size
            else:
              raise ValueError("Input data should be 1D for this model.")

        elif isinstance(train_data, tf.Tensor):
            if len(train_data.shape) == 1:
              if train_data.shape[0] != expected_size:
                if train_data.shape[0] < expected_size:
                  padded_data = tf.pad(tf.expand_dims(train_data, axis=0), [[0, 0], [0, expected_size - train_data.shape[0]]])[0]
                  return padded_data
                else:
                  return train_data[:expected_size]
              return train_data #already correct size

            else:
              raise ValueError("Input data should be 1D for this model.")

        else:
            raise TypeError("train_data must be a NumPy array or TensorFlow tensor.")

    elif len(input_shape) == 2: #2D input (e.g., sequences)
        # Handle 2D input (sequences)
        expected_seq_length = input_shape[0]
        feature_dim = input_shape[1]

        if isinstance(train_data, np.ndarray):
            if len(train_data.shape) == 2:
                if train_data.shape[1] != feature_dim:
                    raise ValueError(f"Feature dimension of input data ({train_data.shape[1]}) does not match model's expected dimension ({feature_dim})")

                if train_data.shape[0] != expected_seq_length:
                    if train_data.shape[0] < expected_seq_length:
                        padded_data = np.pad(train_data, ((0, expected_seq_length - train_data.shape[0]), (0, 0)), 'constant')
                        return padded_data
                    else:
                        return train_data[:expected_seq_length]
                return train_data
            else:
                raise ValueError("Input data should be 2D for this model.")

        elif isinstance(train_data, tf.Tensor):
             if len(train_data.shape) == 2:
                if train_data.shape[1] != feature_dim:
                    raise ValueError(f"Feature dimension of input data ({train_data.shape[1]}) does not match model's expected dimension ({feature_dim})")
                if train_data.shape[0] != expected_seq_length:
                    if train_data.shape[0] < expected_seq_length:
                        padded_data = tf.pad(train_data, [[0, expected_seq_length - train_data.shape[0]], [0, 0]])
                        return padded_data
                    else:
                        return train_data[:expected_seq_length]
                return train_data
             else:
                raise ValueError("Input data should be 2D for this model.")

        else:
            raise TypeError("train_data must be a NumPy array or TensorFlow tensor.")
    else:
        raise ValueError("Model input shape is not supported by this reshape function.")

def create_instance(module_path, class_name, *args, package_name=None, **kwargs):
    """
    Dynamically creates an instance of a class given its module path, class name,
    and constructor arguments.

    Args:
        module_path (str): The dot-separated path to the module containing the class.
        class_name (str): The name of the class to instantiate.
        *args: Positional arguments to pass to the class constructor.
        package_name: The 'package' argument is required when performing a relative import. It specifies
            the package to use as the anchor point from which to resolve the relative import to an absolute import
        **kwargs: Keyword arguments to pass to the class constructor.

    Returns:
        object: An instance of the specified class.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class cannot be found in the module.
    """
    try:
        module = importlib.import_module(module_path, package_name)
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)
    except Exception as e:
        logger.exception(f"Cannot load class='{class_name}' from: '{module_path}'! - {e}")

def bytes_to_megabytes(bytes_size: int):
    try:
        return math.floor(bytes_size / (1024**2))
    except Exception as e:
        logger.exception(f"Cannot convert bytes='{bytes} to MB - {e}")
        return 0

def calculate_max_saved_model_size(model: nn.Module) -> int:
    """
    Calculates the maximum size of a saved PyTorch transformer model.

    Args:
        model: A torch.nn.Module representing the transformer model.

    Returns:
        The maximum size of the saved model in megabytes.
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.pth")
            torch.save(model.state_dict(), model_path)
            model_size = int(os.path.getsize(model_path))
            return bytes_to_megabytes(model_size)
    except Exception as e:
        logger.exception(f"Cannot calculate the maximum size of the saved transformer model - {e}")
        return 0

def estimate_max_transformer_memory(model, input_shape, dtype=torch.float32, optimizer='adam'):
    """
    Estimates the maximum memory usage of a torch.nn Transformer model during training.

    Args:
        model (nn.Module): The Transformer model.
        input_shape (tuple): A tuple representing the shape of the input data (batch_size, sequence_length).
        dtype (torch.dtype): The data type used for the model and data (default: torch.float32).
        optimizer (str): The optimizer used for training ('adam', 'sgd', etc.).

    Returns:
        float: The estimated maximum memory usage in megabytes.
    """

    batch_size, sequence_length = input_shape
    device = next(model.parameters()).device  # Get the device from the model

    # Calculate model parameters size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    encoders = []
    if getattr(model, 'encoder', None):
        encoders.append(getattr(model, 'encoder'))
    elif getattr(model, 'transformer', None):
        encoders.append(getattr(model, 'transformer'))
    else:
        num_layers = len(model.transformer_layers)
        for encoder_layer in model.transformer_layers:
            encoders.append(nn.TransformerEncoder(encoder_layer, num_layers))

    activation_size = 0  # Calculate activation size (simplified estimation)
    element_size = torch.tensor(1, dtype=dtype).element_size()
    for encoder in encoders:
        hidden_size = encoder.layers[0].attn.out_proj.out_features
        activation_size += (batch_size * sequence_length * hidden_size * element_size * encoder.num_layers)

    gradient_size = param_size  # Calculate gradient size (same as parameter size)

    # Calculate optimizer state size (approximation)
    if optimizer.lower() == 'adam':
        optimizer_state_size = param_size * 2  # Adam stores 2 buffers (momentum, variance)
    elif optimizer.lower() == 'sgd':
        optimizer_state_size = param_size  # SGD stores 1 buffer (momentum, if used)
    else:
        optimizer_state_size = param_size #default to one buffer.

    total_memory = param_size + activation_size + gradient_size + optimizer_state_size

    return bytes_to_megabytes(total_memory)

def try_estimate_max_transformer_memory():
    max_len = 128
    embedding_dim = 64
    hidden_dim = 128
    vocab_size = 10000
    num_labels = 12

    sequence_length = 128
    batch_size = 32
    complex_model_path = 'model.ange_transformer_model_final_testing'
    simple_model_path = 'database.app'
    simple_cle = 'SimpleTransformer'
    cls = 'SegmentedTransformer'
    args = (vocab_size, embedding_dim, hidden_dim, num_labels, max_len)
    simple_model = create_instance(simple_model_path, simple_cle, *args)
    complex_model = create_instance(complex_model_path, cls)
    input_shape = (batch_size, sequence_length)

    memory_usage = estimate_max_transformer_memory(simple_model, input_shape)
    logger.info(f"Estimated maximum memory usage simple model: {memory_usage} MB")

    memory_usage_fp16 = estimate_max_transformer_memory(simple_model, input_shape, dtype=torch.float16)
    logger.info(f"Estimated maximum memory usage (fp16) simple model: {memory_usage_fp16} MB")

    memory_usage_sgd = estimate_max_transformer_memory(simple_model, input_shape, optimizer="sgd")
    logger.info(f"Estimated maximum memory usage (SGD) simple model: {memory_usage_sgd} MB")

    ###########################################################################################################

    memory_usage = estimate_max_transformer_memory(complex_model, input_shape)
    logger.info(f"Estimated maximum memory usage complex model: {memory_usage} MB")

    memory_usage_fp16 = estimate_max_transformer_memory(complex_model, input_shape, dtype=torch.float16)
    logger.info(f"Estimated maximum memory usage (fp16) complex model: {memory_usage_fp16} MB")

    memory_usage_sgd = estimate_max_transformer_memory(complex_model, input_shape, optimizer="sgd")
    logger.info(f"Estimated maximum memory usage (SGD) complex model: {memory_usage_sgd} MB")

def parse_tab_separated_data(file_path, headers=None):
    """
    Efficiently parses data from a tab-separated text file.
    Args:
        file_path (str): The path to the text file.
        headers (dict): A dictionary with header replacements as key/value pairs
    Returns:
        list: A list of dictionaries, where each dictionary represents a row
              and the keys are the column headers.
    """

    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split('\t')  # Split header by double tabs
            if isinstance(headers, dict):
                header = [headers.get(x, x) for x in header]
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    values = line.split('\t')  # Split data rows by double tabs
                    if len(values) == len(header):
                        row_dict = dict(zip(header, values))
                        data.append(row_dict)
                    else:
                        logger.warning(f"Skipping line due to inconsistent number of columns: {line}")
    except FileNotFoundError:
        logger.error(f"File not found at {file_path}")

    return data

def save_table_to_mongodb(table_name: str, db_name: str, table: list[dict], indexes: Iterable="", query_keys=None):
    """
    Save table data to mongo database.

    :param table_name: name of the mongodb collection for which to save data
    :param db_name: name of the mongo database
    :param table: list of dictionaries for which to add to the database
    :param indexes: list of keys for which to create indexes for fast lookup.
    :param query_keys: list of keys to use when building query from table to update database.

    :return: None
    """
    from model.util import get_local_mongodb_manager
    database = get_local_mongodb_manager(db_name)
    if query_keys is None:
        query_keys = []
    try:
        database.connect()
        collection = database.get_collection(table_name)
        # Create indexes for fast lookup
        for key in indexes:
            collection.create_index(
                [(key, pymongo.ASCENDING)],
                name=f"{key}_index",
                background=True
            )

        # Add data to database
        records = []
        for data in table:
            query = {x: data[x] for x in query_keys} or data
            update = {"$set": data}
            records.append(UpdateOne(query, update, upsert=True))

        collection.bulk_write(records)

    except Exception as e:
        logger.exception(f"Cannot save table to mongodb. - {e}")
    finally:
        database.disconnect()

def extract_languages_codes_data():
    # Extract country codes
    filepath = 'scrapers/Language_Code_Data_20250221/CountryCodes.tab'
    new_headers = {'Area': 'area', 'CountryID': 'country_id', 'Name': 'country_name'}
    parsed_data = parse_tab_separated_data(filepath, new_headers)
    table_name = 'CountryCodes'
    database = "Language_Codes_Data"
    index = ['area', 'country_id', 'name']
    save_table_to_mongodb(table_name, database, parsed_data, index)

    # Extract language codes
    filepath = 'scrapers/Language_Code_Data_20250221/LanguageCodes.tab'
    new_headers.update({'LangID': 'lang_code', 'LangStatus': 'lang_status', 'Name': 'lang_name'})
    parsed_data = parse_tab_separated_data(filepath, new_headers)
    table_name = 'LanguageCodes'
    database = "Language_Codes_Data"
    index = ['lang_code', 'country_id', 'lang_name']
    save_table_to_mongodb(table_name, database, parsed_data, index)

    # Extract language index
    filepath = 'scrapers/Language_Code_Data_20250221/LanguageIndex.tab'
    new_headers.update({'NameType': 'name_type'})
    parsed_data = parse_tab_separated_data(filepath, new_headers)
    table_name = 'LanguageIndex'
    database = "Language_Codes_Data"
    index = ['lang_code', 'country_id', 'name_type', 'lang_name']
    save_table_to_mongodb(table_name, database, parsed_data, index)

def extract_lexical_data(document, results_queue: Queue=None):
    """
       Extracts definitions, translations, hyponyms, and derived terms
       organized by language and section from mongodb document.
       Args:
           document (dict): A dictionary containing the lexical information.
           results_queue (Queue): a queue for which to add the results once complete.
       Returns:
           dict: A dictionary with lexical data organized by language and section.
       """
    page_content = document.get("page", [])
    page_number = document.get("page_number")
    word = document.get("meta_data", {}).get("title")
    lang_name = document.get("meta_data", {}).get("lang")
    pattern_start = r"#*[\s\*†\|\{\[\:\] ]+|[lQm0-9]+}}|senseid\||desc\||"
    pattern_end = r"\||RQ\||syn\||lb\||tax[fmtlink]+\||([^\|\(\)\[\]\{\}]+)"
    extracted_data = {'word': word, 'page_number': page_number}
    if lang_name and lang_name not in extracted_data:
        extracted_data[lang_name] = {'lang_code': ""}
    current_section = ""
    current_group = "Etymology"
    definition = ""
    lang_code = ""
    embedded_text_pattern = pattern_start + str(lang_code) + pattern_end
    for i, line in enumerate(page_content):
        if i >= 100 and not lang_code:
            extracted_data = {}
            break
        # Identify language sections
        lang_match = regex.match(LANGUAGE_PATTERN, line)
        if lang_match:
            lang_name = lang_match.group(1).strip()
            current_section = ""
            lang_code = ""
            if lang_name not in extracted_data:
                extracted_data[lang_name] = {'lang_code': str(lang_code)}
                current_group = "Etymology"
            continue

        if lang_name and word:
            if not lang_code:
                code_match = re.match(WP_LANG_CODE_PATTERN, line)
                if code_match:
                    lang_code = code_match.group(1).strip()
                    embedded_text_pattern = pattern_start + str(lang_code) + pattern_end
                    extracted_data[lang_name]['lang_code'] = str(lang_code)
            # Identify sections within a language
            section_match = regex.search(WIKI_SECTION_PATTERN, line)
            if section_match:
                if 'Etymology' in section_match.group(1):
                    current_group = section_match.group(1).strip()
                else:
                    current_section = section_match.group(1).strip().lower()
                continue

            if current_section:
                # Extract translations
                if current_section == "translations":
                    trans_top_match = re.search(r"\{\{trans-top\|([^}]+)\}\}", line)
                    translation_pattern = r".*\*:? ([^:]+)?:.*?[{t\+]+\|([a-z]{2,3}).*?\|([^|}]+).*?[}]+.*"
                    if trans_top_match:
                        definition = trans_top_match.group(1).strip()
                        continue

                    trans_match = re.match(translation_pattern, line)
                    if trans_match:
                        name = str(trans_match.group(1)).strip()
                        code = trans_match.group(2)
                        translation = trans_match.group(3).split("=")[-1].strip() if "=" in trans_match.group(
                            3) else trans_match.group(3).strip()
                        lang_key = f"{name} ({code})" if name else code
                        default = {lang_key: translation}
                        try:
                            extracted_data[lang_name][current_group][current_section][definition].update(default)
                        except KeyError as ex:
                            if ex.args[0] == current_group:
                                extracted_data[lang_name][current_group] = {current_section: {definition: default}}
                            elif ex.args[0] == current_section:
                                extracted_data[lang_name][current_group][current_section] = {definition: default}
                            else:
                                extracted_data[lang_name][current_group][current_section][definition] = default
                        continue

                # Extract hyponyms
                elif current_section == "hyponyms":
                    default = {current_section: []}
                    group_info = extracted_data[lang_name].get(current_group, default)
                    pattern = r"[†\|\{\[colvern\] 0-9]+|taxlink|([^\|\(\)\[\]\{\}]+)"
                    match = [x.strip() for x in  re.findall(pattern, line) if x]
                    if not match:
                        continue
                    if len(match) < 2:
                        section = group_info.get(current_section, [])
                        section.append(match[0])
                        group_info[current_section] = section
                        extracted_data[lang_name][current_group] = group_info
                    else:
                        hyponym_term = match[0]
                        if hyponym_term:
                            scientific_name = " ".join(match[1:])
                            section = group_info.get(current_section, [])
                            section.append({hyponym_term: scientific_name})
                            group_info[current_section] = section
                            extracted_data[lang_name][current_group] = group_info

                # Extract derived terms
                elif current_section == "derived terms":
                    default = {current_section: []}
                    group_info = extracted_data[lang_name].get(current_group, default)
                    pattern = r"[* {{col0-9]+?\||([^|}&:;]+)|&lt;t:|&gt"
                    terms = re.findall(pattern, line)
                    if len(terms) > 0:
                        if len(terms[0]) < 1:
                            terms = terms[2:]
                        terms = [x.strip() for x in terms if x]
                        section = group_info.get(current_section, [])
                        section.extend(terms)
                        group_info[current_section] = section
                        extracted_data[lang_name][current_group] = group_info
                else:
                    # Skip pronunciation section
                    if current_section in ['pronunciation']:
                        continue
                    default = {current_section: []}
                    group_info = extracted_data[lang_name].get(current_group, default)
                    definition_match = re.findall(embedded_text_pattern, line)
                    definition_match = " ".join([x.strip() for x in definition_match if x])
                    if definition_match:
                        section = group_info.get(current_section, [])
                        section.append(definition_match)
                        group_info[current_section] = section
                        extracted_data[lang_name][current_group] = group_info

    # Add results to queue
    if results_queue is not None:
        results_queue.put(extracted_data)

    return extracted_data

def save_wiki_page_vocabulary_articles_to_db():
    # Save results to database
    table = "wiktionary_vocabulary_pages"
    database_name = "ASI"
    indexes = ['word', 'page_number']
    query_key = ['word', 'page_number']
    records_processed = 0
    batch = []
    while records_processed < total_records:
        try:
            batch.append(results_que.get(timeout=30))
            records_processed += 1
            results_que.task_done()
            if len(batch) >= 1000:
                save_table_to_mongodb(table, database_name, batch, indexes, query_key)
                batch = []
        except queue.Empty:
            if batch:
                save_table_to_mongodb(table, database_name, batch, indexes, query_key)
                batch = []

def build_wiki_vocabulary_article_pages():
    from model.util import get_local_mongodb_manager

    global results_que, total_records, logger
    wicktionary_database = get_local_mongodb_manager()
    results_que = Queue()
    total_records = 0
    try:
        wicktionary_database.connect()
        record_queue, total_records = wicktionary_database.get_all_records("enwiktionary_latest_pages_articles")
        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            logger = logging.getLogger('wiktionary_parser_worker')
            futures = []
            logger.info(f"Processing {total_records} wiktionary page articles...")
            thread = Thread(target=save_wiki_page_vocabulary_articles_to_db, args=())
            thread.name = f"wiktionary_parser_writer:{thread.name}"
            logger.info(f"Starting {thread.name}...")
            thread.start()
            for _ in range(total_records):
                try:
                    document = record_queue.get(timeout=120)
                    args = document, results_que
                    future = executor.submit(extract_lexical_data, *args)
                    futures.append(future)
                    record_queue.task_done()
                except queue.Empty:
                    break

            # Wait for all submitted tasks to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # To potentially raise any exceptions from the worker
                except Exception as e:
                    logger.exception(f"Exception in worker_with_queue_partial: {e}")
    except Exception as e:
        logger.error(f"Cannot process wiki pages. - {e}")
    finally:
        wicktionary_database.disconnect()

def update_tokenizer_language_database():
    """Update UnicodeTokenizer ISO 639-3 language database info"""
    from model.util import get_local_mongodb_manager
    database = get_local_mongodb_manager("UnicodeTokenizer")
    try:
        database.connect()
        macro_languages = {}
        lang_mapping = {
            "scope": {"I": "Individual", "M": "Macrolanguage", "S": "Special"},
            "type": {"A": "Ancient", "C": "Constructed", "E": "Extinct", "H": "Historical", "L": "Living",
                     "S": "Special"},
            "status": {"A": "Active", "R": "Retired"}
        }
        languages = []
        collection = database.get_collection("languages")
        records, total = database.get_all_records("languages")
        # TODO - country info. e.g. 'a individual language of France."
        descriptions = {
            "I": "an individual language from ISO 639-3 standard.",
            "M": "a macro language from ISO 639-3 standard containing the following individual languages: ",
            "S": "an special language from ISO 639-3 standard.",
        }
        language_info = {}
        for x in range(total):
            record = records.get(timeout=10)
            del record["_id"]
            language_info[record["language_code"]] = record

        for lang in get_all_iso639_languages():
            if lang.scope == "M" and lang.part3 not in macro_languages:
                macro_languages[lang.part3] = set()

            if lang.macrolanguage:
                try:
                    macro_languages[lang.macrolanguage].add(lang.part3)
                except KeyError:
                    macro_languages[lang.macrolanguage] = {lang.part3}


            query = {"language_code": lang.part3}
            record = language_info.get(lang.part3, {})
            update = {
                "$set": {
                    "language_code": lang.part3,
                    "language_name": lang.name,
                    "ancestry": record.get("ancestry", []),
                    "description": record.get("description", "") or descriptions.get(lang.scope, ""),
                    "population": record.get("population", "") or "-1",
                    "vitality": record.get("vitality", "") or "5-Unknown",
                    "scope": lang_mapping["scope"].get(lang.scope, lang.scope),
                    "type": lang_mapping["type"].get(lang.type, lang.type),
                    "status": lang_mapping["status"].get(lang.status, lang.status),
                }
            }
            languages.append(UpdateOne(query, update, upsert=True))

        collection.bulk_write(languages)
        # Update macro languages
        #for code in macro_languages:
    except Exception as e:
        logger.exception(f"Cannot update tokenizer database. - {e}")
    finally:
        database.disconnect()


def find_files_by_pattern(directory_path: str, pattern: str) -> list:
    """
    Returns a list of filenames in the directory that match the given pattern.
    """
    path = Path(directory_path)

    # Ensure the directory exists to avoid errors
    if not path.is_dir():
        print(f"Error: {directory_path} is not a valid directory.")
        return []

    # Using glob with a wildcard prefix to match your pattern
    # This will find files like 'en_greek_latin_roots_cache.json'
    matched_files = path.glob(f"*{pattern}")

    # Convert path objects back to strings for the final list
    return [file.name for file in matched_files if file.is_file()]


def delete_files_from_list(directory_path: str, file_list: list):
    """
    Deletes the specified list of files from the given directory.
    """
    path = Path(directory_path)

    file_count = 0
    for file_name in file_list:
        file_to_rem = path / file_name

        try:
            if file_to_rem.exists() and file_to_rem.is_file():
                file_to_rem.unlink()
                file_count += 1
                print(f"Successfully deleted: {file_name}")
            else:
                print(f"Skip: {file_name} not found in {directory_path}.")
        except Exception as e:
            print(f"Failed to delete {file_name}: {e}")

    return file_count


def get_function_state(func, exclude_globals=None, visited_funcs=None):
    """
    Recursively captures the state of a function, its dependencies, and globals.
    Automatically excludes Python dunder (__) metadata to ensure cross-run stability.
    """
    if visited_funcs is None:
        visited_funcs = set()
    if exclude_globals is None:
        exclude_globals = set()
    else:
        exclude_globals = set(exclude_globals)

    if func in visited_funcs:
        return "<circular_ref>"
    visited_funcs.add(func)

    state = {
        "source": inspect.getsource(func).strip(),
        "globals": {},
        "dependencies": {}
    }

    referenced_names = func.__code__.co_names
    for name in referenced_names:
        # SKIP CRITICAL: Ignore user-excluded globals, the function itself,
        # and all Python internal dunders (like __name__, __file__, etc.)
        if name in exclude_globals or name == func.__name__ or name.startswith('__'):
            continue

        if name in func.__globals__:
            val = func.__globals__[name]

            # Recursive check for local functions in the same module
            if inspect.isfunction(val) and val.__module__ == func.__module__:
                state["dependencies"][name] = get_function_state(val, exclude_globals, visited_funcs)

            # Capture global variables (excluding modules)
            elif not inspect.ismodule(val) and not inspect.isfunction(val):
                state["globals"][name] = val

    return state


def find_and_print_diff(old, new, path="root"):
    """
    Recursively compares two state dictionaries and prints the differences.
    """
    if old == new:
        return

    if type(old) != type(new):
        print(f"  [CHANGE] Type mismatch at {path}: {type(old)} -> {type(new)}")
        return

    if isinstance(old, dict):
        for key in set(old.keys()) | set(new.keys()):
            if key not in old:
                print(f"  [ADDED] {path}.{key}")
            elif key not in new:
                print(f"  [REMOVED] {path}.{key}")
            else:
                find_and_print_diff(old[key], new[key], f"{path}.{key}")
    else:
        # For source code, show a line-by-line diff if it's long
        if "source" in path and isinstance(old, str):
            print(f"  [CHANGE] Source code of {path.split('.')[1]} modified.")
        else:
            print(f"  [CHANGE] Value at {path}:")
            print(f"    Old: {repr(old)[:100]}")
            print(f"    New: {repr(new)[:100]}")


def check_cache_validity(func, exclude_globals=None, log_dir=CACHE_DIR):
    """
    Checks validity and prints specific changes if the cache is invalidated.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    func_name = func.__name__
    state_file = os.path.join(log_dir, f"{func_name}_state.pkl")
    current_state = get_function_state(func, exclude_globals)

    if os.path.exists(state_file):
        try:
            with open(state_file, 'rb') as f:
                old_state = pickle.load(f)

            if old_state == current_state:
                return True

            # If we are here, something changed. Let's find out what.
            print(f"\n[CACHE INVALID] Changes detected in '{func_name}':")
            find_and_print_diff(old_state, current_state, path=func_name)

        except Exception as e:
            print(f"  [ERROR] Could not read old state: {e}")

    # Save the new state
    with open(state_file, 'wb') as f:
        pickle.dump(current_state, f)

    return False


def run_check_cache_validity_demo():
    # Some global state
    global ENCRYPT_KEY, APP_NAME
    ENCRYPT_KEY = 87
    APP_NAME = "MyBlueApp"

    def my_expensive_task(data):
        print(f"my_expensive_task is working. App Name: {APP_NAME}")
        return data * ENCRYPT_KEY

    check_cache_validity(my_expensive_task, exclude_globals={"APP_NAME"})
    my_expensive_task(43)

"""
Okay, let's craft a Python function using Ray Tune to optimize a PyTorch Transformer model. This function will allow you to search for hyperparameters like embed_dim, num_heads, accumulation_steps, sequence_length, batch_size, learning rate, etc., aiming to maximize tokens_per_second while also reporting evaluation accuracy.
Key Components:
 * Model Definition (model_class): Your Transformer model's class. It must be flexible enough to be initialized with varying architectural parameters from the search space (e.g., embed_dim, num_heads).
 * Initial Parameters (initial_params): A dictionary of fixed parameters for your model that are not being tuned (e.g., vocab_size, num_encoder_layers if fixed).
 * Trainable Function: A function that Ray Tune will execute for each trial. It sets up the model, data, optimizer, trains, evaluates, and reports metrics.
 * Search Space (search_space_config): Defines the range and distribution for each hyperparameter you want to tune.
 * Ray Tune Orchestration (optimize_with_ray_tune): The main function that sets up and runs the Ray Tune experiment.
First, ensure you have the necessary libraries:
pip install "ray[tune]" torch torchvision torchaudio hyperopt # hyperopt is optional for Bayesian search

Here's the code:



if _name_ == '_main_':
    # Initialize Ray (adjust resources as needed for your machine)
    if ray.is_initialized():
        ray.shutdown()
    # Set num_gpus based on availability for the Ray cluster
    available_gpus = torch.cuda.device_count()
    ray.init(num_cpus=max(1, torch.get_num_threads() // 2), num_gpus=available_gpus, ignore_reinit_error=True)


    # --- Configuration for the Optimization ---
    MODEL_CLASS = DummyTransformer

    FIXED_MODEL_PARAMS = {
        'vocab_size': 1000, # Example: fixed vocab size
        # 'num_encoder_layers': 2 # Could be fixed here, or tuned in search_space
    }

    # Parameters to construct datasets inside the trainable function
    # sequence_length will be overridden by the tuned value from search_space
    TRAIN_DS_PARAMS = {'num_samples': 2000, 'vocab_size': FIXED_MODEL_PARAMS['vocab_size']}
    EVAL_DS_PARAMS = {'num_samples': 500, 'vocab_size': FIXED_MODEL_PARAMS['vocab_size']}


    # Define the search space for Ray Tune
    # Ensure embed_dim and num_heads are compatible. The model class handles errors.
    # Alternatively, use tune.sample_from to create dependent choices.
    SEARCH_SPACE = {
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([16, 32, 64]),
        "embed_dim": tune.choice([64, 128, 256]), # d_model
        "num_heads": tune.choice([2, 4, 8]),      # Must divide embed_dim
        "num_encoder_layers": tune.choice([1, 2, 3]), # Example of tuning a layer param
        "accumulation_steps": tune.choice([1, 2, 4]),
        "sequence_length": tune.choice([32, 64, 128]), # Model and data will use this
        "optimizer_type": tune.choice(["AdamW", "Adam"]),
        "weight_decay": tune.loguniform(1e-7, 1e-1),
    }

    NUM_RAY_TRIALS = 10 # Number of hyperparameter combinations to try
    EPOCHS_PER_TRIAL = 3  # Number of epochs to train each combination

    # Define resources per trial. If you have 1 GPU, gpu: 0.5 allows 2 trials concurrently on it.
    # If you have more GPUs, you can set gpu: 1 for one trial per GPU.
    RESOURCES = {"cpu": 2, "gpu": 0.5 if available_gpus > 0 else 0}
    if available_gpus >= 1:
        RESOURCES["gpu"] = 1 # Dedicate one full GPU per trial if multiple GPUs are available
        # Or keep it < 1 to pack multiple trials on one GPU if memory allows.

    best_config_found, tps_for_best_config, acc_for_best_config = optimize_transformer_with_ray_tune(
        model_class_to_tune=MODEL_CLASS,
        fixed_initial_params=FIXED_MODEL_PARAMS,
        train_dataset_constructor_params=TRAIN_DS_PARAMS,
        eval_dataset_constructor_params=EVAL_DS_PARAMS,
        search_space=SEARCH_SPACE,
        num_ray_trials=NUM_RAY_TRIALS,
        epochs_per_trial=EPOCHS_PER_TRIAL,
        metric_to_optimize="tokens_per_second", # Primary metric
        optimization_mode="max",
        resources_per_trial_config=RESOURCES,
        experiment_name="MyTransformerTune"
    )

    print("\n--- Overall Optimization Result ---")
    if best_config_found:
        print(f"Best Configuration: {best_config_found}")
        print(f"Achieved Tokens/Second: {tps_for_best_config:.2f}")
        print(f"Achieved Accuracy: {acc_for_best_config:.2f}%")
    else:
        print("Optimization finished, but no best configuration was determined (e.g., all trials failed).")

    ray.shutdown()

Before Running:
 * Replace DummyTransformer and DummySeqDataset: Substitute these with your actual PyTorch Transformer model class and your Dataset class. Ensure your model's _init_ can accept the parameters you're tuning (like d_model, nhead, num_encoder_layers, seq_len).
 * Adapt accuracy_fn: Make sure token_accuracy_counts (or your custom function) correctly calculates accuracy based on your model's output logits and label format. It should return (total_correct_predictions_in_batch, total_valid_predictions_in_batch).
 * Customize SEARCH_SPACE: Adjust the ranges and choices for hyperparameters to suit your model and expectations.
   * embed_dim and num_heads: The DummyTransformer now includes a ValueError if embed_dim is not divisible by num_heads. Ray Tune will mark such trials as ERROR.
   * sequence_length: This is now a tunable parameter. The trainable_function will create DummySeqDataset instances with this specific sequence_length for each trial. Your actual dataset handling logic should accommodate this if sequence_length is dynamic.
 * Adjust Resources: Modify RESOURCES_PER_TRIAL (CPU and GPU allocation per trial) and num_cpus/num_gpus in ray.init() based on your hardware.
 * Install Ray and PyTorch: As mentioned, pip install "ray[tune]" torch torchvision torchaudio. Add hyperopt if you plan to use HyperOptSearch.
 * Run the Script: Execute the Python file. Ray Tune will start the hyperparameter search.
This comprehensive setup provides a powerful and flexible way to optimize your Transformer's training configuration.
"""

"""
    def text_parser(examples):
        return {"inputs": examples["sentence"], "targets": examples["label"],
                "task_type": ["classification"] * len(examples["sentence"])}


    dataset_path = "glue"
    dataset_name = "cola"
    batch_size = 10000  # entire dataset
    data_dir = "cola_parquet"

    train_dataloader, eval_dataloader = prepare_parquet_dataset(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        batch_size=batch_size,
        data_dir=data_dir,
        parser_func=text_parser
    )
    """

if __name__ == "__main__":
    run_check_cache_validity_demo()
    #update_tokenizer_language_database()

    #build_wiki_vocabulary_article_pages()
    output_filename = "enwiktionary-latest-pages-articles.xml.bz2"  # Choose your desired output filename
    #parse_bz2_xml_pages_to_db(output_filename)
    #create_wiktionary_vocab_list()
    #input_filename = "C:/Users/JulianJoseph/Downloads/enwiktionary-latest-pages-articles.xml.bz2"

    # extract_bz2(input_filename, output_filename)
    # process_large_xml_lxml(output_filename)

    #view_xml_by_pages(output_filename)


    skill_keywords = {
        "question_answering": ["answer", "question", "who", "what", "where", "when", "why", "how","answer"], # added duplicate answer.
        "summarization": ["summarize", "summary", "brief", "condense", "main points", "summarize"], # added duplicate summarize.
        "translation": ["translate", "translation", "language", "convert"],
    }
    """
    try_estimate_max_transformer_memory()
    #max_size = calculate_max_saved_model_size()
    save_skill_keywords_to_mongodb(skill_keywords)

    loaded_skill_keywords = load_skill_keywords_from_mongodb()
    logger.debug("\nLoaded Skill Keywords:")
    logger.debug(loaded_skill_keywords)

    #Example of adding to existing keywords.
    new_keywords = {"question_answering": ["test_keyword", "question"]}
    save_skill_keywords_to_mongodb(new_keywords) #add new keyword.
    loaded_skill_keywords = load_skill_keywords_from_mongodb
    logger.debug("\nLoaded Skill Keywords after adding a keyword:")
    logger.debug(loaded_skill_keywords)
    """