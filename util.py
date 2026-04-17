import os
import queue
import sys
import time
import logging
import math
import docker
import glob
import requests
import torch
import torch.nn as nn
import regex as re
from bson import ObjectId
from lxml import html
from lxml.html import HtmlElement
from urllib.parse import urlparse
from copy import deepcopy
from functools import wraps
from datetime import datetime
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from fast_langdetect import detect as fast_detect, detect_multilingual
from langdetect import detect
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import pdfplumber
import fitz  # PyMuPDF
import tempfile
from pymongo import UpdateOne
from rich.progress import Progress
from rich.console import Console
import multiprocessing as mp
from bs4 import BeautifulSoup
from database.manager import MongoDBManager
#from google.cloud import translate_v3 as translate
# from argostranslate import translate as argos_translate, package
from conf import (
    LOG_STR_FORMAT, DATE_FORMAT, FILE_SIZE_PATTERN, WORKSTATION, LOGS_FOLDER, TESTS_FOLDER,
    BOOK_NAME, DESCRIPTION, AUTHORS, MD5_HASH,
    FILE_TYPE, SIZE, NON_FICTION_BOOKS, VERIFIED, LIBRARY_PATHS, console_handler,
    UNICODE_BOOK_NAME_PATTERN
)

# --- Constants ---
PARENT_DIR = Path(__file__).parent
MONGO_BATCH_SIZE = 1000
TRANSLATE_BATCH_SIZE = 100
QUEUE_GET_TIMEOUT = 1 # Seconds to wait for items from the queue

console = Console()


class AsyncFileLogHandler(logging.Handler):
    def __init__(self, log_queue: mp.Queue, formatter=None):
        super().__init__()
        self.log_queue = log_queue
        self.formatter = formatter or logging.Formatter(LOG_STR_FORMAT, DATE_FORMAT)
        self.level = logging.DEBUG

    def emit(self, record):
        try:
            # Format the record here before putting it in the queue
            log_entry = self.format(record)
            self.log_queue.put(log_entry)
        except Exception as e:
            logging.error(f"Error processing log record in queue: {e}")


class AsyncFileLogProcessor(mp.Process):
    def __init__(self, log_queue, control_queue, filename='application.log', batch_size=1000, flush_interval=60):
        super().__init__()
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._shutdown_flag = False
        self.filename = filename
        self.log_queue = log_queue
        self.control_queue = control_queue  # Separate queue for control commands
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_STR_FORMAT, datefmt=DATE_FORMAT))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.filters = {}
        self.buffer = {}
        self.daemon = True
        self.start()

    def run(self):
        last_flush_time = time.time()
        count = 0
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_STR_FORMAT, datefmt=DATE_FORMAT))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.info("AsyncFileLogProcessor Running...")
        log_folder = "logs"
        os.makedirs(log_folder, exist_ok=True)

        while not self._shutdown_flag:
            try:
                # Check for control commands
                try:
                    command, data = self.control_queue.get_nowait()
                    if command == 'add_filter':
                        filter_string, output_filepath = data
                        self.filters[filter_string] = output_filepath
                        self.logger.info(f"Added filter: '{filter_string}' -> '{output_filepath}'")
                    elif command == 'stop_worker':
                        self._shutdown_flag = True
                        self.logger.warning("AsyncFileLogProcess Shutting Down (via control queue)...")
                        break
                except queue.Empty:
                    pass

                # Process log messages
                try:
                    message = self.log_queue.get(timeout=0.1)
                    self.add_log(message)
                    count += 1

                    if count >= self.batch_size or (time.time() - last_flush_time >= self.flush_interval):
                        self.logger.info(f"AsyncFileLogProcessor processing next batch... {count}")
                        self._write_to_file(log_folder)
                        last_flush_time = time.time()
                        count = 0

                except queue.Empty:
                    if time.time() - last_flush_time >= self.flush_interval:
                        self._write_to_file(log_folder)
                        last_flush_time = time.time()
                    continue

            except KeyboardInterrupt:
                self.logger.info("KeyboardInterrupt - AsyncFileLogProcessor Shutting Down...")
                break

            except Exception as e:
                self.logger.error(f"Error in log worker: {e}")
                time.sleep(1)

        self._write_to_file(log_folder)

    def add_filter(self, filter_string, output_filepath):
        """Adds a filter by sending a command to the control queue."""
        self.control_queue.put(('add_filter', (filter_string, output_filepath)))

    def stop_worker(self):
        """Sends a stop signal to the control queue."""
        self.control_queue.put(('stop_worker', None))

    def add_log(self, message):
        added = False
        for filter_str, log_file in self.filters.items():
            if message.find(filter_str) >= 0:
                # Add log to special file
                try:
                    self.buffer[log_file].append(f"{message}\n")
                    added = True
                    break
                except KeyError:
                    self.buffer[log_file] = [f"{message}\n"]
                    added = True
        if not added:
            # Add to default log file
            try:
                self.buffer[self.filename].append(f"{message}\n")
            except KeyError:
                self.buffer[self.filename] = [f"{message}\n"]

    def _write_to_file(self, log_folder):
        try:
            for file_name, logs in self.buffer.items():
                try:
                    log_file = os.path.join(log_folder, file_name)
                    with open(log_file, 'a') as file:
                        file.writelines(logs)
                except Exception as e:
                    self.logger.error(f"Cannot write logs to file={file_name}: {e}")

            self.buffer = {}  # Reset log buffer

        except Exception as e:
            self.logger.exception(f"Cannot write logs to file: {e}")


def download_file_with_rich_progressbar(url, filename):
    """Downloads a file from a URL and displays a rich progress bar.

    Args:
        url (str): The URL of the file to download.
        filename (str): The local filename to save the downloaded file as.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1 Megabyte

        with open(filename, 'wb') as file:
            with Progress(console=console) as progress:
                task = progress.add_task("[green]Downloading...", total=total_size)
                for data in response.iter_content(chunk_size=block_size):
                    size = file.write(data)
                    progress.update(task, advance=size)

        console.print(f"\n[bold green]Successfully downloaded '{filename}'[/bold green]")

    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error downloading '{url}':[/bold red] {e}")
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")


def create_log_file(path, log_name, suffix="_debug.log", log_version=0):
    """
    Creates new log file if the current log is too large. (greater than 1GB) or
    if the current date is different from when the current log was created.

    :param path: directory in which to create log file.
    :param log_name: the name of the log which needs to be created.
        creates a new logger if no logger with provider name exist.
    :param suffix: suffix to add at the end of the log name.
    :param log_version: create log file with a specific version number
    :type path: str
    :type log_name: str
    :type suffix: str
    :type log_version: int

    :return: Logger - logging object with the requested name.
    """
    time_now = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
    if log_version < 1:
        log_name = f"{time_now} {log_name}{suffix}"
        log_file = os.path.join(path, log_name)
    else:
        suffix = suffix.replace('.', ' ({0}).').format(log_version)
        log_name = f"{time_now} {log_name}{suffix}"
        log_file = os.path.join(path, log_name)
    try:
        # Get log size in MBs
        log_size = os.path.getsize(log_file) >> 20
        # Create new log file if current log size is greater than 1GB
        if log_size >= 1024:
            log_version += 1
            log_file = create_log_file(path, log_name, suffix, log_version)
    except FileNotFoundError:
        # Create new log file is now already exist
        with open(log_file, mode="a"):
            pass
    except Exception as e:
        logging.exception(f"{e} - Cannot create new log file")

    return log_file


def update_logs(logger_name, log_file="ange_asi"):
    """
    Creates new log file if the current log is too large. (greater than 1GB) or
    if the current date is different from when the current log was created.

    :param logger_name: the name of the logger which needs to be updated.
        creates a new logger if no logger with provider name exist.
        (default is 'SwumCloudManager')
    :param log_file: name of logging file.
    :type logger_name: str
    :type log_file: str

    :return: Logger - logging object with the requested name.

    """
    date_format = '%Y-%m-%d %H:%M'
    log_format = f'%(asctime)s {LOG_STR_FORMAT}'
    file_formatter = logging.Formatter(log_format, date_format)
    # Create directory to store downloaded files if none exist.
    if not os.path.exists(WORKSTATION):
        os.mkdir(WORKSTATION)
    # Create directory to store system logs if none exist.
    if not os.path.exists(LOGS_FOLDER):
        os.mkdir(LOGS_FOLDER)
    # Create directory to store testing logs if none exist.
    if not os.path.exists(TESTS_FOLDER):
        os.mkdir(TESTS_FOLDER)
    # Setup handler debug logs
    debug_log_file = create_log_file(LOGS_FOLDER, log_file, "_debug.log")
    debug_handler = logging.FileHandler(debug_log_file)
    debug_handler.setFormatter(file_formatter)
    debug_handler.setLevel(logging.DEBUG)
    # Setup handler error logs
    info_log_file = create_log_file(LOGS_FOLDER, log_file, "_info.log")
    info_handler = logging.FileHandler(info_log_file)
    info_handler.setFormatter(file_formatter)
    info_handler.setLevel(logging.INFO)
    # Add handlers to logger
    logging.basicConfig(format=log_format,
                        datefmt=date_format,
                        level=logging.DEBUG,
                        handlers=[console_handler, info_handler, debug_handler],
                        force=True,
                        )
    new_logger = logging.getLogger(logger_name)

    return new_logger


def log_execution_time(func):
    """
    A decorator that logs the execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds.")
        return result
    return wrapper


class _ContextualManager:
   """A helper class to manage the creation of the user's class."""
   def __init__(self, cls, args, kwargs):
       self._cls = cls
       self._args = args
       self._kwargs = kwargs
       self._instance = None

   def __enter__(self):
       """
       Create the actual instance upon entering the context.
       The `as` part of the `with` statement will receive this instance.
       """
       # Instantiate the user's class
       self._instance = self._cls(*self._args, **self._kwargs)
       # If the user's class has its own __enter__, call it.
       if hasattr(self._instance, '__enter__'):
           return self._instance.__enter__()
       return self._instance

   def __exit__(self, exc_type, exc_val, exc_tb):
       """
       Ensure __exit__ is called on the user's instance, if it exists.
       """
       if hasattr(self._instance, '__exit__'):
           # The return value of the user's __exit__ for exception suppression
           return self._instance.__exit__(exc_type, exc_val, exc_tb)

   def __getattr__(self, name):
       """
       Provide a helpful error if methods are called outside a `with` block.
       """
       msg = (
           f"'{self._cls.__name__}' can only be used within a 'with' statement. "
           f"You are trying to access '{name}' on an uninitialized object."
       )
       raise AttributeError(msg)

def require_context_manager(cls):
   """
   A class decorator that refactors the class into a factory.
   It ensures instances are only created and used via a context manager.
   """
   @wraps(cls, updated=())
   def wrapper(*args, **kwargs):
       """
       This wrapper replaces the class constructor. Instead of creating an
       instance of the class, it returns our context manager helper.
       """
       return _ContextualManager(cls, args, kwargs)
   return wrapper


class TreeNode:
    def __init__(self, char):
        self.char = char
        self.children = {}  # Dictionary to store children
        self.is_end = False


class SearchTree:
    def __init__(self):
        self.root = TreeNode(None)

    def insert(self, words):
        for word in words:
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = TreeNode(char)
                    self.root.children[char] = node.children[char]
                node = node.children[char]
            node.is_end = True

    def extract_node_sub_strings(self, node, results_list):
        sub_string = ""
        for char in node.children:
            sub_string += char
            if node.children[char]:
                value = self.extract_node_sub_strings(node.children[char], results_list)
                results_list.extend(value)

        results_list.append(sub_string)

        return results_list

    def get_substrings(self, min_len=2, max_len=5):
        all_sub_strings = []
        node = self.root
        for char in node.children:
            sub_string = self.extract_node_sub_strings(node.children[char], all_sub_strings)
            all_sub_strings.extend(sub_string)

        # remove substring that does not fit criteria
        all_sub_strings = [x for x in all_sub_strings if min_len < len(x) < max_len]

        return all_sub_strings


def generate_substrings(sentence, trie, min_len=2, max_len=5):
    substrings = []
    for i in range(len(sentence)):
        node = trie.root
        current_substring = ""
        for j in range(i, min(i + max_len, len(sentence))):
            char = sentence[j]
            if char in node.children:
                node = node.children[char]
                current_substring += char
                if len(current_substring) >= min_len:
                    substrings.append(current_substring)
                if node.is_end == False and len(current_substring) == max_len:
                  break
            else:
                break
    return list(set(substrings)) #use set to remove duplicates.


class ParquetDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.data = []
        for file_path in self.file_paths:
            table = pq.read_table(file_path)
            self.data.extend(table.to_pandas().to_dict('records'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_parquet_file_name(dataset_name, split_type):
    """
    Create parquet file name for the specified dataset_name and dataset_type

    Args:
        dataset_name (str): the name of the dataset configuration
        split_type (str): can be any of the following: [train, eval]
    """
    return f"{dataset_name}_{split_type}"


def load_dataset_from_parquet_file(batch_size, file_path_prefix):
    """
    Loads a dataset from multiple Parquet files.

    Args:
        batch_size (int): Batch size for DataLoader.
        file_path_prefix (str): The prefix of the Parquet file names
                                         (e.g., "data/train").

    Returns:
        A DataLoader for the loaded dataset.
    """
    all_parquet_files = sorted(glob.glob(f"{file_path_prefix}_part*.parquet"))
    if not all_parquet_files:
        logging.warning(f"No Parquet files found with prefix: {file_path_prefix}")
        return None

    parquet_dataset = ParquetDataset(all_parquet_files)
    dataloader = DataLoader(parquet_dataset, batch_size=batch_size, shuffle=False)  # Shuffle handled during saving if needed
    logging.info(f"Loaded dataset from Parquet files: {all_parquet_files}")
    return dataloader


def save_dataset_to_parquet_file(train_data: Dataset, batch_size: int, tokenize_func: callable,
                                 parquet_file_path_prefix: str, max_file_size_mb: int = 1024):
    """
    Saves a dataset to Parquet files, splitting into multiple files if necessary.

    Args:
        train_data: The dataset to save.
        batch_size: The batch size for the DataLoader (used here to estimate file sizes).
        tokenize_func: A function to tokenize the data. If None, a default function is used.
        parquet_file_path_prefix: The prefix for the Parquet file names (e.g., "data/train").
        max_file_size_mb: The maximum size of each Parquet file in MB (default: 1024).

    Returns:
        A DataLoader for the tokenized data (this might not be the most efficient return here
        if the primary goal is saving. Consider returning the list of saved file paths).
    """
    try:
        if not tokenize_func:
            tokenize_func = lambda data: {key: data[key] for key in ["inputs", "targets", "task_type"]}
        tokenized_data = train_data.map(tokenize_func, batched=True)
        tokenized_data.set_format(type="torch", columns=["inputs", "targets", "task_type"])

        # Estimate the size of a single row in bytes
        sample = tokenized_data[0]
        # Flatten multidimensional arrays before creating the PyArrow table
        sample_table = pa.Table.from_pydict({
            k: [v.cpu().numpy().flatten()] if isinstance(v, torch.Tensor) else [v] for k, v in sample.items()
        })
        row_size_bytes = sample_table.nbytes if sample_table.num_rows > 0 else 1  # Avoid division by zero

        max_file_size_bytes = max_file_size_mb * 1024 * 1024
        max_rows_per_file = max(1, max_file_size_bytes // row_size_bytes)
        num_rows = len(tokenized_data)
        num_files = (num_rows + max_rows_per_file - 1) // max_rows_per_file

        for i in range(num_files):
            start_index = i * max_rows_per_file
            end_index = min((i + 1) * max_rows_per_file, num_rows)
            chunk = tokenized_data.select(range(start_index, end_index))
            # Flatten multidimensional arrays before creating the PyArrow table
            chunk_df = chunk.to_pandas()
            for col in chunk_df.columns:
                if isinstance(chunk_df[col].iloc[0], (list, tuple, np.ndarray)):
                    chunk_df[col] = chunk_df[col].apply(
                        lambda x: x.flatten() if isinstance(x, (list, tuple, np.ndarray)) else x)
            train_table = pa.Table.from_pandas(chunk_df)

            parquet_file_path = f"{parquet_file_path_prefix}_part{i + 1}.parquet"
            pq.write_table(train_table, parquet_file_path, compression='SNAPPY')
            logging.info(f"Parquet dataset part {i + 1} saved: {parquet_file_path}")

        dataloader = DataLoader(tokenized_data, batch_size=batch_size, shuffle=True)
    except Exception as e:
        logging.exception(f"Cannot save dataset to parquet file - {e}")
        dataloader = None

    return dataloader

def prepare_parquet_dataset(dataset_path, dataset_name, batch_size=32, data_dir="parquet_datasets", parser_func=None):
    """
    Retrieves training data from Hugging Face and saves/loads as compressed Parquet files.

    Args:
        dataset_path (str): if path is a dataset repository on the HF hub (list all available datasets
            with [huggingface_hub. list_datasets]) -> load the dataset from supported files in the repository
            (csv, json, parquet, etc.) e.g. 'username/ dataset_name', a dataset repository on the HF hub containing
            the data files. if path is a local directory -> load the dataset from supported files in the directory
            (csv, json, parquet, etc.) e.g. './ path/ to/ directory/ with/ my/ csv/ data'. if path is the name of
            a dataset builder and data_files or data_dir is specified (available builders are "json", "csv", "parquet",
            "arrow", "text", "xml", "webdataset", "imagefolder", "audiofolder", "videofolder") -> load the dataset
            from the files in data_files or data_dir e.g. 'parquet'.
        dataset_name (str): the name of the dataset configuration
        batch_size (int): Batch size for DataLoader.
        data_dir (str): Directory to save/load Parquet files.
        parser_func (callable): a callable function used to filter the dataset.
    Returns:
        (train_dataloader, eval_dataloader)
    """
    train_type = "train"
    eval_type = "eval"
    parquet_train_file_path_prefix = os.path.join(data_dir, get_parquet_file_name(dataset_name, train_type))
    parquet_eval_file_path_prefix = os.path.join(data_dir, get_parquet_file_name(dataset_name, eval_type))

    os.makedirs(data_dir, exist_ok=True)

    # Check if any part files exist for both train and eval
    train_part_files = glob.glob(f"{parquet_train_file_path_prefix}_part*.parquet")
    eval_part_files = glob.glob(f"{parquet_eval_file_path_prefix}_part*.parquet")

    if not train_part_files or not eval_part_files:
        logging.info("Tokenizing and saving dataset to Parquet...")

        dataset = load_dataset(dataset_path, dataset_name)

        train_data = dataset["train"]
        eval_data = dataset["validation"]

        if parser_func is None:
            parser_func = lambda data : {key: data[key] for key in ["inputs", "targets", "task_type"]}

        save_dataset_to_parquet_file(
            train_data, batch_size, parser_func, parquet_train_file_path_prefix
        )
        save_dataset_to_parquet_file(
            eval_data, batch_size, parser_func, parquet_eval_file_path_prefix
        )
        train_dataloader = load_dataset_from_parquet_file(batch_size, parquet_train_file_path_prefix)
        eval_dataloader = load_dataset_from_parquet_file(batch_size, parquet_eval_file_path_prefix)

    else:
        logging.info("Loading dataset from existing Parquet files...")
        train_dataloader = load_dataset_from_parquet_file(batch_size, parquet_train_file_path_prefix)
        eval_dataloader = load_dataset_from_parquet_file(batch_size, parquet_eval_file_path_prefix)

    return train_dataloader, eval_dataloader


def try_prepare_parquet_dataset():
    logging.basicConfig(level=logging.INFO)
    # Assuming you have a dataset on Hugging Face Hub or a local dataset
    dataset_path = "glue"
    dataset_name = "cola"
    batch_size = 16
    data_dir = "cola_parquet"

    def custom_tokenizer(examples):
        # Replace with your actual tokenizer logic
        return {"inputs": examples["sentence"], "targets": examples["label"],
                "task_type": ["classification"] * len(examples["sentence"])}

    train_dataloader, eval_dataloader = prepare_parquet_dataset(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        batch_size=batch_size,
        data_dir=data_dir,
        parser_func=custom_tokenizer
    )
    if train_dataloader:
        print("Train DataLoader created with", len(train_dataloader), "batches.")
        for batch in train_dataloader:
            print("Sample train batch:", batch["inputs"][:2], batch["targets"][:2])
            break
    if eval_dataloader:
        print("Eval DataLoader created with", len(eval_dataloader), "batches.")
        for batch in eval_dataloader:
            print("Sample eval batch:", batch["inputs"][:2], batch["targets"][:2])
            break


def create_directory_in_volume(volume_name, directory_path):
    """
    Creates a directory within the specified Docker volume '/data' if it doesn't already exist.

    Args:
        volume_name (str): The name of the Docker volume.
        directory_path (str): The path to the directory within the volume.

    Returns:
        boolean value specifying if the volume was created successfully.
    """
    status = False
    logging.debug(f"Attempting to create dir={directory_path} in volume={volume_name}...")
    # Create a temporary container to access the volume
    kwargs = {
        'image': 'alpine:latest',
        'volumes': {volume_name: {'bind': '/data', 'mode': 'rw'}},
        'command': f'sh -c "mkdir -p /data/{directory_path}"',
        'detach': False
    }
    container = create_docker_container(volume_name=volume_name, **kwargs)
    container.start()
    result = container.wait()
    output = container.logs().decode('utf-8')

    if result['StatusCode'] == 0:
        status = True
        logging.info(f"Directory '{directory_path}' created (or already existed) in volume '{volume_name}'.")
        logging.debug(f"Volume contents: {output}")
    else:
        logging.exception(f"Failed to create directory. Status code: {result['StatusCode']}")
        logging.debug(f"Logs: {output}")

    container.remove()  # clean up the temporary container.
    return status


def create_docker_container(client=None, volume_name='data_volume', **kwargs):
    """

    Args:
        client (docker.client.DockerClient): the docker client from which to create container.
        volume_name (str): The name of the Docker volume.
        kwargs (dict): args for which to create the Container as key/value pairs
    """
    if client is None:
        import docker
        client = docker.from_env()
    if not kwargs:
        kwargs = {
            'image': 'alpine:latest',
            'volumes': {volume_name: {'bind': '/data', 'mode': 'rw'}},
            'command': 'sh -c "sleep infinity"',  # Keep container running
            'detach': True
        }

    container = client.containers.create(**kwargs)

    return container


def delete_files_from_docker_volume(volume_name: str, container_path: str):
    """Deletes files/directories from a Docker volume.

    Args:
        volume_name: The name of the Docker volume.
        container_path: The path to the files/directories inside the volume.
    """
    try:
        logging.warning(f"Deleting the following dir and files: {container_path} \nfrom: {volume_name}")
        # Create a temporary container to access the volume.
        container_path = container_path.strip('/')
        kwargs = {
            'image': 'alpine:latest',
            'volumes': {volume_name: {'bind': '/data', 'mode': 'rw'}},
            'command': f'sh -c "rm -rf /data/{container_path}"',
            'detach': False
        }
        temp_container = create_docker_container(volume_name=volume_name, **kwargs)
        temp_container.start()
        temp_container.wait()  # wait for the command to finish
    except Exception as e:
        print(f"Error deleting files: {e}")
    finally:
        if 'temp_container' in locals():
            temp_container.stop()
            temp_container.remove()


def copy_files_to_docker_volume(volume_name: str, host_path: str, container_path: str):
    """Copies files from the host to a Docker volume.

    Args:
        volume_name: The name of the Docker volume.
        host_path: The path to the files/directories on the host.
        container_path: The path inside the container (and thus, the volume).
    """
    client = docker.from_env()

    try:
        # Create a temporary container to access the volume.
        temp_container = create_docker_container(client, volume_name)
        temp_container.start()

        # Copy the files using docker cp.
        if os.path.isdir(host_path):
            # Copy directory contents
            for item in os.listdir(host_path):
                source_path = os.path.join(host_path, item)
                docker_cp_path = f"{temp_container.name}:/data/{container_path.strip('/')}/{item}"
                client.api.put_archive(temp_container.id, f"/data/{container_path.strip('/')}", open(source_path, 'rb').read())
        else:
            docker_cp_path = f"{temp_container.name}:/data/{container_path.strip('/')}/{os.path.basename(host_path)}"
            client.api.put_archive(temp_container.id, f"/data/{container_path.strip('/')}", open(host_path, 'rb').read())
    except Exception as e:
        logging.exception(f"Cannot copy files to docker volume - {e}")
    finally:
        if 'temp_container' in locals():
            temp_container.stop()
            temp_container.remove()


def copy_training_dataset_to_docker(dataset_path="CohereForAI/aya_collection_language_split", dataset_name="english"):
    data_dir = "../model_trainer/parquet_datasets"
    prepare_parquet_dataset(dataset_path, dataset_name, data_dir=data_dir)
    # Docker volume details
    volume_name = "database_parquet_datasets"
    container_file_path = "parquet_datasets" # or directory

    copy_files_to_docker_volume(volume_name, data_dir, container_file_path)


def get_url_root(url):
    """
      Extracts the root URL (scheme and hostname) from a given URL.

      Args:
        url: The full URL string.

      Returns:
        The root URL (e.g., "https://example.com"). Returns "" if parsing fails.
      """
    root_url = ""
    try:
        logging.debug(f"Retrieving url root for: {url}")
        parsed_url = urlparse(url)
        if parsed_url.scheme and parsed_url.netloc:
            root_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    except Exception as e:
        logging.exception(f"Cannot get url root - {e}")

    return root_url


def split_filename_extension(filename):
  """
  Separates the filename and extension from the end of a string using regex.

  Args:
    filename: The string representing the filename or path.

  Returns:
    A tuple containing (filename, extension) if found, otherwise (filename, "").
  """
  match = re.search(r'^(.*?)\.([^.]+)$', filename)
  if match:
    name = match.group(1)
    extension = match.group(2)
    return name, extension
  else:
    return filename, ""


def parse_html_links(html_content):
    """
    Parses an HTML document and extracts all text content associated with
    hyperlinks (<a> tags) along with their href attributes.

    Args:
        html_content (str): The HTML content as a string.

    Returns:
        list: A list of dictionaries, where each dictionary contains
              the 'text' of the link and its 'href' value.
    """
    tree = html.fromstring(html_content)
    links_data = []
    for link_element in tree.xpath('//a'):
        text = link_element.text_content().strip()
        href = link_element.get('href')
        if text and href:  # Only include if there's text and a link
            links_data.append({'text': text, 'href': href})
    return links_data


def parse_all_text_and_links(html_content):
    """
    Parses an HTML document and extracts all text content and associates
    href attributes if the text is within an <a> tag.

    Args:
        html_content (str): The HTML content as a string.

    Returns:
        list: A list of dictionaries, where each dictionary contains
              the 'text' and its associated 'href' (which might be None).
    """
    tree = html.fromstring(html_content)
    data = []
    for element in tree.iter():
        text = element.text_content().strip()
        if text:
            href = None
            ancestor_link = element.find_ancestor('a')
            if ancestor_link is not None:
                href = ancestor_link.get('href')
            data.append({'text': text, 'href': href})
    return data


def try_html_parser():
    # Example HTML content
    html_doc = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Example Page</title>
        </head>
        <body>
            <h1>Welcome</h1>
            <p>This is some introductory text.</p>
            <p>Visit our <a href="https://www.example.com">website</a> for more information.</p>
            <ul>
                <li>Item 1</li>
                <li><a href="/about">About Us</a></li>
                <li>Another <a href="mailto:info@example.com">email</a> link.</li>
            </ul>
            <div>
                <p>More text here.</p>
                <a href="https://www.anothersite.net" target="_blank">Another Website</a>
                <span>Some text inside a span.</span>
            </div>
            Text outside any tags.
        </body>
        </html>
        """

    # Option 1: Get text specifically from <a> tags and their hrefs
    links_data = parse_html_links(html_doc)
    print("Text and Hrefs (from <a> tags only):")
    for item in links_data:
        print(item)

    print("\n" + "=" * 30 + "\n")

    # Option 2: Get all text and associate href if within an <a> tag
    all_data = parse_all_text_and_links(html_doc)
    print("All Text with Associated Hrefs:")
    for item in all_data:
        print(item)


def extract_text_per_page_pdf(pdf_path):
    """
    Converts a PDF file to plain text efficiently using PyMuPDF.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted plain text from the PDF.
             Returns an empty string if the PDF cannot be opened.
    """
    pages = []
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                pages.append(page.get_text())
    except Exception as e:
        print(f"Error opening or processing PDF: {e}")

    return pages


def extract_pdf_text_per_page_with_confidence_score(pdf_path, confidence_threshold=0.6):
    """
    Converts a PDF file to plain text and extracts pages and character details where the
    OCR confidence score is below the specified threshold.

    Args:
        pdf_path (str): The path to the PDF file.
        confidence_threshold (float): The confidence threshold (0.0 to 1.0).

    Returns:
        tuple: (page_text, low_confidence_areas) whereas:
            page_text - is a list with the text from each page.
            low_confidence_areas - is a dictionary with the page index and [(char_index, confidence)] as key/value pairs
            whereas: char_index - is the index of the character with low confidence score.
            confidence - is the actual confidence score.

    Example:
        (["text on page 1", "next page text"], {0: (7, 0.55)})
        this indicates that there is an issue with the 8th character on the 1st page
        with a confidence score of 0.55. Closer attention may be required to confirm
        character 'a' was the correct choice. (zero index based)

    """
    low_confidence_areas = {}
    page_text = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text.append(page.get_text())
                text_dict = page.get_text("dict")
                low_confidence_chars_on_page =[]
                char_index = 0

                for block in text_dict.get("blocks", []):
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            for char_data in span.get("chars", []):
                                confidence = char_data.get("confidence", 1.0)
                                if confidence < confidence_threshold:
                                    low_confidence_chars_on_page.append((char_index, confidence))
                                char_index += 1

                if low_confidence_chars_on_page:
                    low_confidence_areas[page_num] = low_confidence_chars_on_page
    except Exception as e:
        print(f"An error occurred: {e}")

    return page_text, low_confidence_areas


def extract_text_per_page_pdfplumber(pdf_path):
    """
    pdfplumber: A more robust library that excels at extracting text and tables from PDFs.
    It often handles complex layouts better than PyPDF2.
    """
    book = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                book.append(page.extract_text())
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

    return book


def extract_text_per_page_djvu(djvu_path):
    """
    Extracts text content for each page of a DJVU file using pydjvutxt.

    Args:
        djvu_path (str): The path to the DJVU file.

    Returns:
        list: A list where each element is the text content of a page.
              Returns empty list if there's an error.
    """
    text_list = []
    logging.debug(f"Extracting text from DJVU file: {djvu_path}...")
    try:
        # conda install conda-forge::djvulibre  # Only available on Linux
        import djvutxt
        document = djvutxt.DjvuDocument(djvu_path)
        for i in range(document.num_pages):
            page_text = document.get_page_text(i)
            text_list.append(page_text)
    except ModuleNotFoundError:
        logging.error("Module Not Found! Please Install: 'conda install conda-forge::python-djvulibrep'")
    except Exception as e:
        logging.exception(f"Cannot process DJVU file: {e}")

    return text_list


def remove_html_comments(html_content: str):
    start="<!--"
    end = "-->"
    remove_end = False
    html_content = html_content.splitlines(keepends=True)
    for idx, line in enumerate(html_content):
        if line.startswith(start) and '<a href=' in line:
            html_content[idx] = line.replace(start, "")
            remove_end = True
            continue

        if remove_end and line.startswith(end):
            html_content[idx] = line.replace(end, "")
            remove_end = False
            continue

    html_content = "".join(html_content)

    return html_content


def extract_html_links_with_cssselect(html_content, href_filter='[href^="/md5/"]') -> list[HtmlElement]:
    """
    Use a CSS selector to find all 'a' tags where the 'href' attribute starts with '/md5/'

    Args:
        html_content (str): html page content. e.g.
            <html>
              <body>
                <p>Some text</p>
                <a href="/md5/abcdef1234567890abcdef1234567890">Link to MD5</a>
                <a href="/other/link">Another Link</a>
                <a href="/md5/0987654321fedcba0987654321fedcba">Another MD5 Link</a>
                <a href="https://example.com">External Link</a>
              </body>
            </html>
        href_filter (str): filter to use for finding specific anchor tags.

    Returns:
        list: a list of [HtmlElement].
    """
    tree = html.fromstring(html_content)
    anchor_tags = tree.cssselect(f'a{href_filter}')

    return anchor_tags


def extract_html_tags_with_cssselect(html_content, tag, tag_filter='') -> list[HtmlElement]:
    """
    Use CSS selector to find all tags where the Tag attribute matches the tag_filter.

    Args:
        html_content (str): html page content. e.g.
            <html>
              <body>
                <p>Some text</p>
                <a href="/md5/abcdef1234567890abcdef1234567890">Link to MD5</a>
                <a href="/other/link">Another Link</a>
                <a href="/md5/0987654321fedcba0987654321fedcba">Another MD5 Link</a>
                <a href="https://example.com">External Link</a>
              </body>
            </html>
        tag (str): html Tag for which to select. e.g. 'a'
        tag_filter (str): filter to use for finding specific anchor tags. e.g.'[href^="/md5/"]'

    Returns:
        list: a list of [HtmlElement].
    """
    tree = html.fromstring(html_content)
    selected_elements = tree.cssselect(f'{tag}{tag_filter}')

    return selected_elements


def is_valid_book_name(name, md5_hash=""):
    valid_book = False
    paths = set(LIBRARY_PATHS)
    if md5_hash:
        paths.add(md5_hash)

    # Check for invalid names
    pattern = "|".join(paths)
    if re.match(pattern, name):
        return valid_book

    # Check for valid names
    valid_book = bool(re.fullmatch(UNICODE_BOOK_NAME_PATTERN, name))

    return valid_book


def mongodb_writer_process(write_queue, log_queue, log_format, date_format, db_name, batch_size, timeout):
    """
    Process that listens on a queue for book data and performs bulk updates/upserts
    to MongoDB in batches.

    Args:
        write_queue (mp.Manager.Queue): The queue to read book update data from.
        log_queue (mp.Manager.Queue): The queue to write logs to file.
        log_format (str): The logging format string.
        date_format (str): The date format string.
        db_name (str): Name of the database to for which to save records.
        batch_size (int): The number of operations per bulk write.
    """
    # Each process needs its own client/manager instance
    books_database = get_local_mongodb_manager(db_name)
    # Dictionary to hold batches per collection
    batches = {}  # e.g. { collection_name: [UpdateOneOp1, UpdateOneOp2,...] }
    total_processed = 0
    file_handler = AsyncFileLogHandler(log_queue, log_format, date_format)
    file_handler.setLevel(logging.INFO)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(log_format, date_format))
    stream.setLevel(logging.WARNING)
    logging.basicConfig(format=log_format,
                        datefmt=date_format,
                        level=logging.INFO,
                        handlers=[file_handler, stream],
                        force=True,
                        )
    logger = logging.getLogger("mongodb_writer_process")
    try:
        logger.info("MongoDB writer process started.")
        item = "Not Started."
        records_updated = []
        while True:
            try:
                # Get data from the queue, with a timeout
                # Format: (collection_name, query_filter, update_data) or None (sentinel)
                item = write_queue.get(timeout=timeout)

                if item is None:  # Sentinel value received
                    logger.info("Sentinel received. Processing final batches...")
                    break  # Exit the loop after processing remaining items

                collection_name, query_filter, update_data = item
                try:
                    records_updated.append(update_data['md5_hash'])
                except Exception as e:
                    logger.exception(f"Cannot retrieve md5_hash for record - {e}: {update_data}")

                # Prepare update operation & use upsert=True to insert if not found, update if found
                update_op = UpdateOne(query_filter, {"$set": update_data}, upsert=True)

                # Add operation to the correct batch
                try:
                    batches[collection_name].append(update_op)
                except KeyError:
                    batches[collection_name] = [update_op]

                total_processed += 1

                # Check if any batch reached the size limit
                if len(batches[collection_name]) >= batch_size:
                    logger.info(
                        f"Writing batch of {len(batches[collection_name])} to collection '{collection_name}'...")
                    collection = books_database.get_collection(collection_name)  # Get collection object
                    collection.bulk_write(batches[collection_name], ordered=False)  # ordered=False can be faster
                    logger.info(f"Batch write to '{collection_name}' complete.")
                    batches[collection_name] = []  # Reset batch for this collection
                    if len(records_updated) > 0:
                        logger.info(f"Last Record Updated: {records_updated[-1]}")
                        records_updated = []
            except queue.Empty:
                # Continue the loop to wait for more items or the sentinel
                logger.debug("Queue empty, continuing to wait...")
                continue
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.exception(f"mongodb writer cannot process item: {item} - {e}")

    except Exception as e:
        print(f"Critical error in MongoDB writer process: {e}")
        logger.exception(f"Critical error in MongoDB writer process: {e}")
    finally:
        logger.info("MongoDB writing final batches...")
        for collection_name, batch in batches.items():
            if batch:  # Check if there are any operations left
                try:
                    logger.info(f"Writing final batch of {len(batch)} to collection '{collection_name}'...")
                    collection = books_database.get_collection(collection_name)
                    collection.bulk_write(batch, ordered=False)
                    logger.info(f"Final batch write to '{collection_name}' complete.")
                except Exception as e:
                    logger.exception(f"Error writing final batch to collection '{collection_name}': {e}")

        logger.info(f"MongoDB writer process finished. Total records updated: {total_processed}")
        books_database.client.close()
        logger.info("Disconnected from MongoDB.")


def try_detect_language():
    text1 = "This is a sample English text."
    text2 = "Ceci est un exemple de texte en français."

    try:
        lang1 = detect(text1)
        lang2 = detect(text2)
        lang3 = detect("你好，世界！"),
        lang4 = detect("Café"),
        print(f"'{text1}' detected as: {lang1}")
        print(f"'{text2}' detected as: {lang2}")
        print(f"'你好，世界！' detected as: {lang3}")
        print(f"'Café' detected as: {lang4}")
        # fast_detect ##################################
        lang1 = fast_detect(text1)
        lang2 = fast_detect(text2)
        lang3 = fast_detect("你好，世界！"),
        lang4 = fast_detect("Café"),
        print(f"\n\n'{text1}' fast_detect as: {lang1}")
        print(f"'{text2}' fast_detect as: {lang2}")
        print(f"'你好，世界！' fast_detect as: {lang3}")
        print(f"'Café' fast_detect as: {lang4}")

        # detect_multilingual ##################################
        lang1 = detect_multilingual(text1, low_memory=False, k=3)
        lang2 = detect_multilingual(text2, low_memory=False, k=3)
        lang3 = detect_multilingual("你好，世界！", low_memory=False, k=3),
        lang4 = detect_multilingual("Café", low_memory=False, k=3),
        print(f"\n\n'{text1}' detect_multilingual as: {lang1}")
        print(f"'{text2}' detect_multilingual as: {lang2}")
        print(f"'你好，世界！' detect_multilingual as: {lang3}")
        print(f"'Café' detect_multilingual as: {lang4}")

    except Exception as e:
        print(f"Error detecting language: {e}")


def translate_text_to_english_google(text, project_id="woven-scene-452520-s0"):
    """
    Translates text to English using the Google Cloud Translate API v3. Requires a PAID subscription.

    You will need to install the google-cloud-translate library if you haven't already:
    pip install google-cloud-translate

    Before running this code, you need to set up your Google Cloud project and authentication:
     * Create a Google Cloud Project: If you don't have one already, create a new project in the Google Cloud Console.
     * Enable the Cloud Translation API: Go to the API Library in your project and enable the "Cloud Translation API".
     * Create a Service Account:
       * Go to the "Service accounts" page in the Google Cloud Console.
       * Click on "Create service account".
       * Give your service account a name and description.
       * Click "Create and continue".
       * Grant this service account the "Cloud Translation API User" role.
       * Click "Continue".
       * Click "Create Key".
       * Choose JSON as the Key type and click "Create". This will download a JSON file containing your service account credentials.
    Set the GOOGLE_APPLICATION_CREDENTIALS environment variable:
        * Replace "path/to/your/credentials.json" in the Python code with the actual path to the downloaded JSON credentials file on your system.

    """

    credentials_path = "C:/Users/JulianJoseph/Desktop/WORKSTATION/AngeAI/woven-scene-452520-s0-747f4d4578fc.json"
    # Set the environment variable for Google Cloud credentials file path
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

    client = translate.TranslationServiceClient()
    location = "global"  # Or the region your project is in
    parent = f"projects/{project_id}/locations/{location}"

    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",  # Can be "text/html"
            "target_language_code": "en",
        }
    )

    for translation in response.translations:
        return translation.translated_text

    return text


def translate_text_to_english_argos(input_text, from_lang_code):
    """
    Efficiently translates a list of texts from a specified language to English
    using the Argos Translate library.

    Args:
        input_text (str): The text to be translated.
        from_lang_code (str): The ISO 639-1 code of the source language
            (e.g., 'fr' for French, 'es' for Spanish).

    Returns:
        str|None: The translated strings in English. or
        None if the language pair is not supported.
    """
    try:
        # TODO - Create translation class and make installed_languages a global variable to increase efficiency
        # Check if the translation package is available
        packages = argos_translate.get_installed_languages()
        source_lang = next((lang for lang in packages if lang.code == from_lang_code), None)
        english_lang = next((lang for lang in packages if lang.code == 'en'), None)

        if not source_lang or not english_lang:
            logging.info(f"Translation from '{from_lang_code}' to 'en' is not supported.")
            return None

        # Get the translation function
        translation_fn = source_lang.get_translation(english_lang)
        if not translation_fn:
            logging.info(f"Translation from '{from_lang_code}' to 'en' is not available.")
            return None

        # Apply translation to the entire list at once (more efficient)
        translated_texts = translation_fn.translate(input_text)

        return translated_texts

    except Exception as e:
        logging.exception(f"Cannot translate text to english. - {e}")
        return None


def download_all_to_english_language_pack_argos():
    """
    Downloads language packages for all available source languages to English for Argos Translate library.
    """
    try:
        # Update the package index to get the latest list of available packages
        package.update_package_index()
        available_packages = package.get_available_packages()
        downloaded_count = 0
        total_count = 0

        installed_packages = {pack.package_path.stem for pack in package.get_installed_packages()}
        working_dir = tempfile.mkdtemp()
        logging.info(f"Using temp dir to download Argos language packages: {working_dir}")
        available_packages = sorted(available_packages, key=lambda x: x.get_description())
        for p in available_packages:
            file_url = p.links[0]
            if p.to_code == 'en':
                total_count += 1
                resp = requests.head(file_url)
                size = resp.headers.get("Content-Length", "0")
                size = int(size) / (1024 * 1024)
                try:
                    file_name = file_url.split('/')[-1]  # Replace with your desired filename
                    file_name = os.path.join(working_dir, file_name)
                    package_path = Path(file_name)
                    if package_path.stem in installed_packages:
                        logging.info(f"Argos '{p.get_description()}' already installed.")
                        continue

                    logging.info(f"Downloading package: {p.get_description()} (Size: {size} MB)")
                    download_file_with_rich_progressbar(file_url, file_name)
                    package.install_from_path(package_path)
                    downloaded_count += 1
                except Exception as e:
                    logging.info(f"Cannot download {p.get_description()} package. - {e}")

        logging.info(f"\nAttempted to download {total_count} packages to English.")
        logging.info(f"Successfully downloaded {downloaded_count} new packages.")

    except Exception as e:
        logging.exception(f"Cannot install argos translate packages: {e}")


def try_translate_text_to_english_argos():
    french_texts = "Bonjour le monde. Comment ça va? Merci beaucoup"
    spanish_texts = "Hola mundo. ¿Cómo estás? Muchas gracias"

    start_time = time.time()
    english_translations_fr = translate_text_to_english_argos(french_texts, 'fr')
    end_time = time.time()

    if english_translations_fr:
        print("French to English:")
        print(f"Original: {french_texts}, Translated: {english_translations_fr}")
        print(f"Translation time: {end_time - start_time:.4f} seconds")
        print("-" * 20)

    start_time = time.time()
    english_translations_es = translate_text_to_english_argos(spanish_texts, 'es')
    end_time = time.time()

    if english_translations_es:
        print("Spanish to English:")
        print(f"Original: {spanish_texts}, Translated: {english_translations_es}")
        print(f"Translation time: {end_time - start_time:.4f} seconds")
        print("-" * 20)

    # Example of an unsupported language pair
    german_texts = "Hallo Welt. Wie geht es dir?"
    start_time = time.time()
    english_translations_de = translate_text_to_english_argos(german_texts, 'de')
    end_time = time.time()
    if english_translations_de is None:
        print("German to English translation was not attempted due to lack of support.")
    else:
        print("German to English:")
        print(f"Original: {german_texts}, Translated: {english_translations_de}")
        print(f"Translation time: {end_time - start_time:.4f} seconds")
        print("-" * 20)


def get_page_content(url, sleep_time=1, max_retry=10):
    server = get_url_root(url)
    html_content = ""
    for x in range(max_retry):
        try:
            response = requests.get(url)
            if response.ok:
                html_content = response.text
                break
            else:
                logging.info(f"Sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)
                response = requests.get(server)
                if response.ok:
                    logging.info(f"Connected to Server: {server}")
                    continue
                else:
                    sleep_time += 5
                    logging.warning(f"Connection Failed: {response.text}")
                    logging.info(f"Sleeping for {sleep_time} seconds...")
                    time.sleep(sleep_time)
        except Exception as e:
            if x < max_retry - 1:
                logging.error(f"Cannot get page content. Retry in 1 minute - {e}")
                time.sleep(60)
            else:
                logging.exception(f"Cannot get page content! - {e}")
        finally:
            time.sleep(0.1)

    return html_content


def get_local_mongodb_manager(database_name="ASI"):
    connect_string = "mongodb://localhost:27017"
    kwargs = {
        "compressors": ['zlib', 'zstd', 'snappy']
    }
    database = MongoDBManager(connect_string, database_name, **kwargs)

    return database


def run_extract_pdf_with_confidence_score():
    pdf_file = "Computer-Algebra-b936bd81d127cef553d773d4736cb9de.pdf"  # Replace with the path to your PDF file
    threshold = 0.6
    pages, low_confidence = extract_pdf_text_per_page_with_confidence_score(pdf_file, threshold)
    if low_confidence:
        for page_number in low_confidence:
            char_idx, confidence = low_confidence[page_number]
            print(f"Character at index:{char_idx} on Page: {page_number} only have a confidence score of: {confidence}")
    else:
        print(f"No low confidence OCR areas found below the threshold of {threshold}.")


def get_book_catalog_with_tags():
    # Granular Book Tags
    grammar_ref = {"Grammar Rules", "Syntax", "Parts of Speech", "Sentence Structure", "Verb Conjugation", "Punctuation"} # Additional keywords: vocabulary|semantics
    build_vocab = {"Word Lists", "Root Words", "Prefixes", "Suffixes", "Idioms", "Phrasal Verbs", "Contextual Learning"}
    reading_practice = {"Graded Readers", "Short Stories", "Novellas", "Articles", "Comprehension Exercises"}
    early_childhood = {"Preschool Activities", "Kindergarten Readiness", "Child Development (Early Years)"}
    general_dictionaries = {"Comprehensive Definitions", "Standard Vocabulary"}
    subject_specific_dictionaries = {"Technical Terms (Specific Field)", "Discipline-Specific Vocabulary"}
    etymological_dictionaries = {"Word Origins", "Historical Development of Words"}
    thesauruses = {"Synonyms", "Antonyms", "Word Choice"}
    encyclopedias = {"General Encyclopedia", "Subject-Specific Encyclopedia"}
    legal_codes = {"Statutory Law", "Legislation"}
    manuals_and_handbooks = {"Quick Lookups", "Practical Information (Reference)"}
    atlases_and_maps = {"Geography", "Cartography", "Navigation"}
    directories = {"Contact Information", "Lists of Organizations"}
    bibliographies_and_catalogs = {"List of Publications", "Research Resources"}
    writing = {"Essay Writing", "Report Writing", "Creative Writing", "Formal Writing", "Informal Writing", "Editing", "Proofreading"}
    dictionary_use = {"Pronunciation Guide", "Definitions", "Word Origins", "Usage Examples"}
    thesaurus_use = {"Synonyms", "Antonyms", "Related Words"}
    test_prep = {"SAT Prep", "GED Prep", "GRE Prep", "TOEFL Prep", "ACT Prep", "IELTS Prep"}
    professional_exams = {"Bar Exam Prep", "Medical Licensing Exam Prep", "Engineering Exam Prep", "Accounting Exam Prep (e.g., CPA)"}
    certificate_exams = {"IT Certifications", "Project Management Certifications", "Language Proficiency Certifications"}
    home_schooling = {"Curriculum Guides", "Activity Ideas (Homeschool)", "Learning Resources (Homeschool)"}

    # Book Categories
    book_categories_with_tags = {
        "Learning & Education": {
            "Early Childhood Education": early_childhood,
            "Reference & Lookup": {
                "Dictionaries": {
                    "General Dictionaries": general_dictionaries,
                    "Subject-Specific Dictionaries": subject_specific_dictionaries,
                    "Etymological Dictionaries": etymological_dictionaries
                },
                "Thesauruses": thesauruses,
                "Encyclopedias": encyclopedias,
                "Legal Codes": legal_codes,
                "Manuals & Handbooks (Reference Focused)": manuals_and_handbooks,
                "Atlases & Maps": atlases_and_maps,
                "Directories": directories,
                "Bibliographies & Catalogs": bibliographies_and_catalogs
            },
            "Language Learning": {
                "Grammar Reference": grammar_ref,
                "Vocabulary Building": build_vocab,
                "Reading Practice": reading_practice,
                "Writing Aid": writing,
                "Dictionary Use": dictionary_use,
                "Thesaurus Use": thesaurus_use
            },
            "Subject Learning": {
                "Law": {
                    "Legal Principles": ["Jurisprudence", "Constitutional Law", "Criminal Law", "Civil Law",
                                         "Contract Law", "Property Law", "Tort Law"],
                    "Case Studies": ["Precedent", "Legal Analysis", "Judicial Review"],
                    "Legislation Reference": ["Statutes", "Acts", "Codes", "Regulations"]
                },
                "Science": {
                    "Biology": ["Cell Biology", "Genetics", "Ecology", "Evolution", "Anatomy", "Physiology", "Botany",
                                "Zoology"],
                    "Chemistry": ["Organic Chemistry", "Inorganic Chemistry", "Physical Chemistry", "Biochemistry",
                                  "Analytical Chemistry"],
                    "Physics": ["Mechanics", "Thermodynamics", "Electromagnetism", "Optics", "Quantum Mechanics",
                                "Relativity"],
                    "Astronomy": ["Cosmology", "Astrophysics", "Planetary Science", "Stellar Evolution"],
                    "Earth Science": ["Geology", "Oceanography", "Meteorology", "Paleontology"],
                    "Environmental Science": ["Ecology", "Conservation", "Pollution", "Climate Change"]
                },
                "History": {
                    "Ancient History": ["Classical Civilizations", "Early Civilizations", "Archaeology"],
                    "Medieval History": ["Middle Ages", "Feudalism", "Crusades"],
                    "Modern History": ["World Wars", "Industrial Revolution", "Post-Colonialism"],
                    "Regional History": ["American History", "European History", "Asian History", "African History",
                                         "Latin American History"],
                    "Biographies & Memoirs (Historical Context)": ["Historical Figures",
                                                                   "Personal Accounts (Historical Significance)"]
                },
                "Mathematics": {
                    "Algebra": ["Equations", "Inequalities", "Functions", "Polynomials", "Linear Algebra"],
                    "Calculus": ["Differential Calculus", "Integral Calculus", "Multivariable Calculus"],
                    "Geometry": ["Euclidean Geometry", "Analytic Geometry", "Trigonometry"],
                    "Statistics": ["Probability", "Data Analysis", "Statistical Inference"],
                    "Discrete Mathematics": ["Logic", "Set Theory", "Graph Theory", "Combinatorics"]
                },
                "Social Sciences": {
                    "Psychology": ["Cognitive Psychology", "Social Psychology", "Developmental Psychology",
                                   "Clinical Psychology", "Behavioral Psychology"],
                    "Sociology": ["Social Theory", "Social Stratification", "Social Change", "Culture",
                                  "Urban Sociology"],
                    "Economics": ["Microeconomics", "Macroeconomics", "Econometrics", "International Economics"],
                    "Political Science": ["Political Theory", "Comparative Politics", "International Relations",
                                          "Public Policy"],
                    "Anthropology": ["Cultural Anthropology", "Archaeology", "Linguistic Anthropology",
                                     "Biological Anthropology"]
                },
                "Technology & Engineering": {
                    "Computer Science": ["Algorithms", "Data Structures", "Programming Languages",
                                         "Software Engineering", "Artificial Intelligence"],
                    "Software Development": ["Web Development", "Mobile Development", "Database Management"],
                    "Electrical Engineering": ["Circuit Theory", "Electronics", "Power Systems", "Signal Processing"],
                    "Mechanical Engineering": ["Thermodynamics", "Fluid Mechanics", "Solid Mechanics", "Design",
                                               "Manufacturing"],
                    "Civil Engineering": ["Structural Engineering", "Transportation Engineering",
                                          "Geotechnical Engineering", "Water Resources"]
                },
                "Arts & Humanities": {
                    "Literature": ["Literary Criticism", "Literary Theory", "Poetry Analysis", "Drama Analysis",
                                   "Fiction Analysis", "World Literature"],
                    "Art History": ["Renaissance Art", "Modern Art", "Contemporary Art", "Specific Art Movements"],
                    "Music History & Theory": ["Musicology", "Music Theory", "Harmony", "Counterpoint"],
                    "Philosophy": ["Epistemology", "Metaphysics", "Ethics", "Logic", "Political Philosophy",
                                   "History of Philosophy"],
                    "Religious Studies": ["Religious History", "Religious Texts", "Religious Practices", "Theology"]
                }
            },
            "Skill Development": {
                "Writing Skills": {
                    "Creative Writing": ["Fiction Writing", "Poetry Writing", "Screenwriting", "Playwriting"],
                    "Technical Writing": ["Documentation", "Reports (Technical)", "User Manuals"],
                    "Business Writing": ["Emails", "Memos", "Proposals", "Business Plans"],
                    "Academic Writing": ["Essays (Academic)", "Research Papers", "Theses", "Dissertations"]
                },
                "Research Skills": ["Information Literacy", "Source Evaluation", "Citation Methods",
                                    "Academic Databases"],
                "Study Skills": ["Note-Taking", "Time Management (Study)", "Exam Preparation Techniques",
                                 "Memory Techniques"],
                "Critical Thinking": ["Logic", "Analysis", "Evaluation", "Problem Solving"],
                "Problem Solving": ["Analytical Thinking", "Decision Making", "Troubleshooting"],
                "Communication Skills": ["Verbal Communication", "Nonverbal Communication", "Active Listening",
                                         "Interpersonal Skills"],
                "Presentation Skills": ["Public Speaking", "Visual Aids", "Delivery Techniques"],
                "Coding & Programming": ["Specific Programming Languages (e.g., Python, Java, C++)",
                                         "Web Development (Coding)", "Data Science (Coding)"],
                "Data Analysis": ["Statistical Analysis", "Data Visualization", "Data Mining"],
                "Foreign Language Acquisition (General)": ["Language Learning Strategies", "Immersion Techniques",
                                                           "Cultural Awareness (Language)"]
            },
            "Test Preparation": {
                "Standardized Tests (SAT, GRE, TOEFL)": test_prep,
                "Professional Exams": professional_exams,
                "Certification Exams": certificate_exams
            },
            "Homeschooling Resources": home_schooling,
        },
        "Entertainment & Leisure": {
            "Reading for Pleasure": {
                "Fiction": {
                    "General Fiction": ["Literary Fiction", "Contemporary Fiction"],
                    "Mystery & Thriller": ["Suspense", "Crime Fiction", "Psychological Thrillers"],
                    "Science Fiction & Fantasy": ["Space Opera", "Dystopian", "Urban Fantasy", "High Fantasy"],
                    "Romance": ["Contemporary Romance", "Historical Romance", "Paranormal Romance"],
                    "Historical Fiction": ["Based on Real Events (Fiction)", "Set in the Past (Fiction)"],
                    "Contemporary Fiction": ["Modern Settings (Fiction)", "Relatable Themes (Fiction)"],
                    "Short Stories": ["Anthologies (Fiction)", "Collections (Single Author)"],
                    "Plays & Drama": ["Scripts", "Dramatic Literature"],
                    "Poetry": ["Collections (Poetry)", "Individual Poems", "Poetry Anthologies"],
                    "Graphic Novels & Comics (Fiction)": ["Superhero Comics", "Manga (Fiction)",
                                                          "Independent Comics (Fiction)"]
                },
                "Non-Fiction (Narrative)": {
                    "Biographies & Memoirs (Personal Accounts)": ["Autobiographies", "Personal Narratives"],
                    "Travel Writing": ["Adventure Travel", "Cultural Exploration"],
                    "Essays & Collections": ["Personal Reflections", "Thematic Essays"],
                    "True Crime": ["Criminal Investigations", "Real-Life Accounts of Crime"],
                    "Narrative History": ["Story-Driven History", "Popular History"],
                    "Personal Development (Leisure Reading Focus)": ["Inspirational", "Motivational (Leisure)"]
                }
            },
            "Activity Books": {
                "Puzzles & Games": ["Crosswords", "Sudoku", "Word Searches", "Logic Puzzles"],
                "Coloring Books": ["Adult Coloring Books", "Children's Coloring Books"],
                "Craft & Hobby Books (Leisure Focused)": ["Knitting", "Sewing", "Painting (Hobby)", "Drawing (Hobby)"],
                "Cookbooks (Leisure Focused)": ["General Cooking", "Baking (Leisure)", "Specific Cuisines (Leisure)"],
                "Gardening (Leisure Focused)": ["Home Gardening", "Flower Gardening", "Vegetable Gardening"]
            },
            "Humor & Satire": ["Comedy Writing", "Parody"]
        },
        "Professional & Practical Use": {
            "Career Development": {
                "Job Searching & Networking": ["Job Hunting Strategies", "Networking Tips", "Online Job Boards"],
                "Interview Skills": ["Behavioral Questions", "Technical Interviews", "Interview Preparation"],
                "Resume & Cover Letter Writing": ["CV Writing", "Application Documents"],
                "Professional Advancement": ["Career Growth", "Leadership Skills (Professional)",
                                             "Management Skills (Professional)"],
                "Career Change": ["Transitioning Careers", "Exploring New Fields"]
            },
            "Business & Finance": {
                "Management & Leadership": ["Team Management", "Strategic Planning", "Organizational Behavior"],
                "Marketing & Sales": ["Digital Marketing", "Social Media Marketing", "Sales Techniques", "Branding"],
                "Finance & Investing": ["Personal Finance", "Stock Market", "Real Estate Investing",
                                        "Financial Analysis"],
                "Entrepreneurship": ["Starting a Business", "Small Business Management", "Business Innovation"],
                "Economics (Practical Application)": ["Business Economics", "Market Analysis"],
                "Business Law": ["Contracts (Business)", "Intellectual Property", "Corporate Law"],
                "Project Management": ["Agile", "Scrum", "Project Planning"]
            },
            "Technical Manuals": {
                "Software & Hardware Manuals": ["User Guides (Software)", "Installation Guides (Hardware)"],
                "Engineering Manuals": ["Design Specifications", "Technical Drawings"],
                "Medical Manuals": ["Diagnostic Procedures", "Treatment Protocols"],
                "Automotive Manuals": ["Repair Guides", "Maintenance Schedules"],
                "Construction Manuals": ["Building Codes", "Safety Procedures (Construction)"]
            },
            "Self-Help & Personal Development (Practical Application)": {
                "Productivity & Time Management": ["Organization Skills", "Efficiency Techniques"],
                "Goal Setting": ["Achieving Goals", "Action Planning"],
                "Stress Management": ["Coping Mechanisms", "Relaxation Techniques"],
                "Relationships & Communication (Practical)": ["Interpersonal Skills (Practical)",
                                                              "Conflict Resolution"],
                "Emotional Intelligence": ["Self-Awareness", "Empathy"]
            },
            "Health & Wellness (Practical Application)": {
                "Fitness & Exercise": ["Workout Routines", "Exercise Guides"],
                "Nutrition & Diet": ["Healthy Eating", "Meal Planning", "Specific Diets"],
                "Mental Health (Practical Guidance)": ["Anxiety Management", "Depression Support", "Well-being"],
                "Specific Health Conditions (Information & Management)": ["Diabetes Management", "Heart Health",
                                                                          "Allergies"]
            },
            "Home & Garden (Practical Application)": {
                "Home Improvement & Repair": ["DIY Projects", "Home Maintenance"],
                "Gardening (Practical Guides)": ["Plant Care", "Landscaping"],
                "Cooking & Baking (Practical Guides)": ["Recipes", "Techniques (Cooking)", "Baking (Practical)"],
                "Interior Design": ["Home Decor", "Space Planning"]
            },
            "Travel Guides & Phrasebooks (Practical Use)": ["Destination Guides", "Language Phrases for Travel"],
            "Craft & Hobby Books (Practical Guides)": ["Instructions (Crafts)", "Techniques (Hobbies)"]
        },
        "Spiritual & Philosophical": {
            "Religion & Spirituality": {
                "Specific Religions": ["Christianity", "Islam", "Judaism", "Buddhism", "Hinduism", "Sikhism",
                                       "Other Religions"],
                "Comparative Religion": ["Interfaith Studies", "Religious Traditions"],
                "Spiritual Practices": ["Meditation (Spiritual)", "Prayer", "Mindfulness (Spiritual)"]
            },
            "Philosophy": {
                "Ethics & Morality": ["Moral Philosophy", "Applied Ethics"],
                "Metaphysics": ["Reality", "Existence", "Time"],
                "Epistemology": ["Theory of Knowledge", "Belief and Justification"],
                "Logic": ["Reasoning", "Argumentation"],
                "Political Philosophy": ["Justice", "Rights", "Government"],
                "History of Philosophy": ["Ancient Philosophy", "Modern Philosophy"]
            },
            "Meditation & Mindfulness": ["Mindfulness Techniques", "Meditation Practices",
                                         "Stress Reduction (Spiritual)"]
        },
        "Social & Cultural Issues": {
            "Sociology (Social Issues Focus)": ["Social Inequality", "Social Justice (Sociology)", "Cultural Norms",
                                                "Social Change (Sociology)"],
            "Political Science (Social Issues Focus)": ["Human Rights", "Social Policy", "Political Activism"],
            "Cultural Studies": ["Popular Culture", "Media Studies", "Identity Studies"],
            "Environmental Issues": ["Sustainability", "Conservation", "Climate Change (Social Impact)"],
            "Social Justice": ["Equality", "Equity", "Human Rights Advocacy"],
            "Psychology (Social Issues Focus)": ["Social Cognition", "Group Dynamics", "Social Influence"]
        }
    }

    return book_categories_with_tags


def populate_tag_keyword_mapping(structure):
    """Create the reverse-lookup dictionary"""
    tag_keyword_mapping = {}
    for key, value in structure.items():
        if isinstance(value, dict):
            populate_tag_keyword_mapping(value)
        elif isinstance(value, set):
            for tag in value:
                tag_lower = tag.lower()
                # Initialize with the tag itself as a key
                if tag_lower not in tag_keyword_mapping:
                    tag_keyword_mapping[tag_lower] = [tag_lower]
                # You can add more related keywords here manually or through some logic
                # For example:
                if tag_lower == "grammar rules":
                    tag_keyword_mapping[tag_lower].extend(["grammatical", "syntax", "sentence structure"])
                elif tag_lower == "vocabulary building":
                    tag_keyword_mapping[tag_lower].extend(["word power", "learn words", "expand vocabulary"])
                elif tag_lower == "essay writing":
                    tag_keyword_mapping[tag_lower].extend(["writing essays", "essay format"])
                elif tag_lower == "legal principles":
                    tag_keyword_mapping[tag_lower].extend(["law fundamentals", "legal basis"])
                elif tag_lower == "cell biology":
                    tag_keyword_mapping[tag_lower].extend(["cellular biology", "biology of cells"])
                elif tag_lower == "python":
                    tag_keyword_mapping[tag_lower].extend(["python programming", "python language"])
                # Add more mappings for other tags as needed


def extract_tags_from_description_enhanced(description, keyword_to_tag_map):
    """
    Extracts a set of granular tags from a book's description using a keyword-to-tag mapping.

    Args:
        description (str): The name or description of the book.
        keyword_to_tag_map (dict): A dictionary mapping keywords to granular tags.

    Returns:
        set: A set of unique granular tags found (via keywords) in the description.
    """
    description = description.lower()
    tags_found = set()

    for tag, keywords in keyword_to_tag_map.items():
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if re.search(r'\b' + re.escape(keyword_lower) + r'\b', description) or re.search(re.escape(keyword_lower), description):
                tags_found.add(tag)
                # Once a keyword for a tag is found, we can move to the next tag
                break

    return tags_found


def suggest_embedding_size(model: nn.Module, vocab_size: int) -> int:
    """
       Suggests an embedding size (integer) based on model complexity and
       the vocabulary size rounded to the nearest thousand.
       Args:
           model: A torch.nn.Module instance representing the model.
           vocab_size: The size of the vocabulary.
       Returns:
           An integer representing the suggested embedding size.
       """
    num_params = sum(p.numel() for p in model.parameters())
    rounded_vocab_size = round(vocab_size / 1000) * 1000
    if rounded_vocab_size == 0:
        rounded_vocab_size = 1000  # Avoid division by zero
    # Heuristic multipliers based on model type and size
    if isinstance(model, nn.Transformer):
        complexity_factor = math.log(num_params + 1, 10) * 1.5 + 5  # Adjust scaling and base
        base_multiplier = 25
    elif isinstance(model, nn.LSTM) or isinstance(model, nn.GRU):
        complexity_factor = math.log(num_params + 1, 10) * 1 + 3
        base_multiplier = 35
    elif isinstance(model, nn.Sequential):
        complexity_factor = math.log(num_params + 1, 10) * 0.5 + 5  # Higher base for simpler models
        base_multiplier = 50
    else:
        complexity_factor = math.log(num_params + 1, 10) * 0.8 + 4
        base_multiplier = 40
    # Vocabulary size influence (inverse, larger vocab -> larger embedding)
    vocab_influence = math.log(rounded_vocab_size + 1, 10) * 1.5
    # Combine factors to get a dynamic multiplier
    dynamic_multiplier = base_multiplier + complexity_factor - vocab_influence
    # Ensure the multiplier is within a reasonable range
    clipped_multiplier = max(10, min(80, dynamic_multiplier))
    # Calculate and return the suggested embedding size
    suggested_dim = int(rounded_vocab_size / clipped_multiplier)
    return suggested_dim

if __name__=='__main__':
    _book_path = "C:/Users/JulianJoseph/Downloads/0/e69b9f480d32e49faa67b8cf1f810403"
    text = extract_text_per_page_djvu(_book_path)
    vocab_size = 20000
    # Example models
    simple_seq_model = nn.Sequential(nn.Embedding(vocab_size, 64), nn.Linear(64, 10))
    lstm_model = nn.LSTM(64, 128, num_layers=2)
    transformer_model = nn.Transformer(d_model=128, nhead=2, dim_feedforward=512, num_encoder_layers=2,
                                       num_decoder_layers=2)
    # Get suggested embedding sizes
    embedding_size_simple = suggest_embedding_size(simple_seq_model, vocab_size)
    embedding_size_lstm = suggest_embedding_size(lstm_model, vocab_size)
    embedding_size_transformer = suggest_embedding_size(transformer_model, vocab_size)
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Simple Sequential Model - Suggested Embedding Size: {embedding_size_simple}")
    print(f"LSTM Model - Suggested Embedding Size: {embedding_size_lstm}")
    print(f"Transformer Model - Suggested Embedding Size: {embedding_size_transformer}")
    # Example with a larger vocabulary
    large_vocab_size = 100000
    embedding_size_transformer_large_vocab = suggest_embedding_size(transformer_model, large_vocab_size)
    print(f"\nLarge Vocabulary Size: {large_vocab_size}")
    print(f"Transformer Model - Suggested Embedding Size: {embedding_size_transformer_large_vocab}")
    #####################################
    #extract_anna_torrent_books_collection()
    #run_html_parser()
    #run_prepare_parquet_dataset()
    #copy_training_dataset_to_docker()

"""
You are absolutely correct! That's a much more robust and practical approach to handle cases where the exact granular tags might not appear in the book's description. Creating a reverse-lookup dictionary will map more common or related keywords found in descriptions to our controlled granular tags.
Here's how we can create that dictionary and modify the extract_tags_from_description function:
import re

# Your existing book_categories_with_tags dictionary (as defined previously)
book_categories_with_tags = {
    "Learning & Education": {
        "Language Learning": {
            "Grammar Reference": ["Grammar Rules", "Syntax", "Parts of Speech", "Sentence Structure", "Verb Conjugation", "Punctuation"],
            "Vocabulary Building": ["Word Lists", "Root Words", "Prefixes", "Suffixes", "Idioms", "Phrasal Verbs", "Contextual Learning"],
            "Reading Practice": ["Graded Readers", "Short Stories (Language)", "Novellas (Language)", "Articles (Language)", "Comprehension Exercises"],
            "Writing Aid": ["Essay Writing", "Report Writing", "Creative Writing (Language)", "Formal Writing", "Informal Writing", "Editing", "Proofreading"],
            "Dictionary Use": ["Pronunciation Guide", "Definitions", "Word Origins", "Usage Examples"],
            "Thesaurus Use": ["Synonyms", "Antonyms", "Related Words"]
        },
        "Subject Learning": {
            "Law": {
                "Legal Principles": ["Jurisprudence", "Constitutional Law", "Criminal Law", "Civil Law", "Contract Law", "Property Law", "Tort Law"],
                "Case Studies": ["Precedent", "Legal Analysis", "Judicial Review"],
                "Legislation Reference": ["Statutes", "Acts", "Codes", "Regulations"]
            },
            "Science": {
                "Biology": ["Cell Biology", "Genetics", "Ecology", "Evolution", "Anatomy", "Physiology", "Botany", "Zoology"],
                "Chemistry": ["Organic Chemistry", "Inorganic Chemistry", "Physical Chemistry", "Biochemistry", "Analytical Chemistry"],
                "Physics": ["Mechanics", "Thermodynamics", "Electromagnetism", "Optics", "Quantum Mechanics", "Relativity"],
                "Astronomy": ["Cosmology", "Astrophysics", "Planetary Science", "Stellar Evolution"],
                "Earth Science": ["Geology", "Oceanography", "Meteorology", "Paleontology"],
                "Environmental Science": ["Ecology", "Conservation", "Pollution", "Climate Change"]
            },
            # ... (rest of your book_categories_with_tags)
        },
        # ... (rest of your book_categories_with_tags)
    },
    # ... (rest of your book_categories_with_tags)
}


populate_tag_keyword_mapping(book_categories_with_tags)
print("Tag Keyword Mapping:")
print(tag_keyword_mapping)
print("-" * 30)



# Example Usage with the enhanced function:
book_description1 = "A guide covering grammatical concepts and sentence syntax."
tags1 = extract_tags_from_description_enhanced(book_description1, tag_keyword_mapping)
print(f"Tags for '{book_description1}': {tags1}")

book_description2 = "Learn to program with Python, focusing on its data structures and algorithms."
tags2 = extract_tags_from_description_enhanced(book_description2, tag_keyword_mapping)
print(f"Tags for '{book_description2}': {tags2}")

book_description3 = "An introduction to the fundamental laws of quantum mechanics."
tags3 = extract_tags_from_description_enhanced(book_description3, tag_keyword_mapping)
print(f"Tags for '{book_description3}': {tags3}")

book_description4 = "A crime novel filled with suspense and intricate mystery."
tags4 = extract_tags_from_description_enhanced(book_description4, tag_keyword_mapping)
print(f"Tags for '{book_description4}': {tags4}")

book_description5 = "Practical techniques for managing stress and achieving relaxation."
tags5 = extract_tags_from_description_enhanced(book_description5, tag_keyword_mapping)
print(f"Tags for '{book_description5}': {tags5}")

Key Changes and Explanation:
 * tag_keyword_mapping Dictionary:
   * This new dictionary has lowercase versions of your granular tags (e.g., "grammar rules") as keys.
   * The value for each key is a list of lowercase keywords or phrases that might appear in a book's description and would indicate that the book falls under that granular tag.
   * Initially, each granular tag is mapped to itself.
   * You'll need to manually populate this dictionary with relevant synonyms, related terms, and common phrasings for each granular tag. The example code shows how you can start adding these mappings.
 * populate_tag_keyword_mapping Function:
   * This function iterates through your book_categories_with_tags dictionary to extract all the granular tags and initialize them as keys in the tag_keyword_mapping.
   * You'll extend this function (or manually edit tag_keyword_mapping) to add the lists of related keywords for each tag.
 * extract_tags_from_description_enhanced Function:
   * This function now takes the keyword_to_tag_map (which is our tag_keyword_mapping) as input.
   * It iterates through each tag and its associated keywords in the mapping.
   * For each keyword, it searches (case-insensitively using re.search) within the book's description.
   * If a keyword is found, the corresponding tag is added to the tags_found set, and we can break out of the inner loop (for keywords of that tag) since we've already identified the tag.
Benefits of this Approach:
 * More Flexible Matching: You can now identify granular tags even if the exact tag words aren't present in the description.
 * Improved Accuracy: By mapping related keywords, you increase the likelihood of correctly categorizing books.
 * Maintainability: You have a central place (tag_keyword_mapping) to manage the relationships between keywords and your controlled vocabulary of granular tags.
Next Steps:
The most important step now is to thoroughly populate the tag_keyword_mapping dictionary. Think about the different ways authors or descriptions might refer to the concepts covered by each granular tag. The more comprehensive this mapping is, the more accurate your tagging system will be.
"""
