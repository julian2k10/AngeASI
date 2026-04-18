import os
import zarr
import zarr.storage
import h5py
import itertools
import shutil
import time
import functools
import json
import logging
from abc import ABC, abstractmethod
from queue import Queue
from threading import Thread
from typing import Any, List, Dict, Mapping, Tuple
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
from pymongo.synchronous.collection import Collection
import numpy as np
import torch

logger = logging.getLogger('AngeASI.database_util')

def time_it(func):
    """Timing Decorator"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds.")
        return result

    return wrapper


class DatabaseManager(ABC):
    """
    Abstract base class for managing database connections and operations.
    Provides a consistent interface for adding, updating, and deleting records,
    regardless of the underlying database type.
    """

    @abstractmethod
    def connect(self) -> None:
        """Establishes a connection to the database."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Closes the connection to the database."""
        pass

    @abstractmethod
    def add_record(self, *args, **kwargs) -> Any:
        """
        Adds a new record to the specified table.
        """
        pass

    @abstractmethod
    def update_record(self, *args, **kwargs) -> bool:
        """
        Updates an existing record in the specified table.

        Returns:
            True if the update was successful, False otherwise.
        """
        pass

    @abstractmethod
    def delete_record(self, *args, **kwargs) -> bool:
        """
        Deletes a record from the specified table.

        Returns:
            True if the deletion was successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_record(self, *args, **kwargs) -> Any | None:
        """
        Retrieves a record from the specified table.
        """
        pass

    @abstractmethod
    def get_all_records(self, table_name: str) -> Queue:
        """
        Retrieves all records from the specified table.

        Args:
            table_name: The name of the table to retrieve records from.

        Returns:
            A queue with records in the specified table.
        """
        pass

    @abstractmethod
    def get_collection(self, collection_name: str) -> Any | None:
        """
        Retrieves the specified collection from the database.

        Args:
            collection_name: The name of the collection to retrieve from the database.

        Returns:
            collection from database or None if not found.
        """
        pass


class MongoDBManager(DatabaseManager):
    """
    Concrete implementation of DatabaseManager for MongoDB.
    """

    def __init__(self, connection_string: str, database_name: str, **kwargs):
        self.connection_string = connection_string
        self.database_name = database_name
        self.client: MongoClient | None = None
        self.db = None
        self.collection = None
        self.kwargs = {'serverSelectionTimeoutMS': 5000, "compressors": ['zlib', 'zstd', 'snappy']}
        self.kwargs.update(kwargs)

    def connect(self) -> None:
        """Establishes a connection to the MongoDB database."""
        try:
            self.client = MongoClient(self.connection_string, **self.kwargs)
            self.client.admin.command('ping')  # Check connection
            self.db = self.client[self.database_name]
            logger.info(f"Connected to {self.database_name} MongoDB")
        except ConnectionFailure as e:
            logger.exception(f"Could not connect to MongoDB - {e}")
            self.client = None
            self.db = None
            self.collection = None
            raise

    def disconnect(self) -> None:
        """Closes the connection to the MongoDB database."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            logger.info(f"Disconnected from {self.database_name} MongoDB")

    def add_record(self, collection: str, record: Dict[str, Any]) -> Any:
        """Adds a new record to the MongoDB collection."""
        try:
            result = self.db[collection].insert_one(record)
            return result.inserted_id
        except PyMongoError as e:
            logger.error(f"Cannot adding record - {e}")
            return None

    def update_record(self, collection: str, query: Dict[str, Any], update_data: Dict[str, Any]) -> bool:
        """Updates an existing record in the MongoDB collection."""
        try:
            result = self.db[collection].update_one(query, {"$set": update_data}, upsert=True)
            return result.modified_count > 0
        except PyMongoError as e:
            logger.error(f"Cannot updating record - {e}")
            return False

    def delete_record(self, collection: str, query: Dict[str, Any]) -> bool:
        """Deletes a record from the MongoDB collection."""
        try:
            result = self.db[collection].delete_one(query)
            return result.deleted_count > 0
        except PyMongoError as e:
            logger.error(f"Cannot deleting record: {e}")
            return False

    def get_record(self, collection: str, query: Dict[str, Any]) -> Any | None:
        """Retrieves a record from the MongoDB collection."""
        try:
            """
            db.your_collection.aggregate(
                        [
                          {
                          "$match": {
                            "$expr": {
                              "$gt": [
                                {
                                  "$size": {
                                    "$regexFindAll": {
                                      input: "$language_name",
                                      regex: "[\\s\\p{P}]+"
                                      }
                                    }
                                  }, 1
                                ]
                              }
                            }
                          }
                        ])
            """
            return self.db[collection].find_one(query)
        except PyMongoError as e:
            logger.error(f"Cannot retrieve db record: {e}")
            return None

    def get_all_records(self, collection_name: str, query=None, chunk_size=1000) -> Tuple[Queue, int]:
        """
        Loads the first chunk of records into a queue and adds the rest in a background thread.

        Args:
            collection_name (str): name of the collection
            query (dict): a dictionary for which to query database.
            chunk_size:
        Return:
            (record_queue, total_records) whereas: record_queue - is the queue with records from the specified collection
            total_records - is the number of records in the specified collection.
        """

        record_queue = Queue()
        total_records = 0
        if not query:
            query = {}
        try:
            record_queue = Queue()
            collection = self.db[collection_name]
            total_records = collection.count_documents({})
            processed_count = 0

            # Load the first chunk
            cursor = collection.find(query).limit(chunk_size)
            for record in cursor:
                processed_count += 1
                record_queue.put(record)

            def enqueue_remaining(collection, processed):
                """Adds the remaining records to the queue in the background."""
                remaining_cursor = collection.find(query).skip(processed)
                for _record in remaining_cursor:
                    record_queue.put(_record)
                logger.info(f"Background enqueuing complete for: {collection_name}")

            # Start background enqueuing of remaining records
            if processed_count < total_records:
                background_thread = Thread(target=enqueue_remaining, args=(collection, processed_count))
                background_thread.daemon = True
                background_thread.start()
                logger.info(
                    f"First {processed_count} records loaded into the queue. "
                    f"Background enqueuing of remaining {total_records - processed_count} records started."
                )
            else:
                logger.info(f"All {total_records} records loaded directly into the queue.")

            return record_queue, total_records

        except PyMongoError as e:
            logger.error(f"Cannot retrieve all records - {e}")
            return record_queue, total_records
        except Exception as e:
            logger.exception(f"Cannot retrieve all records - {e}")
            return record_queue, total_records

    def get_collection(self, collection_name: str) -> Collection[Mapping[str, Any]] | None:
        """Retrieves the specified collection from MongoDB"""
        try:
            return self.db[collection_name]
        except PyMongoError as e:
            logger.error(f"Cannot retrieve all records - {e}")
            return None
        except Exception as e:
            logger.exception(f"Cannot retrieve all records - {e}")
            return None


class BaseDataManager:
    def __init__(self, store_path: str):
        self.store_path = store_path
        self._id_generator = None
        self.root_container = None
        self.segments_metadata_key = 'segments_metadata'

    def _get_next_segment_id_base(self) -> str:
        # This generic method assumes root_container.attrs is available
        # and has 'next_segment_id_counter'
        current_counter_value = self.root_container.attrs.get('next_segment_id_counter', 0)
        # We re-initialize the generator to ensure it starts from the correct persistent value
        self._id_generator = itertools.count(start=current_counter_value)
        segment_id = str(next(self._id_generator))
        self.root_container.attrs['next_segment_id_counter'] = int(segment_id) + 1
        return segment_id

    def save_data_segment(self, data_tensor: torch.Tensor) -> str:
        raise NotImplementedError

    def load_data_segment_by_id(self, segment_id: str) -> np.ndarray:
        raise NotImplementedError

    def get_all_segment_ids(self) -> list:
        segments_meta = self.root_container.attrs.get(self.segments_metadata_key, {})
        if isinstance(segments_meta, str):  # For HDF5 JSON string
            segments_meta = json.loads(segments_meta)
        elif isinstance(segments_meta, bytes):  # For HDF5 bytes string
            segments_meta = json.loads(segments_meta.decode('utf-8'))
        return list(segments_meta.keys())

    def get_segment_info(self, segment_id: str) -> dict:
        segments_meta = self.root_container.attrs.get(self.segments_metadata_key, {})
        if isinstance(segments_meta, str):
            segments_meta = json.loads(segments_meta)
        elif isinstance(segments_meta, bytes):
            segments_meta = json.loads(segments_meta.decode('utf-8'))
        if segment_id not in segments_meta:
            raise ValueError(f"Segment ID '{segment_id}' not found in metadata.")
        return segments_meta[segment_id]

    def cleanup(self):
        """Removes the store directory/file. Ensures store is closed first."""
        # Always attempt to close before cleanup if a 'close' method exists
        if hasattr(self, 'close'):
            self.close()
        if os.path.exists(self.store_path):
            if os.path.isdir(self.store_path):
                print(f"Cleaning up directory: {self.store_path}")
                shutil.rmtree(self.store_path)
            else:
                print(f"Cleaning up file: {self.store_path}")
                os.remove(self.store_path)
            print(f"Cleaned up existing store: {self.store_path}")


class ZarrDirDataManager(BaseDataManager):
    def __init__(self, zarr_store_path: str):
        super().__init__(zarr_store_path)
        self.root_container = zarr.open_group(self.store_path, mode='a')
        if 'next_segment_id_counter' not in self.root_container.attrs:
            self.root_container.attrs['next_segment_id_counter'] = 0
            self.root_container.attrs[self.segments_metadata_key] = {}
        self._id_generator = itertools.count(start=self.root_container.attrs['next_segment_id_counter'])
        print(f"Initialized Zarr Directory Store Manager at {self.store_path}")

    def _get_next_segment_id(self) -> str:
        return self._get_next_segment_id_base()

    @time_it
    def save_data_segment(self, data_tensor: torch.Tensor) -> str:
        segment_id = self._get_next_segment_id()
        flattened_np_array = data_tensor.flatten().numpy(force=True)
        original_shape = list(data_tensor.shape)
        original_dtype = str(flattened_np_array.dtype)
        # Consider specifying chunks for large arrays to optimize performance
        # Example: chunk into 100k elements or full array
        chunk_shape = (min(flattened_np_array.size, 100000),)
        zarr_array = self.root_container.create_array(
            segment_id,
            shape=flattened_np_array.shape,
            dtype=flattened_np_array.dtype,
            chunks=chunk_shape,  # Explicitly specify chunks
            overwrite=True
        )
        zarr_array[:] = flattened_np_array
        zarr_array.attrs['original_shape'] = original_shape
        zarr_array.attrs['original_dtype'] = original_dtype
        segments_meta = self.root_container.attrs[self.segments_metadata_key]
        segments_meta[segment_id] = {
            'original_shape': original_shape,
            'original_dtype': original_dtype,
            'num_elements': flattened_np_array.size
        }
        self.root_container.attrs[self.segments_metadata_key] = segments_meta
        return segment_id

    @time_it
    def load_data_segment_by_id(self, segment_id: str) -> np.ndarray:
        if segment_id not in self.root_container:
            raise ValueError(f"Segment ID '{segment_id}' not found in Zarr store.")
        zarr_array = self.root_container[segment_id]
        original_shape = tuple(zarr_array.attrs['original_shape'])
        original_dtype = np.dtype(zarr_array.attrs['original_dtype'])
        loaded_array_flat = zarr_array[:]
        if loaded_array_flat.dtype != original_dtype:
            loaded_array_flat = loaded_array_flat.astype(original_dtype)
        reshaped_array = loaded_array_flat.reshape(original_shape)
        return reshaped_array

    def close(self):
        """No explicit close needed for Zarr Directory Store, but good practice for consistency."""
        # Zarr directory store relies on underlying file system flushing.
        print(f"Zarr Directory Store at {self.store_path} implicitly closed.")


class ZarrZipDataManager(ZarrDirDataManager):
    def __init__(self, zarr_zip_path: str):
        super(ZarrDirDataManager, self).__init__(zarr_zip_path)
        self.store = zarr.storage.ZipStore(self.store_path, mode='a')
        self.root_container = zarr.group(self.store)

        if 'next_segment_id_counter' not in self.root_container.attrs:
            self.root_container.attrs['next_segment_id_counter'] = 0
            self.root_container.attrs[self.segments_metadata_key] = {}
        # Re-initialize the generator from the *correct* store's counter
        self._id_generator = itertools.count(start=self.root_container.attrs['next_segment_id_counter'])
        print(f"Initialized Zarr Zip Store Manager at {self.store_path}")

    def close(self):
        """Explicitly close the ZipStore."""
        if hasattr(self, 'store') and self.store is not None:
            self.store.close()
            print(f"Zarr Zip Store at {self.store_path} closed.")


class HDF5DataManager(BaseDataManager):
    def __init__(self, h5_file_path: str):
        super().__init__(h5_file_path)
        self.root_container = h5py.File(self.store_path, 'a')
        if 'next_segment_id_counter' not in self.root_container.attrs:
            self.root_container.attrs['next_segment_id_counter'] = 0
            self.root_container.attrs[self.segments_metadata_key] = json.dumps({})
        self._id_generator = itertools.count(start=self.root_container.attrs['next_segment_id_counter'])
        print(f"Initialized HDF5 Data Manager at {self.store_path}")

    def _get_next_segment_id(self) -> str:
        return self._get_next_segment_id_base()

    @time_it
    def save_data_segment(self, data_tensor: torch.Tensor) -> str:
        segment_id = self._get_next_segment_id()
        flattened_np_array = data_tensor.flatten().numpy(force=True)
        original_shape = list(data_tensor.shape)
        original_dtype = str(flattened_np_array.dtype)
        # Create a new dataset in HDF5 file
        chunk_shape = (min(flattened_np_array.size, 100000),)  # Example chunk size
        h5_dataset = self.root_container.create_dataset(
            segment_id,
            data=flattened_np_array,
            chunks=chunk_shape,  # Specify chunks for HDF5
            # compression="lzf" # A fast compression algorithm
            # compression_opts=4 # Compression level if applicable
        )
        h5_dataset.attrs['original_shape'] = original_shape
        h5_dataset.attrs['original_dtype'] = original_dtype
        segments_meta_str = self.root_container.attrs[self.segments_metadata_key]
        segments_meta = json.loads(segments_meta_str)
        segments_meta[segment_id] = {
            'original_shape': original_shape,
            'original_dtype': original_dtype,
            'num_elements': flattened_np_array.size
        }
        self.root_container.attrs[self.segments_metadata_key] = json.dumps(segments_meta)
        return segment_id

    @time_it
    def load_data_segment_by_id(self, segment_id: str) -> np.ndarray:
        if segment_id not in self.root_container:
            raise ValueError(f"Segment ID '{segment_id}' not found in HDF5 store.")
        h5_dataset = self.root_container[segment_id]
        loaded_array_flat = h5_dataset[:]
        original_shape = tuple(h5_dataset.attrs['original_shape'])
        original_dtype = np.dtype(h5_dataset.attrs['original_dtype'])
        if loaded_array_flat.dtype != original_dtype:
            loaded_array_flat = loaded_array_flat.astype(original_dtype)
        reshaped_array = loaded_array_flat.reshape(original_shape)
        return reshaped_array

    def close(self):
        """Explicitly close the HDF5 file."""
        if hasattr(self, 'root_container') and self.root_container:
            self.root_container.close()
            print(f"HDF5 file at {self.store_path} closed.")


def run_performance_test(manager_class: type, store_path: str):
    print(f"\n--- Running test for {manager_class.__name__} at {store_path} ---")
    # Instantiate once to call cleanup, then again for the actual test run
    temp_manager = manager_class(store_path)
    temp_manager.cleanup()
    # Must close the temp_manager if it holds file handles (like ZipStore or HDF5)
    if hasattr(temp_manager, 'close'):
        temp_manager.close()  # Ensure it's fully closed before re-creating
    del temp_manager  # Explicitly delete to release resources immediately
    manager = manager_class(store_path)  # Re-initialize for the actual test
    # Test small writes
    print(f"\n--- {manager_class.__name__}: Saving 10 Small Segments (5x2 float32) ---")
    small_segment_ids = []
    for i in range(10):
        data = torch.randn(5, 2, dtype=torch.float32)
        segment_id = manager.save_data_segment(data)
        small_segment_ids.append(segment_id)
    print(f"  {manager_class.__name__}: All small segments saved.")
    # Test small reads
    print(f"\n--- {manager_class.__name__}: Loading 10 Small Segments ---")
    for segment_id in small_segment_ids:
        _ = manager.load_data_segment_by_id(segment_id)
    print(f"  {manager_class.__name__}: All small segments loaded.")
    # Test larger write
    print(f"\n--- {manager_class.__name__}: Saving Large Segment (1000x100 float32) ---")
    large_data = torch.rand(1000, 100, dtype=torch.float32)  # 100,000 elements
    large_segment_id = manager.save_data_segment(large_data)
    print(f"  {manager_class.__name__}: Large segment saved.")
    # Test larger read
    print(f"\n--- {manager_class.__name__}: Loading Large Segment ---")
    _ = manager.load_data_segment_by_id(large_segment_id)
    print(f"  {manager_class.__name__}: Large segment loaded.")
    manager.close()  # Ensure manager is closed before the final cleanup
    manager.cleanup()  # Clean up after the test


def run_all_performance_tests():
    # Run Zarr Directory Store Test
    run_performance_test(ZarrDirDataManager, 'zarr_dir_store.zarr')
    # Run Zarr Zip Store Test
    run_performance_test(ZarrZipDataManager, 'zarr_zip_store.zip')
    # Run HDF5 Test
    run_performance_test(HDF5DataManager, 'hdf5_store.h5')


if __name__ == '__main__':
    run_all_performance_tests()
