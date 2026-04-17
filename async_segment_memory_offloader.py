import os
import sys
import math
import psutil
import inspect
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import List, Dict, Tuple, Optional, TypeVar, Type, Union
from concurrent.futures import ThreadPoolExecutor, Future
from torch.nn.modules.module import register_module_parameter_registration_hook
from torch.optim.optimizer import register_optimizer_step_pre_hook
from torch.utils.hooks import RemovableHandle
import traceback
import logging
import asyncio
import itertools
import shutil
import tempfile
import time
import functools
import contextlib
import threading

# --- Configuration ---
T_Module = TypeVar("T_Module", bound=nn.Module)

torch.set_float32_matmul_precision('high') # Can significantly speed up training for models that uses float32

# --- Logging Configuration ---
LOG_STR_FORMAT = '%(asctime)s %(name)s.%(threadName)s %(levelname)s: %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=LOG_STR_FORMAT)
logger = logging.getLogger('AngeASI.model.tinyMoE')

# --- Constants for Attribute Names ---
PARAMETERS_REGISTERED = "parameters_registered"
REGISTERED_FULL_NAME = "registered_full_name"
SIZE_IN_BYTES = "size_in_bytes"
INITIALIZED = "initialized"
GRADIENT_ID = "gradient_id"
SEGMENT_ID = "segment_id"
MODULE_ID = "module_id"
DEVICE = "device"
SHAPE = "shape"
DTYPE = "dtype"
NAME = "name"
BYTES_TO_GB = 1024 ** 3
ONE_BILLION = 1_000_000_000
ONE_MILLION = 1_000_000

# --- Global / Helper Functions ---

def time_it(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Execution of {func.__name__} took {end_time - start_time:.4f} seconds.")
        return result

    return wrapper

def _set_optimizer_step_pre_hook(optimizer: torch.optim.Optimizer, args, kwargs):
    state_dict = optimizer.state_dict()
    logger.info(f"[{optimizer.__class__.__name__}] Executing optimizer pre-step hook...")
    logger.info(f"Loading state_dict params: {state_dict}")

def _set_gradient_post_backward_hook(module, grad_input, grad_output):
    full_module_name = getattr(module, REGISTERED_FULL_NAME, module.__class__.__name__)
    logger.info(f"[{full_module_name}] Executing post-backward hook: Gradients available for {full_module_name}.")
    params = list(module.named_parameters(recurse=False))
    logger.info(f"Offloading params: {params}")

def have_parameters(module: nn.Module):
    """Check if a module has any parameters."""
    try:
        return len(vars(module).get('_parameters', {})) > 0
    except TypeError:
        return False

def have_forward(module: nn.Module):
    """Check if a module has a forward function."""
    return callable(module.__class__.__dict__.get('forward', None))

def extract_segment_ids(module: nn.Module) -> Tuple[list[str], list[nn.Parameter], str]:
    """Extract segment ids and Parameters from the specified module."""
    _none = 'not_found_extract_segment_ids'
    parameters = []
    extract_parameters(module, parameters)
    module_name = getattr(module, REGISTERED_FULL_NAME, _none)
    segment_ids = [getattr(param, SEGMENT_ID, _none) for param in parameters]
    return segment_ids, parameters, module_name

def extract_parameters(module: nn.Module, parameters: list):
    """Extract all parameters needed for the module's forward pass."""
    # Extract top level parameters
    for _, param in module.named_parameters(recurse=False):
        parameters.append(param)

    for _, submodule in module.named_children():
        if have_parameters(submodule) and not have_forward(submodule):
            # Extract other required parameters which does not have its own forward pass
            extract_parameters(submodule, parameters)

def get_available_memory(max_memory_usage: float = 0.90, cpu_memory_reserve=0.20):
    """Calculates available memory on CPU and available GPUs."""
    max_device_memory = {}
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        total_gpu_memory = 0
        logger.info(f"Detected {num_gpus} GPU(s).")
        for i in range(num_gpus):
            torch.cuda.empty_cache()
            free_memory, total_memory = torch.cuda.mem_get_info(i)
            max_device_memory[f"cuda:{i}"] = total_memory * max_memory_usage
            total_gpu_memory += total_memory
            logger.info(
                f"GPU {i} ({torch.cuda.get_device_name(i)}): Free = {free_memory / BYTES_TO_GB:.2f} GB, Total = {total_memory / BYTES_TO_GB:.2f} GB")
        logger.info(f"{(total_gpu_memory * max_memory_usage) / BYTES_TO_GB:.2f} GB total GPU RAM available for use.")
    else:
        logger.info("CUDA is not available.")

    virtual_mem = psutil.virtual_memory()
    available_cpu_memory = virtual_mem.available * (1.0 - cpu_memory_reserve)
    max_device_memory["cpu"] = available_cpu_memory
    logger.info(
        f"{(virtual_mem.available / BYTES_TO_GB):.2f} GB total CPU RAM available, reserving {cpu_memory_reserve * 100:.0f}%.")
    return max_device_memory

def create_mmap_tensor_from_file(file_path, num_tensors, tensor_shape, dtype):
    """Creates a file and a memory-mapped tensor linked to it, shaped to hold multiple tensors."""
    tensor_size = torch.tensor([], dtype=dtype).element_size()
    num_elements_per_tensor = math.prod(tensor_shape)
    total_elements = num_tensors * num_elements_per_tensor
    required_bytes = total_elements * tensor_size

    if not os.path.exists(file_path) or os.path.getsize(file_path) < required_bytes:
        with open(file_path, 'wb') as f:
            f.seek(required_bytes - 1)
            f.write(b'\0')

    # The shape is (number_of_tensors_in_file, *original_tensor_shape)
    mmap_tensor = torch.from_file(
        filename=file_path,
        shared=True,
        size=total_elements,
        dtype=dtype
    ).reshape(num_tensors, *tensor_shape)
    return mmap_tensor


class ModuleTracker(nn.Module):
    """
    ModuleTracker is a utility class designed to assist in dynamically instantiating PyTorch
    modules and tracing their call order within a larger model. It provides configurations
    for common `torch.nn` modules and intelligent dummy input generation to facilitate
    testing and understanding of model architectures without requiring actual data.

    This class supports:
    - Instantiation of `torch.nn` modules with sensible default parameters.
    - Generation of appropriate dummy input tensors for various module types.
    - Tracing the execution order of submodules during a dummy forward pass.
    """

    # ======================================================================================
    # --- Global Configuration Constants for Compatibility ---
    # ======================================================================================

    # These constants define compatible dimensions for various layers to ensure seamless
    # dummy forward passes across different module types.
    _DEFAULT_FEATURE_DIM = 64  # A good default for linear layers, embedding_dim, d_model
    _DEFAULT_HIDDEN_SIZE = _DEFAULT_FEATURE_DIM * 2  # Typical for FFNs, RNN hidden states
    _DEFAULT_CHANNELS = 32  # For Convolutional layers
    _DEFAULT_IMAGE_SIZE = 16  # For Convolutional layers (Height, Width)
    _DEFAULT_SEQ_LEN = 10  # For Recurrent Neural Networks (RNNs), Transformers
    _DEFAULT_VOCAB_SIZE = 100  # Default vocabulary size for Embedding layers
    _BATCH_SIZE = 2  # Consistent batch size for all dummy inputs

    # ======================================================================================
    # --- Module Instantiation Configuration ---
    # ======================================================================================
    # Maps torch.nn.Module classes to their minimal __init__ arguments.
    # '_pos_args' explicitly defines the order of positional arguments.
    MODULE_CONFIG_MAP = {
        # --- Linear Layers ---
        nn.Linear: {
            '_pos_args': ('in_features', 'out_features'),
            'in_features': _DEFAULT_FEATURE_DIM,
            'out_features': _DEFAULT_FEATURE_DIM,
        },
        nn.Bilinear: {
            '_pos_args': ('in1_features', 'in2_features', 'out_features'),
            'in1_features': _DEFAULT_FEATURE_DIM,
            'in2_features': _DEFAULT_FEATURE_DIM,
            'out_features': _DEFAULT_FEATURE_DIM // 2,
        },

        # --- Convolutional Layers ---
        nn.Conv2d: {
            '_pos_args': ('in_channels', 'out_channels', 'kernel_size'),
            'in_channels': _DEFAULT_CHANNELS,
            'out_channels': _DEFAULT_CHANNELS,
            'kernel_size': 3,
            'padding': 1,  # Adding padding can simplify subsequent layer compatibility for small kernels
        },
        nn.ConvTranspose2d: {
            '_pos_args': ('in_channels', 'out_channels', 'kernel_size'),
            'in_channels': _DEFAULT_CHANNELS,
            'out_channels': _DEFAULT_CHANNELS,
            'kernel_size': 3,
            'padding': 1,
            'output_padding': 1,
            'stride': 2,
        },

        # --- Pooling Layers ---
        nn.MaxPool2d: {
            '_pos_args': ('kernel_size',),
            'kernel_size': 2
        },
        nn.AdaptiveAvgPool2d: {
            '_pos_args': ('output_size',),
            'output_size': (1, 1)
        },

        # --- Normalization Layers ---
        nn.BatchNorm2d: {
            '_pos_args': ('num_features',),
            'num_features': _DEFAULT_CHANNELS
        },
        nn.LayerNorm: {
            '_pos_args': ('normalized_shape',),
            'normalized_shape': (_DEFAULT_FEATURE_DIM,)
        },
        nn.GroupNorm: {
            '_pos_args': ('num_groups', 'num_channels'),
            'num_groups': 4,
            'num_channels': _DEFAULT_CHANNELS,
        },

        # --- Activation Functions ---
        nn.ReLU: {},
        nn.Dropout: {
            '_pos_args': ('p',),
            'p': 0.1
        },

        # --- Recurrent Layers ---
        nn.RNN: {
            '_pos_args': ('input_size', 'hidden_size'),
            'input_size': _DEFAULT_FEATURE_DIM,
            'hidden_size': _DEFAULT_HIDDEN_SIZE,
            'batch_first': True,
            'num_layers': 1,  # Added for hidden state calculation
        },
        nn.LSTM: {
            '_pos_args': ('input_size', 'hidden_size'),
            'input_size': _DEFAULT_FEATURE_DIM,
            'hidden_size': _DEFAULT_HIDDEN_SIZE,
            'batch_first': True,
            'num_layers': 1,  # Added for hidden state calculation
        },
        nn.GRU: {
            '_pos_args': ('input_size', 'hidden_size'),
            'input_size': _DEFAULT_FEATURE_DIM,
            'hidden_size': _DEFAULT_HIDDEN_SIZE,
            'batch_first': True,
            'num_layers': 1,  # Added for hidden state calculation
        },
        nn.LSTMCell: {
            '_pos_args': ('input_size', 'hidden_size'),
            'input_size': _DEFAULT_FEATURE_DIM,
            'hidden_size': _DEFAULT_HIDDEN_SIZE
        },

        # --- Embedding Layer ---
        nn.Embedding: {
            '_pos_args': ('num_embeddings', 'embedding_dim'),
            'num_embeddings': _DEFAULT_VOCAB_SIZE,
            'embedding_dim': _DEFAULT_FEATURE_DIM
        },

        # --- Transformer Layers ---
        nn.MultiheadAttention: {
            '_pos_args': ('embed_dim', 'num_heads'),
            'embed_dim': _DEFAULT_FEATURE_DIM,
            'num_heads': 8,
            'batch_first': True,
        },
        nn.TransformerEncoderLayer: {
            '_pos_args': ('d_model', 'nhead'),
            'd_model': _DEFAULT_FEATURE_DIM,
            'nhead': 8,
            'batch_first': True,
            'dim_feedforward': _DEFAULT_HIDDEN_SIZE,
        },
        nn.TransformerEncoder: {
            # Note: encoder_layer is a positional argument.
            # We must pass an instance of TransformerEncoderLayer.
            # d_model and nhead for the encoder itself are implicitly
            # derived from its encoder_layer argument.
            '_pos_args': ('encoder_layer', 'num_layers'),
            'encoder_layer': None,  # Placeholder, will be instantiated recursively
            'num_layers': 1
        },
        nn.Transformer: {
            '_pos_args': (),  # No positional args by default
            'd_model': _DEFAULT_FEATURE_DIM,
            'nhead': 8,
            'num_encoder_layers': 1,
            'num_decoder_layers': 1,
            'dim_feedforward': _DEFAULT_HIDDEN_SIZE,
            'batch_first': True,
        },

        # --- Reshaping Layers ---
        nn.Flatten: {
            '_pos_args': ('start_dim', 'end_dim'),
            'start_dim': 1,
            'end_dim': -1
        },
        nn.Unflatten: {
            '_pos_args': ('dim', 'unflattened_size'),
            'dim': 1,
            'unflattened_size': (_DEFAULT_CHANNELS, _DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE)
        },

        # --- Image-specific Reshaping ---
        nn.PixelShuffle: {
            '_pos_args': ('upscale_factor',),
            'upscale_factor': 2
        },
        nn.PixelUnshuffle: {
            '_pos_args': ('downscale_factor',),
            'downscale_factor': 2
        },
        nn.ChannelShuffle: {
            '_pos_args': ('groups',),
            'groups': 2
        },
    }

    def __init__(self, device: torch.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        Initializes the ModuleTracker.

        Args:
            device (torch.device): The default device (CPU or CUDA) for creating tensors.
        """
        super(ModuleTracker, self).__init__()
        self.device = device
        self.dtype = torch.float16
        self._module_hooks = []
        # A flag to control dynamic module replacement during __setattr__
        self._replacing_modules = False

    def __call__(self, cls: Type[T_Module], *args, **kwargs) -> T_Module:
        """
        Wraps the provided torch.nn.Module cls and returns a mini version of the model
        by redeclaring named modules with smaller capacity. The overall logic should remain
        unchanged, ONLY the model parameters should have smaller shape and memory footprint.
        This allows tracking of modules that exceed the available memory resources by
        performing dummy forward passes on the mini version of the model.
        """
        # Step 1: Temporarily replace the target class's __init__
        # with our custom one that handles module "mini-fication".
        original_cls_init = cls.__init__

        # We need a reference to self (ModuleTracker instance) inside the new_init
        tracker_instance = self

        @contextlib.contextmanager
        def _set_default_dtype_context(dtype):
            if dtype is None:  # No-op if dtype is None
                yield
                return
            old_dtype = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            try:
                yield
            finally:
                torch.set_default_dtype(old_dtype)

        @functools.wraps(original_cls_init)
        def new_init(self: nn.Module, *init_args, **init_kwargs):
            # Determine the target device and dtype for the actual (mini) model
            target_device = init_kwargs.pop("device", tracker_instance.device)
            target_dtype = init_kwargs.pop("dtype", tracker_instance.dtype)
            tracker_instance.dtype = target_dtype

            # Store the actual device and dtype for potential later use (e.g., if we were doing a meta-pass first)
            # For this corrected approach, we directly build the mini model.
            self._actual_device = target_device
            self._actual_dtype = target_dtype

            # Temporarily patch nn.Module.__setattr__ to intercept module assignments
            original_setattr = nn.Module.__setattr__

            @functools.wraps(original_setattr)
            def custom_setattr(module_instance, name, value):
                if isinstance(value, nn.Module) and tracker_instance._replacing_modules:
                    # Check if this module type is in our config map for substitution
                    if type(value) in tracker_instance.MODULE_CONFIG_MAP:
                        logger.debug(f"Intercepting module assignment: {name} of type {type(value).__name__}")
                        # Get the arguments for the smaller version of this module
                        new_args, new_kwargs = tracker_instance.get_module_substitute_args_kwargs(value)

                        # Create the new, smaller capacity module instance
                        try:
                            # Pass device/dtype to the submodule's init if it accepts it
                            submodule_init_params = inspect.signature(type(value).__init__).parameters
                            if 'device' in submodule_init_params:
                                new_kwargs['device'] = target_device
                            if 'dtype' in submodule_init_params and target_dtype is not None:
                                new_kwargs['dtype'] = target_dtype

                            mini_module = type(value)(*new_args, **new_kwargs)
                            original_setattr(module_instance, name, mini_module)
                            logger.debug(f"Replaced with mini version of {type(value).__name__}")
                        except Exception as e:
                            logger.error(f"Failed to redeclare submodule '{name}' of type {type(value).__name__} with smaller capacity. Error: {e}")
                            logger.error(f"Attempted args: {new_args}, kwargs: {new_kwargs}")
                            # If substitution fails, fall back to setting the original value
                            original_setattr(module_instance, name, value)
                    else:
                        # If not in config map, just set the original module
                        original_setattr(module_instance, name, value)
                else:
                    # For non-module attributes or when not replacing, use original setattr
                    original_setattr(module_instance, name, value)

            nn.Module.__setattr__ = custom_setattr
            tracker_instance._replacing_modules = True # Activate the replacement logic

            try:
                with contextlib.ExitStack() as stack:
                    # Apply device/dtype context if the model's __init__ doesn't accept them as kwargs
                    init_params = inspect.signature(original_cls_init).parameters
                    if 'device' not in init_params:
                        stack.enter_context(torch.device(target_device))
                    if 'dtype' not in init_params and target_dtype is not None:
                        stack.enter_context(_set_default_dtype_context(target_dtype))

                    # Pass actual device/dtype to the model's original __init__ if it accepts them
                    final_init_kwargs = init_kwargs.copy()
                    if 'device' in init_params:
                        final_init_kwargs['device'] = target_device
                    if 'dtype' in init_params and target_dtype is not None:
                        final_init_kwargs['dtype'] = target_dtype

                    # Call the original __init__. During this call, our custom_setattr
                    # will intercept and replace nn.Module assignments.
                    start_time = time.time()
                    original_cls_init(self, *init_args, **final_init_kwargs)
                    end_time = time.time()
                    logger.debug(f"Mini Model Initialization of {cls.__name__} took {end_time - start_time:.4f} seconds.")

            except Exception as e:
                logger.error(f"Error during mini model initialization for {cls.__name__}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
            finally:
                # Restore original __setattr__ and reset the flag
                nn.Module.__setattr__ = original_setattr
                tracker_instance._replacing_modules = False

        # Temporarily set the class's __init__ to our modified one
        cls.__init__ = new_init
        try:
            # Create an instance of the class. This will now use our new_init
            # which performs the mini-fication on the fly.
            instance = cls(*args, **kwargs)
        finally:
            # Always restore the original __init__ on the class
            cls.__init__ = original_cls_init
        return instance

    def create_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype=None) -> torch.Tensor:
        """
        Utility function to create a random tensor of a given shape and type on the configured device.

        Args:
            shape (Tuple[int, ...]): The desired shape of the tensor.
            dtype (torch.dtype): The data type of the tensor (e.g., torch.float32, torch.long).

        Returns:
            torch.Tensor: A randomly initialized tensor.
        """
        # Ensure the tensor is created on the tracker's device.
        # During the mini-model creation, this will be the actual target device (e.g., CUDA or CPU).
        if dtype == torch.long:
            return torch.randint(0, self._DEFAULT_VOCAB_SIZE, shape, dtype=dtype, device=self.device)
        else:
            dtype = dtype or self.dtype
            return torch.randn(shape, dtype=dtype, device=self.device)

    # ======================================================================================
    # --- Dummy Input Generator Methods (replacing lambdas) ---
    # ======================================================================================

    # These methods are designed to mimic the original lambda logic precisely,
    # ensuring correct input shapes and types for each module.

    def _generate_linear_input(self, module: nn.Linear) -> Tuple[torch.Tensor]:
        return (self.create_tensor((self._BATCH_SIZE, module.in_features)),)

    def _generate_bilinear_input(self, module: nn.Bilinear) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.create_tensor((self._BATCH_SIZE, module.in1_features)),
                self.create_tensor((self._BATCH_SIZE, module.in2_features)))

    def _generate_conv2d_input(self, module: nn.Conv2d) -> Tuple[torch.Tensor]:
        return (self.create_tensor(
            (self._BATCH_SIZE, module.in_channels, self._DEFAULT_IMAGE_SIZE, self._DEFAULT_IMAGE_SIZE)),)

    def _generate_convtranspose2d_input(self, module: nn.ConvTranspose2d) -> Tuple[torch.Tensor]:
        return (self.create_tensor(
            (self._BATCH_SIZE, module.in_channels, self._DEFAULT_IMAGE_SIZE, self._DEFAULT_IMAGE_SIZE)),)

    def _generate_maxpool2d_input(self, module: nn.MaxPool2d) -> Tuple[torch.Tensor]:
        return (self.create_tensor(
            (self._BATCH_SIZE, self._DEFAULT_CHANNELS, self._DEFAULT_IMAGE_SIZE, self._DEFAULT_IMAGE_SIZE)),)

    def _generate_adaptiveavgpool2d_input(self, module: nn.AdaptiveAvgPool2d) -> Tuple[torch.Tensor]:
        return (self.create_tensor(
            (self._BATCH_SIZE, self._DEFAULT_CHANNELS, self._DEFAULT_IMAGE_SIZE, self._DEFAULT_IMAGE_SIZE)),)

    def _generate_batchnorm2d_input(self, module: nn.BatchNorm2d) -> Tuple[torch.Tensor]:
        return (self.create_tensor(
            (self._BATCH_SIZE, module.num_features, self._DEFAULT_IMAGE_SIZE, self._DEFAULT_IMAGE_SIZE)),)

    def _generate_layernorm_input(self, module: nn.LayerNorm) -> Tuple[torch.Tensor]:
        return (self.create_tensor((self._BATCH_SIZE, self._DEFAULT_SEQ_LEN, module.normalized_shape[0])),)

    def _generate_groupnorm_input(self, module: nn.GroupNorm) -> Tuple[torch.Tensor]:
        return (self.create_tensor(
            (self._BATCH_SIZE, module.num_channels, self._DEFAULT_IMAGE_SIZE, self._DEFAULT_IMAGE_SIZE)),)

    def _generate_relu_input(self, module: nn.ReLU) -> Tuple[torch.Tensor]:
        return (self.create_tensor((self._BATCH_SIZE, self._DEFAULT_FEATURE_DIM)),)

    def _generate_dropout_input(self, module: nn.Dropout) -> Tuple[torch.Tensor]:
        return (self.create_tensor((self._BATCH_SIZE, self._DEFAULT_FEATURE_DIM)),)

    def _generate_rnn_input(self, module: nn.RNN) -> Tuple[torch.Tensor, ]:
        # RNN, LSTM, GRU take (input, h_0) for RNN/GRU or (input, (h_0, c_0)) for LSTM
        # h_0 shape: (num_layers * num_directions, batch, hidden_size)
        return (self.create_tensor((self._BATCH_SIZE, self._DEFAULT_SEQ_LEN, module.input_size)),)

    def _generate_lstm_input(self, module: nn.LSTM) -> Tuple[torch.Tensor,]:
        # h_0 and c_0 shape: (num_layers * num_directions, batch, hidden_size)
        num_directions = 2 if module.bidirectional else 1
        return (self.create_tensor((self._BATCH_SIZE, self._DEFAULT_SEQ_LEN, module.input_size)),)

    def _generate_gru_input(self, module: nn.GRU) -> Tuple[torch.Tensor, ]:
        # h_0 shape: (num_layers * num_directions, batch, hidden_size)
        num_directions = 2 if module.bidirectional else 1
        h_0 = self.create_tensor((module.num_layers * num_directions, self._BATCH_SIZE, module.hidden_size))
        return (self.create_tensor((self._BATCH_SIZE, self._DEFAULT_SEQ_LEN, module.input_size)),)

    def _generate_lstmcell_input(self, module: nn.LSTMCell) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # LSTMCell takes (input, (h_0, c_0))
        # input shape: (batch, input_size)
        # h_0, c_0 shape: (batch, hidden_size)
        return (self.create_tensor((self._BATCH_SIZE, module.input_size)),
                (self.create_tensor((self._BATCH_SIZE, module.hidden_size)),
                 self.create_tensor((self._BATCH_SIZE, module.hidden_size))))

    def _generate_embedding_input(self, module: nn.Embedding) -> Tuple[torch.Tensor]:
        return (self.create_tensor((self._BATCH_SIZE, self._DEFAULT_SEQ_LEN), dtype=torch.long),)

    def _generate_multiheadattention_input(self, module: nn.MultiheadAttention) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        # Query, Key, Value all have shape (Batch, Seq_len, Embed_dim) when batch_first=True
        return (self.create_tensor((self._BATCH_SIZE, self._DEFAULT_SEQ_LEN, module.embed_dim)),
                self.create_tensor((self._BATCH_SIZE, self._DEFAULT_SEQ_LEN, module.embed_dim)),
                self.create_tensor((self._BATCH_SIZE, self._DEFAULT_SEQ_LEN, module.embed_dim)))

    def _generate_transformerencoderlayer_input(self, module: nn.TransformerEncoderLayer) -> Tuple[torch.Tensor]:
        # Input is (batch, seq_len, d_model)
        # `d_model` might be an attribute of the module directly or inferred from submodules
        d_model = getattr(module, 'd_model', getattr(module.self_attn, 'embed_dim', self._DEFAULT_FEATURE_DIM))
        return (self.create_tensor((self._BATCH_SIZE, self._DEFAULT_SEQ_LEN, d_model)),)

    def _generate_transformerencoder_input(self, module: nn.TransformerEncoder) -> Tuple[torch.Tensor]:
        # Input is (batch, seq_len, d_model). d_model can be accessed from the first layer.
        d_model = getattr(module.layers[0], 'd_model',
                          getattr(module.layers[0].self_attn, 'embed_dim', self._DEFAULT_FEATURE_DIM))
        return (self.create_tensor((self._BATCH_SIZE, self._DEFAULT_SEQ_LEN, d_model)),)

    def _generate_transformer_input(self, module: nn.Transformer) -> Tuple[torch.Tensor, torch.Tensor]:
        # Transformer takes src and tgt inputs, both (batch, seq_len, d_model)
        return (self.create_tensor((self._BATCH_SIZE, self._DEFAULT_SEQ_LEN, module.d_model)),  # Source
                self.create_tensor((self._BATCH_SIZE, self._DEFAULT_SEQ_LEN, module.d_model)))  # Target

    def _generate_flatten_input(self, module: nn.Flatten) -> Tuple[torch.Tensor]:
        # Provide a typical multi-dimensional input for Flatten
        return (self.create_tensor(
            (self._BATCH_SIZE, self._DEFAULT_CHANNELS, self._DEFAULT_IMAGE_SIZE, self._DEFAULT_IMAGE_SIZE)),)

    def _generate_unflatten_input(self, module: nn.Unflatten) -> Tuple[torch.Tensor]:
        # Input for Unflatten is (Batch, flattened_size)
        flattened_size = math.prod(module.unflattened_size)
        return (self.create_tensor((self._BATCH_SIZE, flattened_size)),)

    def _generate_pixelshuffle_input(self, module: nn.PixelShuffle) -> Tuple[torch.Tensor]:
        # Input channels must be divisible by upscale_factor squared
        in_channels_for_shuffle = self._DEFAULT_CHANNELS * (module.upscale_factor ** 2)
        # Spatial dimensions should be divisible by upscale_factor
        return (self.create_tensor((self._BATCH_SIZE, in_channels_for_shuffle,
                                    self._DEFAULT_IMAGE_SIZE // module.upscale_factor,
                                    self._DEFAULT_IMAGE_SIZE // module.upscale_factor)),)

    def _generate_pixelunshuffle_input(self, module: nn.PixelUnshuffle) -> Tuple[torch.Tensor]:
        # Input spatial dimensions must be divisible by downscale_factor
        return (self.create_tensor((self._BATCH_SIZE, self._DEFAULT_CHANNELS,
                                    self._DEFAULT_IMAGE_SIZE * module.downscale_factor,
                                    self._DEFAULT_IMAGE_SIZE * module.downscale_factor)),)

    def _generate_channelshuffle_input(self, module: nn.ChannelShuffle) -> Tuple[torch.Tensor]:
        # Input channels must be divisible by groups
        in_channels_for_shuffle = self._DEFAULT_CHANNELS * module.groups
        return (self.create_tensor(
            (self._BATCH_SIZE, in_channels_for_shuffle, self._DEFAULT_IMAGE_SIZE, self._DEFAULT_IMAGE_SIZE)),)

    # Mapping of module class to its specific input generation method
    _DUMMY_INPUT_GENERATOR_MAP = {
        nn.Linear: _generate_linear_input,
        nn.Bilinear: _generate_bilinear_input,
        nn.Conv2d: _generate_conv2d_input,
        nn.ConvTranspose2d: _generate_convtranspose2d_input,
        nn.MaxPool2d: _generate_maxpool2d_input,
        nn.AdaptiveAvgPool2d: _generate_adaptiveavgpool2d_input,
        nn.BatchNorm2d: _generate_batchnorm2d_input,
        nn.LayerNorm: _generate_layernorm_input,
        nn.GroupNorm: _generate_groupnorm_input,
        nn.ReLU: _generate_relu_input,
        nn.Dropout: _generate_dropout_input,
        nn.RNN: _generate_rnn_input,
        nn.LSTM: _generate_lstm_input,
        nn.GRU: _generate_gru_input,
        nn.LSTMCell: _generate_lstmcell_input,
        nn.Embedding: _generate_embedding_input,
        nn.MultiheadAttention: _generate_multiheadattention_input,
        nn.TransformerEncoderLayer: _generate_transformerencoderlayer_input,
        nn.TransformerEncoder: _generate_transformerencoder_input,
        nn.Transformer: _generate_transformer_input,
        nn.Flatten: _generate_flatten_input,
        nn.Unflatten: _generate_unflatten_input,
        nn.PixelShuffle: _generate_pixelshuffle_input,
        nn.PixelUnshuffle: _generate_pixelunshuffle_input,
        nn.ChannelShuffle: _generate_channelshuffle_input,
    }

    def get_module_config(self, module_class: type) -> Tuple[Union[Tuple, None], Union[dict, None]]:
        """
        Retrieves the minimal arguments (positional and keyword) required to instantiate a `torch.nn.Module`.
        It prioritizes predefined configurations in `MODULE_CONFIG_MAP` but can fall back to
        inspecting the module's `__init__` signature for unmapped modules.

        Args:
            module_class (type): The `torch.nn.Module` class to configure.

        Returns:
            Tuple[Union[Tuple, None], Union[dict, None]]: A tuple containing (args, kwargs)
            for module instantiation. Returns (None, None) if configuration cannot be determined.
        """
        config = self.MODULE_CONFIG_MAP.get(module_class)

        # Initialize parameters to an empty dictionary to prevent UnboundLocalError
        parameters = {}

        if config is None:
            # Fallback for unmapped modules: try to infer from signature defaults.
            try:
                init_signature = inspect.signature(module_class.__init__)
                parameters = init_signature.parameters

                inferred_args = []
                inferred_kwargs = {}

                for name, param in parameters.items():
                    if name in ['self', '_', '__class__'] or param.kind in (inspect.Parameter.VAR_POSITIONAL,
                                                                            inspect.Parameter.VAR_KEYWORD):
                        continue

                    if param.default is not inspect.Parameter.empty:
                        # If a default exists, use it
                        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                            inferred_args.append(param.default)
                        else:  # KEYWORD_ONLY
                            inferred_kwargs[name] = param.default
                    else:
                        # Try to provide a sensible dummy value for required arguments
                        val = None
                        if name in ['in_features', 'out_features', 'num_features', 'num_channels', 'input_size',
                                    'hidden_size', 'embed_dim', 'd_model', 'nhead', 'dim_feedforward']:
                            val = self._DEFAULT_FEATURE_DIM
                            if name in ['hidden_size', 'dim_feedforward']: val = self._DEFAULT_HIDDEN_SIZE
                            if name == 'nhead': val = 8
                        elif name in ['kernel_size', 'stride', 'padding', 'output_padding', 'num_heads', 'num_layers',
                                      'upscale_factor', 'downscale_factor', 'groups']:
                            val = 2
                        elif name == 'p':
                            val = 0.1
                        elif name == 'normalized_shape':
                            val = (self._DEFAULT_FEATURE_DIM,)
                        elif name == 'output_size':
                            val = (1, 1)
                        elif name in ['num_embeddings']:
                            val = self._DEFAULT_VOCAB_SIZE
                        elif name in ['start_dim', 'end_dim', 'dim']:
                            val = 1
                        elif name == 'unflattened_size':
                            val = (self._DEFAULT_CHANNELS, self._DEFAULT_IMAGE_SIZE, self._DEFAULT_IMAGE_SIZE)
                        elif name in ['encoder_layer', 'decoder_layer', 'custom_encoder', 'custom_decoder']:
                            # Recursive instantiation for Transformer components
                            nested_module_class = None
                            if 'encoder_layer' in name or 'custom_encoder' in name:
                                nested_module_class = nn.TransformerEncoderLayer
                            elif 'decoder_layer' in name or 'custom_decoder' in name:
                                nested_module_class = nn.TransformerDecoderLayer

                            if nested_module_class:
                                # Get config for the nested layer.
                                # Pass d_model from parent context if it's already determined.
                                # IMPORTANT: Ensure d_model is passed only once, either positionally or by keyword.
                                nested_config = self.MODULE_CONFIG_MAP.get(nested_module_class, {})
                                nested_pos_arg_names = nested_config.get('_pos_args', ())

                                # Prepare args and kwargs for the nested module
                                nested_init_args = []
                                nested_init_kwargs = {}

                                current_d_model_for_nested = inferred_kwargs.get('d_model', self._DEFAULT_FEATURE_DIM)

                                for nested_arg_name in nested_pos_arg_names:
                                    if nested_arg_name == 'd_model':
                                        nested_init_args.append(current_d_model_for_nested)
                                    elif nested_arg_name in nested_config:
                                        nested_init_args.append(nested_config[nested_arg_name])
                                    else:
                                        # Fallback for positional args not explicitly in nested_config
                                        # (should be rare if MODULE_CONFIG_MAP is comprehensive)
                                        nested_init_args.append(self._DEFAULT_FEATURE_DIM)  # Generic fallback

                                for nested_kw_name, nested_kw_val in nested_config.items():
                                    if nested_kw_name not in nested_pos_arg_names and nested_kw_name != '_pos_args':
                                        if nested_kw_name == 'd_model':
                                            # Avoid passing d_model as kwarg if already passed positionally
                                            if 'd_model' not in nested_pos_arg_names:
                                                nested_init_kwargs[nested_kw_name] = current_d_model_for_nested
                                        else:
                                            nested_init_kwargs[nested_kw_name] = nested_kw_val

                                try:
                                    val = nested_module_class(*nested_init_args, **nested_init_kwargs)
                                except TypeError as te:
                                    print(f"Error instantiating nested module {nested_module_class.__name__}: {te}")
                                    print(f"  Args: {nested_init_args}, Kwargs: {nested_init_kwargs}")
                                    return None, None
                            else:
                                val = None
                        else:
                            if isinstance(param.default, bool):
                                val = True
                            else:
                                # Generic default for other required params
                                val = 1
                                # print(f"Warning: Could not infer default for required argument '{name}' in '{module_class.__name__}'. Set to {val}.")

                        if val is None:
                            print(
                                f"Error: Failed to get config for required argument '{name}' for '{module_class.__name__}'.")
                            return None, None

                        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                            inferred_args.append(val)
                        else:
                            inferred_kwargs[name] = val

                return tuple(inferred_args), inferred_kwargs

            except Exception as e:
                print(f"Error during signature inspection for '{module_class.__name__}': {e}.")
                print(f"Traceback: {traceback.format_exc()}")
                return None, None

        # For mapped modules, process parameters from the __init__ signature and config map
        final_args = []
        final_kwargs = {}

        pos_arg_names = config.get('_pos_args', ())

        # Get the __init__ signature parameters once. Ensure this is done outside the previous try-except
        # if config is not None, we are guaranteed module_class.__init__ exists.
        init_parameters = inspect.signature(module_class.__init__).parameters

        # First, handle positional arguments explicitly defined in _pos_args
        for arg_name in pos_arg_names:
            if arg_name in config:
                val = config[arg_name]
                # Special handling for nested module instantiation (e.g., Transformer components)
                if (module_class == nn.TransformerEncoder and arg_name == 'encoder_layer') or \
                        (module_class == nn.TransformerDecoder and arg_name == 'decoder_layer') or \
                        (module_class == nn.Transformer and arg_name in ('custom_encoder', 'custom_decoder')):
                    if val is None:  # Only create if placeholder is None
                        nested_module_class = None
                        if arg_name == 'encoder_layer' or arg_name == 'custom_encoder':
                            nested_module_class = nn.TransformerEncoderLayer
                        elif arg_name == 'decoder_layer' or arg_name == 'custom_decoder':
                            nested_module_class = nn.TransformerDecoderLayer

                        if nested_module_class:
                            # Get config for the nested layer.
                            nested_config = self.MODULE_CONFIG_MAP.get(nested_module_class, {})
                            nested_pos_arg_names_for_nested = nested_config.get('_pos_args', ())

                            nested_init_args = []
                            nested_init_kwargs = {}

                            # Get d_model from the parent Transformer's config (or default)
                            current_d_model_for_nested = config.get('d_model', self._DEFAULT_FEATURE_DIM)

                            # Populate args for nested module
                            for nested_arg_name in nested_pos_arg_names_for_nested:
                                if nested_arg_name == 'd_model':
                                    nested_init_args.append(current_d_model_for_nested)
                                elif nested_arg_name in nested_config:
                                    nested_init_args.append(nested_config[nested_arg_name])
                                else:
                                    # Fallback for positional args not explicitly in nested_config
                                    nested_init_args.append(self._DEFAULT_FEATURE_DIM)  # Generic fallback

                            # Populate kwargs for nested module
                            for nested_kw_name, nested_kw_val in nested_config.items():
                                if nested_kw_name not in nested_pos_arg_names_for_nested and nested_kw_name != '_pos_args':
                                    # Crucial: Avoid passing 'd_model' as a keyword arg if it's already a positional arg for the nested module
                                    if nested_kw_name == 'd_model':
                                        if 'd_model' not in nested_pos_arg_names_for_nested:
                                            nested_init_kwargs[nested_kw_name] = current_d_model_for_nested
                                    else:
                                        nested_init_kwargs[nested_kw_name] = nested_kw_val

                            try:
                                val = nested_module_class(*nested_init_args, **nested_init_kwargs)
                            except TypeError as te:
                                print(f"Error instantiating nested module {nested_module_class.__name__}: {te}")
                                print(f"  Args: {nested_init_args}, Kwargs: {nested_init_kwargs}")
                                return None, None
                        else:
                            val = None  # Should not happen for these names
                final_args.append(val)
            else:
                # If a positional argument is missing from config map, check init signature default.
                if arg_name in init_parameters and init_parameters[arg_name].default is not inspect.Parameter.empty:
                    final_args.append(init_parameters[arg_name].default)
                else:
                    print(
                        f"Error: Positional argument '{arg_name}' for {module_class.__name__} not found in config or has no default.")
                    return None, None

        # Then, handle keyword arguments from the config map
        for name, value in config.items():
            # Only add to kwargs if it's not a positional argument already handled, and not the internal _pos_args key
            if name not in pos_arg_names and name != '_pos_args':
                # Check if the parameter exists in the actual __init__ signature
                if name in init_parameters:
                    final_kwargs[name] = value

        return tuple(final_args), final_kwargs

    def generate_dummy_input_for_module(self, module: nn.Module) -> Tuple[Union[torch.Tensor, Tuple], ...]:
        """
        Generates a valid dummy input tuple for a given module using a dictionary map of
        input generator methods. This method is primarily used by the pre-forward hook
        to ensure modules receive appropriate inputs during tracing.

        Args:
            module (nn.Module): The module for which to generate input.

        Returns:
            Tuple[Union[torch.Tensor, Tuple], ...]: A tuple containing tensors appropriate
            for the module's forward method.
        """
        generator_method = self._DUMMY_INPUT_GENERATOR_MAP.get(type(module))

        if generator_method:
            # Call the bound method, passing the module instance itself
            return generator_method(self, module)
        else:
            # Fallback if no specific generator is found for this module type.
            print(f"Warning: No specific dummy input generator for {type(module).__name__}. Providing generic input.")
            return (self.create_tensor((self._BATCH_SIZE, self._DEFAULT_FEATURE_DIM)),)

    def generate_dummy_input_for_model_entry(self, model: nn.Module) -> torch.Tensor:
        """
        Generates a suitable initial dummy input tensor for the entire model.
        It attempts to infer the expected input shape based on the model's
        first major processing layer (e.g., Linear, Conv2d, RNN, Transformer).

        Args:
            model (nn.Module): The model for which to generate the initial input.

        Returns:
            torch.Tensor: An initial dummy input tensor for the model.
        """
        # Iterate through the model's direct children first to find the most likely input layer.
        for _, module in model.named_children():
            if isinstance(module, nn.Linear):
                return self.create_tensor((self._BATCH_SIZE, module.in_features))
            if isinstance(module, nn.Conv2d):
                return self.create_tensor(
                    (self._BATCH_SIZE, module.in_channels, self._DEFAULT_IMAGE_SIZE, self._DEFAULT_IMAGE_SIZE))
            if isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
                return self.create_tensor((self._BATCH_SIZE, self._DEFAULT_SEQ_LEN, module.input_size))
            if isinstance(module, nn.Embedding):
                return self.create_tensor((self._BATCH_SIZE, self._DEFAULT_SEQ_LEN), dtype=torch.long)
            if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerEncoder, nn.Transformer)):
                # For Transformer-like inputs, expect (batch, seq_len, d_model)
                d_model = getattr(module, 'd_model', None)
                if d_model is None and hasattr(module, 'layers') and len(module.layers) > 0:
                    # Access d_model from the internal self_attn layer of the first TransformerEncoderLayer
                    if hasattr(module.layers[0], 'self_attn') and hasattr(module.layers[0].self_attn, 'embed_dim'):
                        d_model = module.layers[0].self_attn.embed_dim
                if d_model is None and hasattr(module, 'self_attn'):  # For TransformerEncoderLayer directly
                    d_model = getattr(module.self_attn, 'embed_dim', None)

                d_model = d_model if d_model is not None else self._DEFAULT_FEATURE_DIM  # Fallback
                return self.create_tensor((self._BATCH_SIZE, self._DEFAULT_SEQ_LEN, d_model))

            # If the first child is a container (e.g., nn.Sequential, nn.ModuleList, nn.ModuleDict),
            # provide a generic input and let the hooks handle the inner modules.
            if isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)) and len(module) > 0:
                # If the first actual layer is a Linear, use its in_features
                if isinstance(module[0], nn.Linear):
                    return self.create_tensor((self._BATCH_SIZE, module[0].in_features))
                # Otherwise, fall back to a generic feature dim
                return self.create_tensor((self._BATCH_SIZE, self._DEFAULT_FEATURE_DIM))

        # Fallback if no specific input-consuming direct child is found
        print(
            f"Warning: No specific input-consuming direct child layer found in the model '{model.__class__.__name__}'. Returning a generic ({self._BATCH_SIZE}, {self._DEFAULT_FEATURE_DIM}) tensor.")
        return self.create_tensor((self._BATCH_SIZE, self._DEFAULT_FEATURE_DIM))

    def run_gpu_training_warmup(self, device: torch.device = None):
        if device is not None:
            self.device = device

        modules_to_test = [
            nn.Linear, nn.Bilinear, nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d,
            nn.AdaptiveAvgPool2d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.ReLU,
            nn.Dropout, nn.RNN, nn.LSTM, nn.GRU, nn.LSTMCell, nn.Embedding,
            nn.MultiheadAttention, nn.TransformerEncoderLayer, nn.TransformerEncoder,
            nn.Transformer, nn.Flatten, nn.Unflatten, nn.PixelShuffle,
            nn.PixelUnshuffle, nn.ChannelShuffle, nn.Transformer, nn.Transformer
        ]

        for mod_class in modules_to_test:
            args, kwargs = self.get_module_config(mod_class)
            if args is None:
                continue
            try:
                instance = mod_class(*args, **kwargs).to(self.device, self.dtype)
                dummy_input_tuple = self.generate_dummy_input_for_module(instance)

                with torch.no_grad():
                    _ = instance(*dummy_input_tuple)
            except Exception as e:
                print(f"{e} - Cannot Perform GPU Warmup on Device: {self.device} \n{traceback.format_exc()}")

    def pre_forward_hook_input_shaper(self, module: nn.Module, input_tuple: Tuple[Union[torch.Tensor, Tuple], ...]) -> \
            Tuple[Union[torch.Tensor, Tuple], ...]:
        """
        A pre-forward hook that dynamically reshapes or generates input for the module.
        This hook is registered on submodules during tracing. It ensures that each
        module receives a valid input, either by respecting existing compatible inputs
        or by generating new dummy inputs, especially for modules requiring specific
        initial states (like RNNs).

        Args:
            module (nn.Module): The module for which the forward pass is about to occur.
            input_tuple (Tuple[Union[torch.Tensor, Tuple], ...]): The input that was
                                                                   passed to the module's forward method.

        Returns:
            Tuple[Union[torch.Tensor, Tuple], ...]: The modified or newly generated input
                                                      tuple for the module.
        """
        # If the incoming input_tuple is empty or contains None (e.g., initial call to a module for testing),
        # or if it's a recurrent module (which often expects (input, hidden_state) tuples),
        # we generate a fresh, appropriate dummy input.
        if not input_tuple or \
                (len(input_tuple) == 1 and input_tuple[0] is None) or \
                isinstance(module, (nn.RNN, nn.LSTM, nn.GRU, nn.LSTMCell)):
            return self.generate_dummy_input_for_module(module)
        else:
            # If a valid input tensor is already present from the preceding module, pass it through.
            # This handles sequential flow and residual connections.
            return input_tuple

    def get_module_substitute_args_kwargs(self, module: nn.Module) -> Tuple[Tuple, Dict]:
        """
        Retrieves the (args, kwargs) for instantiating a smaller version of the given module.
        It uses the predefined MODULE_CONFIG_MAP to get the smaller capacity parameters.
        """
        module_class = type(module)
        config = self.MODULE_CONFIG_MAP.get(module_class)

        if config is None:
            logger.warning(f"No specific configuration for {module_class.__name__} found. Attempting to infer smaller parameters.")
            # Fallback: Try to use existing parameters but attempt to reduce their size
            init_signature = inspect.signature(module_class.__init__)
            substitute_args = []
            substitute_kwargs = {}

            # Retrieve parameters from the current module instance if they exist, or use defaults/inferred.
            for param_name, param in init_signature.parameters.items():
                if param_name in ['self', '_', '__class__'] or param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue

                value = None
                # Try to get the original value from the module instance
                if hasattr(module, param_name):
                    value = getattr(module, param_name)
                elif param.default is not inspect.Parameter.empty:
                    value = param.default

                # Apply "mini-fication" logic to numerical parameters
                if isinstance(value, int) and value > 1:
                    if 'features' in param_name or 'dim' in param_name or 'size' in param_name or 'channels' in param_name:
                        value = max(1, value // 2) # Halve numerical dimensions, but at least 1
                    elif 'num_heads' in param_name:
                        value = max(1, value // 2) # Halve number of heads, but at least 1
                    elif 'num_layers' in param_name:
                        value = max(1, value // 2) # Halve number of layers, but at least 1
                elif isinstance(value, tuple) and all(isinstance(x, int) for x in value):
                    value = tuple(max(1, x // 2) for x in value) # Halve tuple dimensions

                # Special handling for nested modules (like TransformerEncoderLayer inside TransformerEncoder)
                if isinstance(value, nn.Module) and (
                    param_name == 'encoder_layer' or param_name == 'decoder_layer' or \
                    param_name == 'custom_encoder' or param_name == 'custom_decoder'
                ):
                    # Recursively get mini args/kwargs for the nested module
                    nested_sub_args, nested_sub_kwargs = self.get_module_substitute_args_kwargs(value)
                    try:
                        # Instantiate the nested mini module
                        value = type(value)(*nested_sub_args, **nested_sub_kwargs)
                    except Exception as e:
                        logger.error(f"Failed to recursively instantiate mini nested module {type(value).__name__} for '{param_name}'. Error: {e}")
                        value = nn.Identity() # Fallback

                # Fallback for required args without default or inferrable value
                if value is None and param.default is inspect.Parameter.empty:
                    logger.warning(f"Could not infer value for required argument '{param_name}' for {module_class.__name__}. Using generic default.")
                    if 'size' in param_name or 'features' in param_name or 'dim' in param_name:
                        value = self._DEFAULT_FEATURE_DIM
                    elif 'channels' in param_name:
                        value = self._DEFAULT_CHANNELS
                    elif 'kernel_size' in param_name:
                        value = 3
                    elif 'normalized_shape' in param_name:
                        value = (self._DEFAULT_FEATURE_DIM,)
                    elif 'output_size' in param_name:
                        value = (1, 1)
                    elif 'num_embeddings' in param_name:
                        value = self._DEFAULT_VOCAB_SIZE
                    elif 'start_dim' in param_name or 'end_dim' in param_name or 'dim' in param_name:
                        value = 1
                    elif 'unflattened_size' in param_name:
                        value = (self._DEFAULT_CHANNELS, self._DEFAULT_IMAGE_SIZE, self._DEFAULT_IMAGE_SIZE)
                    elif 'p' == param_name:
                        value = 0.1
                    else:
                        value = 1 # Generic numerical default

                if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                    substitute_args.append(value)
                else:
                    substitute_kwargs[param_name] = value

            return tuple(substitute_args), substitute_kwargs

        # If a specific configuration exists, use it
        substitute_args = []
        substitute_kwargs = {}
        pos_arg_names = config.get('_pos_args', ())

        # Populate positional arguments from config
        for arg_name in pos_arg_names:
            if arg_name in config:
                value = config[arg_name]
                if value is None:
                    # Handle recursive instantiation for Transformer layers if placeholder is None
                    if (module_class == nn.TransformerEncoder and arg_name == 'encoder_layer') or \
                       (module_class == nn.TransformerDecoder and arg_name == 'decoder_layer') or \
                       (module_class == nn.Transformer and arg_name in ('custom_encoder', 'custom_decoder')):
                        nested_module_class = None
                        if arg_name == 'encoder_layer' or arg_name == 'custom_encoder':
                            nested_module_class = nn.TransformerEncoderLayer
                        elif arg_name == 'decoder_layer' or arg_name == 'custom_decoder':
                            nested_module_class = nn.TransformerDecoderLayer

                        if nested_module_class:
                            # Get the original nested module from the `module` instance if it exists
                            original_nested_module = getattr(module, arg_name, None)
                            if original_nested_module is None or not isinstance(original_nested_module, nn.Module):
                                # Fallback if original nested module cannot be retrieved (e.g., not yet set)
                                # Or if it's not a module (e.g., custom_encoder might be a function)
                                nested_sub_args, nested_sub_kwargs = self.get_module_config(nested_module_class)
                            else:
                                nested_sub_args, nested_sub_kwargs = self.get_module_substitute_args_kwargs(original_nested_module)

                            # Ensure d_model for nested layer is consistent with the parent's (mini) d_model
                            d_model_for_nested = config.get('d_model', self._DEFAULT_FEATURE_DIM)
                            if d_model_for_nested is not None:
                                if 'd_model' in nested_sub_kwargs:
                                    nested_sub_kwargs['d_model'] = d_model_for_nested
                                else: # If d_model is positional for nested_module_class
                                    nested_init_signature = inspect.signature(nested_module_class.__init__)
                                    nested_pos_names = [p.name for p in nested_init_signature.parameters.values()
                                                        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and p.name != 'self']
                                    if 'd_model' in nested_pos_names and nested_pos_names.index('d_model') < len(nested_sub_args):
                                        nested_sub_args = list(nested_sub_args)
                                        nested_sub_args[nested_pos_names.index('d_model')] = d_model_for_nested
                                        nested_sub_args = tuple(nested_sub_args)

                            try:
                                value = nested_module_class(*nested_sub_args, **nested_sub_kwargs)
                            except Exception as e:
                                logger.error(f"Failed to instantiate nested module {nested_module_class.__name__} for {module_class.__name__}. Error: {e}")
                                value = nn.Identity() # Fallback to Identity to avoid breaking
                        else:
                            value = None # Should not happen for these names
                substitute_args.append(value)
            else:
                # If a positional argument is in _pos_args but not in the config map itself,
                # try to get it from the original module's attributes.
                if hasattr(module, arg_name):
                    substitute_args.append(getattr(module, arg_name))
                else:
                    logger.warning(f"Positional argument '{arg_name}' for {module_class.__name__} not found in config or module attributes.")
                    # Fallback to default if not found
                    if 'features' in arg_name or 'dim' in arg_name or 'size' in arg_name:
                        substitute_args.append(self._DEFAULT_FEATURE_DIM)
                    elif 'channels' in arg_name:
                        substitute_args.append(self._DEFAULT_CHANNELS)
                    elif 'kernel_size' in arg_name:
                        substitute_args.append(3) # Common default
                    elif 'normalized_shape' in arg_name:
                        substitute_args.append((self._DEFAULT_FEATURE_DIM,))
                    elif 'output_size' in arg_name:
                        substitute_args.append((1, 1))
                    elif 'upscale_factor' in arg_name or 'downscale_factor' in arg_name or 'groups' in arg_name:
                        substitute_args.append(2) # Common default for these
                    else:
                        substitute_args.append(1) # Generic default

        # Then, handle keyword arguments from the config map
        for name, value in config.items():
            if name not in pos_arg_names and name != '_pos_args':
                # Check if the parameter exists in the actual __init__ signature
                if name in inspect.signature(module_class.__init__).parameters:
                    substitute_kwargs[name] = value

        return tuple(substitute_args), substitute_kwargs


    def get_module_call_order(self, model: nn.Module) -> List[str]:
        """
        Performs a dummy forward pass on a model to trace the call order of its submodules.
        It registers hooks on all submodules to record their class names as they are called.

        Args:
            model (nn.Module): The model to inspect.

        Returns:
            List[str]: A list of submodule class names in the order they were called
                       during the dummy forward pass.
        """
        module_call_order = []
        try:
            self._register_module_hooks(model, module_call_order)
            d_input = self.generate_dummy_input_for_model_entry(model)
            logger.debug(f"Generated initial dummy input shape for model: {d_input.shape}, dtype: {d_input.dtype}")
            with torch.no_grad():
                _ = model(d_input)
        except Exception as e:
            logger.error(f"Error during model trace: {e}")
            logger.error(f"Traceback (most recent call last):\n{traceback.format_exc()}")
        finally:
            # Ensure all hooks are removed even if an error occurs to prevent memory leaks
            self.remove_module_hooks()

        return module_call_order

    def _register_module_hooks(self, model: torch.nn.Module, module_call_order: list[str]):
        """Register hooks on all submodules to record their class names as they are called."""
        for name, module in model.named_modules():
            # Exclude the top-level model itself as it's the wrapper
            if module is model: continue

            # Skip modules without parameters
            if not list(module.named_parameters(recurse=False)): continue

            # Register the input shaper hook to ensure valid inputs
            self._module_hooks.append(
                module.register_forward_pre_hook(self.pre_forward_hook_input_shaper)
            )
            # Register a hook to record the module class name after its forward pass
            def _forward_hook(module, input, output, name=name):
                module_call_order.append(name)

            self._module_hooks.append(module.register_forward_hook(_forward_hook))

        if self._module_hooks:
            logger.debug(f"{len(self._module_hooks)} new modules were registered for model tracing.")

    def remove_module_hooks(self):
        for hook in self._module_hooks:
            hook.remove()

        if self._module_hooks:
            logger.debug(f"{len(self._module_hooks)} hooks were removed from model tracing.")
            self._module_hooks.clear()


class AsyncSegmentMemory:
    """
    Manages model parameters by offloading them to memory-mapped files and loading them
    asynchronously just-in-time for computation, preventing the main thread from blocking on I/O.
    """

    def __init__(self, cache_dir: str = "", max_memory_usage: float = 0.9):
        logger.info("Initializing AsyncSegmentMemory...")
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix="segment_memory_cache_")
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Using cache directory: {self.cache_dir}")
        self.module_tracker = ModuleTracker()
        self.max_device_memory = get_available_memory(max_memory_usage)

        # --- Core Asynchronous Components ---
        self._stop_event = threading.Event()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.executor = ThreadPoolExecutor(thread_name_prefix='AsyncMem-IO')
        self._event_loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)

        # --- State Management ---
        self.segment_info: Dict[str, dict] = {}
        self.mmap_tensors: Dict[str, torch.Tensor] = {}  # Cache for open mmap tensors
        self.registered_parameters: Dict[str, nn.Parameter] = {}
        self.module_exec_order: List[str] = []
        self.module_to_segment_ids: Dict[str, List[str]] = {}
        self.segment_locks: Dict[str, asyncio.Lock] = {}

        # --- Futures for Synchronization ---
        self.pending_loads: Dict[str, Future] = {}  # Module_name -> Future for its parameters
        self.pending_offloads: Dict[str, Future] = {}  # Module_name -> Future for offloading

        self._id_generator = itertools.count()
        self._is_initialized = False

    def __call__(self, cls: Type[T_Module], *args, **kwargs):
        """
        Initialize the specified class with active parameter offloading to disk.
        """

        """
        s_model = self.module_tracker(cls, *args, **kwargs).to(self.module_tracker.device)
        self.registered_modules = [""]  # Add an empty string as placeholder for the top level module.
        self.registered_modules.extend(self.module_tracker.get_module_call_order(s_model))
        """


        cls = self.register_and_offload_parameters(cls, *args, **kwargs)
        temp_kwargs = kwargs.copy()
        temp_kwargs["_is_meta_pass"] = True

        # First, create a dummy instance for meta-pass, which sets _actual_device and _actual_dtype on it
        dummy_instance = cls(*args, **temp_kwargs)  # This will call new_init with _is_meta_pass=True

        # Retrieve the intended device and dtype from the dummy instance
        device = getattr(dummy_instance, '_actual_device', None)
        dtype = getattr(dummy_instance, '_actual_dtype', None)

        init_kwargs = kwargs.copy()
        # Pass the original kwargs, but ensure device/dtype are present if they were determined.
        if device is not None:
            init_kwargs["device"] = device
        if dtype is not None:
            init_kwargs["dtype"] = dtype

        self._event_loop_thread.start()
        while self.loop is None: time.sleep(0.01)

        self._is_initialized = True
        logger.info("Tracing complete. Returning decorated model class.")

        instance = cls(*args, **kwargs)

        return instance

    def initialize(self, model_class: Type[T_Module], *args, **kwargs):
        """
        Analyzes the model and prepares the class for asynchronous offloading.
        """
        if self._is_initialized:
            logger.warning("AsyncSegmentMemory is already initialized.")
            return

        self._event_loop_thread.start()
        while self.loop is None: time.sleep(0.01)

        logger.info("--- Phase 1: Model Tracing and Structure Analysis ---")
        # Instantiate model on 'meta' device for ultra-fast, memoryless tracing
        device = torch.device('cpu')

        module_tracker = ModuleTracker(device)
        t_model = module_tracker(model_class, *args, **kwargs, device='cpu', dtype=torch.float32)
        self.module_exec_order = module_tracker.get_module_call_order(t_model)
        logger.info(f"Model execution order: {self.module_exec_order}")
        del t_model  # Clean up trace model

        self._is_initialized = True
        logger.info("Tracing complete. Returning decorated model class.")
        return self._decorate_class(model_class)

    def _decorate_class(self, cls: Type[T_Module]) -> Type[T_Module]:
        """Applies a wrapper to the model's __init__ method."""
        original_init = cls.__init__
        asm_self = self

        @functools.wraps(original_init)
        def new_init(self: nn.Module, *args, **kwargs):
            # We must not initialize parameters on a device yet.
            # So we build the model on the 'meta' device first.
            with torch.device('meta'):
                original_init(self, *args, **kwargs)

            # Now, setup the real instance for offloading
            asm_self._setup_real_instance(self, target_device=kwargs.get('device'), target_dtype=kwargs.get('dtype'))

        cls.__init__ = new_init
        return cls

    def _setup_real_instance(self, model: nn.Module, target_device: torch.device, target_dtype):
        """Sets up a fully constructed model instance for offloading."""
        if target_device.type != "meta":
            logger.info("--- Phase 2: Registering Parameters and Attaching Hooks ---")
            # Register each parameter, giving it a segment ID and metadata
            for name, param in model.named_parameters():
                try:
                    # Update parameter with actual name
                    segment_id = getattr(param, SEGMENT_ID)
                    self.segment_info[segment_id][NAME] = name
                except AttributeError:
                    logger.error(f"Parameter {name} not registered. Adding new parameter...")
                    self._register_parameter(param, name)

            total_params_gb = sum(info[SIZE_IN_BYTES] for info in self.segment_info.values()) / BYTES_TO_GB
            logger.info(f"Registered {len(self.segment_info)} parameter segments, totaling {total_params_gb:.2f} GB.")

            # Create the memory-mapped files on disk. This is a blocking call.
            self._submit_to_loop(self._create_mmap_files_async()).result()

            # Attach hooks for JIT loading/offloading
            self._post_init_setup(model)
        else:
            logger.info(f"Skipping meta pass...")

    def register_and_offload_parameters(self, cls, *args, **kwargs):
        """Decorator with automatic naming and just-in-time parameter offloading."""
        original_init = cls.__init__
        hooks: Dict[str, RemovableHandle] = dict()
        hooks["optimizer_hook"] = register_optimizer_step_pre_hook(_set_optimizer_step_pre_hook)
        logger.info("--- Phase 1: Model Tracing and Structure Analysis ---")

        _start = time.time()
        t_model = self.module_tracker(cls, *args, **kwargs)
        self.module_exec_order = self.module_tracker.get_module_call_order(t_model)
        logger.info(f"Retrieved Model Execution Order in {time.time() - _start}: \n{self.module_exec_order}")
        del t_model  # Clean up trace model
        asm = self

        def add_global_hooks():
            """Register global module and parameter registration hooks for offloading parameters."""
            logger.info("--- Adding global module and parameter registration hooks ---")
            hooks["param_hook"] = register_module_parameter_registration_hook(asm._set_param_registration_hook)

        def remove_global_hooks():
            """
            Remove global module and parameter registration hooks.
            <<Always remove hooks to prevent side effects on other model instantiations>>"""
            logger.info("--- Removing global module and parameter registration hooks ---")
            for name in ["param_hook"]:
                if name in hooks:  # Check if hook exists before trying to remove
                    hooks[name].remove()
                    del hooks[name]

        @contextlib.contextmanager
        def _set_default_dtype_context(dtype):
            if dtype is None:  # No-op if dtype is None
                yield
                return
            old_dtype = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            try:
                yield
            finally:
                torch.set_default_dtype(old_dtype)

        @functools.wraps(original_init)
        def new_init(self: nn.Module, *args, **kwargs):
            # Determine if this is the "meta pass" based on a kwarg, not global state
            is_meta_pass = kwargs.pop("_is_meta_pass", False)

            if is_meta_pass:
                parameters = inspect.signature(original_init).parameters

                # Store the intended device and dtype for the actual instance
                self._actual_device = kwargs.pop("device", None)
                self._actual_dtype = kwargs.pop("dtype", None)

                # Context for meta device creation
                with contextlib.ExitStack() as stack:
                    stack.enter_context(torch.device("meta"))
                    original_default_dtype = torch.get_default_dtype()
                    # Apply the default dtype for meta pass if specified
                    stack.enter_context(_set_default_dtype_context(self._actual_dtype or original_default_dtype))

                    temp_kwargs = kwargs.copy()
                    if 'device' in parameters:
                        temp_kwargs["device"] = torch.device("meta")
                    # Ensure dtype is also passed to original_init if it accepts it
                    if 'dtype' in parameters and self._actual_dtype is not None:
                        temp_kwargs["dtype"] = self._actual_dtype

                    start_time = time.time()
                    # Call original_init on 'self' in the meta context
                    original_init(self, *args, **temp_kwargs)
                    end_time = time.time()
                    logger.info(f"Meta Device Execution of {cls.__name__} took {end_time - start_time:.4f} seconds.")

                    total_params = 0
                    total_param_bytes = 0
                    for name, param in self.named_parameters():
                        total_params += param.numel()
                        total_param_bytes += (param.dtype.itemsize * param.numel())
                    if total_params > ONE_BILLION:
                        logger.info(f"Total Parameters: {total_params / ONE_BILLION:.2f} Billion")
                    elif total_params > ONE_MILLION:
                        logger.info(f"Total Parameters: {total_params / ONE_BILLION:.2f} Million")
                    else:
                        logger.info(f"Total Parameters: {total_params}")
                    logger.info(f"Total Memory Requirement (without offload): {total_param_bytes / BYTES_TO_GB:.2f} GB")
                    logger.info(f"--------------------------------------------------")
            else:  # This is the actual instance creation path
                logger.info(f"Initializing {cls.__name__} with parameter offloading...")
                # Determine the actual device to use
                actual_device = kwargs.get("device")
                if actual_device is None:
                    actual_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    kwargs["device"] = actual_device  # Ensure kwargs has the device for original_init

                actual_dtype = kwargs.get("dtype")  # Get dtype from kwargs if provided
                original_default_dtype = torch.get_default_dtype()  # Store current default dtype

                add_global_hooks()  # Add hooks before original_init for parameter registration

                try:
                    # Call original __init__ to create actual tensors and capture registered modules/parameters.
                    start_time = time.time()
                    # Use context managers to set device/dtype for the original_init call
                    with contextlib.ExitStack() as stack:
                        # If the module itself accepts 'device' in its __init__, let it handle it.
                        # Otherwise, use torch.device context.
                        if 'device' not in inspect.signature(original_init).parameters:
                            stack.enter_context(actual_device)
                        # If the module itself accepts 'dtype' in its __init__, let it handle it.
                        # Otherwise, use the _set_default_dtype_context.
                        if 'dtype' not in inspect.signature(original_init).parameters:
                            stack.enter_context(_set_default_dtype_context(actual_dtype or original_default_dtype))

                        original_init(self, *args, **kwargs)

                    end_time = time.time()
                    logger.info(f"Execution of {cls.__name__} took {end_time - start_time:.4f} seconds.")
                except Exception as e:
                    logger.error(f"Cannot setup offloading hooks - {e}", exc_info=True)
                finally:
                    remove_global_hooks()

                # Set up the real instance for offloading
                asm._setup_real_instance(self, target_device=actual_device, target_dtype=actual_dtype)

        @classmethod
        async def create(cls, *args, **kwargs):
            """--- Async Class Method (Factory) ---"""
            temp_kwargs = kwargs.copy()
            temp_kwargs["_is_meta_pass"] = True

            # First, create a dummy instance for meta-pass, which sets _actual_device and _actual_dtype on it
            dummy_instance = cls(*args, **temp_kwargs)  # This will call new_init with _is_meta_pass=True

            # Retrieve the intended device and dtype from the dummy instance
            device = getattr(dummy_instance, '_actual_device', None)
            dtype = getattr(dummy_instance, '_actual_dtype', None)

            init_kwargs = kwargs.copy()
            # Pass the original kwargs, but ensure device/dtype are present if they were determined.
            if device is not None:
                init_kwargs["device"] = device
            if dtype is not None:
                init_kwargs["dtype"] = dtype

            # Then, create the actual instance in a separate thread to not block the event loop
            # This will call new_init with _is_meta_pass=False
            instance = await asyncio.to_thread(lambda: cls(*args, **init_kwargs))

            return instance

        cls.__init__ = new_init
        cls.create = create

        return cls

    def _run_event_loop(self):
        """The target function for the dedicated event loop thread."""
        logger.info("Event loop thread started.")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.set_default_executor(self.executor)

        try:
            self.loop.run_forever()
        finally:
            logger.info("Event loop is shutting down.")
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.executor.shutdown(wait=True)
            self.loop.close()
            logger.info("Event loop shut down completely.")

    def _submit_to_loop(self, coro):
        """Thread-safe submission of a coroutine to the running event loop."""
        if not self.loop or not self.loop.is_running():
            raise RuntimeError("Event loop is not running.")
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    async def _create_mmap_files_async(self):
        """Coroutine to create all necessary memory-mapped files and cache the mmap tensor objects."""
        mmap_file_specs = {}
        for seg_id, info in self.segment_info.items():
            key = (info[SHAPE], info[DTYPE])
            if key not in mmap_file_specs:
                mmap_file_specs[key] = []
            mmap_file_specs[key].append(seg_id)

        for (shape, dtype), seg_ids in mmap_file_specs.items():
            _shape = "_X_".join([str(x) for x in shape])
            filename = f"segment_{_shape}_{str(dtype).split('.')[-1]}.bin"
            file_path = os.path.join(self.cache_dir, filename)

            # Create one mmap tensor for all segments of the same shape/dtype
            mmap_tensor = create_mmap_tensor_from_file(file_path, len(seg_ids), shape, dtype)
            self.mmap_tensors[file_path] = mmap_tensor

            # Assign each segment its file path and its index within that file
            for i, seg_id in enumerate(seg_ids):
                self.segment_info[seg_id]['mmap_path'] = file_path
                self.segment_info[seg_id]['mmap_index'] = i
        logger.info("All memory-mapped files have been created.")

    def _register_parameter(self, param: nn.Parameter, name: str):
        """Gathers parameter metadata, assigns a segment ID, and offloads it immediately."""
        if hasattr(param, SEGMENT_ID): return

        segment_id = str(next(self._id_generator))
        setattr(param, SEGMENT_ID, segment_id)

        self.segment_info[segment_id] = {
            NAME: name,
            DEVICE: param.device,
            SHAPE: tuple(param.shape),
            DTYPE: param.dtype,
            INITIALIZED: False,
            SIZE_IN_BYTES: param.numel() * torch.tensor([], dtype=param.dtype).element_size()
        }
        self.segment_locks[segment_id] = asyncio.Lock()

        # Replace parameter data with an empty tensor to free memory.
        # The materialization happens on the target device later.
        with torch.no_grad():
            if 'weight' in name:
                param.data = torch.empty((2, 2), device=param.device, dtype=param.dtype)  # keep data on original device
            else:
                param.data = torch.empty((2,), device=param.device, dtype=param.dtype)  # keep data on original device
            #param.requires_grad = False  # Turn off grads until loaded

    def _post_init_setup(self, model: nn.Module):
        """Adds forward hooks and maps modules to their parameter segments."""
        self.registered_parameters = {
            getattr(p, SEGMENT_ID): p for _, p in model.named_parameters() if hasattr(p, SEGMENT_ID)
        }

        module_map = dict(model.named_modules())
        for module_name in self.module_exec_order:
            try:
                module = module_map[module_name]
                segment_ids = extract_segment_ids(module)[0]  # Find all parameters that are direct children
                self.module_to_segment_ids[module_name] = segment_ids
                # Bind the module_name to the hook functions
                pre_hook_with_name = functools.partial(self._pre_forward_hook, module_name=module_name)
                post_hook_with_name = functools.partial(self._post_forward_hook, module_name=module_name)
                module.register_forward_pre_hook(pre_hook_with_name)
                module.register_forward_hook(post_hook_with_name)
            except KeyError:
                logger.error(f"Could not find module {module_name} during _post_init_setup.", exc_info=True)

        # Proactively start loading the parameters for the very first module in the execution order
        if self.module_exec_order:
            self._schedule_preload(self.module_exec_order[0])

    def _set_param_registration_hook(self, module: nn.Module, name: str, param: nn.Parameter) -> None:
        """
        A parameter registration hook that sets a 'name_in_parent' attribute on the registered parameter.
        Also handles immediate initialization and offloading if not on meta device.
        """
        if isinstance(param, nn.Parameter) and param.device.type != 'meta':
            self._register_parameter(param, name)  # Add parameter to tracker

    def _pre_forward_hook(self, module: nn.Module, inputs: Tuple, module_name: str):
        """Hook that blocks until parameters are loaded and then schedules the next load."""
        if module_name in self.pending_loads:
            future = self.pending_loads.pop(module_name)
            try:
                # This is a blocking call that waits for the async load to finish
                future.result(timeout=300)  # Wait up to 20 seconds
            except Exception as e:
                logger.error(f"FATAL: Parameter loading for module '{module_name}' failed or timed out.",
                             exc_info=False)
                # Re-raise the original exception from the async task
                raise RuntimeError(f"Parameter loading failed for {module_name}") from e

        # Schedule the preload for the *next* module in the sequence
        try:
            current_module_idx = self.module_exec_order.index(module_name)
            if current_module_idx + 1 < len(self.module_exec_order):
                next_module_name = self.module_exec_order[current_module_idx + 1]
                self._schedule_preload(next_module_name)
            else:
                next_module_name = self.module_exec_order[0]
                self._schedule_preload(next_module_name)
        except (ValueError, IndexError):
            logger.warning(f"Module '{module_name}' not in exec order or is the last module. Cannot preload next.")

    def _post_forward_hook(self, module: nn.Module, inputs: Tuple, outputs, module_name: str):
        """Schedules the non-blocking offloading of the current module's parameters."""
        logger.info(f"Forward pass completed for module '{module_name}'.")
        segment_ids = self.module_to_segment_ids.get(module_name, [])
        if not segment_ids: return

        # This is a fire-and-forget operation
        future = self._submit_to_loop(self._offload_segments_async(segment_ids, module_name))
        future.result(timeout=300)  # Wait up to 20 seconds
        """
        self.pending_offloads[module_name] = self._submit_to_loop(
            self._offload_segments_async(segment_ids, module_name)
        )
        """

    def _schedule_preload(self, module_name: str):
        """Schedules the asynchronous loading of a module's parameters if not already in flight."""
        if module_name in self.pending_loads:
            logger.error(f"Module '{module_name}' already preloaded.")
            return

        if not module_name:
            logger.error(f"Cannot preload module '{module_name}', Invalid Name.")
            return
        segment_ids = self.module_to_segment_ids.get(module_name, [])
        if not segment_ids:
            logger.error(f"Cannot preload module '{module_name}'. No segment ids found: {segment_ids}")
            return

        logger.info(f"Scheduling preload for module: '{module_name}'")
        future = self._submit_to_loop(self._load_segments_async(segment_ids, module_name))
        self.pending_loads[module_name] = future

    async def _load_segments_async(self, segment_ids: List[str], module_name: str):
        """Coroutine to load parameters, using segment-level locking and robust error handling."""
        logger.info(f"Loading parameters for '{module_name}'...")
        tasks = [self._read_and_init_param(seg_id) for seg_id in segment_ids]

        # Use return_exceptions=True to prevent gather from failing fast
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # This part must run without interruption to ensure model state is consistent
        with torch.no_grad():
            for seg_id, data_or_exc in zip(segment_ids, results):
                if isinstance(data_or_exc, Exception):
                    # If any sub-task failed, we must raise to notify the waiting main thread
                    logger.error(f"Sub-task failed for segment {seg_id} in module {module_name}: {data_or_exc}")
                    raise RuntimeError(f"Failed to load segment {seg_id}") from data_or_exc

                param = self.registered_parameters[seg_id]
                param.data = data_or_exc
                param.requires_grad = True  # Enable gradients now that it's on device
        logger.info(f"Finished loading for '{module_name}'.")

    def empty_cache(self):
        try:
            task = self.loop.run_in_executor(None, torch.cuda.empty_cache)
        except Exception as e:
            logger.error(f"Failed to clear cuda cache. {e}")

    async def _read_and_init_param(self, seg_id: str) -> torch.Tensor:
        """Coroutine-wrapped function to read from mmap file and initialize if needed."""
        # This async with block ensures the lock is handled correctly, even with exceptions.
        async with self.segment_locks[seg_id]:
            # This is the actual blocking I/O, run in the executor thread pool.
            return await self.loop.run_in_executor(None, self._read_from_mmap_sync, seg_id)

    def _read_from_mmap_sync(self, seg_id: str) -> torch.Tensor|None:
        """Synchronous function (run in executor) that handles disk read and initialization."""
        try:
            info = self.segment_info[seg_id]
            mmap_path = info['mmap_path']
            mmap_index = info['mmap_index']
            mmap_tensor = self.mmap_tensors[mmap_path]

            # Check if this segment has been initialized with weights yet
            if not info[INITIALIZED]:
                # Initialize on CPU first to avoid memory spikes on GPU
                temp_data = torch.empty(info[SHAPE], dtype=info[DTYPE], device='cpu')

                # Use standard initialization schemes
                param_name = info[NAME]
                if "weight" in param_name:
                    if temp_data.dim() > 1:
                        init.kaiming_uniform_(temp_data, nonlinearity="relu")  #a=math.sqrt(5)
                    else:
                        init.uniform_(temp_data)  # Fallback for 1D weights
                elif "bias" in param_name:
                    init.zeros_(temp_data)
                else:  # Default for other params like LayerNorm
                    init.ones_(temp_data)

                # Copy the initialized data to the memory-mapped file
                #mmap_tensor[mmap_index].copy_(temp_data)
                info[INITIALIZED] = True
                logger.info(f"Parameter {param_name} with id {seg_id} and shape {info[SHAPE]} Initialized for '{mmap_path}'.")

                # Return data moved to the final target device
                return temp_data.clone().to(info[DEVICE])
            else:
                # If already initialized, just read from disk and move to target device
                return mmap_tensor[mmap_index].clone().to(info[DEVICE])
        except Exception as e:
            logger.error(f"Could not read segment {seg_id} - {e}", exc_info=True)


    async def _offload_segments_async(self, segment_ids: List[str], module_name: str):
        """Coroutine to offload parameters, freeing device memory."""
        logger.info(f"Offloading segments {segment_ids} for '{module_name}'...")
        tasks = []
        _start = time.time()
        for seg_id in segment_ids:
            # Each offload operation acquires and releases its own lock
            async def offload_one(s_id):
                logger.info(f"Acquiring offload lock for segment {s_id}.")
                async with self.segment_locks[s_id]:
                    logger.info(f"Starting offload for segment {s_id}.")
                    param = self.registered_parameters.get(s_id)
                    with torch.no_grad():
                        start = time.time()
                        try:
                            info = self.segment_info[s_id]
                            mmap_path = info['mmap_path']
                            mmap_index = info['mmap_index']
                            mmap_tensor = self.mmap_tensors[mmap_path]
                            mmap_tensor[mmap_index].copy_(param.data.cpu())  # Save the updated data
                            logger.info(f"Executed mmap_tensor copy_ of {param.shape} in {time.time() - start:.2f} seconds.")
                            param.data = torch.zeros((2, 2), device=param.device, dtype=param.dtype)
                            # param.requires_grad = False
                        except Exception as e:
                            logger.error(f"Cannot offload segment {s_id} with shape {param.shape} to file {mmap_path} - {e}", exc_info=True)
                        finally:
                            logger.info(f"Executed offloaded segment {s_id} in {time.time() - start:.2f} seconds.")

            tasks.append(offload_one(seg_id))

        await asyncio.gather(*tasks)

        logger.info(f"Finished offloading '{module_name}' - {segment_ids} in {time.time() - _start:.2f} seconds.")

    def shutdown(self):
        """Gracefully shuts down the event loop and background thread."""
        logger.info("Shutdown requested.")
        if self._event_loop_thread.is_alive():
            self._stop_event.set()
            if self.loop and self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
            self._event_loop_thread.join(timeout=10)

        try:
            # Clean up memory-mapped tensors and the cache directory
            self.mmap_tensors.clear()
            shutil.rmtree(self.cache_dir)
            logger.info(f"Cleaned up cache directory: {self.cache_dir}")
        except OSError as e:
            logger.error(f"Error removing cache directory {self.cache_dir}: {e}")


class SimpleTransformerBlock(nn.Module):
    def __init__(self, in_features, num_heads=8, device=None, dtype=None):
        super().__init__()
        self.attention = nn.MultiheadAttention(in_features, num_heads, batch_first=True, device=device, dtype=dtype)
        self.norm1 = nn.LayerNorm(in_features, device=device, dtype=dtype)
        self.linear1 = nn.Linear(in_features, in_features * 4, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features * 4, in_features, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(in_features, device=device, dtype=dtype)

    def forward(self, x):
        try:
            # Pre-normalization (standard in many modern transformers)
            norm_x = self.norm1(x)
            attn_output, _ = self.attention(norm_x, norm_x, norm_x)
            x = x + attn_output

            norm_x_ffn = self.norm2(x)
            ffn_output = self.linear2(self.relu(self.linear1(norm_x_ffn)))
            x = x + ffn_output
            return x
        except Exception as e:
            logger.error(f"Cannot process layer - {e}", exc_info=e)
            logger.info(f"Named Parameters: {list(self.named_parameters())}")
            raise


class MySimpleModel(nn.Module):
    def __init__(self, num_layers=4, in_features=512, num_heads=8, device=None, dtype=None):
        super().__init__()
        # Use nn.Sequential for a simpler forward pass
        self.layers = nn.Sequential(*[
            SimpleTransformerBlock(in_features, num_heads, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(in_features, device=device, dtype=dtype)

    def forward(self, x):
        x = self.layers(x)
        return self.output_norm(x)


def gpu_warmup(device=torch.device("cuda"), num_segments=10):
    """
       Warms up the specified GPU unit by allocating and performing operations on tensors,
       then synchronizing the device.
       """
    if device.type == "cuda":
        # If device is "cuda" without an index, default to "cuda:0"
        if device.index is None:
            device = torch.device("cuda:0")
        # Ensure the specified device is valid and available
        if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
            logger.warning(f"Specified device {device} is not available or invalid. Skipping GPU warmup.")
            return
        logger.info(f"Warming up GPU: {device}")
        try:
            # Get available memory for the specific device
            available_memory_map = get_available_memory(max_memory_usage=0.95)  # Use 95% of available memory for warmup
            if str(device) not in available_memory_map:
                logger.warning(f"Could not get memory information for {device}. Skipping GPU warmup.")
                return
            max_device_memory = available_memory_map[str(device)]
            # Calculate the number of elements that can be held in memory
            # Use a slightly smaller fraction of memory to avoid OOM errors during operations
            num_elements = int(max_device_memory // torch.float32.itemsize)
            if num_elements <= 0:
                logger.warning(f"Not enough memory on {device} to perform GPU warmup.")
                return
            # Determine a reasonable shape for the tensors
            # Aim for roughly square matrices for better memory access patterns
            elements_per_segment = num_elements // num_segments
            if elements_per_segment <= 0:
                elements_per_segment = 1  # Ensure at least one element per segment
            # Calculate segment_size for a roughly square tensor
            segment_dim = int(math.sqrt(elements_per_segment))
            if segment_dim == 0:
                segment_dim = 1  # Ensure dimension is at least 1
            shape = (segment_dim, segment_dim)
            if shape[1] == 0:
                shape = (segment_dim, 1)  # Ensure the second dimension is not zero
            if shape[0] * shape[1] == 0:
                logger.warning(
                    f"Calculated tensor shape {shape} is too small for meaningful warmup on {device}. Skipping.")
                return
            data_tensors = []
            for x in range(num_segments):
                # Allocate tensor on the specified device
                data = torch.zeros(shape, dtype=torch.float32, device=device)
                # Perform a simple operation to ensure memory allocation and computation
                # This prevents lazy loading and forces the GPU to "warm up"
                # data.fill_(x) # Fill with a value
                # Perform a simple element-wise operations
                result = data + 1.7565
                result = result * 2.65
                data_tensors.append(result)

            """
            # Perform a final operation on all tensors (optional but good for more thorough warmup)
            if data_tensors:
                # Concatenate or sum if reasonable, or just iterate and touch
                for t in data_tensors:
                    _ = t.sum() # Forces computation
            """
            # Synchronize the device to ensure all operations are completed
            torch.cuda.synchronize(device)
            logger.info(f"GPU {device} warmup complete.")
            del data_tensors  # Release memory
            torch.cuda.empty_cache() # Clear cache after warmup
        except RuntimeError as e:
            logger.error(f"Error during GPU warmup on {device}: {e}")
            torch.cuda.empty_cache()  # Attempt to clear cache on error
    elif device.type == "cpu":
        logger.info("CPU device specified, no GPU warmup performed.")
    else:
        logger.info(f"Device type {device.type} not supported for specific warmup operations.")

# Main Execution
if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model parameters
    D_MODEL = 5120
    NUM_LAYERS = 5
    BATCH_SIZE = 4
    SEQ_LEN = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float32

    # Initialize the memory manager
    memory_manager = AsyncSegmentMemory()
    # Instantiating the model triggers the setup and hooking process that prepares the model for offloading
    model = memory_manager(
        MySimpleModel, num_layers=NUM_LAYERS, in_features=D_MODEL, dtype=DTYPE
    )

    # Note: For this setup to be effective, optimizer instantiation should be handled carefully.
    # A custom optimizer that can handle parameters moving in and out of the device might be needed
    # for a real use-case. For this simulation, we create it after the first forward pass
    # when all parameters have been initialized.
    optimizer = None

    print("\n--- Starting Training Simulation ---")
    model.train()  # This only affects modules like Dropout, etc.

    try:
        start_time = time.time()
        for i in range(5):
            print(f"\n--- Step {i + 1}/5 ---")
            dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, device=DEVICE, dtype=DTYPE)

            if optimizer:
                optimizer.zero_grad()

            _time = time.time()
            output = model(dummy_input)
            print(f"Step {i + 1} completed in {time.time() - _time:.2f} seconds.")

            # Since parameters are offloaded after use, backward() and step() need careful handling.
            # For this simulation, we'll skip the backward pass which would require all params on device.
            # loss.backward()
            # optimizer.step()
            current_params = dict(model.named_parameters())
            current_modules = list(model.named_modules())
        print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    except Exception as e:
        logger.error("An error occurred during the training loop.", exc_info=True)
    finally:
        print("\n--- Shutting down memory manager ---")
        memory_manager.shutdown()
        print("Shutdown complete.")
