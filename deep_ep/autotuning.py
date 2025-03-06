import os
import json
import time
import torch
import torch.distributed as dist
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from pathlib import Path

from .buffer import Buffer
from .utils import EventOverlap


class AutoTuner:
    """
    Automatic tuning system for DeepEP parameters.
    
    This class provides functionality for:
    1. Automatically benchmarking different configurations
    2. Finding optimal parameters for specific hardware setups
    3. Persisting configuration profiles for later use
    """
    
    def __init__(
        self,
        group: dist.ProcessGroup,
        config_dir: Optional[str] = None,
        cache_file: Optional[str] = None
    ):
        """
        Initialize the auto-tuner.
        
        Args:
            group: The communication group
            config_dir: Directory to store configuration profiles (defaults to ~/.deep_ep/configs)
            cache_file: File to store cached results (defaults to ~/.deep_ep/cache.json)
        """
        self.rank = group.rank()
        self.group_size = group.size()
        self.group = group
        
        # Set up configuration directory
        if config_dir is None:
            home_dir = os.path.expanduser("~")
            config_dir = os.path.join(home_dir, ".deep_ep", "configs")
        
        # Set up cache file
        if cache_file is None:
            home_dir = os.path.expanduser("~")
            cache_file = os.path.join(home_dir, ".deep_ep", "cache.json")
        
        self.config_dir = config_dir
        self.cache_file = cache_file
        
        # Ensure directories exist
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        # Load cache if it exists
        self.cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load cache file: {e}")
    
    def _get_hardware_signature(self) -> str:
        """
        Generate a unique signature for the current hardware setup.
        
        Returns:
            A string representing the hardware configuration
        """
        # Get GPU information
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        gpu_count = torch.cuda.device_count()
        
        # Get NCCL version
        nccl_version = torch.cuda.nccl.version() if hasattr(torch.cuda, 'nccl') else "unknown"
        
        # Create a hardware signature
        signature = f"{gpu_name}_{gpu_count}_{nccl_version}_{self.group_size}"
        
        return signature
    
    def _save_cache(self):
        """Save the current cache to disk"""
        if self.rank == 0:  # Only leader rank saves
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.cache, f)
            except Exception as e:
                print(f"Warning: Could not save cache file: {e}")
    
    def _save_config(self, config_name: str, config: Dict[str, Any]):
        """
        Save a configuration to disk.
        
        Args:
            config_name: Name of the configuration
            config: Configuration dictionary
        """
        if self.rank == 0:  # Only leader rank saves
            try:
                file_path = os.path.join(self.config_dir, f"{config_name}.json")
                with open(file_path, 'w') as f:
                    json.dump(config, f)
            except Exception as e:
                print(f"Warning: Could not save configuration file: {e}")
    
    def _load_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a configuration from disk.
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            Configuration dictionary if it exists, None otherwise
        """
        file_path = os.path.join(self.config_dir, f"{config_name}.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load configuration file: {e}")
        return None
    
    def _benchmark_sm_counts(
        self,
        hidden_size: int,
        num_tokens: int,
        num_experts: int,
        dtype: torch.dtype = torch.bfloat16,
        sm_values: Optional[List[int]] = None,
        num_trials: int = 10
    ) -> Tuple[int, float]:
        """
        Benchmark different SM count values to find the optimal one.
        
        Args:
            hidden_size: Hidden dimension size
            num_tokens: Number of tokens per batch
            num_experts: Number of experts
            dtype: Data type
            sm_values: List of SM counts to try (defaults to [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48])
            num_trials: Number of trials for each SM count
            
        Returns:
            Tuple of (optimal_sm_count, best_time)
        """
        if sm_values is None:
            sm_values = [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
        
        # Filter SM values based on available SMs
        device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
        max_sms = device_props.multi_processor_count
        sm_values = [sm for sm in sm_values if sm <= max_sms]
        
        # If no valid SM values, return default
        if not sm_values:
            return 20, float('inf')
        
        # Set up test data
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device='cuda')
        topk_idx = torch.randint(0, num_experts, (num_tokens, 2), device='cuda')
        topk_weights = torch.ones((num_tokens, 2), dtype=dtype, device='cuda') / 2
        
        best_sm_count = sm_values[0]
        best_time = float('inf')
        
        for sm_count in sm_values:
            # Create buffer with current SM count
            Buffer.set_num_sms(sm_count)
            buffer = Buffer(self.group, 
                            num_nvl_bytes=hidden_size * num_tokens * 4, 
                            num_rdma_bytes=hidden_size * num_tokens * 4)
            
            # Warm up
            for _ in range(3):
                layout_result = buffer.get_dispatch_layout(topk_idx, num_experts)
                buffer.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights, 
                               num_tokens_per_rank=layout_result[0],
                               num_tokens_per_rdma_rank=layout_result[1],
                               is_token_in_rank=layout_result[3],
                               num_tokens_per_expert=layout_result[2])
                torch.cuda.synchronize()
            
            # Benchmark
            times = []
            for _ in range(num_trials):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                layout_result = buffer.get_dispatch_layout(topk_idx, num_experts)
                dispatch_result = buffer.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights, 
                                               num_tokens_per_rank=layout_result[0],
                                               num_tokens_per_rdma_rank=layout_result[1],
                                               is_token_in_rank=layout_result[3],
                                               num_tokens_per_expert=layout_result[2])
                end_event.record()
                
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event) / 1000.0)  # Convert to seconds
            
            avg_time = np.mean(times[1:])  # Skip first time
            
            if avg_time < best_time:
                best_time = avg_time
                best_sm_count = sm_count
                
        return best_sm_count, best_time
    
    def tune_sm_count(
        self,
        hidden_size: int,
        num_tokens: int,
        num_experts: int,
        dtype: torch.dtype = torch.bfloat16,
        force_refresh: bool = False
    ) -> int:
        """
        Find the optimal SM count for the current hardware and model configuration.
        
        Args:
            hidden_size: Hidden dimension size
            num_tokens: Number of tokens per batch
            num_experts: Number of experts
            dtype: Data type
            force_refresh: Whether to force a new benchmark even if cached results exist
            
        Returns:
            Optimal SM count
        """
        # Create a cache key
        hw_signature = self._get_hardware_signature()
        config_key = f"sm_count_{hw_signature}_{hidden_size}_{num_tokens}_{num_experts}_{dtype}"
        
        # Check cache unless forced refresh
        if not force_refresh and config_key in self.cache:
            return self.cache[config_key]["optimal_value"]
        
        # Run benchmark
        optimal_sm_count, _ = self._benchmark_sm_counts(
            hidden_size=hidden_size,
            num_tokens=num_tokens,
            num_experts=num_experts,
            dtype=dtype
        )
        
        # Update cache
        self.cache[config_key] = {
            "optimal_value": optimal_sm_count,
            "timestamp": time.time()
        }
        self._save_cache()
        
        return optimal_sm_count
    
    def create_optimized_buffer(
        self,
        hidden_size: int,
        num_tokens: int,
        num_experts: int,
        dtype: torch.dtype = torch.bfloat16,
        low_latency_mode: bool = False,
        num_qps_per_rank: int = 1,
        optimize_sm_count: bool = True
    ) -> Buffer:
        """
        Create a Buffer with optimized parameters for the current hardware.
        
        Args:
            hidden_size: Hidden dimension size
            num_tokens: Number of tokens per batch
            num_experts: Number of experts
            dtype: Data type
            low_latency_mode: Whether to enable low-latency mode
            num_qps_per_rank: Number of QPs per rank
            optimize_sm_count: Whether to optimize SM count
            
        Returns:
            Optimized Buffer instance
        """
        # Determine nvl and rdma buffer sizes
        nvl_bytes = 0
        rdma_bytes = 0
        
        if low_latency_mode:
            # For low latency mode
            rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
                num_tokens, hidden_size, self.group_size, num_experts
            )
        else:
            # For normal mode, calculate buffer sizes
            element_size = torch.finfo(dtype).bits // 8
            # Add 20% buffer for safety
            nvl_bytes = int(hidden_size * num_tokens * element_size * 1.2)
            rdma_bytes = int(hidden_size * num_tokens * element_size * 1.2)
        
        # Optimize SM count if requested
        if optimize_sm_count and not low_latency_mode:
            optimal_sm_count = self.tune_sm_count(
                hidden_size=hidden_size,
                num_tokens=num_tokens,
                num_experts=num_experts,
                dtype=dtype
            )
            Buffer.set_num_sms(optimal_sm_count)
        
        # Create and return buffer
        return Buffer(
            self.group,
            num_nvl_bytes=nvl_bytes,
            num_rdma_bytes=rdma_bytes,
            low_latency_mode=low_latency_mode,
            num_qps_per_rank=num_qps_per_rank
        )
    
    def tune_and_save_profile(
        self,
        profile_name: str,
        hidden_size: int,
        num_tokens: int,
        num_experts: int,
        dtype: torch.dtype = torch.bfloat16
    ) -> Dict[str, Any]:
        """
        Run a complete tuning process and save the results as a named profile.
        
        Args:
            profile_name: Name for the configuration profile
            hidden_size: Hidden dimension size
            num_tokens: Number of tokens per batch
            num_experts: Number of experts
            dtype: Data type
            
        Returns:
            Configuration dictionary
        """
        # Tune SM count
        optimal_sm_count = self.tune_sm_count(
            hidden_size=hidden_size,
            num_tokens=num_tokens,
            num_experts=num_experts,
            dtype=dtype,
            force_refresh=True
        )
        
        # Create configuration dictionary
        config = {
            "hardware_signature": self._get_hardware_signature(),
            "created_at": time.time(),
            "parameters": {
                "hidden_size": hidden_size,
                "num_tokens": num_tokens,
                "num_experts": num_experts,
                "dtype": str(dtype),
            },
            "tuned_values": {
                "optimal_sm_count": optimal_sm_count
            }
        }
        
        # Save configuration
        self._save_config(profile_name, config)
        
        return config
    
    def load_profile(self, profile_name: str) -> bool:
        """
        Load and apply a named configuration profile.
        
        Args:
            profile_name: Name of the configuration profile
            
        Returns:
            True if profile was successfully loaded and applied, False otherwise
        """
        config = self._load_config(profile_name)
        if config is None:
            return False
        
        # Apply configuration
        if "tuned_values" in config and "optimal_sm_count" in config["tuned_values"]:
            Buffer.set_num_sms(config["tuned_values"]["optimal_sm_count"])
            return True
        
        return False 