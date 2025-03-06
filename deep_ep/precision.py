import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Type
from enum import Enum, auto
import contextlib


class PrecisionMode(Enum):
    """Enumeration of precision modes for MoE operations."""
    FULL = auto()       # Use full precision (FP32)
    MIXED = auto()      # Use mixed precision (BF16/FP16)
    LOW = auto()        # Use low precision (FP8)
    DYNAMIC = auto()    # Dynamically choose precision based on stability metrics
    HYBRID = auto()     # Use different precision for different stages


class FP8Format(Enum):
    """Enumeration of FP8 formats."""
    E4M3 = auto()     # 4 exponent bits, 3 mantissa bits
    E5M2 = auto()     # 5 exponent bits, 2 mantissa bits
    
    @classmethod
    def from_string(cls, format_str: str) -> 'FP8Format':
        """Convert string representation to FP8Format."""
        if format_str.lower() in ['e4m3', 'e4m3fn']:
            return cls.E4M3
        elif format_str.lower() in ['e5m2', 'e5m2fn']:
            return cls.E5M2
        else:
            raise ValueError(f"Unknown FP8 format: {format_str}")
    
    def to_torch_dtype(self) -> torch.dtype:
        """Convert FP8Format to torch dtype."""
        if self == FP8Format.E4M3:
            return torch.float8_e4m3fn
        elif self == FP8Format.E5M2:
            return torch.float8_e5m2fn
        else:
            raise ValueError(f"Unsupported FP8 format: {self}")


class StageConfig:
    """Configuration for precision settings at a specific MoE computation stage."""
    
    def __init__(
        self,
        dtype: torch.dtype = torch.bfloat16,
        fp8_format: Optional[FP8Format] = None,
        scaling_factor: float = 1.0,
        amax_history_len: int = 16,
        amax_compute_algo: str = "max",
        has_fp8_weights: bool = False,
        dynamic_threshold: float = 1e-4,
    ):
        """
        Initialize stage precision configuration.
        
        Args:
            dtype: Default data type when not using FP8
            fp8_format: FP8 format to use if low precision is enabled
            scaling_factor: Initial scaling factor for FP8 quantization
            amax_history_len: Length of history for amax statistics
            amax_compute_algo: Algorithm for computing amax from history ('max' or 'moving_average')
            has_fp8_weights: Whether this stage has pre-quantized FP8 weights
            dynamic_threshold: Threshold for numerical stability in dynamic precision mode
        """
        self.dtype = dtype
        self.fp8_format = fp8_format
        self.scaling_factor = scaling_factor
        self.amax_history_len = amax_history_len
        self.amax_compute_algo = amax_compute_algo
        self.has_fp8_weights = has_fp8_weights
        self.dynamic_threshold = dynamic_threshold
        
        # Runtime statistics
        self.amax_history = []
        self.current_amax = None
        
    def is_fp8_enabled(self) -> bool:
        """Check if FP8 is enabled for this stage."""
        return self.fp8_format is not None
    
    def get_current_dtype(self, mode: PrecisionMode) -> torch.dtype:
        """
        Get the active data type based on precision mode.
        
        Args:
            mode: The current precision mode
            
        Returns:
            The appropriate torch dtype
        """
        if mode == PrecisionMode.FULL:
            return torch.float32
        elif mode == PrecisionMode.LOW and self.is_fp8_enabled():
            return self.fp8_format.to_torch_dtype()
        else:
            return self.dtype


class PrecisionManager:
    """
    Manager for precision control in DeepEP.
    
    This class provides utilities to control precision at different stages
    of MoE computation, including FP8 quantization.
    """
    
    def __init__(
        self,
        default_mode: PrecisionMode = PrecisionMode.MIXED,
        default_dtype: torch.dtype = torch.bfloat16,
        default_fp8_format: FP8Format = FP8Format.E4M3,
    ):
        """
        Initialize the precision manager.
        
        Args:
            default_mode: Default precision mode
            default_dtype: Default data type for non-FP8 operations
            default_fp8_format: Default FP8 format for low precision operations
        """
        self.default_mode = default_mode
        self.default_dtype = default_dtype
        self.default_fp8_format = default_fp8_format
        
        # Current active mode
        self.current_mode = default_mode
        
        # Stage configurations
        self.stage_configs = {
            # Default configurations for each MoE stage
            "router": StageConfig(dtype=default_dtype),
            "dispatch": StageConfig(dtype=default_dtype, fp8_format=default_fp8_format),
            "expert_compute": StageConfig(dtype=default_dtype),
            "combine": StageConfig(dtype=default_dtype, fp8_format=default_fp8_format),
        }
        
    def configure_stage(self, stage_name: str, **kwargs) -> None:
        """
        Configure precision settings for a specific stage.
        
        Args:
            stage_name: Name of the stage to configure
            **kwargs: Configuration parameters to override
        """
        if stage_name not in self.stage_configs:
            self.stage_configs[stage_name] = StageConfig(
                dtype=self.default_dtype,
                fp8_format=self.default_fp8_format if "fp8_format" not in kwargs else None
            )
            
        config = self.stage_configs[stage_name]
        
        # Update configuration with provided kwargs
        for key, value in kwargs.items():
            if key == "fp8_format" and isinstance(value, str):
                value = FP8Format.from_string(value)
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        
    def set_mode(self, mode: PrecisionMode) -> None:
        """
        Set the active precision mode.
        
        Args:
            mode: Precision mode to activate
        """
        self.current_mode = mode
        
    @contextlib.contextmanager
    def with_mode(self, mode: PrecisionMode):
        """
        Context manager for temporarily changing precision mode.
        
        Args:
            mode: Precision mode to use within the context
            
        Yields:
            None
        """
        previous_mode = self.current_mode
        self.current_mode = mode
        try:
            yield
        finally:
            self.current_mode = previous_mode
            
    def get_active_dtype(self, stage: str) -> torch.dtype:
        """
        Get the active data type for a specific stage.
        
        Args:
            stage: Stage name
            
        Returns:
            Active torch dtype for the stage
        """
        if stage not in self.stage_configs:
            return self.default_dtype
            
        config = self.stage_configs[stage]
        return config.get_current_dtype(self.current_mode)
    
    def enable_fp8(self, stages: Optional[List[str]] = None, fp8_format: Optional[FP8Format] = None) -> None:
        """
        Enable FP8 for specific stages.
        
        Args:
            stages: List of stages to enable FP8 for (if None, enable for all)
            fp8_format: FP8 format to use (if None, use default)
        """
        format_to_use = fp8_format or self.default_fp8_format
        stages_to_configure = stages or list(self.stage_configs.keys())
        
        for stage in stages_to_configure:
            self.configure_stage(stage, fp8_format=format_to_use)
    
    def disable_fp8(self, stages: Optional[List[str]] = None) -> None:
        """
        Disable FP8 for specific stages.
        
        Args:
            stages: List of stages to disable FP8 for (if None, disable for all)
        """
        stages_to_configure = stages or list(self.stage_configs.keys())
        
        for stage in stages_to_configure:
            self.configure_stage(stage, fp8_format=None)
            
    def supports_fp8(self) -> bool:
        """Check if the current environment supports FP8."""
        return hasattr(torch, "float8_e4m3fn")


class FP8Converter:
    """
    Utilities for FP8 conversion and scaling.
    
    This class provides methods for converting between different precisions,
    with focus on FP8 operations.
    """
    
    @staticmethod
    def per_tensor_quantize(
        x: torch.Tensor,
        dtype: torch.dtype,
        amax: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize a tensor to FP8 using per-tensor scaling.
        
        Args:
            x: Input tensor
            dtype: Target dtype (should be an FP8 type)
            amax: Optional pre-computed amax value
            scale: Optional pre-computed scale factor
            
        Returns:
            Tuple of (quantized tensor, amax, scale)
        """
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("FP8 is not supported in this PyTorch version")
            
        if amax is None:
            # Compute amax (max absolute value in the tensor)
            amax = torch.max(torch.abs(x.float())).detach()
            
        if scale is None:
            # Compute scaling factor
            if dtype == torch.float8_e4m3fn:
                scale = 448.0 / (amax + 1e-10)  # e4m3 has amax of 448.0
            elif dtype == torch.float8_e5m2fn:
                scale = 57344.0 / (amax + 1e-10)  # e5m2 has amax of 57344.0
            else:
                raise ValueError(f"Unsupported FP8 dtype: {dtype}")
                
        # Scale and quantize
        x_scaled = x.float() * scale
        x_quant = x_scaled.to(dtype)
        
        return x_quant, amax, scale
    
    @staticmethod
    def per_tensor_dequantize(
        x: torch.Tensor,
        scale: torch.Tensor,
        target_dtype: torch.dtype = torch.bfloat16
    ) -> torch.Tensor:
        """
        Dequantize an FP8 tensor to higher precision.
        
        Args:
            x: Quantized tensor
            scale: Scale factor used in quantization
            target_dtype: Target dtype for dequantized tensor
            
        Returns:
            Dequantized tensor
        """
        # Convert to float and apply inverse scaling
        return (x.float() / scale).to(target_dtype)
    
    @staticmethod
    def per_token_quantize(
        x: torch.Tensor,
        dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a tensor to FP8 using per-token scaling.
        
        Args:
            x: Input tensor with shape [num_tokens, hidden_size]
            dtype: Target dtype (should be an FP8 type)
            
        Returns:
            Tuple of (quantized tensor, scales)
        """
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("FP8 is not supported in this PyTorch version")
            
        # Get shape information
        num_tokens, hidden_size = x.shape
        
        # Ensure hidden dimension is divisible by 128 (for optimal performance)
        if hidden_size % 128 != 0:
            # Pad to nearest multiple of 128
            pad_size = 128 - (hidden_size % 128)
            padded_size = hidden_size + pad_size
            x_padded = torch.zeros((num_tokens, padded_size), dtype=x.dtype, device=x.device)
            x_padded[:, :hidden_size] = x
            x = x_padded
            hidden_size = padded_size
            
        # Reshape for per-token scaling (each token gets its own scaling factors for each 128 elements)
        x_view = x.view(num_tokens, -1, 128)
        
        # Compute amax for each token's 128-element chunks
        amax = x_view.abs().float().amax(dim=2).view(num_tokens, -1).clamp(1e-4)
        
        # Compute scale factors
        if dtype == torch.float8_e4m3fn:
            scale = 448.0 / amax
        elif dtype == torch.float8_e5m2fn:
            scale = 57344.0 / amax
        else:
            raise ValueError(f"Unsupported FP8 dtype: {dtype}")
            
        # Scale and quantize
        x_scaled = (x_view * scale.unsqueeze(-1))
        x_quant = x_scaled.to(dtype).view(num_tokens, hidden_size)
        
        # Return quantized tensor and scales (for dequantization)
        if hidden_size != x.shape[1]:
            # Remove padding in result
            x_quant = x_quant[:, :x.shape[1]]
            
        return x_quant, scale
    
    @staticmethod
    def per_token_dequantize(
        x: torch.Tensor,
        scale: torch.Tensor,
        target_dtype: torch.dtype = torch.bfloat16
    ) -> torch.Tensor:
        """
        Dequantize a per-token FP8 tensor to higher precision.
        
        Args:
            x: Quantized tensor with shape [num_tokens, hidden_size]
            scale: Scale factors with shape [num_tokens, hidden_size//128]
            target_dtype: Target dtype for dequantized tensor
            
        Returns:
            Dequantized tensor
        """
        # Get shape information
        num_tokens, hidden_size = x.shape
        
        # Check if padding was added
        original_hidden_size = hidden_size
        if hidden_size % 128 != 0:
            # Pad to nearest multiple of 128
            pad_size = 128 - (hidden_size % 128)
            padded_size = hidden_size + pad_size
            x_padded = torch.zeros((num_tokens, padded_size), dtype=x.dtype, device=x.device)
            x_padded[:, :hidden_size] = x
            x = x_padded
            hidden_size = padded_size
            
        # Reshape for per-token dequantization
        x_view = x.view(num_tokens, -1, 128)
        
        # Apply inverse scaling
        x_dequant = (x_view.float() / scale.unsqueeze(-1)).to(target_dtype)
        
        # Reshape back to original shape
        x_dequant = x_dequant.view(num_tokens, hidden_size)
        
        # Remove padding if needed
        if hidden_size != original_hidden_size:
            x_dequant = x_dequant[:, :original_hidden_size]
            
        return x_dequant


class HybridPrecisionDispatch:
    """
    Implementation of hybrid precision dispatch for DeepEP.
    
    This class provides methods to perform dispatch operations with 
    different precision for different parts of the computation.
    """
    
    def __init__(
        self,
        precision_manager: PrecisionManager,
        buffer_provider: Callable
    ):
        """
        Initialize hybrid precision dispatch.
        
        Args:
            precision_manager: Manager for precision control
            buffer_provider: Function that returns a Buffer instance
        """
        self.precision_manager = precision_manager
        self.get_buffer = buffer_provider
        
    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        use_fp8: Optional[bool] = None,
        **kwargs
    ) -> Tuple:
        """
        Perform dispatch with potentially different precision at different stages.
        
        Args:
            x: Input tensor or tuple of tensors
            topk_idx: Top-k expert indices
            topk_weights: Top-k expert weights
            num_experts: Total number of experts
            use_fp8: Override to enable/disable FP8 (if None, use precision manager settings)
            **kwargs: Additional arguments to pass to buffer.dispatch
            
        Returns:
            Dispatch results (see buffer.dispatch)
        """
        buffer = self.get_buffer()
        dispatch_dtype = self.precision_manager.get_active_dtype("dispatch")
        
        # Force FP8 setting if explicitly provided
        if use_fp8 is not None:
            with self.precision_manager.with_mode(
                PrecisionMode.LOW if use_fp8 else PrecisionMode.MIXED
            ):
                dispatch_dtype = self.precision_manager.get_active_dtype("dispatch")
        
        # Handle FP8 conversion if needed
        if dispatch_dtype in [torch.float8_e4m3fn, torch.float8_e5m2fn]:
            # Convert tensor to use as input for dispatch
            input_tensor = x[0] if isinstance(x, tuple) else x
            
            # For FP8 dispatch, we use per-token quantization
            x_fp8, scales = FP8Converter.per_token_quantize(input_tensor, dispatch_dtype)
            
            # Create input (use tuple format with scales for FP8)
            x_input = (x_fp8, scales)
        else:
            # No conversion needed
            x_input = x
        
        # Calculate layout
        layout_result = buffer.get_dispatch_layout(topk_idx, num_experts)
        
        # Perform dispatch
        dispatch_result = buffer.dispatch(
            x_input, 
            topk_idx=topk_idx, 
            topk_weights=topk_weights,
            num_tokens_per_rank=layout_result[0],
            num_tokens_per_rdma_rank=layout_result[1],
            is_token_in_rank=layout_result[3],
            num_tokens_per_expert=layout_result[2],
            **kwargs
        )
        
        return dispatch_result
    
    def combine(
        self,
        x: torch.Tensor,
        handle: Tuple,
        topk_weights: Optional[torch.Tensor] = None,
        use_fp8: Optional[bool] = None,
        **kwargs
    ) -> Tuple:
        """
        Perform combine with potentially different precision at different stages.
        
        Args:
            x: Input tensor from experts
            handle: Communication handle from dispatch
            topk_weights: Top-k expert weights
            use_fp8: Override to enable/disable FP8 (if None, use precision manager settings)
            **kwargs: Additional arguments to pass to buffer.combine
            
        Returns:
            Combine results (see buffer.combine)
        """
        buffer = self.get_buffer()
        combine_dtype = self.precision_manager.get_active_dtype("combine")
        
        # Force FP8 setting if explicitly provided
        if use_fp8 is not None:
            with self.precision_manager.with_mode(
                PrecisionMode.LOW if use_fp8 else PrecisionMode.MIXED
            ):
                combine_dtype = self.precision_manager.get_active_dtype("combine")
        
        # Handle FP8 conversion if needed
        if combine_dtype in [torch.float8_e4m3fn, torch.float8_e5m2fn]:
            # For FP8 combine, we use per-token quantization
            x_fp8, scales = FP8Converter.per_token_quantize(x, combine_dtype)
            x_input = x_fp8
            
            # Store scales as additional data to be used during dequantization
            kwargs["fp8_scales"] = scales
        else:
            # No conversion needed
            x_input = x
        
        # Perform combine
        combine_result = buffer.combine(
            x_input,
            handle,
            topk_weights=topk_weights,
            **kwargs
        )
        
        # Handle FP8 dequantization if needed
        if combine_dtype in [torch.float8_e4m3fn, torch.float8_e5m2fn] and "fp8_scales" in kwargs:
            # Get the target dtype for dequantization
            target_dtype = self.precision_manager.stage_configs["combine"].dtype
            
            # Dequantize the result
            combined_x = combine_result[0]
            fp8_scales = kwargs["fp8_scales"]
            
            # Only dequantize if the result is still in FP8
            if combined_x.dtype in [torch.float8_e4m3fn, torch.float8_e5m2fn]:
                dequantized_x = FP8Converter.per_token_dequantize(
                    combined_x, fp8_scales, target_dtype=target_dtype
                )
                
                # Replace the original result with the dequantized one
                combine_result = (dequantized_x,) + combine_result[1:]
        
        return combine_result


# Create a default global precision manager
default_precision_manager = PrecisionManager() 