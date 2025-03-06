import os
import argparse
import torch
import torch.distributed as dist
import time
import numpy as np
from typing import Tuple, Optional

from deep_ep import (
    Buffer, 
    PrecisionMode, 
    FP8Format, 
    PrecisionManager, 
    HybridPrecisionDispatch, 
    FP8Converter
)


def init_dist(local_rank, num_local_ranks):
    """Initialize PyTorch distributed environment"""
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))

    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{ip}:{port}',
        world_size=num_nodes * num_local_ranks,
        rank=node_rank * num_local_ranks + local_rank
    )
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    torch.cuda.set_device(local_rank)

    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))


class PrecisionControlledMoE:
    """
    MoE model with precision control for different stages of computation.
    
    This class demonstrates how to use the precision control features
    for controlling FP8 and other precision modes throughout MoE computation.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int,
        group: dist.ProcessGroup,
        top_k: int = 2,
        precision_config: Optional[dict] = None
    ):
        """
        Initialize the precision-controlled MoE.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            num_experts: Number of experts
            group: Distributed group
            top_k: Number of experts per token
            precision_config: Configuration for precision settings
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.group = group
        self.top_k = top_k
        
        # Create experts (simple linear layers for this example)
        self.experts = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, output_dim) for _ in range(num_experts)
        ])
        
        # Create router
        self.router = torch.nn.Linear(input_dim, num_experts)
        
        # Create precision manager (or use default config)
        if precision_config is None:
            precision_config = {
                "default_mode": PrecisionMode.MIXED,
                "default_dtype": torch.bfloat16,
                "default_fp8_format": FP8Format.E4M3
            }
            
        self.precision_manager = PrecisionManager(**precision_config)
        
        # Create buffer for communication
        self._buffer = None
        
        # Create hybrid precision dispatch
        self.hybrid_dispatch = HybridPrecisionDispatch(
            precision_manager=self.precision_manager,
            buffer_provider=self.get_buffer
        )
        
        # FP8 detection
        self.supports_fp8 = self.precision_manager.supports_fp8()
        
    def get_buffer(self) -> Buffer:
        """Get or create the communication buffer."""
        if self._buffer is None:
            # Calculate buffer size based on input dimension
            element_size = torch.finfo(torch.bfloat16).bits // 8
            buffer_size = self.input_dim * 1024 * element_size * 2  # Reasonable default size
            
            self._buffer = Buffer(
                self.group,
                num_nvl_bytes=buffer_size,
                num_rdma_bytes=buffer_size
            )
            
        return self._buffer
    
    def configure_precision(self, config_dict: dict) -> None:
        """
        Configure precision settings for different stages.
        
        Args:
            config_dict: Dictionary mapping stage names to configuration dicts
        """
        for stage, stage_config in config_dict.items():
            self.precision_manager.configure_stage(stage, **stage_config)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with configurable precision.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Tuple of (output tensor, precision stats)
        """
        batch_size = x.shape[0]
        precision_stats = {}
        
        # 1. Router (typically in mixed precision)
        router_start = time.time()
        with self.precision_manager.with_mode(PrecisionMode.MIXED):
            router_logits = self.router(x)
            router_probs = torch.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        router_weights, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)  # Re-normalize
        
        precision_stats["router_dtype"] = self.precision_manager.get_active_dtype("router")
        precision_stats["router_time"] = time.time() - router_start
        
        # 2. Dispatch (can use FP8 for communication efficiency)
        dispatch_start = time.time()
        
        # Use the hybrid precision dispatch which handles FP8 conversion
        dispatch_result = self.hybrid_dispatch.dispatch(
            x=x,
            topk_idx=router_indices,
            topk_weights=router_weights,
            num_experts=self.num_experts
        )
        
        precision_stats["dispatch_dtype"] = self.precision_manager.get_active_dtype("dispatch")
        precision_stats["dispatch_time"] = time.time() - dispatch_start
        
        # 3. Expert computation (typically in mixed precision)
        expert_start = time.time()
        
        # Use appropriate precision for expert computation
        with self.precision_manager.with_mode(PrecisionMode.MIXED):
            expert_outputs = torch.zeros_like(dispatch_result[0])
            
            # Get the number of tokens received for each expert
            num_tokens_per_expert = dispatch_result[3]
            
            # Process each expert's tokens
            start_idx = 0
            for expert_idx in range(self.num_experts):
                if num_tokens_per_expert[expert_idx] > 0:
                    # Get this expert's tokens
                    end_idx = start_idx + num_tokens_per_expert[expert_idx]
                    
                    # Process with the expert
                    expert_input = dispatch_result[0][start_idx:end_idx]
                    expert_output = self.experts[expert_idx](expert_input)
                    
                    # Store the expert output
                    expert_outputs[start_idx:end_idx] = expert_output
                    
                    # Update start index for next expert
                    start_idx = end_idx
        
        precision_stats["expert_dtype"] = self.precision_manager.get_active_dtype("expert_compute")
        precision_stats["expert_time"] = time.time() - expert_start
        
        # 4. Combine (can use FP8 for communication efficiency)
        combine_start = time.time()
        
        # Use the hybrid precision combine which handles FP8 conversion
        combined_output, _, _ = self.hybrid_dispatch.combine(
            x=expert_outputs,
            handle=dispatch_result[4],  # handle
            topk_weights=dispatch_result[2]
        )
        
        precision_stats["combine_dtype"] = self.precision_manager.get_active_dtype("combine")
        precision_stats["combine_time"] = time.time() - combine_start
        
        # Final timing stats
        precision_stats["total_time"] = (
            precision_stats["router_time"] + 
            precision_stats["dispatch_time"] + 
            precision_stats["expert_time"] + 
            precision_stats["combine_time"]
        )
        
        return combined_output, precision_stats
    
    
def compare_precision_modes(model, inputs, batch_labels=None):
    """
    Compare different precision modes.
    
    Args:
        model: PrecisionControlledMoE model
        inputs: Input tensor
        batch_labels: Optional batch labels for accuracy evaluation
    """
    rank = dist.get_rank()
    if not model.supports_fp8:
        if rank == 0:
            print("FP8 is not supported in this PyTorch version. Skipping FP8 tests.")
        return
    
    # Define precision configurations to test
    precision_configs = {
        "Mixed Precision (BF16)": {
            "mode": PrecisionMode.MIXED,
            "config": {}
        },
        "FP8 Dispatch Only": {
            "mode": PrecisionMode.HYBRID,
            "config": {
                "dispatch": {"fp8_format": FP8Format.E4M3},
                "combine": {"fp8_format": None}
            }
        },
        "FP8 Combine Only": {
            "mode": PrecisionMode.HYBRID,
            "config": {
                "dispatch": {"fp8_format": None},
                "combine": {"fp8_format": FP8Format.E4M3}
            }
        },
        "Full FP8 Communication": {
            "mode": PrecisionMode.HYBRID,
            "config": {
                "dispatch": {"fp8_format": FP8Format.E4M3},
                "combine": {"fp8_format": FP8Format.E4M3}
            }
        },
        "E5M2 FP8 Format": {
            "mode": PrecisionMode.HYBRID,
            "config": {
                "dispatch": {"fp8_format": FP8Format.E5M2},
                "combine": {"fp8_format": FP8Format.E5M2}
            }
        }
    }
    
    results = {}
    reference_output = None
    
    # Run tests for each configuration
    for name, config in precision_configs.items():
        if rank == 0:
            print(f"\nTesting: {name}")
        
        # Apply configuration
        model.precision_manager.set_mode(config["mode"])
        if config["config"]:
            model.configure_precision(config["config"])
        
        # Warm-up
        for _ in range(3):
            model(inputs)
            torch.cuda.synchronize()
        
        # Benchmark
        timings = []
        for _ in range(10):
            torch.cuda.synchronize()
            output, stats = model(inputs)
            torch.cuda.synchronize()
            timings.append(stats["total_time"])
            
            # Store reference output from first config
            if reference_output is None:
                reference_output = output.detach()
        
        # Compute numerical error compared to reference
        rel_error = 0
        if reference_output is not None:
            with torch.no_grad():
                abs_diff = (output - reference_output).abs().mean().item()
                abs_ref = reference_output.abs().mean().item()
                rel_error = abs_diff / (abs_ref + 1e-9)
        
        # Collect results
        results[name] = {
            "avg_time": np.mean(timings) * 1000,  # ms
            "min_time": np.min(timings) * 1000,  # ms
            "rel_error": rel_error,
            "dtype_config": {
                "router": str(stats["router_dtype"]),
                "dispatch": str(stats["dispatch_dtype"]),
                "expert": str(stats["expert_dtype"]),
                "combine": str(stats["combine_dtype"])
            },
            "breakdown": {
                "router": stats["router_time"] * 1000,
                "dispatch": stats["dispatch_time"] * 1000,
                "expert": stats["expert_time"] * 1000,
                "combine": stats["combine_time"] * 1000
            }
        }
    
    # Report results
    if rank == 0:
        print("\n" + "=" * 100)
        print("PRECISION MODE COMPARISON RESULTS")
        print("=" * 100)
        
        headers = ["Configuration", "Avg Time (ms)", "Speedup", "Rel. Error", "Router", "Dispatch", "Expert", "Combine"]
        rows = []
        
        baseline_time = results["Mixed Precision (BF16)"]["avg_time"]
        
        for name, result in results.items():
            speedup = baseline_time / result["avg_time"] if result["avg_time"] > 0 else 0
            rows.append([
                name,
                f"{result['avg_time']:.2f}",
                f"{speedup:.2f}x",
                f"{result['rel_error']:.2e}",
                f"{result['breakdown']['router']:.2f}",
                f"{result['breakdown']['dispatch']:.2f}",
                f"{result['breakdown']['expert']:.2f}",
                f"{result['breakdown']['combine']:.2f}"
            ])
        
        # Print as a formatted table
        col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        
        # Print header
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        print(header_line)
        print("-" * len(header_line))
        
        # Print rows
        for row in rows:
            print(" | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)))
        
        print("\n" + "=" * 100)
        print("DATA TYPE CONFIGURATION DETAILS")
        print("=" * 100)
        
        for name, result in results.items():
            print(f"\n{name}:")
            for stage, dtype in result["dtype_config"].items():
                print(f"  {stage.capitalize()}: {dtype}")


def main():
    parser = argparse.ArgumentParser(description="DeepEP Enhanced FP8 and Precision Control Example")
    
    parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden dimension size")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--top_k", type=int, default=2, help="Number of experts per token")
    parser.add_argument("--fp8_format", type=str, choices=["e4m3", "e5m2"], default="e4m3", 
                       help="FP8 format to use")
    parser.add_argument("--precision_mode", type=str, 
                       choices=["full", "mixed", "low", "dynamic", "hybrid"], 
                       default="hybrid", help="Precision mode")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank")
    parser.add_argument("--num_local_ranks", type=int, default=1, help="Number of local ranks")
    
    args = parser.parse_args()
    
    # Initialize distributed
    rank, world_size, group = init_dist(args.local_rank, args.num_local_ranks)
    
    # Map string arguments to enums
    mode_map = {
        "full": PrecisionMode.FULL,
        "mixed": PrecisionMode.MIXED,
        "low": PrecisionMode.LOW,
        "dynamic": PrecisionMode.DYNAMIC,
        "hybrid": PrecisionMode.HYBRID
    }
    
    format_map = {
        "e4m3": FP8Format.E4M3,
        "e5m2": FP8Format.E5M2
    }
    
    # Create precision configuration
    precision_config = {
        "default_mode": mode_map[args.precision_mode],
        "default_dtype": torch.bfloat16,
        "default_fp8_format": format_map[args.fp8_format]
    }
    
    # Create model with precision control
    model = PrecisionControlledMoE(
        input_dim=args.hidden_size,
        output_dim=args.hidden_size,
        num_experts=args.num_experts,
        group=group,
        top_k=args.top_k,
        precision_config=precision_config
    )
    
    if rank == 0:
        print("\n--- DeepEP Enhanced FP8 Support and Precision Control Demo ---")
        print(f"Number of experts: {args.num_experts}")
        print(f"Hidden size: {args.hidden_size}")
        print(f"Batch size: {args.batch_size}")
        print(f"Top-k experts: {args.top_k}")
        print(f"Default precision mode: {args.precision_mode}")
        print(f"Default FP8 format: {args.fp8_format}")
        print(f"FP8 support detected: {model.supports_fp8}")
        print("-----------------------------------------------------------\n")
    
    # Create test input
    inputs = torch.randn(args.batch_size, args.hidden_size, device='cuda', dtype=torch.bfloat16)
    
    # Compare different precision modes
    compare_precision_modes(model, inputs)
    
    if rank == 0:
        print("\n--- FP8 Precision Control Examples ---")
        
        # Example: Per-tensor quantization
        if model.supports_fp8:
            print("\nPer-tensor quantization example:")
            # Create a test tensor
            test_tensor = torch.randn(10, 1024, device='cuda', dtype=torch.bfloat16)
            # Quantize to FP8
            fp8_tensor, amax, scale = FP8Converter.per_tensor_quantize(
                test_tensor, torch.float8_e4m3fn
            )
            # Dequantize back
            dequantized = FP8Converter.per_tensor_dequantize(
                fp8_tensor, scale, target_dtype=torch.bfloat16
            )
            # Calculate error
            rel_error = (test_tensor - dequantized).abs().mean() / test_tensor.abs().mean()
            
            print(f"Original shape: {test_tensor.shape}, dtype: {test_tensor.dtype}")
            print(f"FP8 shape: {fp8_tensor.shape}, dtype: {fp8_tensor.dtype}")
            print(f"Dequantized shape: {dequantized.shape}, dtype: {dequantized.dtype}")
            print(f"Relative error: {rel_error:.2e}")
            
            # Print numerical statistics
            print(f"Original tensor stats: min={test_tensor.min().item():.4f}, max={test_tensor.max().item():.4f}")
            print(f"Dequantized tensor stats: min={dequantized.min().item():.4f}, max={dequantized.max().item():.4f}")
        else:
            print("FP8 is not supported in this PyTorch version.")
    
    # Examples of different precision configurations
    if rank == 0:
        print("\nExample precision configurations:")
        
        print("\n1. High throughput configuration (low precision communication):")
        print("   - Router: Mixed precision (BF16)")
        print("   - Dispatch: Low precision (FP8 E4M3)")
        print("   - Expert computation: Mixed precision (BF16)")
        print("   - Combine: Low precision (FP8 E4M3)")
        
        print("\n2. High accuracy configuration:")
        print("   - Router: Mixed precision (BF16)")
        print("   - Dispatch: Mixed precision (BF16)")
        print("   - Expert computation: Full precision (FP32)")
        print("   - Combine: Mixed precision (BF16)")
        
        print("\n3. Balanced configuration:")
        print("   - Router: Mixed precision (BF16)")
        print("   - Dispatch: Mixed precision (BF16)")
        print("   - Expert computation: Mixed precision (BF16)")
        print("   - Combine: Mixed precision (BF16)")
        
        print("\n4. Memory-optimized configuration:")
        print("   - Router: Low precision (FP8 E5M2)")
        print("   - Dispatch: Low precision (FP8 E4M3)")
        print("   - Expert computation: Mixed precision (BF16)")
        print("   - Combine: Low precision (FP8 E4M3)")


if __name__ == "__main__":
    main() 