import os
import json
import time
import torch
import torch.distributed as dist
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from pathlib import Path
import argparse
import tabulate

from .buffer import Buffer
from .utils import EventOverlap
from .autotuning import AutoTuner


class Benchmark:
    """
    Benchmarking tools for DeepEP library performance testing.
    
    This class provides methods to test different configurations of the DeepEP
    library and report performance metrics.
    """
    
    def __init__(self, group: dist.ProcessGroup):
        """
        Initialize the benchmarking tool.
        
        Args:
            group: The communication group for testing
        """
        self.rank = group.rank()
        self.group_size = group.size()
        self.group = group
        self.auto_tuner = AutoTuner(group)
    
    def _create_test_data(
        self,
        num_tokens: int,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        dtype: torch.dtype = torch.bfloat16
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create test data for benchmarking.
        
        Args:
            num_tokens: Number of tokens
            hidden_size: Hidden dimension size
            num_experts: Number of experts
            top_k: Number of top experts per token
            dtype: Data type
            
        Returns:
            Tuple of (input_tensor, topk_idx, topk_weights)
        """
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device='cuda')
        topk_idx = torch.randint(0, num_experts, (num_tokens, top_k), device='cuda')
        topk_weights = torch.ones((num_tokens, top_k), dtype=dtype, device='cuda') / top_k
        return x, topk_idx, topk_weights
    
    def benchmark_dispatch_combine(
        self,
        hidden_size: int,
        num_tokens: int,
        num_experts: int,
        sm_count: Optional[int] = None,
        num_trials: int = 20,
        warmup_trials: int = 5,
        dtype: torch.dtype = torch.bfloat16,
        print_results: bool = True
    ) -> Dict[str, float]:
        """
        Benchmark the dispatch and combine operations.
        
        Args:
            hidden_size: Hidden dimension size
            num_tokens: Number of tokens per batch
            num_experts: Number of experts
            sm_count: Number of SMs to use (if None, use default or auto-tuned)
            num_trials: Number of trials to run
            warmup_trials: Number of warmup trials
            dtype: Data type
            print_results: Whether to print results to console
            
        Returns:
            Dictionary of benchmark results
        """
        # Set SM count if provided, otherwise use auto-tuned value
        if sm_count is not None:
            Buffer.set_num_sms(sm_count)
        else:
            # Use auto-tuned value
            optimal_sm_count = self.auto_tuner.tune_sm_count(
                hidden_size=hidden_size,
                num_tokens=num_tokens,
                num_experts=num_experts,
                dtype=dtype
            )
            Buffer.set_num_sms(optimal_sm_count)
            sm_count = optimal_sm_count
        
        # Create buffer
        element_size = torch.finfo(dtype).bits // 8
        buffer_size = hidden_size * num_tokens * element_size * 2  # x2 for safety
        
        buffer = Buffer(
            self.group,
            num_nvl_bytes=buffer_size,
            num_rdma_bytes=buffer_size
        )
        
        # Create test data
        x, topk_idx, topk_weights = self._create_test_data(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            num_experts=num_experts,
            dtype=dtype
        )
        
        # Warmup
        for _ in range(warmup_trials):
            # Dispatch
            layout_result = buffer.get_dispatch_layout(topk_idx, num_experts)
            dispatch_result = buffer.dispatch(
                x, 
                topk_idx=topk_idx, 
                topk_weights=topk_weights,
                num_tokens_per_rank=layout_result[0],
                num_tokens_per_rdma_rank=layout_result[1],
                is_token_in_rank=layout_result[3],
                num_tokens_per_expert=layout_result[2]
            )
            
            # Combine
            combined_x, _, _ = buffer.combine(
                dispatch_result[0],
                dispatch_result[4],  # handle
                topk_weights=dispatch_result[2]
            )
            
            torch.cuda.synchronize()
        
        # Benchmark dispatch
        dispatch_times = []
        for _ in range(num_trials):
            # Dispatch benchmark
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            layout_result = buffer.get_dispatch_layout(topk_idx, num_experts)
            dispatch_result = buffer.dispatch(
                x, 
                topk_idx=topk_idx, 
                topk_weights=topk_weights,
                num_tokens_per_rank=layout_result[0],
                num_tokens_per_rdma_rank=layout_result[1],
                is_token_in_rank=layout_result[3],
                num_tokens_per_expert=layout_result[2]
            )
            end_event.record()
            
            torch.cuda.synchronize()
            dispatch_times.append(start_event.elapsed_time(end_event) / 1000.0)  # Convert to seconds
        
        # Benchmark combine
        combine_times = []
        for _ in range(num_trials):
            # Use the last dispatch result
            layout_result = buffer.get_dispatch_layout(topk_idx, num_experts)
            dispatch_result = buffer.dispatch(
                x, 
                topk_idx=topk_idx, 
                topk_weights=topk_weights,
                num_tokens_per_rank=layout_result[0],
                num_tokens_per_rdma_rank=layout_result[1],
                is_token_in_rank=layout_result[3],
                num_tokens_per_expert=layout_result[2]
            )
            
            # Combine benchmark
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            combined_x, _, _ = buffer.combine(
                dispatch_result[0],
                dispatch_result[4],  # handle
                topk_weights=dispatch_result[2]
            )
            end_event.record()
            
            torch.cuda.synchronize()
            combine_times.append(start_event.elapsed_time(end_event) / 1000.0)  # Convert to seconds
        
        # Calculate statistics
        dispatch_avg = np.mean(dispatch_times)
        dispatch_std = np.std(dispatch_times)
        dispatch_min = np.min(dispatch_times)
        dispatch_max = np.max(dispatch_times)
        
        combine_avg = np.mean(combine_times)
        combine_std = np.std(combine_times)
        combine_min = np.min(combine_times)
        combine_max = np.max(combine_times)
        
        # Calculate bandwidth (bytes/second)
        bytes_moved = num_tokens * hidden_size * element_size
        dispatch_bandwidth = bytes_moved / dispatch_avg  # bytes/second
        combine_bandwidth = bytes_moved / combine_avg    # bytes/second
        
        # Convert to GB/s
        dispatch_bandwidth_gb = dispatch_bandwidth / 1e9
        combine_bandwidth_gb = combine_bandwidth / 1e9
        
        results = {
            "sm_count": sm_count,
            "hidden_size": hidden_size,
            "num_tokens": num_tokens,
            "num_experts": num_experts,
            "dtype": str(dtype),
            "dispatch_avg_time": dispatch_avg,
            "dispatch_std_time": dispatch_std,
            "dispatch_min_time": dispatch_min,
            "dispatch_max_time": dispatch_max,
            "dispatch_bandwidth_gb": dispatch_bandwidth_gb,
            "combine_avg_time": combine_avg,
            "combine_std_time": combine_std,
            "combine_min_time": combine_min,
            "combine_max_time": combine_max,
            "combine_bandwidth_gb": combine_bandwidth_gb
        }
        
        # Print results if requested
        if print_results and self.rank == 0:
            print("\n" + "=" * 50)
            print(f"Benchmark Results (SM Count: {sm_count})")
            print("=" * 50)
            
            print(f"\nConfiguration:")
            print(f"  Hidden Size: {hidden_size}")
            print(f"  Tokens: {num_tokens}")
            print(f"  Experts: {num_experts}")
            print(f"  Data Type: {dtype}")
            
            print(f"\nDispatch:")
            print(f"  Average Time: {dispatch_avg*1000:.2f} ms")
            print(f"  Std Dev: {dispatch_std*1000:.2f} ms")
            print(f"  Min Time: {dispatch_min*1000:.2f} ms")
            print(f"  Max Time: {dispatch_max*1000:.2f} ms")
            print(f"  Bandwidth: {dispatch_bandwidth_gb:.2f} GB/s")
            
            print(f"\nCombine:")
            print(f"  Average Time: {combine_avg*1000:.2f} ms")
            print(f"  Std Dev: {combine_std*1000:.2f} ms")
            print(f"  Min Time: {combine_min*1000:.2f} ms")
            print(f"  Max Time: {combine_max*1000:.2f} ms")
            print(f"  Bandwidth: {combine_bandwidth_gb:.2f} GB/s")
            
            print("\n" + "=" * 50)
        
        return results
    
    def compare_sm_counts(
        self,
        hidden_size: int,
        num_tokens: int,
        num_experts: int,
        sm_counts: List[int],
        num_trials: int = 10,
        warmup_trials: int = 5,
        dtype: torch.dtype = torch.bfloat16
    ) -> None:
        """
        Compare the performance of different SM count configurations.
        
        Args:
            hidden_size: Hidden dimension size
            num_tokens: Number of tokens per batch
            num_experts: Number of experts
            sm_counts: List of SM counts to compare
            num_trials: Number of trials to run
            warmup_trials: Number of warmup trials
            dtype: Data type
        """
        if self.rank == 0:
            print("\n" + "=" * 60)
            print(f"SM Count Comparison")
            print("=" * 60)
            print(f"\nConfiguration:")
            print(f"  Hidden Size: {hidden_size}")
            print(f"  Tokens: {num_tokens}")
            print(f"  Experts: {num_experts}")
            print(f"  Data Type: {dtype}")
            print("=" * 60)
        
        results = []
        for sm_count in sm_counts:
            result = self.benchmark_dispatch_combine(
                hidden_size=hidden_size,
                num_tokens=num_tokens,
                num_experts=num_experts,
                sm_count=sm_count,
                num_trials=num_trials,
                warmup_trials=warmup_trials,
                dtype=dtype,
                print_results=False
            )
            results.append(result)
        
        # Print comparison table
        if self.rank == 0:
            headers = ["SM Count", "Dispatch Time (ms)", "Dispatch BW (GB/s)", "Combine Time (ms)", "Combine BW (GB/s)"]
            table_data = []
            
            for result in results:
                row = [
                    result["sm_count"],
                    f"{result['dispatch_avg_time']*1000:.2f}",
                    f"{result['dispatch_bandwidth_gb']:.2f}",
                    f"{result['combine_avg_time']*1000:.2f}",
                    f"{result['combine_bandwidth_gb']:.2f}"
                ]
                table_data.append(row)
            
            print(tabulate.tabulate(table_data, headers=headers, tablefmt="grid"))
            
            # Find and highlight the best configuration
            best_dispatch_idx = np.argmax([r["dispatch_bandwidth_gb"] for r in results])
            best_combine_idx = np.argmax([r["combine_bandwidth_gb"] for r in results])
            
            print(f"\nBest configuration for Dispatch: SM Count = {results[best_dispatch_idx]['sm_count']}")
            print(f"Best configuration for Combine: SM Count = {results[best_combine_idx]['sm_count']}")
            
            # Recommend a good overall configuration
            if best_dispatch_idx == best_combine_idx:
                print(f"\nRecommended SM Count: {results[best_dispatch_idx]['sm_count']}")
            else:
                # Average the rankings
                sm_counts = [r["sm_count"] for r in results]
                dispatch_ranks = np.argsort([r["dispatch_bandwidth_gb"] for r in results])[::-1]
                combine_ranks = np.argsort([r["combine_bandwidth_gb"] for r in results])[::-1]
                
                avg_ranks = {}
                for i, sm in enumerate(sm_counts):
                    d_rank = np.where(dispatch_ranks == i)[0][0]
                    c_rank = np.where(combine_ranks == i)[0][0]
                    avg_ranks[sm] = (d_rank + c_rank) / 2
                
                best_sm = min(avg_ranks, key=avg_ranks.get)
                print(f"\nRecommended SM Count (balanced): {best_sm}")
    
    def benchmark_low_latency(
        self,
        hidden_size: int,
        num_tokens: int,
        num_experts: int,
        num_trials: int = 20,
        warmup_trials: int = 5,
        dtype: torch.dtype = torch.bfloat16,
        print_results: bool = True
    ) -> Dict[str, float]:
        """
        Benchmark the low latency operations.
        
        Args:
            hidden_size: Hidden dimension size
            num_tokens: Number of tokens per batch
            num_experts: Number of experts
            num_trials: Number of trials to run
            warmup_trials: Number of warmup trials
            dtype: Data type
            print_results: Whether to print results to console
            
        Returns:
            Dictionary of benchmark results
        """
        # Create buffer in low latency mode
        rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
            num_tokens, hidden_size, self.group_size, num_experts
        )
        
        # For low latency mode, number of QPs per rank should equal number of local experts
        num_qps_per_rank = num_experts // self.group_size
        if num_qps_per_rank < 1:
            num_qps_per_rank = 1
        
        buffer = Buffer(
            self.group,
            num_nvl_bytes=0,
            num_rdma_bytes=rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_qps_per_rank
        )
        
        # Create test data
        x, topk_idx, topk_weights = self._create_test_data(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            num_experts=num_experts,
            dtype=dtype
        )
        
        # Warmup
        for _ in range(warmup_trials):
            dispatch_result = buffer.low_latency_dispatch(
                x,
                topk_idx,
                num_tokens,
                num_experts
            )
            
            combine_result = buffer.low_latency_combine(
                dispatch_result[0][0],  # x
                dispatch_result[1],     # topk_idx
                topk_weights,
                dispatch_result[2]      # handle
            )
            
            torch.cuda.synchronize()
        
        # Benchmark dispatch
        dispatch_times = []
        for _ in range(num_trials):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            dispatch_result = buffer.low_latency_dispatch(
                x,
                topk_idx,
                num_tokens,
                num_experts
            )
            end_event.record()
            
            torch.cuda.synchronize()
            dispatch_times.append(start_event.elapsed_time(end_event) / 1000.0)  # Convert to seconds
        
        # Benchmark combine
        combine_times = []
        for _ in range(num_trials):
            # Create new dispatch result for each trial
            dispatch_result = buffer.low_latency_dispatch(
                x,
                topk_idx,
                num_tokens,
                num_experts
            )
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            combine_result = buffer.low_latency_combine(
                dispatch_result[0][0],  # x
                dispatch_result[1],     # topk_idx
                topk_weights,
                dispatch_result[2]      # handle
            )
            end_event.record()
            
            torch.cuda.synchronize()
            combine_times.append(start_event.elapsed_time(end_event) / 1000.0)  # Convert to seconds
        
        # Calculate statistics
        dispatch_avg = np.mean(dispatch_times)
        dispatch_std = np.std(dispatch_times)
        dispatch_min = np.min(dispatch_times)
        dispatch_max = np.max(dispatch_times)
        
        combine_avg = np.mean(combine_times)
        combine_std = np.std(combine_times)
        combine_min = np.min(combine_times)
        combine_max = np.max(combine_times)
        
        # Calculate bandwidth (bytes/second)
        bytes_moved = num_tokens * hidden_size * (torch.finfo(dtype).bits // 8)
        dispatch_bandwidth = bytes_moved / dispatch_avg  # bytes/second
        combine_bandwidth = bytes_moved / combine_avg    # bytes/second
        
        # Convert to GB/s
        dispatch_bandwidth_gb = dispatch_bandwidth / 1e9
        combine_bandwidth_gb = combine_bandwidth / 1e9
        
        results = {
            "hidden_size": hidden_size,
            "num_tokens": num_tokens,
            "num_experts": num_experts,
            "dtype": str(dtype),
            "dispatch_avg_time": dispatch_avg,
            "dispatch_std_time": dispatch_std,
            "dispatch_min_time": dispatch_min,
            "dispatch_max_time": dispatch_max,
            "dispatch_bandwidth_gb": dispatch_bandwidth_gb,
            "combine_avg_time": combine_avg,
            "combine_std_time": combine_std,
            "combine_min_time": combine_min,
            "combine_max_time": combine_max,
            "combine_bandwidth_gb": combine_bandwidth_gb
        }
        
        # Print results if requested
        if print_results and self.rank == 0:
            print("\n" + "=" * 50)
            print(f"Low Latency Benchmark Results")
            print("=" * 50)
            
            print(f"\nConfiguration:")
            print(f"  Hidden Size: {hidden_size}")
            print(f"  Tokens: {num_tokens}")
            print(f"  Experts: {num_experts}")
            print(f"  Data Type: {dtype}")
            
            print(f"\nLow Latency Dispatch:")
            print(f"  Average Time: {dispatch_avg*1000:.2f} ms")
            print(f"  Std Dev: {dispatch_std*1000:.2f} ms")
            print(f"  Min Time: {dispatch_min*1000:.2f} ms")
            print(f"  Max Time: {dispatch_max*1000:.2f} ms")
            print(f"  Bandwidth: {dispatch_bandwidth_gb:.2f} GB/s")
            print(f"  Latency: {dispatch_avg*1000*1000:.2f} us")
            
            print(f"\nLow Latency Combine:")
            print(f"  Average Time: {combine_avg*1000:.2f} ms")
            print(f"  Std Dev: {combine_std*1000:.2f} ms")
            print(f"  Min Time: {combine_min*1000:.2f} ms")
            print(f"  Max Time: {combine_max*1000:.2f} ms")
            print(f"  Bandwidth: {combine_bandwidth_gb:.2f} GB/s")
            print(f"  Latency: {combine_avg*1000*1000:.2f} us")
            
            print("\n" + "=" * 50)
        
        return results


def main():
    """Command line interface for running benchmarks."""
    parser = argparse.ArgumentParser(description="DeepEP Benchmarking Tool")
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=["normal", "low_latency", "compare_sm"], 
                        default="normal", help="Benchmark mode")
    
    # Common parameters
    parser.add_argument("--hidden_size", type=int, default=7168, help="Hidden dimension size")
    parser.add_argument("--num_tokens", type=int, default=4096, help="Number of tokens per batch")
    parser.add_argument("--num_experts", type=int, default=32, help="Number of experts")
    parser.add_argument("--num_trials", type=int, default=20, help="Number of trials to run")
    parser.add_argument("--warmup_trials", type=int, default=5, help="Number of warmup trials")
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float32", "float16"], 
                        default="bfloat16", help="Data type")
    
    # SM count parameters (for normal and compare_sm modes)
    parser.add_argument("--sm_count", type=int, default=None, 
                        help="Number of SMs to use (if None, use auto-tuned)")
    parser.add_argument("--sm_counts", type=str, default="8,16,24,32,40,48", 
                        help="Comma-separated list of SM counts to compare")
    
    # Distributed parameters
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank")
    parser.add_argument("--num_local_ranks", type=int, default=1, help="Number of local ranks")
    
    args = parser.parse_args()
    
    # Initialize distributed
    from .utils import init_dist
    rank, world_size, group = init_dist(args.local_rank, args.num_local_ranks)
    
    # Map dtype string to torch type
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float16": torch.float16
    }
    dtype = dtype_map[args.dtype]
    
    # Create benchmark instance
    benchmark = Benchmark(group)
    
    # Run benchmark according to mode
    if args.mode == "normal":
        benchmark.benchmark_dispatch_combine(
            hidden_size=args.hidden_size,
            num_tokens=args.num_tokens,
            num_experts=args.num_experts,
            sm_count=args.sm_count,
            num_trials=args.num_trials,
            warmup_trials=args.warmup_trials,
            dtype=dtype
        )
    elif args.mode == "low_latency":
        benchmark.benchmark_low_latency(
            hidden_size=args.hidden_size,
            num_tokens=args.num_tokens,
            num_experts=args.num_experts,
            num_trials=args.num_trials,
            warmup_trials=args.warmup_trials,
            dtype=dtype
        )
    elif args.mode == "compare_sm":
        sm_counts = [int(x) for x in args.sm_counts.split(",")]
        benchmark.compare_sm_counts(
            hidden_size=args.hidden_size,
            num_tokens=args.num_tokens,
            num_experts=args.num_experts,
            sm_counts=sm_counts,
            num_trials=args.num_trials,
            warmup_trials=args.warmup_trials,
            dtype=dtype
        )


if __name__ == "__main__":
    main() 