import os
import argparse
import torch
import torch.distributed as dist

from deep_ep import Benchmark


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


def main():
    parser = argparse.ArgumentParser(description="DeepEP Benchmarking Example")
    
    # Common parameters
    parser.add_argument("--hidden_size", type=int, default=7168, help="Hidden dimension size")
    parser.add_argument("--num_tokens", type=int, default=4096, help="Number of tokens per batch")
    parser.add_argument("--num_experts", type=int, default=32, help="Number of experts")
    parser.add_argument("--num_trials", type=int, default=20, help="Number of trials to run")
    parser.add_argument("--warmup_trials", type=int, default=5, help="Number of warmup trials")
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=["normal", "low_latency", "compare_sm"], 
                      default="normal", help="Benchmark mode")
    
    # SM count parameters
    parser.add_argument("--sm_count", type=int, default=None, 
                      help="Number of SMs to use for normal mode (if None, use auto-tuned)")
    parser.add_argument("--sm_counts", type=str, default="8,16,24,32,40,48", 
                      help="Comma-separated list of SM counts to compare in compare_sm mode")
    
    # Distributed parameters
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank")
    parser.add_argument("--num_local_ranks", type=int, default=1, help="Number of local ranks")
    
    args = parser.parse_args()
    
    # Initialize distributed
    rank, world_size, group = init_dist(args.local_rank, args.num_local_ranks)
    
    # Create benchmark instance
    benchmark = Benchmark(group)
    
    # Print hardware info
    if rank == 0:
        device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
        print(f"\nHardware Information:")
        print(f"  Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"  Compute Capability: {device_props.major}.{device_props.minor}")
        print(f"  Number of SMs: {device_props.multi_processor_count}")
        print(f"  Number of Ranks: {world_size}")
        print(f"  CUDA Version: {torch.version.cuda}")
        
        print(f"\nBenchmark Configuration:")
        print(f"  Hidden Size: {args.hidden_size}")
        print(f"  Number of Tokens: {args.num_tokens}")
        print(f"  Number of Experts: {args.num_experts}")
        print(f"  Number of Trials: {args.num_trials}")
        print(f"  Warmup Trials: {args.warmup_trials}")
        print(f"  Mode: {args.mode}")
        
        if args.mode == "normal" and args.sm_count:
            print(f"  SM Count: {args.sm_count}")
        elif args.mode == "compare_sm":
            print(f"  SM Counts to Compare: {args.sm_counts}")
    
    # Run benchmark according to selected mode
    if args.mode == "normal":
        result = benchmark.benchmark_dispatch_combine(
            hidden_size=args.hidden_size,
            num_tokens=args.num_tokens,
            num_experts=args.num_experts,
            sm_count=args.sm_count,
            num_trials=args.num_trials,
            warmup_trials=args.warmup_trials
        )
        
        if rank == 0:
            # Print some summary statistics
            print(f"\nSummary:")
            print(f"  Dispatch Bandwidth: {result['dispatch_bandwidth_gb']:.2f} GB/s")
            print(f"  Combine Bandwidth: {result['combine_bandwidth_gb']:.2f} GB/s")
            print(f"  Effective Bandwidth: {(result['dispatch_bandwidth_gb'] + result['combine_bandwidth_gb'])/2:.2f} GB/s")
            
    elif args.mode == "low_latency":
        result = benchmark.benchmark_low_latency(
            hidden_size=args.hidden_size,
            num_tokens=args.num_tokens,
            num_experts=args.num_experts,
            num_trials=args.num_trials,
            warmup_trials=args.warmup_trials
        )
        
        if rank == 0:
            # Print some summary statistics
            print(f"\nSummary:")
            print(f"  Dispatch Latency: {result['dispatch_avg_time']*1000*1000:.2f} μs")
            print(f"  Combine Latency: {result['combine_avg_time']*1000*1000:.2f} μs")
            print(f"  Total Latency: {(result['dispatch_avg_time'] + result['combine_avg_time'])*1000*1000:.2f} μs")
            print(f"  Dispatch Bandwidth: {result['dispatch_bandwidth_gb']:.2f} GB/s")
            print(f"  Combine Bandwidth: {result['combine_bandwidth_gb']:.2f} GB/s")
            
    elif args.mode == "compare_sm":
        sm_counts = [int(x) for x in args.sm_counts.split(",")]
        benchmark.compare_sm_counts(
            hidden_size=args.hidden_size,
            num_tokens=args.num_tokens,
            num_experts=args.num_experts,
            sm_counts=sm_counts,
            num_trials=args.num_trials,
            warmup_trials=args.warmup_trials
        )


if __name__ == "__main__":
    main() 