import os
import argparse
import torch
import torch.distributed as dist

from deep_ep import AutoTuner, Buffer


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


def create_test_moe_data(num_tokens, hidden_size, num_experts, top_k=2, dtype=torch.bfloat16):
    """Create test data for MoE operations"""
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device='cuda')
    topk_idx = torch.randint(0, num_experts, (num_tokens, top_k), device='cuda')
    topk_weights = torch.ones((num_tokens, top_k), dtype=dtype, device='cuda') / top_k
    return x, topk_idx, topk_weights


def run_moe_forward(buffer, x, topk_idx, topk_weights, num_experts):
    """Run MoE forward pass using the provided buffer"""
    # Get dispatch layout
    layout_result = buffer.get_dispatch_layout(topk_idx, num_experts)
    
    # Dispatch
    dispatch_result = buffer.dispatch(
        x, 
        topk_idx=topk_idx, 
        topk_weights=topk_weights,
        num_tokens_per_rank=layout_result[0],
        num_tokens_per_rdma_rank=layout_result[1],
        is_token_in_rank=layout_result[3],
        num_tokens_per_expert=layout_result[2]
    )
    
    # For a real MoE, you would do expert computation here
    expert_output = dispatch_result[0]  # Just pass through for this example
    
    # Combine
    combined_output, _, _ = buffer.combine(
        expert_output,
        dispatch_result[4],  # handle
        topk_weights=dispatch_result[2]
    )
    
    return combined_output


def main():
    parser = argparse.ArgumentParser(description="DeepEP Auto-tuning Example")
    
    parser.add_argument("--hidden_size", type=int, default=7168, help="Hidden dimension size")
    parser.add_argument("--num_tokens", type=int, default=4096, help="Number of tokens per batch")
    parser.add_argument("--num_experts", type=int, default=32, help="Number of experts")
    parser.add_argument("--profile_name", type=str, default="default_profile", help="Name for the tuned profile")
    parser.add_argument("--save_profile", action="store_true", help="Whether to save the tuned profile")
    parser.add_argument("--load_profile", action="store_true", help="Whether to load the tuned profile")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank")
    parser.add_argument("--num_local_ranks", type=int, default=1, help="Number of local ranks")
    
    args = parser.parse_args()
    
    # Initialize distributed
    rank, world_size, group = init_dist(args.local_rank, args.num_local_ranks)
    
    # Create the auto-tuner
    auto_tuner = AutoTuner(group)
    
    # Print current hardware signature
    if rank == 0:
        print(f"Hardware signature: {auto_tuner._get_hardware_signature()}")
    
    # Load a saved profile if requested
    if args.load_profile:
        if rank == 0:
            print(f"Loading profile: {args.profile_name}")
        success = auto_tuner.load_profile(args.profile_name)
        if rank == 0:
            if success:
                print(f"Successfully loaded profile")
            else:
                print(f"Failed to load profile, will auto-tune instead")
    
    # Create test data
    x, topk_idx, topk_weights = create_test_moe_data(
        args.num_tokens, 
        args.hidden_size, 
        args.num_experts
    )
    
    # Method 1: Auto-tune and create optimized buffer in one step
    if rank == 0:
        print("\nMethod 1: Creating optimized buffer with auto-tuning")
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    optimized_buffer = auto_tuner.create_optimized_buffer(
        hidden_size=args.hidden_size,
        num_tokens=args.num_tokens,
        num_experts=args.num_experts,
        dtype=torch.bfloat16
    )
    end_event.record()
    
    torch.cuda.synchronize()
    
    if rank == 0:
        print(f"  Optimized SM count: {Buffer.num_sms}")
        print(f"  Buffer creation time: {start_event.elapsed_time(end_event):.2f} ms")
    
    # Run MoE operations with optimized buffer
    output = run_moe_forward(optimized_buffer, x, topk_idx, topk_weights, args.num_experts)
    
    if rank == 0:
        print(f"  Output shape: {output.shape}")
    
    # Method 2: Explicitly tune SM count first
    if rank == 0:
        print("\nMethod 2: Explicitly tuning SM count first")
    
    # Reset to default SM count
    Buffer.set_num_sms(20)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    optimal_sm_count = auto_tuner.tune_sm_count(
        hidden_size=args.hidden_size,
        num_tokens=args.num_tokens,
        num_experts=args.num_experts,
        dtype=torch.bfloat16,
        force_refresh=True  # Force re-tuning
    )
    end_event.record()
    
    torch.cuda.synchronize()
    
    if rank == 0:
        print(f"  Optimal SM count: {optimal_sm_count}")
        print(f"  Tuning time: {start_event.elapsed_time(end_event):.2f} ms")
    
    # Set the tuned SM count
    Buffer.set_num_sms(optimal_sm_count)
    
    # Create a buffer with the tuned SM count
    element_size = torch.finfo(torch.bfloat16).bits // 8
    buffer_size = args.hidden_size * args.num_tokens * element_size * 2  # x2 for safety
    
    manual_buffer = Buffer(
        group,
        num_nvl_bytes=buffer_size,
        num_rdma_bytes=buffer_size
    )
    
    # Run MoE operations with manually tuned buffer
    output = run_moe_forward(manual_buffer, x, topk_idx, topk_weights, args.num_experts)
    
    if rank == 0:
        print(f"  Output shape: {output.shape}")
    
    # Save the tuned profile if requested
    if args.save_profile:
        if rank == 0:
            print(f"\nSaving tuned profile as: {args.profile_name}")
        
        config = auto_tuner.tune_and_save_profile(
            profile_name=args.profile_name,
            hidden_size=args.hidden_size,
            num_tokens=args.num_tokens,
            num_experts=args.num_experts,
            dtype=torch.bfloat16
        )
        
        if rank == 0:
            print(f"  Profile saved with optimal SM count: {config['tuned_values']['optimal_sm_count']}")


if __name__ == "__main__":
    main() 