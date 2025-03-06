import os
import argparse
import torch
import torch.distributed as dist
import numpy as np
import time
from typing import Tuple, List, Optional

from deep_ep import DynamicRouter, LoadBalancer, Buffer


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


class SimpleMoELayer:
    """
    Simple Mixture of Experts layer implementation with dynamic load balancing.
    
    This is a basic example for demonstration purposes and does not include all
    optimizations that would be used in a production setting.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int,
        group: dist.ProcessGroup,
        top_k: int = 2,
        enable_auto_balancing: bool = True,
        balance_threshold: float = 0.2,
        window_size: int = 50
    ):
        """
        Initialize the MoE layer.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            num_experts: Number of experts
            group: Distributed group
            top_k: Number of experts per token
            enable_auto_balancing: Whether to enable automatic load balancing
            balance_threshold: Threshold for triggering load balancing
            window_size: Number of batches to consider for load balancing
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
        
        # Create dynamic router for load balancing
        self.dynamic_router = DynamicRouter(
            num_experts=num_experts,
            group=group,
            balance_threshold=balance_threshold,
            window_size=window_size,
            enable_auto_balancing=enable_auto_balancing
        )
        
        # Create DeepEP buffer for inter-expert communication
        self.buffer = None
        
    def create_buffer(self, batch_size: int):
        """Create or update the communication buffer"""
        element_size = torch.finfo(torch.bfloat16).bits // 8
        buffer_size = self.input_dim * batch_size * element_size * 2  # x2 for safety
        
        if self.buffer is None:
            self.buffer = Buffer(
                self.group,
                num_nvl_bytes=buffer_size,
                num_rdma_bytes=buffer_size
            )
        elif self.buffer.num_nvl_bytes < buffer_size or self.buffer.num_rdma_bytes < buffer_size:
            self.buffer = Buffer(
                self.group,
                num_nvl_bytes=buffer_size,
                num_rdma_bytes=buffer_size
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dynamic load balancing.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        batch_size = x.shape[0]
        
        # Ensure buffer is properly sized
        self.create_buffer(batch_size)
        
        # Get routing logits
        router_logits = self.router(x)
        
        # Apply dynamic routing with load balancing
        route_indices, route_weights = self.dynamic_router.route(
            router_logits=router_logits,
            top_k=self.top_k,
            use_capacity=True  # Apply capacity weights for load balancing
        )
        
        # Dispatch tokens to experts
        layout_result = self.buffer.get_dispatch_layout(route_indices, self.num_experts)
        dispatch_result = self.buffer.dispatch(
            x, 
            topk_idx=route_indices,
            topk_weights=route_weights,
            num_tokens_per_rank=layout_result[0],
            num_tokens_per_rdma_rank=layout_result[1],
            is_token_in_rank=layout_result[3],
            num_tokens_per_expert=layout_result[2]
        )
        
        # Process tokens with experts
        expert_outputs = torch.zeros_like(dispatch_result[0])
        recv_topk_idx = dispatch_result[1]
        
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
        
        # Combine expert outputs
        combined_output, _, _ = self.buffer.combine(
            expert_outputs,
            dispatch_result[4],  # handle
            topk_weights=dispatch_result[2]
        )
        
        return combined_output
    
    def plot_utilization(self, save_path: Optional[str] = None, show: bool = True):
        """Plot expert utilization"""
        self.dynamic_router.plot_utilization(save_path, show)
    
    def plot_balancing_events(self, save_path: Optional[str] = None, show: bool = True):
        """Plot load balancing events"""
        self.dynamic_router.plot_balancing_events(save_path, show)
    
    def get_expert_capacities(self) -> torch.Tensor:
        """Get current expert capacity weights"""
        return self.dynamic_router.get_expert_capacity()
    
    def get_expert_utilization(self) -> torch.Tensor:
        """Get current expert utilization"""
        return self.dynamic_router.get_expert_utilization()


def simulate_skewed_routing(model: SimpleMoELayer, num_batches: int, batch_size: int, input_dim: int, 
                           phases: List[Tuple[int, List[float]]], print_interval: int = 10):
    """
    Simulate training with artificially skewed expert routing.
    
    Args:
        model: The MoE model
        num_batches: Number of batches to simulate
        batch_size: Number of tokens per batch
        input_dim: Input dimension
        phases: List of (num_batches, expert_preferences) tuples for different routing patterns
        print_interval: How often to print status
    """
    rank = dist.get_rank()
    
    # Prepare for simulation
    current_phase = 0
    batches_in_phase = 0
    total_batches = 0
    phase_length, expert_preferences = phases[current_phase]
    
    # Normalize expert preferences to sum to 1
    expert_preferences = torch.tensor(expert_preferences, dtype=torch.float)
    expert_preferences = expert_preferences / expert_preferences.sum()
    
    # Run simulation
    for i in range(num_batches):
        # Check if we need to switch to the next phase
        if batches_in_phase >= phase_length and current_phase < len(phases) - 1:
            current_phase += 1
            batches_in_phase = 0
            phase_length, expert_preferences = phases[current_phase]
            expert_preferences = torch.tensor(expert_preferences, dtype=torch.float)
            expert_preferences = expert_preferences / expert_preferences.sum()
            
            if rank == 0:
                print(f"\n--- Switching to phase {current_phase + 1} ---")
                print(f"Expert preferences: {expert_preferences.tolist()}")
        
        # Create input data
        x = torch.randn(batch_size, input_dim, device='cuda', dtype=torch.bfloat16)
        
        # Override the router to use our expert preferences
        with torch.no_grad():
            # Replace the normal router output with our skewed distribution
            # We'll use a distribution where each token strongly prefers experts according
            # to the expert_preferences distribution
            original_logits = model.router(x)
            
            # Create skewed logits
            skewed_logits = torch.zeros_like(original_logits)
            
            # For each token, sample experts according to our preference distribution
            # and give them a high score
            for token_idx in range(batch_size):
                # Sample expert indices based on preference distribution
                preferred_experts = torch.multinomial(
                    expert_preferences, 
                    num_samples=min(model.top_k * 2, model.num_experts), 
                    replacement=False
                )
                
                # Assign high scores to preferred experts (decreasing by position)
                for i, expert_idx in enumerate(preferred_experts):
                    skewed_logits[token_idx, expert_idx] = 10.0 - i * 0.5
            
            # Save original forward method
            original_forward = model.dynamic_router.route
            
            # Create a temporary override
            def temp_route(router_logits, top_k=2, use_capacity=True):
                return original_forward(skewed_logits, top_k, use_capacity)
            
            # Replace method temporarily
            model.dynamic_router.route = temp_route
        
        # Forward pass
        start_time = time.time()
        output = model(x)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        # Restore original method
        model.dynamic_router.route = original_forward
        
        # Print status
        batches_in_phase += 1
        total_batches += 1
        
        if rank == 0 and total_batches % print_interval == 0:
            capacities = model.get_expert_capacities()
            utilization = model.get_expert_utilization()
            
            print(f"Batch {total_batches}/{num_batches} " 
                  f"(Phase {current_phase + 1}, Batch {batches_in_phase}/{phase_length})")
            print(f"Forward pass time: {elapsed * 1000:.2f} ms")
            
            # Only print a few experts if there are many
            if model.num_experts <= 8:
                print(f"Expert capacities: {capacities.tolist()}")
                print(f"Expert utilization: {utilization.tolist()}")
            else:
                print(f"Expert capacities: {capacities[:4].tolist()} ... {capacities[-4:].tolist()}")
                print(f"Expert utilization: {utilization[:4].tolist()} ... {utilization[-4:].tolist()}")
            print()


def main():
    parser = argparse.ArgumentParser(description="DeepEP Dynamic Load Balancing Example")
    
    parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden dimension size")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_batches", type=int, default=200, help="Number of batches")
    parser.add_argument("--top_k", type=int, default=2, help="Number of experts per token")
    parser.add_argument("--enable_balancing", action="store_true", help="Enable auto balancing")
    parser.add_argument("--balance_threshold", type=float, default=0.2, 
                      help="Threshold for triggering load balancing")
    parser.add_argument("--window_size", type=int, default=50, 
                      help="Number of batches to consider for load balancing")
    parser.add_argument("--plot_dir", type=str, default="plots", help="Directory to save plots")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank")
    parser.add_argument("--num_local_ranks", type=int, default=1, help="Number of local ranks")
    
    args = parser.parse_args()
    
    # Initialize distributed
    rank, world_size, group = init_dist(args.local_rank, args.num_local_ranks)
    
    # Create MoE model with dynamic load balancing
    model = SimpleMoELayer(
        input_dim=args.hidden_size,
        output_dim=args.hidden_size,
        num_experts=args.num_experts,
        group=group,
        top_k=args.top_k,
        enable_auto_balancing=args.enable_balancing,
        balance_threshold=args.balance_threshold,
        window_size=args.window_size
    )
    
    if rank == 0:
        print("\n--- DeepEP Dynamic Load Balancing Demonstration ---")
        print(f"Number of experts: {args.num_experts}")
        print(f"Hidden size: {args.hidden_size}")
        print(f"Batch size: {args.batch_size}")
        print(f"Top-k experts: {args.top_k}")
        print(f"Auto-balancing: {'Enabled' if args.enable_balancing else 'Disabled'}")
        if args.enable_balancing:
            print(f"Balance threshold: {args.balance_threshold}")
            print(f"Window size: {args.window_size}")
        print(f"Number of batches: {args.num_batches}")
        print("---------------------------------------------------\n")
    
    # Define phases with different routing patterns to demonstrate load balancing
    # Each phase is (num_batches, expert_preferences)
    phases = [
        # Phase 1: Heavily skewed to first 2 experts
        (50, [5.0, 4.0] + [0.2] * (args.num_experts - 2)),
        
        # Phase 2: Transition to different experts
        (50, [0.2, 0.2] + [3.0, 4.0] + [0.2] * (args.num_experts - 4)),
        
        # Phase 3: More balanced but with peaks
        (50, [0.5] * (args.num_experts // 2) + [2.0] * (args.num_experts - args.num_experts // 2)),
        
        # Phase 4: Return to initial pattern
        (50, [5.0, 4.0] + [0.2] * (args.num_experts - 2))
    ]
    
    # Simulate training with skewed routing
    simulate_skewed_routing(
        model=model,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        input_dim=args.hidden_size,
        phases=phases,
        print_interval=10
    )
    
    # Create plot directory if needed
    if rank == 0 and args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
    
    # Generate plots
    if rank == 0:
        print("\nGenerating plots...")
        model.plot_utilization(
            save_path=os.path.join(args.plot_dir, "expert_utilization.png"),
            show=False
        )
        
        if args.enable_balancing:
            model.plot_balancing_events(
                save_path=os.path.join(args.plot_dir, "balancing_events.png"),
                show=False
            )
        
        print(f"Plots saved to {args.plot_dir}")
        print("\nFinal expert capacities:")
        print(model.get_expert_capacities().tolist())
        print("\nFinal expert utilization:")
        print(model.get_expert_utilization().tolist())


if __name__ == "__main__":
    main() 