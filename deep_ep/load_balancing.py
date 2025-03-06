import os
import json
import time
import torch
import torch.distributed as dist
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from pathlib import Path
import matplotlib.pyplot as plt
from collections import deque

from .buffer import Buffer
from .utils import EventOverlap


class ExpertStats:
    """
    Tracks and manages statistics for expert utilization.
    
    This class provides methods to monitor and analyze the workload
    distribution across experts.
    """
    
    def __init__(self, num_experts: int, window_size: int = 100):
        """
        Initialize the expert statistics tracker.
        
        Args:
            num_experts: Total number of experts in the system
            window_size: Number of batches to include in the moving average
        """
        self.num_experts = num_experts
        self.window_size = window_size
        
        # Initialize counters and histories
        self.total_calls = torch.zeros(num_experts, dtype=torch.long)
        self.total_tokens = torch.zeros(num_experts, dtype=torch.long)
        self.call_history = [deque(maxlen=window_size) for _ in range(num_experts)]
        self.token_history = [deque(maxlen=window_size) for _ in range(num_experts)]
        
        # Initialize timestamped utilization records for visualization
        self.utilization_history = []
        self.timestamps = []
        
    def update(self, expert_counts: torch.Tensor):
        """
        Update expert statistics with the latest batch information.
        
        Args:
            expert_counts: Tensor with shape [num_experts] containing the number of 
                          tokens assigned to each expert in the current batch
        """
        timestamp = time.time()
        
        # Record the current timestamp
        self.timestamps.append(timestamp)
        
        # Update cumulative counts
        self.total_tokens += expert_counts
        
        # Update call counts (any non-zero count is a call)
        calls = (expert_counts > 0).long()
        self.total_calls += calls
        
        # Update histories
        for i in range(self.num_experts):
            self.call_history[i].append(calls[i].item())
            self.token_history[i].append(expert_counts[i].item())
        
        # Record utilization snapshot for visualization
        utilization = expert_counts.float() / expert_counts.sum()
        self.utilization_history.append(utilization.tolist())
        
        # Trim history if it's getting too long (keep last 1000 entries)
        if len(self.timestamps) > 1000:
            self.timestamps = self.timestamps[-1000:]
            self.utilization_history = self.utilization_history[-1000:]
    
    def get_utilization(self) -> torch.Tensor:
        """
        Get the current utilization distribution across experts.
        
        Returns:
            Tensor with shape [num_experts] containing the fraction of tokens 
            processed by each expert
        """
        total = self.total_tokens.sum().float()
        if total == 0:
            return torch.ones(self.num_experts) / self.num_experts
        return self.total_tokens.float() / total
    
    def get_recent_utilization(self) -> torch.Tensor:
        """
        Get the recent utilization based on the moving window.
        
        Returns:
            Tensor with shape [num_experts] containing the recent utilization pattern
        """
        recent_tokens = torch.zeros(self.num_experts, dtype=torch.float)
        
        for i in range(self.num_experts):
            if self.token_history[i]:
                recent_tokens[i] = sum(self.token_history[i])
        
        total = recent_tokens.sum()
        if total == 0:
            return torch.ones(self.num_experts) / self.num_experts
        return recent_tokens / total
    
    def get_call_frequency(self) -> torch.Tensor:
        """
        Get the frequency at which each expert is called.
        
        Returns:
            Tensor with shape [num_experts] containing the call frequency for each expert
        """
        total_batches = max(1, len(self.timestamps))
        return self.total_calls.float() / total_batches
    
    def get_recent_call_frequency(self) -> torch.Tensor:
        """
        Get the recent call frequency based on the moving window.
        
        Returns:
            Tensor with shape [num_experts] containing the recent call frequency
        """
        recent_calls = torch.zeros(self.num_experts, dtype=torch.float)
        
        for i in range(self.num_experts):
            if self.call_history[i]:
                recent_calls[i] = sum(self.call_history[i]) / len(self.call_history[i])
        
        return recent_calls
    
    def reset(self):
        """Reset all statistics"""
        self.total_calls = torch.zeros(self.num_experts, dtype=torch.long)
        self.total_tokens = torch.zeros(self.num_experts, dtype=torch.long)
        self.call_history = [deque(maxlen=self.window_size) for _ in range(self.num_experts)]
        self.token_history = [deque(maxlen=self.window_size) for _ in range(self.num_experts)]
        self.utilization_history = []
        self.timestamps = []
    
    def plot_utilization(self, save_path: Optional[str] = None, show: bool = True):
        """
        Generate a visualization of expert utilization over time.
        
        Args:
            save_path: Path to save the plot (if None, plot is not saved)
            show: Whether to display the plot
        """
        if not self.utilization_history:
            print("No utilization data available for plotting")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Convert timestamps to relative time (seconds from start)
        rel_timestamps = [t - self.timestamps[0] for t in self.timestamps]
        
        # Plot utilization heatmap over time
        utilization_array = np.array(self.utilization_history)
        im = ax1.imshow(
            utilization_array.T, 
            aspect='auto', 
            cmap='viridis',
            extent=[rel_timestamps[0], rel_timestamps[-1], -0.5, self.num_experts - 0.5]
        )
        ax1.set_title('Expert Utilization Over Time')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Expert ID')
        fig.colorbar(im, ax=ax1, label='Utilization')
        
        # Plot line chart of cumulative utilization
        overall_utilization = self.get_utilization().numpy()
        ax2.bar(range(self.num_experts), overall_utilization)
        ax2.set_title('Cumulative Expert Utilization')
        ax2.set_xlabel('Expert ID')
        ax2.set_ylabel('Utilization')
        ax2.set_xticks(range(self.num_experts))
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()


class LoadBalancer:
    """
    Dynamic load balancing system for DeepEP.
    
    This class provides functionality to:
    1. Monitor expert utilization
    2. Detect load imbalances
    3. Dynamically adjust routing strategies
    4. Visualize workload distribution
    """
    
    def __init__(
        self,
        group: dist.ProcessGroup,
        num_experts: int,
        balance_threshold: float = 0.2,
        window_size: int = 100,
        enable_auto_balancing: bool = False
    ):
        """
        Initialize the load balancer.
        
        Args:
            group: The communication group
            num_experts: Total number of experts
            balance_threshold: Threshold for imbalance detection (0.0-1.0)
            window_size: Number of batches to include in moving averages
            enable_auto_balancing: Whether to automatically apply load balancing
        """
        self.rank = group.rank()
        self.group_size = group.size()
        self.group = group
        self.num_experts = num_experts
        self.balance_threshold = balance_threshold
        self.enable_auto_balancing = enable_auto_balancing
        
        # Initialize expert statistics
        self.expert_stats = ExpertStats(num_experts, window_size)
        
        # Initialize expert capacity weights (for load balancing)
        self.expert_capacity = torch.ones(num_experts, dtype=torch.float)
        
        # Initialize tracking for load balancing events
        self.balance_events = []
        
    def update_stats(self, expert_counts: torch.Tensor):
        """
        Update expert utilization statistics with latest batch information.
        
        Args:
            expert_counts: Tensor with shape [num_experts] containing the number of 
                          tokens assigned to each expert in the current batch
        """
        self.expert_stats.update(expert_counts)
        
        # Check for imbalance and possibly trigger rebalancing
        if self.enable_auto_balancing:
            imbalance = self.detect_imbalance()
            if imbalance > self.balance_threshold:
                self.rebalance()
    
    def detect_imbalance(self) -> float:
        """
        Detect the level of load imbalance across experts.
        
        Returns:
            Imbalance score (0.0 = perfectly balanced, 1.0 = completely imbalanced)
        """
        utilization = self.expert_stats.get_recent_utilization()
        
        # Calculate standard deviation of utilization
        mean_util = utilization.mean()
        if mean_util == 0:
            return 0.0
        
        # Normalized standard deviation as imbalance metric
        std_util = utilization.std()
        imbalance = std_util / mean_util
        
        # Cap at 1.0 for consistency
        return min(float(imbalance), 1.0)
    
    def rebalance(self) -> Dict[str, Any]:
        """
        Perform load balancing by adjusting expert capacity weights.
        
        Returns:
            Dictionary containing the balancing event details
        """
        # Get current utilization
        utilization = self.expert_stats.get_recent_utilization()
        
        # Calculate target capacity (inverse of utilization)
        # High utilization → lower capacity, Low utilization → higher capacity
        # Add small epsilon to avoid division by zero
        epsilon = 1e-5
        inverse_util = 1.0 / (utilization + epsilon)
        
        # Normalize to ensure the sum equals num_experts
        self.expert_capacity = inverse_util * (self.num_experts / inverse_util.sum())
        
        # Record the balancing event
        event = {
            'timestamp': time.time(),
            'utilization_before': utilization.tolist(),
            'capacity_after': self.expert_capacity.tolist(),
            'imbalance_score': self.detect_imbalance()
        }
        self.balance_events.append(event)
        
        return event
    
    def get_expert_capacities(self) -> torch.Tensor:
        """
        Get the current expert capacity weights for routing.
        
        Returns:
            Tensor with shape [num_experts] containing capacity weights
        """
        return self.expert_capacity.clone()
    
    def apply_capacity_to_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Apply capacity weights to expert routing scores.
        
        Args:
            scores: Tensor with shape [batch_size, num_experts] containing 
                  raw expert routing scores
                  
        Returns:
            Tensor with shape [batch_size, num_experts] containing 
            capacity-weighted scores
        """
        # Reshape for broadcasting
        capacity_weights = self.expert_capacity.view(1, -1)
        
        # Apply capacity weights to scores
        weighted_scores = scores * capacity_weights
        
        return weighted_scores
    
    def reset_balancing(self):
        """Reset load balancing to equal distribution"""
        self.expert_capacity = torch.ones(self.num_experts, dtype=torch.float)
        
    def reset_stats(self):
        """Reset all statistics"""
        self.expert_stats.reset()
        self.balance_events = []
        
    def plot_utilization(self, save_path: Optional[str] = None, show: bool = True):
        """
        Generate a visualization of expert utilization.
        
        Args:
            save_path: Path to save the plot (if None, plot is not saved)
            show: Whether to display the plot
        """
        self.expert_stats.plot_utilization(save_path, show)
        
    def plot_balancing_events(self, save_path: Optional[str] = None, show: bool = True):
        """
        Generate a visualization of load balancing events.
        
        Args:
            save_path: Path to save the plot (if None, plot is not saved)
            show: Whether to display the plot
        """
        if not self.balance_events:
            print("No balancing events to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Extract data from balance events
        timestamps = [event['timestamp'] for event in self.balance_events]
        relative_times = [t - timestamps[0] for t in timestamps]
        imbalance_scores = [event['imbalance_score'] for event in self.balance_events]
        
        # Get capacities over time
        capacities = np.array([event['capacity_after'] for event in self.balance_events])
        
        # Plot capacities
        im = ax1.imshow(
            capacities.T, 
            aspect='auto', 
            cmap='coolwarm',
            extent=[relative_times[0], relative_times[-1], -0.5, self.num_experts - 0.5]
        )
        ax1.set_title('Expert Capacity Adjustments Over Time')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Expert ID')
        fig.colorbar(im, ax=ax1, label='Capacity')
        
        # Plot imbalance scores
        ax2.plot(relative_times, imbalance_scores, marker='o')
        ax2.axhline(y=self.balance_threshold, color='r', linestyle='--', alpha=0.7, 
                   label=f'Threshold ({self.balance_threshold})')
        ax2.set_title('Load Imbalance Over Time')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Imbalance Score')
        ax2.set_ylim(0, 1)
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_stats(self, file_path: str):
        """
        Save statistics and balancing events to a file.
        
        Args:
            file_path: Path to save the data
        """
        data = {
            'num_experts': self.num_experts,
            'balance_threshold': self.balance_threshold,
            'total_calls': self.expert_stats.total_calls.tolist(),
            'total_tokens': self.expert_stats.total_tokens.tolist(),
            'utilization_history': self.expert_stats.utilization_history,
            'timestamps': self.expert_stats.timestamps,
            'current_capacities': self.expert_capacity.tolist(),
            'balance_events': self.balance_events
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f)
    
    def load_stats(self, file_path: str):
        """
        Load statistics and balancing events from a file.
        
        Args:
            file_path: Path to load the data from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Verify compatibility
            if data['num_experts'] != self.num_experts:
                print(f"Warning: Loaded data has {data['num_experts']} experts, " 
                      f"but current system has {self.num_experts}")
                return False
            
            # Load data
            self.balance_threshold = data['balance_threshold']
            self.expert_stats.total_calls = torch.tensor(data['total_calls'], dtype=torch.long)
            self.expert_stats.total_tokens = torch.tensor(data['total_tokens'], dtype=torch.long)
            self.expert_stats.utilization_history = data['utilization_history']
            self.expert_stats.timestamps = data['timestamps']
            self.expert_capacity = torch.tensor(data['current_capacities'], dtype=torch.float)
            self.balance_events = data['balance_events']
            
            return True
        
        except Exception as e:
            print(f"Error loading stats: {e}")
            return False
            
            
class DynamicRouter:
    """
    Dynamic expert router with load balancing capabilities.
    
    This class extends standard MoE routing with dynamic load balancing.
    """
    
    def __init__(
        self,
        num_experts: int,
        group: dist.ProcessGroup,
        balance_threshold: float = 0.2,
        window_size: int = 100,
        enable_auto_balancing: bool = False
    ):
        """
        Initialize the dynamic router.
        
        Args:
            num_experts: Total number of experts
            group: The communication group
            balance_threshold: Threshold for imbalance detection (0.0-1.0)
            window_size: Number of batches to include in moving averages
            enable_auto_balancing: Whether to automatically apply load balancing
        """
        self.num_experts = num_experts
        self.group = group
        
        # Initialize load balancer
        self.load_balancer = LoadBalancer(
            group=group,
            num_experts=num_experts,
            balance_threshold=balance_threshold,
            window_size=window_size,
            enable_auto_balancing=enable_auto_balancing
        )
    
    def route(
        self, 
        router_logits: torch.Tensor,
        top_k: int = 2,
        use_capacity: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts with dynamic load balancing.
        
        Args:
            router_logits: Tensor with shape [batch_size, num_experts] containing
                          raw routing scores
            top_k: Number of experts to select per token
            use_capacity: Whether to apply capacity weights
            
        Returns:
            Tuple of (expert_indices, expert_weights), where:
              - expert_indices has shape [batch_size, top_k]
              - expert_weights has shape [batch_size, top_k]
        """
        batch_size = router_logits.size(0)
        
        # Apply capacity weights if enabled
        if use_capacity:
            router_logits = self.load_balancer.apply_capacity_to_scores(router_logits)
        
        # Get top-k experts and weights using softmax
        router_probs = torch.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = torch.topk(router_probs, top_k, dim=-1)
        
        # Normalize weights to sum to 1
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        # Collect statistics for load balancing
        with torch.no_grad():
            # Count how many tokens are routed to each expert
            expert_counts = torch.zeros(self.num_experts, device=router_logits.device)
            for k in range(top_k):
                for expert_idx in range(self.num_experts):
                    # Sum the number of tokens that have this expert in their top-k
                    count = (expert_indices[:, k] == expert_idx).sum().item()
                    expert_counts[expert_idx] += count
            
            # Update load balancing statistics
            self.load_balancer.update_stats(expert_counts.cpu())
        
        return expert_indices, expert_weights
    
    def get_expert_utilization(self) -> torch.Tensor:
        """
        Get the current expert utilization distribution.
        
        Returns:
            Tensor with shape [num_experts] containing the fraction of tokens 
            processed by each expert
        """
        return self.load_balancer.expert_stats.get_utilization()
    
    def get_expert_capacity(self) -> torch.Tensor:
        """
        Get the current expert capacity weights.
        
        Returns:
            Tensor with shape [num_experts] containing capacity weights
        """
        return self.load_balancer.get_expert_capacities()
    
    def reset_stats(self):
        """Reset all statistics"""
        self.load_balancer.reset_stats()
    
    def plot_utilization(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plot expert utilization.
        
        Args:
            save_path: Path to save the plot (if None, plot is not saved)
            show: Whether to display the plot
        """
        self.load_balancer.plot_utilization(save_path, show)
    
    def plot_balancing_events(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plot load balancing events.
        
        Args:
            save_path: Path to save the plot (if None, plot is not saved)
            show: Whether to display the plot
        """
        self.load_balancer.plot_balancing_events(save_path, show)
    
    def save_stats(self, file_path: str):
        """
        Save statistics and balancing events to a file.
        
        Args:
            file_path: Path to save the data
        """
        self.load_balancer.save_stats(file_path)
    
    def load_stats(self, file_path: str) -> bool:
        """
        Load statistics and balancing events from a file.
        
        Args:
            file_path: Path to load the data from
            
        Returns:
            True if successful, False otherwise
        """
        return self.load_balancer.load_stats(file_path) 