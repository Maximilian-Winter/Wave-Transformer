"""
Distributed training utilities for Wave Transformer analysis.

Provides utilities for running analysis operations in distributed training environments:
- Process rank checking
- Tensor gathering across processes
- Dictionary reduction
- Synchronization barriers
- Wrapper for callbacks to work in DDP
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Any, Callable
import functools


class DistributedAnalysisHelper:
    """
    Utilities for distributed training analysis.

    Handles common distributed operations needed for analysis:
    - Checking if running in distributed mode
    - Identifying the main process
    - Gathering tensors from all processes
    - Reducing metrics across processes
    - Synchronization

    Example usage:
        helper = DistributedAnalysisHelper()

        if helper.is_main_process():
            # Only run expensive analysis on main process
            plot_wave_evolution()

        # Gather tensors from all ranks
        all_losses = helper.gather_tensors(loss_tensor)

        # Reduce dictionary of metrics
        metrics = {'loss': loss, 'accuracy': acc}
        averaged_metrics = helper.reduce_dict(metrics)
    """

    def __init__(self):
        """Initialize distributed helper."""
        self._is_distributed = dist.is_available() and dist.is_initialized()

    def is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return self._is_distributed

    def is_main_process(self) -> bool:
        """
        Check if this is the main process (rank 0).

        Returns:
            True if this is rank 0 or not in distributed mode
        """
        if not self._is_distributed:
            return True
        return dist.get_rank() == 0

    def get_rank(self) -> int:
        """
        Get the rank of the current process.

        Returns:
            Rank (0 if not distributed)
        """
        if not self._is_distributed:
            return 0
        return dist.get_rank()

    def get_world_size(self) -> int:
        """
        Get the total number of processes.

        Returns:
            World size (1 if not distributed)
        """
        if not self._is_distributed:
            return 1
        return dist.get_world_size()

    def gather_tensors(
        self,
        tensor: torch.Tensor,
        dst: int = 0,
    ) -> Optional[List[torch.Tensor]]:
        """
        Gather tensors from all processes to a destination process.

        Args:
            tensor: Tensor to gather
            dst: Destination rank (default: 0)

        Returns:
            List of tensors from all ranks (only on dst rank, None elsewhere)
        """
        if not self._is_distributed:
            return [tensor]

        # Prepare tensor list on destination
        world_size = self.get_world_size()
        if self.get_rank() == dst:
            tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        else:
            tensor_list = None

        # Gather
        dist.gather(tensor, gather_list=tensor_list, dst=dst)

        return tensor_list

    def all_gather_tensors(
        self,
        tensor: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Gather tensors from all processes to all processes.

        Args:
            tensor: Tensor to gather

        Returns:
            List of tensors from all ranks (on all ranks)
        """
        if not self._is_distributed:
            return [tensor]

        world_size = self.get_world_size()
        tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]

        dist.all_gather(tensor_list, tensor)

        return tensor_list

    def reduce_tensor(
        self,
        tensor: torch.Tensor,
        op: str = 'mean',
        dst: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Reduce a tensor across all processes.

        Args:
            tensor: Tensor to reduce
            op: Reduction operation ('mean', 'sum', 'max', 'min')
            dst: Destination rank (None for all_reduce)

        Returns:
            Reduced tensor
        """
        if not self._is_distributed:
            return tensor

        # Clone to avoid modifying original
        tensor = tensor.clone()

        # Determine reduction operation
        if op == 'mean':
            reduce_op = dist.ReduceOp.SUM
            divide_by_world_size = True
        elif op == 'sum':
            reduce_op = dist.ReduceOp.SUM
            divide_by_world_size = False
        elif op == 'max':
            reduce_op = dist.ReduceOp.MAX
            divide_by_world_size = False
        elif op == 'min':
            reduce_op = dist.ReduceOp.MIN
            divide_by_world_size = False
        else:
            raise ValueError(f"Unsupported reduction operation: {op}")

        # Perform reduction
        if dst is None:
            # All-reduce (result available on all ranks)
            dist.all_reduce(tensor, op=reduce_op)
        else:
            # Reduce to specific rank
            dist.reduce(tensor, dst=dst, op=reduce_op)

        # Divide by world size for mean
        if divide_by_world_size:
            tensor = tensor / self.get_world_size()

        return tensor

    def reduce_dict(
        self,
        input_dict: Dict[str, Any],
        op: str = 'mean',
        dst: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Reduce a dictionary of metrics across all processes.

        Args:
            input_dict: Dictionary with string keys and tensor/scalar values
            op: Reduction operation ('mean', 'sum', 'max', 'min')
            dst: Destination rank (None for all_reduce)

        Returns:
            Dictionary with reduced values
        """
        if not self._is_distributed:
            return input_dict

        output_dict = {}

        for key, value in input_dict.items():
            # Convert scalars to tensors
            if isinstance(value, (int, float)):
                value = torch.tensor(value, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

            # Reduce tensor
            if isinstance(value, torch.Tensor):
                output_dict[key] = self.reduce_tensor(value, op=op, dst=dst)
            else:
                # Non-tensor values are not reduced
                output_dict[key] = value

        return output_dict

    def synchronize(self):
        """
        Synchronize all processes (barrier).

        Blocks until all processes reach this point.
        """
        if self._is_distributed:
            dist.barrier()

    def broadcast_object(
        self,
        obj: Any,
        src: int = 0,
    ) -> Any:
        """
        Broadcast a Python object from source to all processes.

        Args:
            obj: Object to broadcast (only used on src rank)
            src: Source rank

        Returns:
            The object (same on all ranks after broadcast)
        """
        if not self._is_distributed:
            return obj

        # Create list to hold object
        obj_list = [obj]

        # Broadcast
        dist.broadcast_object_list(obj_list, src=src)

        return obj_list[0]

    def execute_on_main_process(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function only on the main process.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result on main process, None elsewhere
        """
        if self.is_main_process():
            return func(*args, **kwargs)
        return None

    def execute_and_broadcast(
        self,
        func: Callable,
        *args,
        src: int = 0,
        **kwargs
    ) -> Any:
        """
        Execute a function on a specific rank and broadcast the result.

        Args:
            func: Function to execute
            *args: Positional arguments
            src: Source rank to execute on
            **kwargs: Keyword arguments

        Returns:
            Function result (same on all ranks)
        """
        if self.get_rank() == src:
            result = func(*args, **kwargs)
        else:
            result = None

        # Broadcast result to all ranks
        result = self.broadcast_object(result, src=src)

        return result


def main_process_only(func: Callable) -> Callable:
    """
    Decorator to run a function only on the main process.

    Example usage:
        @main_process_only
        def expensive_visualization():
            plot_wave_evolution()
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        helper = DistributedAnalysisHelper()
        if helper.is_main_process():
            return func(*args, **kwargs)
        return None

    return wrapper


def synchronized(func: Callable) -> Callable:
    """
    Decorator to synchronize all processes before and after function execution.

    Example usage:
        @synchronized
        def checkpoint_model():
            torch.save(model.state_dict(), 'checkpoint.pt')
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        helper = DistributedAnalysisHelper()
        helper.synchronize()
        result = func(*args, **kwargs)
        helper.synchronize()
        return result

    return wrapper


class DistributedCallback:
    """
    Wrapper for callbacks to work properly in distributed training.

    Automatically handles:
    - Running callbacks only on main process (when specified)
    - Reducing metrics across processes
    - Synchronization

    Example usage:
        # Original callback
        evolution_callback = WaveEvolutionCallback(...)

        # Wrap for distributed training
        dist_callback = DistributedCallback(
            callback=evolution_callback,
            main_process_only=True,  # Only run on rank 0
            reduce_metrics=True,     # Reduce metrics from all ranks
        )

        # Use in training loop
        dist_callback.on_batch_end(batch, logs={'loss': loss})
    """

    def __init__(
        self,
        callback: Any,
        main_process_only: bool = True,
        reduce_metrics: bool = True,
        reduction_op: str = 'mean',
    ):
        """
        Args:
            callback: The callback to wrap
            main_process_only: Whether to run callback only on main process
            reduce_metrics: Whether to reduce metrics before passing to callback
            reduction_op: Operation for metric reduction ('mean', 'sum', etc.)
        """
        self.callback = callback
        self.main_process_only = main_process_only
        self.reduce_metrics = reduce_metrics
        self.reduction_op = reduction_op
        self.helper = DistributedAnalysisHelper()

    def _should_execute(self) -> bool:
        """Check if callback should execute on this process."""
        if not self.main_process_only:
            return True
        return self.helper.is_main_process()

    def _prepare_logs(self, logs: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Prepare logs by reducing metrics if needed."""
        if logs is None:
            return None

        if self.reduce_metrics and self.helper.is_distributed():
            # Reduce all metrics across processes
            return self.helper.reduce_dict(logs, op=self.reduction_op)

        return logs

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Forward to callback if should execute."""
        if self._should_execute():
            logs = self._prepare_logs(logs)
            self.callback.on_train_begin(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Forward to callback if should execute."""
        if self._should_execute():
            logs = self._prepare_logs(logs)
            self.callback.on_epoch_begin(epoch, logs)

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Forward to callback if should execute."""
        if self._should_execute():
            logs = self._prepare_logs(logs)
            self.callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Forward to callback if should execute."""
        if self._should_execute():
            logs = self._prepare_logs(logs)
            self.callback.on_batch_end(batch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Forward to callback if should execute."""
        if self._should_execute():
            logs = self._prepare_logs(logs)
            self.callback.on_epoch_end(epoch, logs)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Forward to callback if should execute."""
        if self._should_execute():
            logs = self._prepare_logs(logs)
            self.callback.on_train_end(logs)

    def __getattr__(self, name: str):
        """Forward attribute access to wrapped callback."""
        return getattr(self.callback, name)


class DistributedMetricsAggregator:
    """
    Aggregates metrics across distributed processes.

    Useful for collecting and averaging metrics during training.

    Example usage:
        aggregator = DistributedMetricsAggregator()

        for batch in dataloader:
            loss = model(batch)
            aggregator.update({'loss': loss})

        # Get averaged metrics across all ranks
        avg_metrics = aggregator.compute()
    """

    def __init__(self, reduction_op: str = 'mean'):
        """
        Args:
            reduction_op: Operation for reduction ('mean', 'sum', 'max', 'min')
        """
        self.reduction_op = reduction_op
        self.helper = DistributedAnalysisHelper()
        self.metrics: Dict[str, List[float]] = {}

    def update(self, metrics: Dict[str, Any]):
        """
        Update metrics with new values.

        Args:
            metrics: Dictionary of metric names to values
        """
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()

            if key not in self.metrics:
                self.metrics[key] = []

            self.metrics[key].append(value)

    def compute(self, reset: bool = True) -> Dict[str, float]:
        """
        Compute aggregated metrics across all processes.

        Args:
            reset: Whether to reset metrics after computing

        Returns:
            Dictionary of averaged metrics
        """
        # Compute local averages
        local_metrics = {}
        for key, values in self.metrics.items():
            if values:
                local_metrics[key] = sum(values) / len(values)
            else:
                local_metrics[key] = 0.0

        # Reduce across processes
        aggregated = self.helper.reduce_dict(local_metrics, op=self.reduction_op)

        # Reset if requested
        if reset:
            self.reset()

        return aggregated

    def reset(self):
        """Reset all collected metrics."""
        self.metrics.clear()

    def get_local_metrics(self) -> Dict[str, float]:
        """Get metrics for this process only (without reduction)."""
        local_metrics = {}
        for key, values in self.metrics.items():
            if values:
                local_metrics[key] = sum(values) / len(values)
            else:
                local_metrics[key] = 0.0
        return local_metrics
