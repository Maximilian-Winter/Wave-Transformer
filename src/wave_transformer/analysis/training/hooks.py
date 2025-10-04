"""
PyTorch hooks for capturing wave representations, gradients, and attention patterns during training.

This module provides three main hook classes:
- WaveForwardHook: Captures wave representations during forward pass
- WaveGradientHook: Captures gradients flowing through wave components
- AttentionHook: Captures attention patterns (when not using flash attention)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

from wave_transformer.core.wave import Wave


@dataclass
class HookStorage:
    """Storage for captured hook data with minimal memory overhead."""

    # Forward pass data
    forward_outputs: Dict[str, List[torch.Tensor]] = field(default_factory=lambda: defaultdict(list))
    wave_outputs: Dict[str, List[Wave]] = field(default_factory=lambda: defaultdict(list))

    # Gradient data
    gradients: Dict[str, List[torch.Tensor]] = field(default_factory=lambda: defaultdict(list))
    wave_gradients: Dict[str, List[Dict[str, torch.Tensor]]] = field(default_factory=lambda: defaultdict(list))

    # Attention data
    attention_weights: Dict[str, List[torch.Tensor]] = field(default_factory=lambda: defaultdict(list))

    def clear(self):
        """Clear all stored data."""
        self.forward_outputs.clear()
        self.wave_outputs.clear()
        self.gradients.clear()
        self.wave_gradients.clear()
        self.attention_weights.clear()

    def get_memory_usage_mb(self) -> float:
        """Estimate memory usage in megabytes."""
        total_bytes = 0

        # Forward outputs
        for tensors in self.forward_outputs.values():
            for t in tensors:
                if isinstance(t, torch.Tensor):
                    total_bytes += t.element_size() * t.nelement()

        # Wave outputs (frequencies, amplitudes, phases)
        for waves in self.wave_outputs.values():
            for w in waves:
                total_bytes += w.frequencies.element_size() * w.frequencies.nelement()
                total_bytes += w.amplitudes.element_size() * w.amplitudes.nelement()
                total_bytes += w.phases.element_size() * w.phases.nelement()

        # Gradients
        for grads in self.gradients.values():
            for g in grads:
                if isinstance(g, torch.Tensor):
                    total_bytes += g.element_size() * g.nelement()

        # Wave gradients
        for wave_grads in self.wave_gradients.values():
            for grad_dict in wave_grads:
                for g in grad_dict.values():
                    if isinstance(g, torch.Tensor):
                        total_bytes += g.element_size() * g.nelement()

        # Attention weights
        for attn in self.attention_weights.values():
            for a in attn:
                if isinstance(a, torch.Tensor):
                    total_bytes += a.element_size() * a.nelement()

        return total_bytes / (1024 ** 2)  # Convert to MB


class WaveForwardHook:
    """
    Hook for capturing wave representations during forward pass.

    Handles both Wave objects and tensor representations, converting tensors
    back to Wave objects when needed.

    Example usage:
        storage = HookStorage()
        hook = WaveForwardHook(storage=storage, name="encoder_layer_0")
        handle = module.register_forward_hook(hook)
        # ... training ...
        handle.remove()  # Clean up
    """

    def __init__(
        self,
        storage: HookStorage,
        name: str,
        capture_input: bool = False,
        capture_output: bool = True,
        detach: bool = True,
        to_cpu: bool = False,
        num_harmonics: Optional[int] = None,
    ):
        """
        Args:
            storage: HookStorage instance to store captured data
            name: Identifier for this hook (e.g., "encoder_layer_0")
            capture_input: Whether to capture input to the module
            capture_output: Whether to capture output from the module
            detach: Whether to detach tensors from computation graph
            to_cpu: Whether to move captured tensors to CPU (saves GPU memory)
            num_harmonics: Number of harmonics per wave component (for tensor->Wave conversion)
        """
        self.storage = storage
        self.name = name
        self.capture_input = capture_input
        self.capture_output = capture_output
        self.detach = detach
        self.to_cpu = to_cpu
        self.num_harmonics = num_harmonics

    def __call__(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: torch.Tensor
    ) -> None:
        """Hook function called during forward pass."""

        if self.capture_input:
            self._process_data(input[0] if isinstance(input, tuple) else input, is_input=True)

        if self.capture_output:
            self._process_data(output, is_input=False)

    def _process_data(self, data: Any, is_input: bool):
        """Process and store data (either tensor or Wave object)."""
        prefix = f"{self.name}_input" if is_input else f"{self.name}_output"

        # Handle Wave objects directly
        if isinstance(data, Wave):
            wave = data
            if self.detach:
                wave = Wave(
                    frequencies=wave.frequencies.detach(),
                    amplitudes=wave.amplitudes.detach(),
                    phases=wave.phases.detach()
                )
            if self.to_cpu:
                wave = Wave(
                    frequencies=wave.frequencies.cpu(),
                    amplitudes=wave.amplitudes.cpu(),
                    phases=wave.phases.cpu()
                )
            self.storage.wave_outputs[prefix].append(wave)

        # Handle tensor representations
        elif isinstance(data, torch.Tensor):
            tensor = data
            if self.detach:
                tensor = tensor.detach()
            if self.to_cpu:
                tensor = tensor.cpu()

            # Try to convert to Wave if we know the number of harmonics
            if self.num_harmonics is not None and tensor.shape[-1] == 3 * self.num_harmonics:
                try:
                    wave = Wave.from_representation(tensor)
                    self.storage.wave_outputs[prefix].append(wave)
                except Exception:
                    # If conversion fails, store as tensor
                    self.storage.forward_outputs[prefix].append(tensor)
            else:
                self.storage.forward_outputs[prefix].append(tensor)

        # Handle tuple outputs (e.g., from attention mechanisms)
        elif isinstance(data, tuple):
            for i, item in enumerate(data):
                self._process_data(item, is_input=is_input)


class WaveGradientHook:
    """
    Hook for capturing gradients flowing through wave components.

    This hook captures gradients during backward pass and can separate them
    into frequency, amplitude, and phase components.

    Example usage:
        storage = HookStorage()
        hook = WaveGradientHook(storage=storage, name="encoder_layer_0")
        tensor.register_hook(hook)
    """

    def __init__(
        self,
        storage: HookStorage,
        name: str,
        separate_components: bool = True,
        compute_norms: bool = True,
        detach: bool = True,
        to_cpu: bool = True,
        num_harmonics: Optional[int] = None,
    ):
        """
        Args:
            storage: HookStorage instance to store captured gradients
            name: Identifier for this hook
            separate_components: Whether to separate freq/amp/phase gradients
            compute_norms: Whether to compute gradient norms
            detach: Whether to detach gradients (required for storage)
            to_cpu: Whether to move to CPU
            num_harmonics: Number of harmonics (for component separation)
        """
        self.storage = storage
        self.name = name
        self.separate_components = separate_components
        self.compute_norms = compute_norms
        self.detach = detach
        self.to_cpu = to_cpu
        self.num_harmonics = num_harmonics

        # Store computed statistics
        self.gradient_norms: List[float] = []
        self.component_norms: Dict[str, List[float]] = defaultdict(list)

    def __call__(self, grad: torch.Tensor) -> None:
        """Hook function called during backward pass."""
        if grad is None:
            return

        # Process gradient
        processed_grad = grad
        if self.detach:
            processed_grad = processed_grad.detach()
        if self.to_cpu:
            processed_grad = processed_grad.cpu()

        # Store full gradient
        self.storage.gradients[self.name].append(processed_grad)

        # Compute gradient norm
        if self.compute_norms:
            norm = processed_grad.norm().item()
            self.gradient_norms.append(norm)

        # Separate into wave components if requested
        if self.separate_components and self.num_harmonics is not None:
            if processed_grad.shape[-1] == 3 * self.num_harmonics:
                try:
                    # Split into freq, amp, phase gradients
                    chunks = processed_grad.chunk(3, dim=-1)
                    component_grads = {
                        'frequencies': chunks[0],
                        'amplitudes': chunks[1],
                        'phases': chunks[2]
                    }

                    self.storage.wave_gradients[self.name].append(component_grads)

                    # Compute component-wise norms
                    if self.compute_norms:
                        for comp_name, comp_grad in component_grads.items():
                            norm = comp_grad.norm().item()
                            self.component_norms[comp_name].append(norm)

                except Exception as e:
                    # If separation fails, just store the full gradient
                    pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get gradient flow statistics."""
        stats = {
            'mean_grad_norm': sum(self.gradient_norms) / len(self.gradient_norms) if self.gradient_norms else 0.0,
            'max_grad_norm': max(self.gradient_norms) if self.gradient_norms else 0.0,
            'min_grad_norm': min(self.gradient_norms) if self.gradient_norms else 0.0,
        }

        if self.component_norms:
            stats['component_norms'] = {}
            for comp_name, norms in self.component_norms.items():
                stats['component_norms'][comp_name] = {
                    'mean': sum(norms) / len(norms) if norms else 0.0,
                    'max': max(norms) if norms else 0.0,
                    'min': min(norms) if norms else 0.0,
                }

        return stats


class AttentionHook:
    """
    Hook for capturing attention patterns (when not using flash attention).

    Captures attention weights to analyze which positions the model focuses on.
    Note: This is incompatible with Flash Attention which doesn't materialize
    attention weights.

    Example usage:
        storage = HookStorage()
        hook = AttentionHook(storage=storage, name="layer_0_self_attn")
        handle = attention_module.register_forward_hook(hook)
    """

    def __init__(
        self,
        storage: HookStorage,
        name: str,
        capture_pattern: bool = True,
        compute_entropy: bool = True,
        sample_heads: Optional[List[int]] = None,
        detach: bool = True,
        to_cpu: bool = True,
    ):
        """
        Args:
            storage: HookStorage instance
            name: Identifier for this hook
            capture_pattern: Whether to capture full attention pattern
            compute_entropy: Whether to compute attention entropy
            sample_heads: List of head indices to sample (None = all heads)
            detach: Whether to detach from computation graph
            to_cpu: Whether to move to CPU
        """
        self.storage = storage
        self.name = name
        self.capture_pattern = capture_pattern
        self.compute_entropy = compute_entropy
        self.sample_heads = sample_heads
        self.detach = detach
        self.to_cpu = to_cpu

        # Store computed statistics
        self.attention_entropies: List[float] = []
        self.head_entropies: Dict[int, List[float]] = defaultdict(list)

    def __call__(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: Any
    ) -> None:
        """Hook function called during forward pass."""

        # Try to extract attention weights from output
        # Common patterns: (output, attn_weights) or just output
        attn_weights = None

        if isinstance(output, tuple) and len(output) >= 2:
            # Assume second element is attention weights
            potential_attn = output[1]
            if isinstance(potential_attn, torch.Tensor):
                # Check if it looks like attention weights: should be [B, H, S, S] or [B, S, S]
                if len(potential_attn.shape) >= 3:
                    attn_weights = potential_attn

        if attn_weights is None:
            return

        # Process attention weights
        processed_attn = attn_weights
        if self.detach:
            processed_attn = processed_attn.detach()

        # Sample specific heads if requested
        if self.sample_heads is not None and len(processed_attn.shape) == 4:
            # Shape: [B, num_heads, S, S]
            processed_attn = processed_attn[:, self.sample_heads, :, :]

        if self.to_cpu:
            processed_attn = processed_attn.cpu()

        # Store attention pattern
        if self.capture_pattern:
            self.storage.attention_weights[self.name].append(processed_attn)

        # Compute attention entropy
        if self.compute_entropy:
            entropy = self._compute_attention_entropy(processed_attn)
            self.attention_entropies.append(entropy)

            # Per-head entropy if multi-head
            if len(processed_attn.shape) == 4:
                for head_idx in range(processed_attn.shape[1]):
                    head_attn = processed_attn[:, head_idx, :, :]
                    head_entropy = self._compute_attention_entropy(head_attn)
                    self.head_entropies[head_idx].append(head_entropy)

    @staticmethod
    def _compute_attention_entropy(attn_weights: torch.Tensor) -> float:
        """
        Compute Shannon entropy of attention distribution.

        Higher entropy = more uniform attention (less focused)
        Lower entropy = more peaked attention (more focused)
        """
        # Add small epsilon for numerical stability
        eps = 1e-10
        attn_weights = attn_weights + eps

        # Compute entropy: -sum(p * log(p))
        entropy = -(attn_weights * torch.log(attn_weights)).sum(dim=-1).mean()
        return entropy.item()

    def get_statistics(self) -> Dict[str, Any]:
        """Get attention statistics."""
        stats = {
            'mean_entropy': sum(self.attention_entropies) / len(self.attention_entropies) if self.attention_entropies else 0.0,
            'max_entropy': max(self.attention_entropies) if self.attention_entropies else 0.0,
            'min_entropy': min(self.attention_entropies) if self.attention_entropies else 0.0,
        }

        if self.head_entropies:
            stats['head_entropies'] = {}
            for head_idx, entropies in self.head_entropies.items():
                stats['head_entropies'][head_idx] = {
                    'mean': sum(entropies) / len(entropies) if entropies else 0.0,
                    'max': max(entropies) if entropies else 0.0,
                    'min': min(entropies) if entropies else 0.0,
                }

        return stats


class HookManager:
    """
    Manages multiple hooks and their lifecycle.

    Provides utilities for registering, removing, and managing hooks across
    multiple modules.

    Example usage:
        manager = HookManager()
        manager.register_forward_hook(module, "encoder", hook_fn)
        # ... training ...
        manager.remove_all_hooks()
    """

    def __init__(self):
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self.storage = HookStorage()

    def register_wave_forward_hook(
        self,
        module: nn.Module,
        name: str,
        **kwargs
    ) -> WaveForwardHook:
        """Register a WaveForwardHook on a module."""
        hook = WaveForwardHook(storage=self.storage, name=name, **kwargs)
        handle = module.register_forward_hook(hook)
        self.hook_handles.append(handle)
        return hook

    def register_wave_gradient_hook(
        self,
        tensor: torch.Tensor,
        name: str,
        **kwargs
    ) -> WaveGradientHook:
        """Register a WaveGradientHook on a tensor."""
        hook = WaveGradientHook(storage=self.storage, name=name, **kwargs)
        handle = tensor.register_hook(hook)
        if handle is not None:
            self.hook_handles.append(handle)
        return hook

    def register_attention_hook(
        self,
        module: nn.Module,
        name: str,
        **kwargs
    ) -> AttentionHook:
        """Register an AttentionHook on a module."""
        hook = AttentionHook(storage=self.storage, name=name, **kwargs)
        handle = module.register_forward_hook(hook)
        self.hook_handles.append(handle)
        return hook

    def remove_all_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def clear_storage(self):
        """Clear all stored data."""
        self.storage.clear()

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.storage.get_memory_usage_mb()

    def __del__(self):
        """Cleanup on deletion."""
        self.remove_all_hooks()
