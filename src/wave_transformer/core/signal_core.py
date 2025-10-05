import dataclasses
from typing import List, Union, Optional, Dict, Callable, Any

import torch

from .normalization import NormalizationSpec, identity_norm


@dataclasses.dataclass
class SignalConfig:
    """
    Configuration for a single signal dimension.

    Attributes:
        signal_name: Descriptive name for the signal
        torch_activation_function: Activation function (callable or nn.Module)
        normalization: Normalization specification applied AFTER activation
        num_dimensions: Number of dimensions for this signal
    """
    signal_name: str
    torch_activation_function: Callable
    normalization: Optional[NormalizationSpec] = None
    num_dimensions: int = 32

    def __post_init__(self):
        """Set default normalization if not provided."""
        if self.normalization is None:
            self.normalization = identity_norm()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary.

        Note: torch_activation_function is saved as string name for common activations,
        or marked as 'custom' for non-standard activations.
        """
        # Try to get activation name
        activation_name = self._get_activation_name()

        return {
            'signal_name': self.signal_name,
            'activation_function': activation_name,
            'normalization': self.normalization.to_dict(),
            'num_dimensions': self.num_dimensions
        }

    def _get_activation_name(self) -> str:
        """Get string name for activation function."""
        if hasattr(self.torch_activation_function, '__name__'):
            return self.torch_activation_function.__name__
        elif hasattr(self.torch_activation_function, '__class__'):
            return self.torch_activation_function.__class__.__name__
        else:
            return 'custom'

    @classmethod
    def from_dict(
            cls,
            config: Dict[str, Any],
            activation_registry: Optional[Dict[str, Callable]] = None,
            custom_norm_registry: Optional[Dict[str, Callable]] = None
    ) -> 'SignalConfig':
        """
        Deserialize from dictionary.

        Args:
            config: Dictionary configuration
            activation_registry: Optional registry mapping names to activation functions
            custom_norm_registry: Optional registry for custom normalization functions
        """
        # Get activation function
        activation_name = config['activation_function']
        activation_fn = cls._resolve_activation(activation_name, activation_registry)

        # Get normalization
        norm_config = config.get('normalization')
        normalization = NormalizationSpec.from_dict(
            norm_config,
            custom_fn_registry=custom_norm_registry
        ) if norm_config else identity_norm()

        return cls(
            signal_name=config['signal_name'],
            torch_activation_function=activation_fn,
            normalization=normalization,
            num_dimensions=config.get('num_dimensions', 32)
        )

    @staticmethod
    def _resolve_activation(name: str, registry: Optional[Dict[str, Callable]] = None):
        """Resolve activation function from name."""
        import torch.nn as nn

        # Default activations
        defaults = {
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'relu': torch.relu,
            'gelu': nn.functional.gelu,
            'silu': nn.functional.silu,
            'softplus': nn.functional.softplus,
            'identity': lambda x: x,
        }

        if registry and name in registry:
            return registry[name]
        elif name in defaults:
            return defaults[name]
        else:
            raise ValueError(f"Unknown activation '{name}'. Provide activation_registry.")


@dataclasses.dataclass
class MultiSignal:
    """
    Multi-signal representation where each signal has its own learned dimensions.

    Supports variable dimensions per signal.
    Shape: [batch_size, seq_len, sum(dimensions)]
    """
    num_signals: int = 3
    num_dimensions: Union[int, List[int]] = 32  # int for uniform, List for variable
    representation_data: torch.Tensor = None

    def __post_init__(self):
        if self.representation_data is not None:
            expected_last_dim = self.total_dimensions
            actual_last_dim = self.representation_data.shape[-1]
            assert actual_last_dim == expected_last_dim, (
                f"Expected last dimension {expected_last_dim}, got {actual_last_dim}"
            )

    @property
    def dimensions_list(self) -> List[int]:
        """Get dimensions as list (handles both int and List[int])."""
        if isinstance(self.num_dimensions, int):
            return [self.num_dimensions] * self.num_signals
        return self.num_dimensions

    @property
    def total_dimensions(self) -> int:
        """Total flattened dimension."""
        if isinstance(self.num_dimensions, int):
            return self.num_signals * self.num_dimensions
        return sum(self.num_dimensions)

    @property
    def shape(self) -> torch.Size:
        return self.representation_data.shape if self.representation_data is not None else None

    @property
    def device(self) -> torch.device:
        return self.representation_data.device if self.representation_data is not None else None

    def get_signal_data(self, signal_idx: int) -> torch.Tensor:
        """
        Extract a specific signal's data.

        Args:
            signal_idx: Index of signal to extract (0 to num_signals-1)

        Returns:
            Tensor of shape [batch_size, seq_len, num_dimensions]
        """
        if self.representation_data is None:
            raise ValueError("representation_data is None")

        if not 0 <= signal_idx < self.num_signals:
            raise ValueError(f"signal_idx {signal_idx} out of range [0, {self.num_signals})")

        dims_list = self.dimensions_list
        start_idx = sum(dims_list[:signal_idx])
        end_idx = start_idx + dims_list[signal_idx]

        return self.representation_data[..., start_idx:end_idx]

    def get_all_signals(self) -> List[torch.Tensor]:
        """
        Split representation into list of individual signals.

        Returns:
            List of tensors, each of shape [batch_size, seq_len, num_dimensions]
        """
        if self.representation_data is None:
            raise ValueError("representation_data is None")

        return [
            self.get_signal_data(i)
            for i in range(self.num_signals)
        ]

    def to_flat(self) -> torch.Tensor:
        """Return the full concatenated representation."""
        return self.representation_data

    @classmethod
    def from_signals(cls, signals: List[torch.Tensor]) -> 'MultiSignal':
        """
        Construct MultiSignal from list of signal tensors.

        Args:
            signals: List of tensors, each [batch_size, seq_len, signal_dim]
                     Can have variable dimensions per signal.
        """
        num_signals = len(signals)

        # Get dimensions of each signal
        signal_dimensions = [signal.shape[-1] for signal in signals]

        # Check if all dimensions are the same (uniform case)
        if len(set(signal_dimensions)) == 1:
            num_dimensions = signal_dimensions[0]
        else:
            # Variable dimensions
            num_dimensions = signal_dimensions

        representation_data = torch.cat(signals, dim=-1)

        return cls(
            num_signals=num_signals,
            num_dimensions=num_dimensions,
            representation_data=representation_data
        )

    def __repr__(self) -> str:
        shape_str = str(self.shape) if self.shape is not None else "None"
        return (
            f"MultiSignal(num_signals={self.num_signals}, "
            f"num_dimensions={self.num_dimensions}, "
            f"shape={shape_str})"
        )


