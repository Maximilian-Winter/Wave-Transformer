"""
Frequency Curriculum Learning (FCL)

Implements progressive frequency unmasking for Wave Transformer training.
The curriculum gradually introduces higher frequencies during training, allowing
the model to first learn low-frequency patterns before tackling complex high-frequency details.

Key Components:
- FrequencyCurriculumScheduler: Manages curriculum progression
- FrequencyMask: Applies frequency-based masking to wave representations
- Adaptive scheduling based on validation loss
- Smooth sigmoid transitions between frequency bands

References:
- "Curriculum Learning" (Bengio et al., 2009)
- "Self-Paced Learning for Latent Variable Models" (Kumar et al., 2010)
- "On The Power of Curriculum Learning in Training Deep Networks" (Hacohen & Weinshall, 2019)
"""

import math
from typing import Optional, Tuple, Dict, List
import warnings

import torch
import torch.nn as nn


class FrequencyMask(nn.Module):
    """
    Applies smooth frequency-based masking to wave representations.

    Uses a sigmoid-based transition to smoothly interpolate between masked
    and unmasked frequencies, avoiding sharp cutoffs that could harm training.

    Args:
        num_harmonics: Number of harmonics in wave representation
        mask_slope: Slope of sigmoid transition (higher = sharper cutoff)

    Shape:
        - Input: (B, S, 3*H) - wave representation [freqs, amps, phases]
        - Output: (B, S, 3*H) - masked wave representation

    Example:
        >>> mask = FrequencyMask(num_harmonics=64, mask_slope=10.0)
        >>> wave_repr = torch.randn(2, 128, 192)
        >>> masked = mask(wave_repr, freq_limit=0.5)  # Mask top 50% of frequencies
    """

    def __init__(
        self,
        num_harmonics: int,
        mask_slope: float = 10.0,
    ):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.mask_slope = mask_slope

        # Create harmonic indices for masking
        self.register_buffer(
            'harmonic_indices',
            torch.arange(num_harmonics, dtype=torch.float32)
        )

    def create_smooth_mask(
        self,
        freq_limit: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create smooth frequency mask using sigmoid transition.

        Args:
            freq_limit: Fraction of frequencies to keep (0.0 to 1.0)
            device: Device to create mask on

        Returns:
            mask: (H,) - smooth mask values in [0, 1]
        """
        # Cutoff point in harmonic space
        cutoff = freq_limit * self.num_harmonics

        # Normalized distance from cutoff
        indices = self.harmonic_indices.to(device)
        distance = (indices - cutoff) / self.num_harmonics

        # Sigmoid transition: 1 for low freqs, 0 for high freqs
        mask = torch.sigmoid(-self.mask_slope * distance)

        return mask

    def forward(
        self,
        wave_repr: torch.Tensor,
        freq_limit: float,
    ) -> torch.Tensor:
        """
        Apply frequency masking to wave representation.

        Args:
            wave_repr: (B, S, 3*H) - wave representation
            freq_limit: Fraction of frequencies to keep (0.0 to 1.0)

        Returns:
            masked_wave: (B, S, 3*H) - frequency-masked representation
        """
        if freq_limit >= 1.0:
            # No masking needed
            return wave_repr

        # Create smooth mask
        mask = self.create_smooth_mask(freq_limit, wave_repr.device)  # (H,)

        # Split wave representation
        freqs, amps, phases = wave_repr.chunk(3, dim=-1)  # Each (B, S, H)

        # Apply mask to each component
        # Note: We primarily mask amplitudes, but also apply to freqs/phases for consistency
        freqs_masked = freqs * mask.unsqueeze(0).unsqueeze(0)
        amps_masked = amps * mask.unsqueeze(0).unsqueeze(0)
        phases_masked = phases * mask.unsqueeze(0).unsqueeze(0)

        # Recombine
        masked_wave = torch.cat([freqs_masked, amps_masked, phases_masked], dim=-1)

        return masked_wave


class FrequencyCurriculumScheduler:
    """
    Manages progressive frequency curriculum during training.

    The scheduler gradually increases the fraction of frequencies available
    to the model, starting with low frequencies and progressively adding
    higher frequencies. Supports both fixed and adaptive scheduling.

    Args:
        total_steps: Total training steps
        start_freq_limit: Initial frequency limit (default: 0.1 = 10% of frequencies)
        end_freq_limit: Final frequency limit (default: 1.0 = all frequencies)
        warmup_steps: Steps to reach start_freq_limit from 0
        curriculum_mode: 'linear', 'exponential', or 'cosine'
        adaptive: Whether to use adaptive scheduling based on validation loss
        patience: Steps to wait before increasing frequency limit (adaptive mode)

    Example:
        >>> scheduler = FrequencyCurriculumScheduler(
        ...     total_steps=10000,
        ...     start_freq_limit=0.1,
        ...     end_freq_limit=1.0,
        ...     curriculum_mode='cosine',
        ... )
        >>> for step in range(10000):
        ...     freq_limit = scheduler.get_freq_limit(step)
        ...     # Use freq_limit in training
        ...     if step % 100 == 0:
        ...         val_loss = validate()
        ...         scheduler.step(step, val_loss)
    """

    def __init__(
        self,
        total_steps: int,
        start_freq_limit: float = 0.1,
        end_freq_limit: float = 1.0,
        warmup_steps: int = 0,
        curriculum_mode: str = 'cosine',
        adaptive: bool = True,
        patience: int = 500,
    ):
        assert 0.0 < start_freq_limit < end_freq_limit <= 1.0, \
            "Must have 0 < start_freq_limit < end_freq_limit <= 1.0"
        assert curriculum_mode in ['linear', 'exponential', 'cosine'], \
            "curriculum_mode must be 'linear', 'exponential', or 'cosine'"

        self.total_steps = total_steps
        self.start_freq_limit = start_freq_limit
        self.end_freq_limit = end_freq_limit
        self.warmup_steps = warmup_steps
        self.curriculum_mode = curriculum_mode
        self.adaptive = adaptive
        self.patience = patience

        # State tracking
        self.current_step = 0
        self.current_freq_limit = 0.0 if warmup_steps > 0 else start_freq_limit
        self.best_val_loss = float('inf')
        self.steps_since_improvement = 0

        # History for logging
        self.freq_limit_history: List[Tuple[int, float]] = []

    def _compute_base_schedule(self, step: int) -> float:
        """
        Compute frequency limit based on step and curriculum mode.

        Args:
            step: Current training step

        Returns:
            freq_limit: Frequency limit for this step
        """
        # Handle warmup phase
        if step < self.warmup_steps:
            return (step / self.warmup_steps) * self.start_freq_limit

        # Normalize step to [0, 1] in curriculum phase
        curriculum_step = step - self.warmup_steps
        curriculum_total = self.total_steps - self.warmup_steps
        progress = min(1.0, curriculum_step / max(1, curriculum_total))

        # Compute frequency limit based on mode
        freq_range = self.end_freq_limit - self.start_freq_limit

        if self.curriculum_mode == 'linear':
            # Linear interpolation
            freq_limit = self.start_freq_limit + progress * freq_range

        elif self.curriculum_mode == 'exponential':
            # Exponential growth (slow start, fast end)
            freq_limit = self.start_freq_limit + (progress ** 2) * freq_range

        elif self.curriculum_mode == 'cosine':
            # Cosine annealing (smooth S-curve)
            cosine_progress = (1 - math.cos(progress * math.pi)) / 2
            freq_limit = self.start_freq_limit + cosine_progress * freq_range

        else:
            raise ValueError(f"Unknown curriculum_mode: {self.curriculum_mode}")

        return min(freq_limit, self.end_freq_limit)

    def get_freq_limit(self, step: Optional[int] = None) -> float:
        """
        Get current frequency limit.

        Args:
            step: Optional step override (uses internal counter if None)

        Returns:
            freq_limit: Current frequency limit (0.0 to 1.0)
        """
        if step is not None:
            self.current_step = step

        freq_limit = self._compute_base_schedule(self.current_step)
        self.current_freq_limit = freq_limit

        return freq_limit

    def step(
        self,
        step: int,
        val_loss: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Update scheduler state and optionally adapt based on validation loss.

        Args:
            step: Current training step
            val_loss: Optional validation loss for adaptive scheduling

        Returns:
            info: Dictionary with scheduling information
        """
        self.current_step = step

        # Get base frequency limit
        freq_limit = self._compute_base_schedule(step)

        # Adaptive adjustment based on validation loss
        if self.adaptive and val_loss is not None:
            if val_loss < self.best_val_loss:
                # Improvement: reset counter
                self.best_val_loss = val_loss
                self.steps_since_improvement = 0
            else:
                # No improvement: increment counter
                self.steps_since_improvement += 1

            # If stagnant, slow down curriculum
            if self.steps_since_improvement > self.patience:
                # Reduce frequency limit slightly to give model more time
                freq_limit *= 0.95
                freq_limit = max(self.start_freq_limit, freq_limit)

                warnings.warn(
                    f"Step {step}: No validation improvement for {self.patience} steps. "
                    f"Reducing freq_limit to {freq_limit:.3f}"
                )

        # Update current limit
        self.current_freq_limit = freq_limit

        # Record in history
        self.freq_limit_history.append((step, freq_limit))

        # Prepare info dict
        info = {
            'step': step,
            'freq_limit': freq_limit,
            'curriculum_progress': min(1.0, step / self.total_steps),
            'steps_since_improvement': self.steps_since_improvement,
        }

        if val_loss is not None:
            info['val_loss'] = val_loss
            info['best_val_loss'] = self.best_val_loss

        return info

    def state_dict(self) -> Dict:
        """Get scheduler state for checkpointing."""
        return {
            'current_step': self.current_step,
            'current_freq_limit': self.current_freq_limit,
            'best_val_loss': self.best_val_loss,
            'steps_since_improvement': self.steps_since_improvement,
            'freq_limit_history': self.freq_limit_history,
        }

    def load_state_dict(self, state_dict: Dict):
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict['current_step']
        self.current_freq_limit = state_dict['current_freq_limit']
        self.best_val_loss = state_dict['best_val_loss']
        self.steps_since_improvement = state_dict['steps_since_improvement']
        self.freq_limit_history = state_dict['freq_limit_history']

    def plot_schedule(self, save_path: Optional[str] = None):
        """
        Visualize the frequency curriculum schedule.

        Args:
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not available, cannot plot schedule")
            return

        # Generate schedule for all steps
        steps = list(range(0, self.total_steps, max(1, self.total_steps // 1000)))
        freq_limits = [self._compute_base_schedule(s) for s in steps]

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(steps, freq_limits, linewidth=2, label='Frequency Limit')
        plt.axhline(y=self.start_freq_limit, color='r', linestyle='--',
                    alpha=0.5, label=f'Start ({self.start_freq_limit:.2f})')
        plt.axhline(y=self.end_freq_limit, color='g', linestyle='--',
                    alpha=0.5, label=f'End ({self.end_freq_limit:.2f})')

        if self.warmup_steps > 0:
            plt.axvline(x=self.warmup_steps, color='orange', linestyle='--',
                        alpha=0.5, label=f'Warmup End ({self.warmup_steps})')

        plt.xlabel('Training Step')
        plt.ylabel('Frequency Limit (fraction)')
        plt.title(f'Frequency Curriculum Schedule ({self.curriculum_mode})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


# ==================== INTEGRATION EXAMPLE ====================

def example_usage():
    """
    Example of integrating FCL into WaveTransformer training.
    """
    # Setup
    batch_size = 4
    seq_len = 128
    num_harmonics = 64
    d_model = 3 * num_harmonics

    total_steps = 10000

    # Create scheduler
    scheduler = FrequencyCurriculumScheduler(
        total_steps=total_steps,
        start_freq_limit=0.1,  # Start with 10% of frequencies
        end_freq_limit=1.0,    # End with all frequencies
        warmup_steps=500,      # 500 step warmup
        curriculum_mode='cosine',
        adaptive=True,
        patience=200,
    )

    # Create frequency mask
    freq_mask = FrequencyMask(
        num_harmonics=num_harmonics,
        mask_slope=10.0,
    )

    # Simulate training loop
    print("Simulating training with frequency curriculum...\n")

    for step in range(0, total_steps, 1000):
        # Get current frequency limit
        freq_limit = scheduler.get_freq_limit(step)

        # Simulate wave representation
        wave_repr = torch.randn(batch_size, seq_len, d_model)

        # Apply frequency masking
        masked_wave = freq_mask(wave_repr, freq_limit)

        # Simulate validation
        if step % 1000 == 0:
            # Simulate decreasing validation loss
            val_loss = 2.0 - (step / total_steps) * 1.5 + torch.rand(1).item() * 0.1

            # Update scheduler with validation loss
            info = scheduler.step(step, val_loss)

            print(f"Step {step:5d}: freq_limit={info['freq_limit']:.3f}, "
                  f"val_loss={val_loss:.3f}, progress={info['curriculum_progress']:.1%}")

            # Check masking effect
            original_energy = (wave_repr ** 2).sum()
            masked_energy = (masked_wave ** 2).sum()
            reduction = 1.0 - (masked_energy / original_energy)
            print(f"           Energy reduction: {reduction:.1%}\n")

    # Plot the schedule
    print("\nGenerating schedule visualization...")
    scheduler.plot_schedule(save_path='frequency_curriculum_schedule.png')
    print("Saved to: frequency_curriculum_schedule.png")

    return scheduler, freq_mask


def training_callback_example():
    """
    Example of using FCL in a PyTorch training loop.
    """
    print("\n" + "="*60)
    print("TRAINING CALLBACK EXAMPLE")
    print("="*60 + "\n")

    # Pseudocode for integration
    code = '''
# In your training script:

from wave_transformer.enhancements import (
    FrequencyCurriculumScheduler,
    FrequencyMask,
)

# Initialize scheduler
curriculum_scheduler = FrequencyCurriculumScheduler(
    total_steps=num_epochs * steps_per_epoch,
    start_freq_limit=0.1,
    end_freq_limit=1.0,
    curriculum_mode='cosine',
    adaptive=True,
)

# Initialize frequency mask
freq_mask = FrequencyMask(
    num_harmonics=config.num_harmonics,
    mask_slope=10.0,
)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_loader):
        step = epoch * len(train_loader) + batch_idx

        # Get current frequency limit
        freq_limit = curriculum_scheduler.get_freq_limit(step)

        # Encode to wave representation
        wave_repr = wave_encoder(**batch)
        wave_tensor = wave_repr.to_representation()

        # Apply frequency curriculum masking
        masked_wave = freq_mask(wave_tensor, freq_limit)

        # Continue with normal forward pass
        # (replace wave_tensor with masked_wave in transformer)
        output = transformer.forward_from_wave(masked_wave, ...)

        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

    # Validation and scheduler update
    val_loss = validate(model, val_loader)
    info = curriculum_scheduler.step(
        step=epoch * len(train_loader),
        val_loss=val_loss
    )

    print(f"Epoch {epoch}: freq_limit={info['freq_limit']:.3f}, "
          f"val_loss={val_loss:.4f}")
    '''

    print(code)


if __name__ == "__main__":
    # Run examples
    example_usage()
    training_callback_example()
