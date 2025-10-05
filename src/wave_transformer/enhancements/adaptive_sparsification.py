"""
Adaptive Harmonic Sparsification (AHS)

Implements learnable top-k harmonic selection with context-aware importance scoring.
This enhancement reduces computational cost by selecting only the most important harmonics
at each position while maintaining differentiability through Gumbel-Softmax.

Key Components:
- AdaptiveHarmonicSelector: Main module for dynamic harmonic selection
- HarmonicSparsificationLoss: Regularization losses for sparsity control
- Straight-through estimators for gradient flow through discrete operations

References:
- "Adaptive Sparse Transformers" (Correia et al., 2019)
- "Gumbel-Softmax for Categorical Reparameterization" (Jang et al., 2017)
"""

import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveHarmonicSelector(nn.Module):
    """
    Adaptive Harmonic Sparsification module with learnable top-k selection.

    This module computes context-aware importance scores for each harmonic
    and selects the top-k most important ones per position. It uses Gumbel-Softmax
    for differentiable sampling during training and hard top-k during inference.

    Args:
        num_harmonics: Number of input harmonics (H)
        d_model: Model dimension (should be 3 * num_harmonics for wave representation)
        k_ratio: Ratio of harmonics to keep (0.0 to 1.0)
        temperature: Gumbel-Softmax temperature (higher = softer selection)
        min_k: Minimum number of harmonics to always keep
        use_dynamic_k: If True, k can vary per position based on complexity

    Shape:
        - Input: (B, S, 3*H) - wave representation [freqs, amps, phases]
        - Output: (B, S, 3*H) - sparsified wave representation

    Example:
        >>> selector = AdaptiveHarmonicSelector(num_harmonics=64, d_model=192, k_ratio=0.5)
        >>> wave_repr = torch.randn(2, 128, 192)  # batch=2, seq=128, dim=192
        >>> sparse_repr, stats = selector(wave_repr)
        >>> print(f"Actual sparsity: {stats['actual_k_mean']:.1f} harmonics")
    """

    def __init__(
        self,
        num_harmonics: int,
        d_model: int,
        k_ratio: float = 0.5,
        temperature: float = 1.0,
        min_k: int = 8,
        use_dynamic_k: bool = True,
    ):
        super().__init__()

        assert d_model == 3 * num_harmonics, \
            f"d_model ({d_model}) must equal 3 * num_harmonics ({3 * num_harmonics})"
        assert 0.0 < k_ratio <= 1.0, "k_ratio must be in (0.0, 1.0]"

        self.num_harmonics = num_harmonics
        self.d_model = d_model
        self.k_ratio = k_ratio
        self.temperature = temperature
        self.min_k = min(min_k, num_harmonics)
        self.use_dynamic_k = use_dynamic_k

        # Base k value
        self.base_k = max(self.min_k, int(num_harmonics * k_ratio))

        # Importance scoring network
        # Projects wave representation to importance scores
        self.importance_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, num_harmonics),
        )

        # Dynamic k predictor (if enabled)
        if use_dynamic_k:
            self.k_predictor = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.LayerNorm(d_model // 4),
                nn.SiLU(),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid(),  # Output in [0, 1]
            )

        # Statistics tracking (not learned)
        self.register_buffer('ema_k', torch.tensor(float(self.base_k)))
        self.register_buffer('ema_sparsity', torch.tensor(1.0 - k_ratio))

    def compute_importance_scores(
        self,
        wave_repr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute importance score for each harmonic based on context.

        Args:
            wave_repr: Wave representation (B, S, 3*H)

        Returns:
            importance_scores: (B, S, H)
        """
        # Get importance scores from the network
        scores = self.importance_net(wave_repr)  # (B, S, H)

        # Add amplitude bias: harmonics with higher amplitude are inherently important
        # Extract amplitudes from wave representation
        _, amps, _ = wave_repr.chunk(3, dim=-1)  # Each is (B, S, H)

        # Normalize amplitudes per position
        amp_normalized = amps / (amps.sum(dim=-1, keepdim=True) + 1e-8)

        # Combine learned scores with amplitude bias
        combined_scores = scores + amp_normalized

        return combined_scores

    def compute_dynamic_k(
        self,
        wave_repr: torch.Tensor,
        base_k: int,
    ) -> torch.Tensor:
        """
        Compute position-specific k values based on complexity.

        Args:
            wave_repr: Wave representation (B, S, 3*H)
            base_k: Base k value

        Returns:
            k_values: (B, S) - k value per position
        """
        # Predict k adjustment factor in [0, 1]
        k_factor = self.k_predictor(wave_repr).squeeze(-1)  # (B, S)

        # Scale to [min_k, num_harmonics]
        k_range = self.num_harmonics - self.min_k
        k_values = self.min_k + k_factor * k_range

        # Round to integers
        k_values = k_values.round().long()

        return k_values

    def select_topk_gumbel(
        self,
        importance_scores: torch.Tensor,
        k: int,
        temperature: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable top-k selection using Gumbel-Softmax.

        Args:
            importance_scores: (B, S, H)
            k: Number of harmonics to select
            temperature: Gumbel temperature

        Returns:
            soft_mask: (B, S, H) - soft selection mask during training
            hard_mask: (B, S, H) - hard binary mask
        """
        B, S, H = importance_scores.shape

        if self.training:
            # Add Gumbel noise for stochastic selection
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(importance_scores) + 1e-8
            ) + 1e-8)

            # Perturbed scores
            perturbed_scores = (importance_scores + gumbel_noise) / temperature

            # Soft top-k using softmax
            soft_mask = F.softmax(perturbed_scores, dim=-1)

            # Hard top-k for straight-through
            _, topk_indices = torch.topk(perturbed_scores, k, dim=-1)
            hard_mask = torch.zeros_like(importance_scores)
            hard_mask.scatter_(-1, topk_indices, 1.0)

            # Straight-through estimator: use hard mask in forward, soft mask in backward
            mask = hard_mask + (soft_mask - soft_mask.detach())

        else:
            # Inference: deterministic top-k
            _, topk_indices = torch.topk(importance_scores, k, dim=-1)
            mask = torch.zeros_like(importance_scores)
            mask.scatter_(-1, topk_indices, 1.0)
            hard_mask = mask

        return mask, hard_mask

    def apply_sparsification(
        self,
        wave_repr: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply harmonic sparsification mask to wave representation.

        Args:
            wave_repr: (B, S, 3*H)
            mask: (B, S, H) - binary selection mask

        Returns:
            sparse_wave: (B, S, 3*H) - sparsified representation
        """
        # Split into components
        freqs, amps, phases = wave_repr.chunk(3, dim=-1)  # Each (B, S, H)

        # Apply mask to each component
        freqs_sparse = freqs * mask
        amps_sparse = amps * mask
        phases_sparse = phases * mask

        # Recombine
        sparse_wave = torch.cat([freqs_sparse, amps_sparse, phases_sparse], dim=-1)

        return sparse_wave

    def forward(
        self,
        wave_repr: torch.Tensor,
        return_stats: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass with adaptive harmonic sparsification.

        Args:
            wave_repr: Wave representation (B, S, 3*H)
            return_stats: Whether to return selection statistics

        Returns:
            sparse_wave: Sparsified wave representation (B, S, 3*H)
            stats: Dictionary of statistics (if return_stats=True)
        """
        B, S, _ = wave_repr.shape

        # Compute importance scores
        importance_scores = self.compute_importance_scores(wave_repr)  # (B, S, H)

        # Determine k values
        if self.use_dynamic_k and self.training:
            # Dynamic k per position
            k_values = self.compute_dynamic_k(wave_repr, self.base_k)  # (B, S)

            # For simplicity, use mean k for the batch
            # In practice, you could process each position separately
            k_mean = k_values.float().mean().round().long().item()
            k = max(self.min_k, min(k_mean, self.num_harmonics))
        else:
            k = self.base_k
            k_values = torch.full((B, S), k, device=wave_repr.device)

        # Select top-k harmonics
        mask, hard_mask = self.select_topk_gumbel(
            importance_scores,
            k,
            self.temperature
        )

        # Apply sparsification
        sparse_wave = self.apply_sparsification(wave_repr, mask)

        # Compute statistics
        stats = None
        if return_stats:
            actual_k = hard_mask.sum(dim=-1).float().mean()
            sparsity = 1.0 - (actual_k / self.num_harmonics)

            # Update EMA statistics
            if self.training:
                momentum = 0.99
                self.ema_k.mul_(momentum).add_(actual_k * (1 - momentum))
                self.ema_sparsity.mul_(momentum).add_(sparsity * (1 - momentum))

            stats = {
                'importance_scores': importance_scores.detach(),
                'selection_mask': hard_mask.detach(),
                'actual_k_mean': actual_k.detach(),
                'sparsity': sparsity.detach(),
                'ema_k': self.ema_k.detach(),
                'ema_sparsity': self.ema_sparsity.detach(),
            }

            if self.use_dynamic_k:
                stats['k_values'] = k_values.detach()

        return sparse_wave, stats


class HarmonicSparsificationLoss(nn.Module):
    """
    Regularization losses for harmonic sparsification.

    Combines multiple loss terms to encourage:
    1. Target sparsity level
    2. Smooth selection across harmonics
    3. Temporal consistency of selection

    Args:
        target_sparsity: Target sparsity ratio (0.0 to 1.0)
        smoothness_weight: Weight for smoothness loss
        temporal_weight: Weight for temporal consistency loss

    Example:
        >>> loss_fn = HarmonicSparsificationLoss(target_sparsity=0.5, smoothness_weight=0.1)
        >>> # After getting stats from selector
        >>> reg_loss = loss_fn(stats)
        >>> total_loss = task_loss + 0.01 * reg_loss
    """

    def __init__(
        self,
        target_sparsity: float = 0.5,
        smoothness_weight: float = 0.1,
        temporal_weight: float = 0.05,
    ):
        super().__init__()
        self.target_sparsity = target_sparsity
        self.smoothness_weight = smoothness_weight
        self.temporal_weight = temporal_weight

    def sparsity_loss(
        self,
        selection_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encourage target sparsity level.

        Args:
            selection_mask: (B, S, H) - binary selection mask

        Returns:
            loss: Scalar tensor
        """
        actual_sparsity = 1.0 - selection_mask.mean()
        loss = (actual_sparsity - self.target_sparsity) ** 2
        return loss

    def smoothness_loss(
        self,
        importance_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encourage smooth importance distributions (avoid selecting scattered harmonics).

        Args:
            importance_scores: (B, S, H)

        Returns:
            loss: Scalar tensor
        """
        # Compute variance of importance scores along harmonic dimension
        # Lower variance = more peaked distribution = better
        variance = importance_scores.var(dim=-1).mean()

        # We want high variance (peaked selection), so minimize negative variance
        loss = -variance

        return loss

    def temporal_consistency_loss(
        self,
        selection_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encourage temporal consistency in harmonic selection.

        Args:
            selection_mask: (B, S, H)

        Returns:
            loss: Scalar tensor
        """
        if selection_mask.size(1) <= 1:
            return torch.tensor(0.0, device=selection_mask.device)

        # Compute differences between consecutive positions
        temporal_diff = selection_mask[:, 1:, :] - selection_mask[:, :-1, :]

        # Minimize changes (encourage consistency)
        loss = (temporal_diff ** 2).mean()

        return loss

    def forward(
        self,
        stats: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined sparsification loss.

        Args:
            stats: Statistics dictionary from AdaptiveHarmonicSelector

        Returns:
            losses: Dictionary with individual and total losses
        """
        selection_mask = stats['selection_mask']
        importance_scores = stats['importance_scores']

        # Individual losses
        l_sparsity = self.sparsity_loss(selection_mask)
        l_smoothness = self.smoothness_loss(importance_scores)
        l_temporal = self.temporal_consistency_loss(selection_mask)

        # Combined loss
        total_loss = (
            l_sparsity +
            self.smoothness_weight * l_smoothness +
            self.temporal_weight * l_temporal
        )

        return {
            'total': total_loss,
            'sparsity': l_sparsity,
            'smoothness': l_smoothness,
            'temporal': l_temporal,
        }


# ==================== INTEGRATION EXAMPLE ====================

def example_usage():
    """
    Example of integrating AHS into WaveTransformer.
    """
    # Setup
    batch_size = 4
    seq_len = 128
    num_harmonics = 64
    d_model = 3 * num_harmonics  # 192

    # Create selector
    selector = AdaptiveHarmonicSelector(
        num_harmonics=num_harmonics,
        d_model=d_model,
        k_ratio=0.5,  # Keep 50% of harmonics
        temperature=1.0,
        use_dynamic_k=True,
    )

    # Create loss function
    sparsity_loss_fn = HarmonicSparsificationLoss(
        target_sparsity=0.5,
        smoothness_weight=0.1,
        temporal_weight=0.05,
    )

    # Simulate wave representation
    wave_repr = torch.randn(batch_size, seq_len, d_model)

    # Apply sparsification
    sparse_wave, stats = selector(wave_repr, return_stats=True)

    print(f"Input shape: {wave_repr.shape}")
    print(f"Output shape: {sparse_wave.shape}")
    print(f"Actual k (mean): {stats['actual_k_mean']:.1f} / {num_harmonics}")
    print(f"Sparsity: {stats['sparsity']:.2%}")

    # Compute regularization loss
    reg_losses = sparsity_loss_fn(stats)
    print(f"\nRegularization losses:")
    for name, value in reg_losses.items():
        print(f"  {name}: {value.item():.4f}")

    # In training loop, you would combine with task loss:
    # total_loss = task_loss + 0.01 * reg_losses['total']

    return sparse_wave, stats, reg_losses


if __name__ == "__main__":
    # Run example
    example_usage()
