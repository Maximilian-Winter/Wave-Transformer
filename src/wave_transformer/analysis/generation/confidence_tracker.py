"""
Track confidence and uncertainty during generation.

This module analyzes the model's confidence at each generation step through
probability distributions, entropy, and correlations with wave properties.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import warnings

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available. Correlation analysis will use numpy fallback.")


@dataclass
class ConfidenceStats:
    """Container for confidence statistics at each generation step."""
    step: int
    max_probability: float
    entropy: float
    top_k_mass: float  # Probability mass in top-k tokens
    perplexity: float
    token_id: int


class GenerationConfidenceTracker:
    """
    Track model confidence and uncertainty during autoregressive generation.

    Monitors probability distributions, entropy, and correlates confidence
    with wave properties to understand the model's decision-making process.

    Example:
        >>> tracker = GenerationConfidenceTracker(k=10)
        >>> for step in range(max_length):
        ...     logits, wave = model(input_ids, return_encoder_outputs=True)
        ...     next_token = sample(logits)
        ...     tracker.track_step(step, logits[:, -1, :], next_token, wave)
        >>> tracker.plot_confidence_trajectory()
        >>> uncertain_tokens = tracker.identify_uncertain_regions(threshold=0.5)
    """

    def __init__(self, k: int = 10, batch_idx: int = 0):
        """
        Initialize confidence tracker.

        Args:
            k: Number of top tokens to track for top-k probability mass
            batch_idx: Which batch element to track
        """
        self.k = k
        self.batch_idx = batch_idx
        self.confidence_history: List[ConfidenceStats] = []
        self.wave_history: List = []
        self.logits_history: List[torch.Tensor] = []

    def track_step(
        self,
        step: int,
        logits: torch.Tensor,
        token_id: int,
        wave
    ) -> ConfidenceStats:
        """
        Track confidence statistics for a single generation step.

        Args:
            step: Generation step number
            logits: Logits for next token prediction [batch_size, vocab_size] or [vocab_size]
            token_id: ID of sampled/selected token
            wave: Wave object from encoder

        Returns:
            ConfidenceStats for this step
        """
        # Ensure logits is 1D
        if logits.dim() > 1:
            logits = logits[self.batch_idx]

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)

        # Max probability (confidence in top token)
        max_prob = probs.max().item()

        # Entropy: -sum(p * log(p))
        # Using log base 2 for bits, or natural log for nats
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

        # Top-k probability mass
        top_k_probs, _ = torch.topk(probs, min(self.k, probs.size(0)))
        top_k_mass = top_k_probs.sum().item()

        # Perplexity: 2^entropy or exp(entropy)
        perplexity = np.exp(entropy)

        stats = ConfidenceStats(
            step=step,
            max_probability=max_prob,
            entropy=entropy,
            top_k_mass=top_k_mass,
            perplexity=perplexity,
            token_id=token_id
        )

        self.confidence_history.append(stats)
        self.wave_history.append(wave)
        self.logits_history.append(logits.cpu())

        return stats

    @torch.no_grad()
    def track_generation(
        self,
        model: torch.nn.Module,
        initial_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, List[ConfidenceStats]]:
        """
        Track confidence during full generation process.

        Args:
            model: Wave Transformer model
            initial_ids: Starting token IDs [1, initial_len]
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            device: Device for computation

        Returns:
            Tuple of (generated_ids, confidence_stats_list)
        """
        if device is None:
            device = next(model.parameters()).device

        model.eval()
        current_ids = initial_ids.to(device)
        self.confidence_history.clear()
        self.wave_history.clear()
        self.logits_history.clear()

        for step in range(max_length):
            # Forward pass
            logits, wave = model(
                encoder_input={'token_ids': current_ids},
                return_encoder_outputs=True
            )

            # Get logits for next token
            next_logits = logits[:, -1, :] / temperature

            # Sample token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()

            # Track this step
            self.track_step(step, next_logits, token_id, wave)

            # Append token
            current_ids = torch.cat([current_ids, next_token], dim=1)

        return current_ids, self.confidence_history

    def plot_confidence_trajectory(
        self,
        figsize: Tuple[int, int] = (16, 8),
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Visualize confidence metrics over generation trajectory.

        Args:
            figsize: Figure size
            save_path: Optional path to save figure

        Returns:
            Tuple of (figure, axes)
        """
        if len(self.confidence_history) == 0:
            raise ValueError("No confidence data available. Run track_generation first.")

        steps = [s.step for s in self.confidence_history]
        max_probs = [s.max_probability for s in self.confidence_history]
        entropies = [s.entropy for s in self.confidence_history]
        top_k_masses = [s.top_k_mass for s in self.confidence_history]
        perplexities = [s.perplexity for s in self.confidence_history]

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Max probability (confidence)
        axes[0, 0].plot(steps, max_probs, linewidth=2, color='blue', marker='o', markersize=3)
        axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
        axes[0, 0].set_title('Token Confidence (Max Probability)')
        axes[0, 0].set_xlabel('Generation Step')
        axes[0, 0].set_ylabel('Probability')
        axes[0, 0].set_ylim([0, 1.05])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Entropy
        axes[0, 1].plot(steps, entropies, linewidth=2, color='green', marker='s', markersize=3)
        axes[0, 1].set_title('Prediction Entropy')
        axes[0, 1].set_xlabel('Generation Step')
        axes[0, 1].set_ylabel('Entropy (nats)')
        axes[0, 1].grid(True, alpha=0.3)

        # Top-k probability mass
        axes[1, 0].plot(steps, top_k_masses, linewidth=2, color='purple', marker='^', markersize=3)
        axes[1, 0].axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
        axes[1, 0].set_title(f'Top-{self.k} Probability Mass')
        axes[1, 0].set_xlabel('Generation Step')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].set_ylim([0, 1.05])
        axes[1, 0].grid(True, alpha=0.3)

        # Perplexity
        axes[1, 1].plot(steps, perplexities, linewidth=2, color='orange', marker='d', markersize=3)
        axes[1, 1].set_title('Perplexity')
        axes[1, 1].set_xlabel('Generation Step')
        axes[1, 1].set_ylabel('Perplexity (exp(entropy))')
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle('Generation Confidence Trajectory', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, axes

    def identify_uncertain_regions(
        self,
        threshold: float = 0.5,
        metric: str = 'max_probability'
    ) -> List[Dict]:
        """
        Identify tokens/steps where model was uncertain.

        Args:
            threshold: Threshold for uncertainty detection
                      - For 'max_probability': tokens below threshold are uncertain
                      - For 'entropy': tokens above threshold are uncertain
            metric: 'max_probability', 'entropy', or 'top_k_mass'

        Returns:
            List of dictionaries with uncertain token information
        """
        uncertain_tokens = []

        for stats in self.confidence_history:
            is_uncertain = False

            if metric == 'max_probability':
                is_uncertain = stats.max_probability < threshold
            elif metric == 'entropy':
                is_uncertain = stats.entropy > threshold
            elif metric == 'top_k_mass':
                is_uncertain = stats.top_k_mass < threshold
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if is_uncertain:
                uncertain_tokens.append({
                    'step': stats.step,
                    'token_id': stats.token_id,
                    'max_probability': stats.max_probability,
                    'entropy': stats.entropy,
                    'top_k_mass': stats.top_k_mass,
                })

        return uncertain_tokens

    def correlate_wave_confidence(
        self,
        wave_metric: str = 'energy'
    ) -> Dict:
        """
        Correlate wave properties with confidence metrics.

        Analyzes relationship between wave characteristics and model confidence
        to understand how internal representations relate to prediction certainty.

        Args:
            wave_metric: Wave property to correlate
                        ('energy', 'amplitude', 'centroid', 'entropy')

        Returns:
            Dictionary with correlation results
        """
        if len(self.confidence_history) == 0 or len(self.wave_history) == 0:
            raise ValueError("No data available for correlation")

        # Extract confidence metrics
        max_probs = np.array([s.max_probability for s in self.confidence_history])
        entropies = np.array([s.entropy for s in self.confidence_history])

        # Extract wave metrics
        wave_values = []
        for wave in self.wave_history:
            amps = wave.amplitudes[self.batch_idx].detach().cpu().numpy()
            freqs = wave.frequencies[self.batch_idx].detach().cpu().numpy()

            if wave_metric == 'energy':
                value = (amps ** 2).sum()
            elif wave_metric == 'amplitude':
                value = amps.mean()
            elif wave_metric == 'centroid':
                eps = 1e-8
                value = (freqs * amps).sum() / (amps.sum() + eps)
            elif wave_metric == 'entropy':
                # Harmonic entropy
                amp_sum = amps.sum() + 1e-10
                probs = amps / amp_sum
                value = -(probs * np.log(probs + 1e-10)).sum()
            else:
                raise ValueError(f"Unknown wave_metric: {wave_metric}")

            wave_values.append(value)

        wave_values = np.array(wave_values)

        # Compute correlations
        if HAS_SCIPY:
            # Pearson correlation
            corr_prob_wave, p_prob_wave = scipy_stats.pearsonr(max_probs, wave_values)
            corr_entropy_wave, p_entropy_wave = scipy_stats.pearsonr(entropies, wave_values)

            # Spearman correlation (rank-based, more robust)
            spearman_prob_wave, sp_p_prob_wave = scipy_stats.spearmanr(max_probs, wave_values)
            spearman_entropy_wave, sp_p_entropy_wave = scipy_stats.spearmanr(entropies, wave_values)

            return {
                'wave_metric': wave_metric,
                'pearson': {
                    'confidence_vs_wave': {
                        'correlation': float(corr_prob_wave),
                        'p_value': float(p_prob_wave),
                    },
                    'entropy_vs_wave': {
                        'correlation': float(corr_entropy_wave),
                        'p_value': float(p_entropy_wave),
                    },
                },
                'spearman': {
                    'confidence_vs_wave': {
                        'correlation': float(spearman_prob_wave),
                        'p_value': float(sp_p_prob_wave),
                    },
                    'entropy_vs_wave': {
                        'correlation': float(spearman_entropy_wave),
                        'p_value': float(sp_p_entropy_wave),
                    },
                },
                'data_points': len(max_probs),
            }
        else:
            # Fallback to numpy correlation (Pearson only, no p-values)
            corr_prob_wave = float(np.corrcoef(max_probs, wave_values)[0, 1])
            corr_entropy_wave = float(np.corrcoef(entropies, wave_values)[0, 1])

            return {
                'wave_metric': wave_metric,
                'pearson': {
                    'confidence_vs_wave': {
                        'correlation': corr_prob_wave,
                        'p_value': None,  # Not available without scipy
                    },
                    'entropy_vs_wave': {
                        'correlation': corr_entropy_wave,
                        'p_value': None,
                    },
                },
                'spearman': {
                    'confidence_vs_wave': {
                        'correlation': None,  # Not available without scipy
                        'p_value': None,
                    },
                    'entropy_vs_wave': {
                        'correlation': None,
                        'p_value': None,
                    },
                },
                'data_points': len(max_probs),
                'warning': 'scipy not available - Spearman correlation and p-values not computed'
            }

    def plot_wave_confidence_correlation(
        self,
        wave_metric: str = 'energy',
        figsize: Tuple[int, int] = (14, 5),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize correlation between wave properties and confidence.

        Args:
            wave_metric: Wave property to plot ('energy', 'amplitude', 'centroid')
            figsize: Figure size
            save_path: Optional save path

        Returns:
            Matplotlib figure
        """
        if len(self.confidence_history) == 0 or len(self.wave_history) == 0:
            raise ValueError("No data available")

        # Extract metrics
        max_probs = np.array([s.max_probability for s in self.confidence_history])
        entropies = np.array([s.entropy for s in self.confidence_history])
        steps = np.array([s.step for s in self.confidence_history])

        # Extract wave values
        wave_values = []
        for wave in self.wave_history:
            amps = wave.amplitudes[self.batch_idx].detach().cpu().numpy()
            freqs = wave.frequencies[self.batch_idx].detach().cpu().numpy()

            if wave_metric == 'energy':
                value = (amps ** 2).sum()
            elif wave_metric == 'amplitude':
                value = amps.mean()
            elif wave_metric == 'centroid':
                value = (freqs * amps).sum() / (amps.sum() + 1e-8)
            else:
                value = amps.mean()

            wave_values.append(value)

        wave_values = np.array(wave_values)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Dual-axis plot: confidence and wave metric over time
        ax1 = axes[0]
        ax2 = ax1.twinx()

        line1 = ax1.plot(steps, max_probs, color='blue', linewidth=2, marker='o',
                        markersize=3, label='Confidence', alpha=0.7)
        line2 = ax2.plot(steps, wave_values, color='red', linewidth=2, marker='s',
                        markersize=3, label=f'Wave {wave_metric}', alpha=0.7)

        ax1.set_xlabel('Generation Step')
        ax1.set_ylabel('Confidence (Max Prob)', color='blue')
        ax2.set_ylabel(f'Wave {wave_metric}', color='red')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        ax1.set_title('Confidence vs Wave Metric Over Time')
        ax1.grid(True, alpha=0.3)

        # Scatter plot: confidence vs wave metric
        axes[1].scatter(wave_values, max_probs, alpha=0.6, c=steps, cmap='viridis', s=50)
        axes[1].set_xlabel(f'Wave {wave_metric}')
        axes[1].set_ylabel('Confidence (Max Prob)')
        axes[1].set_title('Confidence vs Wave Metric')
        axes[1].grid(True, alpha=0.3)

        # Add regression line
        z = np.polyfit(wave_values, max_probs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(wave_values.min(), wave_values.max(), 100)
        axes[1].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        # Scatter plot: entropy vs wave metric
        axes[2].scatter(wave_values, entropies, alpha=0.6, c=steps, cmap='plasma', s=50)
        axes[2].set_xlabel(f'Wave {wave_metric}')
        axes[2].set_ylabel('Entropy (nats)')
        axes[2].set_title('Entropy vs Wave Metric')
        axes[2].grid(True, alpha=0.3)

        # Add regression line
        z2 = np.polyfit(wave_values, entropies, 1)
        p2 = np.poly1d(z2)
        axes[2].plot(x_line, p2(x_line), "r--", alpha=0.8, linewidth=2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def get_confidence_summary(self) -> Dict:
        """
        Get summary statistics of confidence metrics.

        Returns:
            Dictionary with aggregate confidence statistics
        """
        if len(self.confidence_history) == 0:
            return {}

        max_probs = np.array([s.max_probability for s in self.confidence_history])
        entropies = np.array([s.entropy for s in self.confidence_history])
        top_k_masses = np.array([s.top_k_mass for s in self.confidence_history])
        perplexities = np.array([s.perplexity for s in self.confidence_history])

        return {
            'num_steps': len(self.confidence_history),
            'max_probability': {
                'mean': float(max_probs.mean()),
                'std': float(max_probs.std()),
                'min': float(max_probs.min()),
                'max': float(max_probs.max()),
                'median': float(np.median(max_probs)),
            },
            'entropy': {
                'mean': float(entropies.mean()),
                'std': float(entropies.std()),
                'min': float(entropies.min()),
                'max': float(entropies.max()),
                'median': float(np.median(entropies)),
            },
            'top_k_mass': {
                'mean': float(top_k_masses.mean()),
                'std': float(top_k_masses.std()),
                'min': float(top_k_masses.min()),
                'max': float(top_k_masses.max()),
                'k': self.k,
            },
            'perplexity': {
                'mean': float(perplexities.mean()),
                'std': float(perplexities.std()),
                'min': float(perplexities.min()),
                'max': float(perplexities.max()),
            },
        }
