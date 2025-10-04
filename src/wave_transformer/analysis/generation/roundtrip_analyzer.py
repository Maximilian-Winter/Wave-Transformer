"""
Round-trip token-wave-token analysis.

This module analyzes the encoder-decoder round-trip process: how well tokens
can be reconstructed from their wave representations, and how distinct wave
representations are for different tokens.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class RoundTripResult:
    """Container for round-trip analysis results."""
    original_tokens: torch.Tensor
    reconstructed_tokens: torch.Tensor
    reconstruction_accuracy: float
    per_position_accuracy: torch.Tensor
    wave: object  # Wave object


class RoundTripAnalyzer:
    """
    Analyze token → wave → token reconstruction quality.

    Evaluates how well the encoder-decoder architecture preserves token
    information through the wave representation, and measures wave
    distinguishability for different tokens.

    Example:
        >>> analyzer = RoundTripAnalyzer(model)
        >>> result = analyzer.analyze_roundtrip(token_ids)
        >>> print(f"Accuracy: {result.reconstruction_accuracy:.2%}")
        >>> analyzer.plot_roundtrip_analysis(result)
        >>> correlation = analyzer.analyze_wave_token_correlation(dataloader)
    """

    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        """
        Initialize round-trip analyzer.

        Args:
            model: Wave Transformer model
            device: Device for computation (defaults to model's device)
        """
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device

    @torch.no_grad()
    def analyze_roundtrip(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> RoundTripResult:
        """
        Perform token → wave → token round-trip analysis.

        Args:
            token_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]

        Returns:
            RoundTripResult with reconstruction statistics
        """
        self.model.eval()
        token_ids = token_ids.to(self.device)

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Forward pass to get wave and reconstruction
        logits, wave = self.model(
            encoder_input={'token_ids': token_ids},
            attention_mask=attention_mask,
            return_encoder_outputs=True
        )

        # Reconstruct tokens from logits
        reconstructed_tokens = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]

        # Compute accuracy
        if attention_mask is not None:
            # Only consider non-padded positions
            correct = (reconstructed_tokens == token_ids) & (attention_mask == 1)
            total = attention_mask.sum()
            accuracy = correct.sum().float() / total.float()

            # Per-position accuracy (averaged across batch, only valid positions)
            per_position = (reconstructed_tokens == token_ids).float()
            per_position = per_position * attention_mask
            counts = attention_mask.sum(dim=0).float()
            per_position_accuracy = per_position.sum(dim=0) / (counts + 1e-8)
        else:
            # All positions are valid
            correct = (reconstructed_tokens == token_ids)
            accuracy = correct.float().mean()
            per_position_accuracy = correct.float().mean(dim=0)

        return RoundTripResult(
            original_tokens=token_ids.cpu(),
            reconstructed_tokens=reconstructed_tokens.cpu(),
            reconstruction_accuracy=accuracy.item(),
            per_position_accuracy=per_position_accuracy.cpu(),
            wave=wave
        )

    @torch.no_grad()
    def analyze_batch(
        self,
        dataloader,
        max_batches: Optional[int] = None
    ) -> Dict:
        """
        Analyze round-trip reconstruction over multiple batches.

        Args:
            dataloader: DataLoader providing (token_ids, attention_mask) or just token_ids
            max_batches: Maximum number of batches to process (None = all)

        Returns:
            Dictionary with aggregate statistics
        """
        self.model.eval()

        total_tokens = 0
        total_correct = 0
        position_correct = defaultdict(int)
        position_total = defaultdict(int)

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Unpack batch
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                token_ids, attention_mask = batch[0], batch[1]
            elif isinstance(batch, dict):
                token_ids = batch.get('input_ids', batch.get('token_ids'))
                attention_mask = batch.get('attention_mask', None)
            else:
                token_ids = batch
                attention_mask = None

            # Analyze
            result = self.analyze_roundtrip(token_ids, attention_mask)

            # Accumulate statistics
            if attention_mask is not None:
                mask = attention_mask.cpu()
                valid_tokens = mask.sum().item()
                correct = ((result.reconstructed_tokens == result.original_tokens) & (mask == 1)).sum().item()

                total_tokens += valid_tokens
                total_correct += correct

                # Per-position
                for pos in range(token_ids.size(1)):
                    pos_mask = mask[:, pos]
                    pos_total = pos_mask.sum().item()
                    pos_correct = ((result.reconstructed_tokens[:, pos] == result.original_tokens[:, pos]) & (pos_mask == 1)).sum().item()

                    position_total[pos] += pos_total
                    position_correct[pos] += pos_correct
            else:
                total_tokens += token_ids.numel()
                total_correct += (result.reconstructed_tokens == result.original_tokens).sum().item()

                # Per-position
                for pos in range(token_ids.size(1)):
                    position_total[pos] += token_ids.size(0)
                    position_correct[pos] += (result.reconstructed_tokens[:, pos] == result.original_tokens[:, pos]).sum().item()

        # Compute aggregate statistics
        overall_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

        position_accuracies = {}
        for pos in sorted(position_total.keys()):
            if position_total[pos] > 0:
                position_accuracies[pos] = position_correct[pos] / position_total[pos]

        return {
            'overall_accuracy': overall_accuracy,
            'total_tokens': total_tokens,
            'total_correct': total_correct,
            'position_accuracies': position_accuracies,
            'num_batches_processed': batch_idx + 1 if 'batch_idx' in locals() else 0,
        }

    def plot_roundtrip_analysis(
        self,
        result: RoundTripResult,
        tokenizer: Optional[object] = None,
        batch_idx: int = 0,
        max_display_length: int = 50,
        figsize: Tuple[int, int] = (16, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize round-trip reconstruction results.

        Args:
            result: RoundTripResult from analyze_roundtrip
            tokenizer: Optional tokenizer for decoding tokens
            batch_idx: Which batch element to visualize
            max_display_length: Maximum sequence length to display
            figsize: Figure size
            save_path: Optional save path

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Extract data for visualization
        orig = result.original_tokens[batch_idx].numpy()
        recon = result.reconstructed_tokens[batch_idx].numpy()
        seq_len = min(len(orig), max_display_length)

        orig = orig[:seq_len]
        recon = recon[:seq_len]

        # 1. Token comparison
        ax = axes[0, 0]
        matches = (orig == recon).astype(int)
        x = np.arange(seq_len)

        # Plot original and reconstructed as heatmap
        data = np.vstack([orig, recon])
        im = ax.imshow(data, aspect='auto', cmap='tab20', interpolation='nearest')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Original', 'Reconstructed'])
        ax.set_xlabel('Sequence Position')
        ax.set_title('Token ID Comparison')
        plt.colorbar(im, ax=ax, label='Token ID')

        # 2. Match/mismatch visualization
        ax = axes[0, 1]
        colors = ['red' if m == 0 else 'green' for m in matches]
        ax.bar(x, np.ones(seq_len), color=colors, alpha=0.6, edgecolor='black')
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Match')
        ax.set_title('Reconstruction Accuracy per Position')
        ax.set_ylim([0, 1.5])
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.3)

        # Add text
        accuracy = matches.mean()
        ax.text(0.5, 1.2, f'Accuracy: {accuracy:.2%}',
               transform=ax.transData, ha='left', fontsize=10, fontweight='bold')

        # 3. Per-position accuracy (from full batch)
        ax = axes[1, 0]
        per_pos_acc = result.per_position_accuracy.numpy()[:seq_len]
        ax.plot(x, per_pos_acc, linewidth=2, marker='o', markersize=4, color='blue')
        ax.fill_between(x, 0, per_pos_acc, alpha=0.3, color='blue')
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Accuracy')
        ax.set_title('Per-Position Accuracy (Batch Average)')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

        # 4. Token error distribution
        ax = axes[1, 1]
        errors = orig != recon
        if errors.any():
            error_positions = np.where(errors)[0]
            ax.hist(error_positions, bins=min(20, seq_len // 2), alpha=0.7, color='red', edgecolor='black')
            ax.set_xlabel('Position of Error')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Reconstruction Errors')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No reconstruction errors!',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, fontweight='bold', color='green')

        fig.suptitle(f'Round-Trip Analysis (Overall Accuracy: {result.reconstruction_accuracy:.2%})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    @torch.no_grad()
    def compute_wave_distinguishability(
        self,
        token_ids_list: List[torch.Tensor],
        metric: str = 'cosine'
    ) -> Dict:
        """
        Measure how distinct wave representations are for different token sequences.

        Args:
            token_ids_list: List of token ID tensors to compare
            metric: Distance metric ('cosine', 'l2', 'l1')

        Returns:
            Dictionary with distinguishability metrics
        """
        self.model.eval()

        # Get wave representations
        waves = []
        for token_ids in token_ids_list:
            token_ids = token_ids.to(self.device)
            _, wave = self.model(
                encoder_input={'token_ids': token_ids},
                return_encoder_outputs=True
            )
            # Convert to flat representation
            wave_repr = wave.to_representation()  # [B, S, H*3]
            waves.append(wave_repr)

        # Compute pairwise distances
        n = len(waves)
        distance_matrix = torch.zeros(n, n)

        for i in range(n):
            for j in range(i + 1, n):
                w1 = waves[i].flatten()
                w2 = waves[j].flatten()

                # Ensure same length (truncate to shorter if needed)
                min_len = min(w1.size(0), w2.size(0))
                w1 = w1[:min_len]
                w2 = w2[:min_len]

                if metric == 'cosine':
                    # Cosine similarity -> distance
                    similarity = F.cosine_similarity(w1.unsqueeze(0), w2.unsqueeze(0))
                    distance = 1 - similarity.item()
                elif metric == 'l2':
                    distance = torch.norm(w1 - w2, p=2).item()
                elif metric == 'l1':
                    distance = torch.norm(w1 - w2, p=1).item()
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        # Statistics
        upper_triangle = distance_matrix[torch.triu(torch.ones_like(distance_matrix), diagonal=1) == 1]

        return {
            'metric': metric,
            'num_sequences': n,
            'distance_matrix': distance_matrix.numpy(),
            'mean_distance': float(upper_triangle.mean()),
            'std_distance': float(upper_triangle.std()),
            'min_distance': float(upper_triangle.min()),
            'max_distance': float(upper_triangle.max()),
        }

    @torch.no_grad()
    def analyze_wave_token_correlation(
        self,
        dataloader,
        max_batches: Optional[int] = 10,
        sample_positions: Optional[List[int]] = None
    ) -> Dict:
        """
        Analyze correlation between token properties and wave properties.

        Args:
            dataloader: DataLoader providing token sequences
            max_batches: Maximum batches to process
            sample_positions: Specific positions to analyze (None = all)

        Returns:
            Dictionary with correlation analysis
        """
        self.model.eval()

        # Collect data
        token_frequencies = defaultdict(list)  # token_id -> list of wave energies
        position_stats = defaultdict(lambda: {'energy': [], 'amplitude': [], 'centroid': []})

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Unpack batch
            if isinstance(batch, (tuple, list)):
                token_ids = batch[0]
                attention_mask = batch[1] if len(batch) > 1 else None
            elif isinstance(batch, dict):
                token_ids = batch.get('input_ids', batch.get('token_ids'))
                attention_mask = batch.get('attention_mask', None)
            else:
                token_ids = batch
                attention_mask = None

            token_ids = token_ids.to(self.device)

            # Get wave
            _, wave = self.model(
                encoder_input={'token_ids': token_ids},
                attention_mask=attention_mask,
                return_encoder_outputs=True
            )

            # Extract wave properties per position
            B, S, H = wave.frequencies.shape

            for b in range(B):
                for s in range(S):
                    # Skip if masked
                    if attention_mask is not None and attention_mask[b, s] == 0:
                        continue

                    # Skip if not in sample positions
                    if sample_positions is not None and s not in sample_positions:
                        continue

                    token_id = token_ids[b, s].item()
                    amps = wave.amplitudes[b, s].cpu().numpy()
                    freqs = wave.frequencies[b, s].cpu().numpy()

                    energy = float((amps ** 2).sum())
                    amplitude = float(amps.mean())
                    centroid = float((freqs * amps).sum() / (amps.sum() + 1e-8))

                    # Store by token ID
                    token_frequencies[token_id].append({
                        'energy': energy,
                        'amplitude': amplitude,
                        'centroid': centroid,
                    })

                    # Store by position
                    position_stats[s]['energy'].append(energy)
                    position_stats[s]['amplitude'].append(amplitude)
                    position_stats[s]['centroid'].append(centroid)

        # Compute statistics per token
        token_statistics = {}
        for token_id, wave_props_list in token_frequencies.items():
            if len(wave_props_list) == 0:
                continue

            energies = [p['energy'] for p in wave_props_list]
            amplitudes = [p['amplitude'] for p in wave_props_list]
            centroids = [p['centroid'] for p in wave_props_list]

            token_statistics[int(token_id)] = {
                'count': len(wave_props_list),
                'energy': {
                    'mean': float(np.mean(energies)),
                    'std': float(np.std(energies)),
                },
                'amplitude': {
                    'mean': float(np.mean(amplitudes)),
                    'std': float(np.std(amplitudes)),
                },
                'centroid': {
                    'mean': float(np.mean(centroids)),
                    'std': float(np.std(centroids)),
                },
            }

        # Compute statistics per position
        position_statistics = {}
        for pos, stats in position_stats.items():
            if len(stats['energy']) == 0:
                continue

            position_statistics[int(pos)] = {
                'count': len(stats['energy']),
                'energy': {
                    'mean': float(np.mean(stats['energy'])),
                    'std': float(np.std(stats['energy'])),
                },
                'amplitude': {
                    'mean': float(np.mean(stats['amplitude'])),
                    'std': float(np.std(stats['amplitude'])),
                },
                'centroid': {
                    'mean': float(np.mean(stats['centroid'])),
                    'std': float(np.std(stats['centroid'])),
                },
            }

        return {
            'token_statistics': token_statistics,
            'position_statistics': position_statistics,
            'num_unique_tokens': len(token_statistics),
            'num_positions_analyzed': len(position_statistics),
            'total_samples': sum(s['count'] for s in token_statistics.values()),
        }

    def plot_wave_distinguishability(
        self,
        distinguishability_result: Dict,
        figsize: Tuple[int, int] = (12, 5),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize wave distinguishability matrix.

        Args:
            distinguishability_result: Result from compute_wave_distinguishability
            figsize: Figure size
            save_path: Optional save path

        Returns:
            Matplotlib figure
        """
        distance_matrix = distinguishability_result['distance_matrix']
        metric = distinguishability_result['metric']

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Heatmap
        im = axes[0].imshow(distance_matrix, cmap='viridis', interpolation='nearest')
        axes[0].set_title(f'Pairwise Wave Distance ({metric})')
        axes[0].set_xlabel('Sequence Index')
        axes[0].set_ylabel('Sequence Index')
        plt.colorbar(im, ax=axes[0], label='Distance')

        # Distribution
        upper_triangle = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        axes[1].hist(upper_triangle, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[1].axvline(distinguishability_result['mean_distance'], color='red',
                       linestyle='--', linewidth=2, label='Mean')
        axes[1].set_xlabel('Distance')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distance Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        fig.suptitle(f'Wave Distinguishability Analysis\n'
                    f'Mean: {distinguishability_result["mean_distance"]:.4f}, '
                    f'Std: {distinguishability_result["std_distance"]:.4f}',
                    fontsize=12, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig
