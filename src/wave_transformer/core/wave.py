import torch
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import seaborn as sns


@dataclass
class Wave:
    """
    Wave representation with shape [batch_size, seq_len, num_harmonics]

    frequencies: Tensor of shape [B, S, H] - oscillation rates
    amplitudes: Tensor of shape [B, S, H] - intensity of each harmonic
    phases: Tensor of shape [B, S, H] - temporal offsets
    """
    frequencies: torch.Tensor
    amplitudes: torch.Tensor
    phases: torch.Tensor

    def to_representation(self) -> torch.Tensor:
        return torch.cat([
            self.frequencies,
            self.amplitudes,
            self.phases
        ], dim=-1)

    @classmethod
    def from_representation(cls, x: torch.Tensor):
        chunks = x.chunk(3, dim=-1)
        return cls(
            frequencies=chunks[0],
            amplitudes=chunks[1],
            phases=chunks[2]
        )

    def synthesize(self, t: torch.Tensor) -> torch.Tensor:
        """Generate wave signal at time points t."""
        t = t.unsqueeze(-1)
        return (self.amplitudes * torch.sin(
            2 * np.pi * self.frequencies * t + self.phases
        )).sum(dim=-1)

    # ==================== CORE VISUALIZATION METHODS ====================

    def plot_sequence_heatmaps(
            self,
            batch_idx: int = 0,
            seq_slice: Optional[Tuple[int, int]] = None,
            figsize: Tuple[int, int] = (18, 5),
            save_path: Optional[str] = None
    ):
        """
        Plot heatmaps of frequencies, amplitudes, and phases for a sequence.

        Args:
            batch_idx: Which batch element to visualize
            seq_slice: (start, end) to visualize subset of sequence, or None for all
            figsize: Figure size
            save_path: Optional path to save figure
        """
        B, S, H = self.frequencies.shape

        if seq_slice:
            start, end = seq_slice
        else:
            start, end = 0, S

        freq = self.frequencies[batch_idx, start:end].detach().cpu().numpy()
        amp = self.amplitudes[batch_idx, start:end].detach().cpu().numpy()
        phase = self.phases[batch_idx, start:end].detach().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Frequencies
        im1 = axes[0].imshow(freq.T, aspect='auto', cmap='viridis', interpolation='nearest')
        axes[0].set_title(f'Frequencies [Batch {batch_idx}]', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Sequence Position')
        axes[0].set_ylabel('Harmonic Index')
        plt.colorbar(im1, ax=axes[0], label='Frequency (Hz)')

        # Amplitudes
        im2 = axes[1].imshow(amp.T, aspect='auto', cmap='plasma', interpolation='nearest')
        axes[1].set_title(f'Amplitudes [Batch {batch_idx}]', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Sequence Position')
        axes[1].set_ylabel('Harmonic Index')
        plt.colorbar(im2, ax=axes[1], label='Amplitude')

        # Phases
        im3 = axes[2].imshow(phase.T, aspect='auto', cmap='twilight', interpolation='nearest')
        axes[2].set_title(f'Phases [Batch {batch_idx}]', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Sequence Position')
        axes[2].set_ylabel('Harmonic Index')
        plt.colorbar(im3, ax=axes[2], label='Phase (radians)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, axes

    def plot_harmonic_spectra(
            self,
            batch_idx: int = 0,
            seq_positions: Optional[List[int]] = None,
            max_positions: int = 5,
            figsize: Tuple[int, int] = (15, 10),
            save_path: Optional[str] = None
    ):
        """
        Plot frequency spectra (amplitude vs frequency) for specific sequence positions.

        Args:
            batch_idx: Which batch element
            seq_positions: List of sequence positions to plot, or None for auto-select
            max_positions: Max number of positions to show if seq_positions is None
            figsize: Figure size
            save_path: Optional path to save
        """
        B, S, H = self.frequencies.shape

        if seq_positions is None:
            # Auto-select evenly spaced positions
            step = max(1, S // max_positions)
            seq_positions = list(range(0, S, step))[:max_positions]

        freq = self.frequencies[batch_idx].detach().cpu().numpy()
        amp = self.amplitudes[batch_idx].detach().cpu().numpy()

        fig, axes = plt.subplots(len(seq_positions), 1, figsize=figsize, sharex=True)
        if len(seq_positions) == 1:
            axes = [axes]

        for i, pos in enumerate(seq_positions):
            # Sort by frequency for better visualization
            sorted_indices = np.argsort(freq[pos])
            sorted_freq = freq[pos][sorted_indices]
            sorted_amp = amp[pos][sorted_indices]

            axes[i].stem(sorted_freq, sorted_amp, linefmt='C0-', markerfmt='C0o', basefmt='C0-')
            axes[i].set_ylabel('Amplitude')
            axes[i].set_title(f'Position {pos} Spectrum', fontsize=10)
            axes[i].grid(True, alpha=0.3)

        axes[-1].set_xlabel('Frequency (Hz)')
        fig.suptitle(f'Harmonic Spectra [Batch {batch_idx}]', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, axes

    def plot_waveforms(
            self,
            batch_idx: int = 0,
            seq_positions: Optional[List[int]] = None,
            duration: float = 1.0,
            sample_rate: int = 1000,
            max_positions: int = 4,
            figsize: Tuple[int, int] = (15, 10),
            save_path: Optional[str] = None
    ):
        """
        Plot synthesized waveforms for specific sequence positions.

        Args:
            batch_idx: Which batch element
            seq_positions: Positions to plot
            duration: Time duration to synthesize
            sample_rate: Sampling rate
            max_positions: Max positions if auto-selecting
            figsize: Figure size
            save_path: Optional save path
        """
        B, S, H = self.frequencies.shape

        if seq_positions is None:
            step = max(1, S // max_positions)
            seq_positions = list(range(0, S, step))[:max_positions]

        t = torch.linspace(0, duration, int(duration * sample_rate))

        fig, axes = plt.subplots(len(seq_positions), 1, figsize=figsize, sharex=True)
        if len(seq_positions) == 1:
            axes = [axes]

        for i, pos in enumerate(seq_positions):
            # Extract single position's harmonics
            freqs = self.frequencies[batch_idx, pos]
            amps = self.amplitudes[batch_idx, pos]
            phases = self.phases[batch_idx, pos]

            # Synthesize
            signal = (amps.unsqueeze(0) * torch.sin(
                2 * np.pi * freqs.unsqueeze(0) * t.unsqueeze(-1).to(freqs.device) + phases.unsqueeze(0)
            )).sum(dim=-1).detach().cpu().numpy()

            axes[i].plot(t.numpy(), signal.flatten(), linewidth=1)
            axes[i].set_ylabel('Amplitude')
            axes[i].set_title(f'Position {pos} Waveform', fontsize=10)
            axes[i].grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(f'Synthesized Waveforms [Batch {batch_idx}]', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, axes

    def plot_phase_relationships(
            self,
            batch_idx: int = 0,
            seq_slice: Optional[Tuple[int, int]] = None,
            figsize: Tuple[int, int] = (12, 8),
            save_path: Optional[str] = None
    ):
        """
        Plot phase relationships using polar coordinates and phase differences.

        Args:
            batch_idx: Which batch element
            seq_slice: Optional (start, end) for sequence subset
            figsize: Figure size
            save_path: Optional save path
        """
        B, S, H = self.frequencies.shape

        if seq_slice:
            start, end = seq_slice
        else:
            start, end = 0, min(S, 20)  # Default to first 20 positions

        phases = self.phases[batch_idx, start:end].detach().cpu().numpy()
        amps = self.amplitudes[batch_idx, start:end].detach().cpu().numpy()

        fig = plt.figure(figsize=figsize)

        # Polar plot for first position
        ax1 = plt.subplot(221, projection='polar')
        theta = phases[0]
        r = amps[0]
        ax1.scatter(theta, r, c=np.arange(H), cmap='hsv', s=50, alpha=0.7)
        ax1.set_title(f'Phase Distribution (Pos {start})', fontsize=11)

        # Phase difference heatmap
        ax2 = plt.subplot(222)
        phase_diff = np.diff(phases, axis=0)  # Difference between consecutive positions
        im = ax2.imshow(phase_diff.T, aspect='auto', cmap='coolwarm',
                        vmin=-np.pi, vmax=np.pi, interpolation='nearest')
        ax2.set_title('Phase Changes (Sequential)', fontsize=11)
        ax2.set_xlabel('Sequence Position')
        ax2.set_ylabel('Harmonic Index')
        plt.colorbar(im, ax=ax2, label='Phase Δ (rad)')

        # Phase unwrapped over sequence for selected harmonics
        ax3 = plt.subplot(223)
        num_harmonics_to_show = min(8, H)
        harmonic_indices = np.linspace(0, H - 1, num_harmonics_to_show, dtype=int)
        for h_idx in harmonic_indices:
            ax3.plot(range(start, end), phases[:, h_idx],
                     label=f'H{h_idx}', alpha=0.7, marker='o', markersize=3)
        ax3.set_xlabel('Sequence Position')
        ax3.set_ylabel('Phase (rad)')
        ax3.set_title('Phase Evolution', fontsize=11)
        ax3.legend(ncol=2, fontsize=8)
        ax3.grid(True, alpha=0.3)

        # Amplitude-weighted phase histogram
        ax4 = plt.subplot(224)
        all_phases = phases.flatten()
        all_amps = amps.flatten()
        ax4.hist(all_phases, bins=50, weights=all_amps, alpha=0.7, color='purple')
        ax4.set_xlabel('Phase (rad)')
        ax4.set_ylabel('Amplitude-weighted Count')
        ax4.set_title('Phase Distribution (Amplitude-weighted)', fontsize=11)
        ax4.grid(True, alpha=0.3)

        fig.suptitle(f'Phase Analysis [Batch {batch_idx}]', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_statistics_summary(
            self,
            batch_idx: int = 0,
            figsize: Tuple[int, int] = (16, 10),
            save_path: Optional[str] = None
    ):
        """
        Comprehensive statistical summary of wave components.

        Args:
            batch_idx: Which batch element
            figsize: Figure size
            save_path: Optional save path
        """
        freq = self.frequencies[batch_idx].detach().cpu().numpy()
        amp = self.amplitudes[batch_idx].detach().cpu().numpy()
        phase = self.phases[batch_idx].detach().cpu().numpy()

        S, H = freq.shape

        fig, axes = plt.subplots(3, 3, figsize=figsize)

        # Row 1: Frequencies
        # Mean freq per position
        axes[0, 0].plot(freq.mean(axis=1), color='C0')
        axes[0, 0].fill_between(range(S),
                                freq.mean(axis=1) - freq.std(axis=1),
                                freq.mean(axis=1) + freq.std(axis=1),
                                alpha=0.3, color='C0')
        axes[0, 0].set_title('Mean Frequency per Position')
        axes[0, 0].set_xlabel('Sequence Position')
        axes[0, 0].set_ylabel('Frequency (Hz)')
        axes[0, 0].grid(True, alpha=0.3)

        # Freq distribution
        axes[0, 1].hist(freq.flatten(), bins=50, alpha=0.7, color='C0')
        axes[0, 1].set_title('Frequency Distribution')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].grid(True, alpha=0.3)

        # Freq variance per harmonic
        axes[0, 2].plot(freq.var(axis=0), marker='o', markersize=3, color='C0')
        axes[0, 2].set_title('Frequency Variance per Harmonic')
        axes[0, 2].set_xlabel('Harmonic Index')
        axes[0, 2].set_ylabel('Variance')
        axes[0, 2].grid(True, alpha=0.3)

        # Row 2: Amplitudes
        # Mean amp per position
        axes[1, 0].plot(amp.mean(axis=1), color='C1')
        axes[1, 0].fill_between(range(S),
                                amp.mean(axis=1) - amp.std(axis=1),
                                amp.mean(axis=1) + amp.std(axis=1),
                                alpha=0.3, color='C1')
        axes[1, 0].set_title('Mean Amplitude per Position')
        axes[1, 0].set_xlabel('Sequence Position')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].grid(True, alpha=0.3)

        # Amp distribution
        axes[1, 1].hist(amp.flatten(), bins=50, alpha=0.7, color='C1')
        axes[1, 1].set_title('Amplitude Distribution')
        axes[1, 1].set_xlabel('Amplitude')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3)

        # Top harmonics by mean amplitude
        mean_amps = amp.mean(axis=0)
        top_k = min(20, H)
        top_indices = np.argsort(mean_amps)[-top_k:]
        axes[1, 2].barh(range(top_k), mean_amps[top_indices], color='C1', alpha=0.7)
        axes[1, 2].set_title(f'Top {top_k} Harmonics by Amplitude')
        axes[1, 2].set_xlabel('Mean Amplitude')
        axes[1, 2].set_ylabel('Harmonic Index')
        axes[1, 2].grid(True, alpha=0.3, axis='x')

        # Row 3: Phases
        # Mean phase per position
        axes[2, 0].plot(phase.mean(axis=1), color='C2')
        axes[2, 0].fill_between(range(S),
                                phase.mean(axis=1) - phase.std(axis=1),
                                phase.mean(axis=1) + phase.std(axis=1),
                                alpha=0.3, color='C2')
        axes[2, 0].set_title('Mean Phase per Position')
        axes[2, 0].set_xlabel('Sequence Position')
        axes[2, 0].set_ylabel('Phase (rad)')
        axes[2, 0].grid(True, alpha=0.3)

        # Phase distribution
        axes[2, 1].hist(phase.flatten(), bins=50, alpha=0.7, color='C2')
        axes[2, 1].set_title('Phase Distribution')
        axes[2, 1].set_xlabel('Phase (rad)')
        axes[2, 1].set_ylabel('Count')
        axes[2, 1].grid(True, alpha=0.3)

        # Energy (amp^2) per position
        energy = (amp ** 2).sum(axis=1)
        axes[2, 2].plot(energy, color='purple', linewidth=2)
        axes[2, 2].set_title('Total Energy per Position')
        axes[2, 2].set_xlabel('Sequence Position')
        axes[2, 2].set_ylabel('Energy (Σ amp²)')
        axes[2, 2].grid(True, alpha=0.3)

        fig.suptitle(f'Wave Statistics Summary [Batch {batch_idx}]',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, axes

    def plot_batch_comparison(
            self,
            batch_indices: Optional[List[int]] = None,
            max_batches: int = 4,
            metric: str = 'mean_amplitude',
            figsize: Tuple[int, int] = (15, 8),
            save_path: Optional[str] = None
    ):
        """
        Compare statistics across multiple batch elements.

        Args:
            batch_indices: Which batches to compare, or None for auto-select
            max_batches: Max batches if auto-selecting
            metric: 'mean_amplitude', 'mean_frequency', 'energy', or 'phase_std'
            figsize: Figure size
            save_path: Optional save path
        """
        B, S, H = self.frequencies.shape

        if batch_indices is None:
            batch_indices = list(range(min(max_batches, B)))

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        for b_idx in batch_indices:
            freq = self.frequencies[b_idx].detach().cpu().numpy()
            amp = self.amplitudes[b_idx].detach().cpu().numpy()
            phase = self.phases[b_idx].detach().cpu().numpy()

            label = f'Batch {b_idx}'

            # Mean amplitude
            axes[0, 0].plot(amp.mean(axis=1), label=label, alpha=0.7)

            # Mean frequency
            axes[0, 1].plot(freq.mean(axis=1), label=label, alpha=0.7)

            # Energy
            energy = (amp ** 2).sum(axis=1)
            axes[1, 0].plot(energy, label=label, alpha=0.7)

            # Phase std
            axes[1, 1].plot(phase.std(axis=1), label=label, alpha=0.7)

        axes[0, 0].set_title('Mean Amplitude per Position')
        axes[0, 0].set_xlabel('Sequence Position')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_title('Mean Frequency per Position')
        axes[0, 1].set_xlabel('Sequence Position')
        axes[0, 1].set_ylabel('Frequency (Hz)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].set_title('Energy per Position')
        axes[1, 0].set_xlabel('Sequence Position')
        axes[1, 0].set_ylabel('Energy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].set_title('Phase Std Dev per Position')
        axes[1, 1].set_xlabel('Sequence Position')
        axes[1, 1].set_ylabel('Phase Std')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle('Batch Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, axes

    def quick_viz(
            self,
            batch_idx: int = 0,
            save_prefix: Optional[str] = None
    ):
        """
        Generate all visualizations at once with sensible defaults.

        Args:
            batch_idx: Which batch to visualize
            save_prefix: If provided, save all plots with this prefix
        """
        print(f"🌊 Generating visualizations for batch {batch_idx}...")

        # Heatmaps
        print("  📊 Creating heatmaps...")
        self.plot_sequence_heatmaps(
            batch_idx=batch_idx,
            save_path=f"{save_prefix}_heatmaps.png" if save_prefix else None
        )

        # Spectra
        print("  🎵 Creating harmonic spectra...")
        self.plot_harmonic_spectra(
            batch_idx=batch_idx,
            save_path=f"{save_prefix}_spectra.png" if save_prefix else None
        )

        # Waveforms
        print("  〰️ Creating waveforms...")
        self.plot_waveforms(
            batch_idx=batch_idx,
            save_path=f"{save_prefix}_waveforms.png" if save_prefix else None
        )

        # Phase analysis
        print("  🔄 Creating phase analysis...")
        self.plot_phase_relationships(
            batch_idx=batch_idx,
            save_path=f"{save_prefix}_phases.png" if save_prefix else None
        )

        # Statistics
        print("  📈 Creating statistics summary...")
        self.plot_statistics_summary(
            batch_idx=batch_idx,
            save_path=f"{save_prefix}_stats.png" if save_prefix else None
        )

        print("✨ All visualizations complete!")

        if not save_prefix:
            plt.show()

    # ==================== UTILITY METHODS ====================

    def get_statistics(self, batch_idx: int = 0) -> dict:
        """
        Get numerical statistics for a batch element.

        Returns dict with comprehensive stats.
        """
        freq = self.frequencies[batch_idx].detach().cpu()
        amp = self.amplitudes[batch_idx].detach().cpu()
        phase = self.phases[batch_idx].detach().cpu()

        return {
            'frequencies': {
                'mean': freq.mean().item(),
                'std': freq.std().item(),
                'min': freq.min().item(),
                'max': freq.max().item(),
                'median': freq.median().item(),
            },
            'amplitudes': {
                'mean': amp.mean().item(),
                'std': amp.std().item(),
                'min': amp.min().item(),
                'max': amp.max().item(),
                'median': amp.median().item(),
                'total_energy': (amp ** 2).sum().item(),
            },
            'phases': {
                'mean': phase.mean().item(),
                'std': phase.std().item(),
                'min': phase.min().item(),
                'max': phase.max().item(),
                'median': phase.median().item(),
            },
            'shape': {
                'sequence_length': freq.shape[0],
                'num_harmonics': freq.shape[1],
            }
        }

    def print_statistics(self, batch_idx: int = 0):
        """Pretty print statistics."""
        stats = self.get_statistics(batch_idx)

        print(f"\n{'=' * 60}")
        print(f"Wave Statistics [Batch {batch_idx}]")
        print(f"{'=' * 60}")
        print(f"Shape: {stats['shape']['sequence_length']} positions × {stats['shape']['num_harmonics']} harmonics")
        print(f"\n{'Frequencies':-^60}")
        for k, v in stats['frequencies'].items():
            print(f"  {k:.<20} {v:>10.4f} Hz")
        print(f"\n{'Amplitudes':-^60}")
        for k, v in stats['amplitudes'].items():
            print(f"  {k:.<20} {v:>10.4f}")
        print(f"\n{'Phases':-^60}")
        for k, v in stats['phases'].items():
            print(f"  {k:.<20} {v:>10.4f} rad")
        print(f"{'=' * 60}\n")


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Create dummy wave data
    batch_size = 4
    seq_len = 128
    num_harmonics = 64

    frequencies = torch.nn.functional.sigmoid(torch.randn(batch_size, seq_len, num_harmonics)) * 20 + 0.1
    amplitudes = torch.nn.functional.softplus(torch.randn(batch_size, seq_len, num_harmonics))
    phases = torch.nn.functional.tanh(torch.randn(batch_size, seq_len, num_harmonics)) * np.pi

    wave = Wave(frequencies, amplitudes, phases)

    # Print stats
    wave.print_statistics(batch_idx=0)

    # Generate all visualizations
    wave.quick_viz(batch_idx=0, save_prefix="demo_wave")

    # Or individual plots
    # wave.plot_sequence_heatmaps(batch_idx=0)
    # wave.plot_harmonic_spectra(batch_idx=0, seq_positions=[0, 32, 64, 96])
    # wave.plot_waveforms(batch_idx=0)
    # wave.plot_phase_relationships(batch_idx=0)
    # wave.plot_statistics_summary(batch_idx=0)
    # wave.plot_batch_comparison(batch_indices=[0, 1, 2, 3])