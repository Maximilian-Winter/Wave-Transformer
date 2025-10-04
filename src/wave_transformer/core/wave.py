import torch
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import seaborn as sns
from torch import nn
import torch.nn.functional as F

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

    def to_complex(self) -> torch.Tensor:
        """Convert to complex representation for FFT operations"""
        # Complex form: A * exp(i * (2π * f * t + φ))
        return self.amplitudes * torch.exp(1j * self.phases)

    def interfere_with(self, other: 'Wave', mode='constructive') -> 'Wave':
        """Wave interference operations"""
        if mode == 'constructive':
            # Constructive: phases align, amplitudes add
            new_phases = (self.phases + other.phases) / 2
            new_amps = self.amplitudes + other.amplitudes
        elif mode == 'destructive':
            # Destructive: phases oppose, amplitudes subtract
            new_phases = (self.phases - other.phases)
            new_amps = torch.abs(self.amplitudes - other.amplitudes)
        else:  # 'modulate'
            # Frequency modulation
            new_phases = self.phases + other.phases
            new_amps = self.amplitudes * other.amplitudes

        # Frequency mixing through harmonic mean
        new_freqs = 2 * self.frequencies * other.frequencies / (self.frequencies + other.frequencies + 1e-6)

        return Wave(new_freqs, new_amps, new_phases)

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

### Advanced Parts
class HarmonicResonator(nn.Module):
    """Learn harmonic relationships between concepts"""

    def __init__(self, num_harmonics: int, num_resonant_modes: int = 16):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.num_modes = num_resonant_modes

        # Learnable resonant frequencies (fundamental frequencies of the language)
        self.resonant_freqs = nn.Parameter(torch.logspace(-1, 1.3, num_resonant_modes))
        self.resonant_decay = nn.Parameter(torch.ones(num_resonant_modes) * 0.1)

        # Coupling strengths
        self.coupling = nn.Linear(num_harmonics, num_resonant_modes)

    def forward(self, wave: Wave) -> Wave:
        B, L, H = wave.frequencies.shape

        # Compute coupling to resonant modes
        coupling_strengths = torch.sigmoid(self.coupling(wave.amplitudes))

        # Each resonant mode influences the wave
        resonance = torch.zeros_like(wave.frequencies)
        for i in range(self.num_modes):
            # Distance from resonant frequency
            freq_diff = torch.abs(wave.frequencies - self.resonant_freqs[i].view(1, 1, 1))

            # Lorentzian resonance curve
            resonance_strength = self.resonant_decay[i] / (freq_diff ** 2 + self.resonant_decay[i] ** 2)

            # Apply coupling
            resonance += resonance_strength * coupling_strengths[..., i:i + 1]

        # Modulate amplitudes based on resonance
        enhanced_amps = wave.amplitudes * (1 + resonance)

        # Slight frequency pulling toward resonant modes
        freq_pull = torch.sum(
            self.resonant_freqs.view(1, 1, 1, -1) * coupling_strengths.unsqueeze(-2),
            dim=-1
        ) / (coupling_strengths.sum(dim=-1, keepdim=True) + 1e-6)

        new_freqs = 0.9 * wave.frequencies + 0.1 * freq_pull

        return Wave(new_freqs, enhanced_amps, wave.phases)


class WaveNormalization(nn.Module):
    """Normalize waves while preserving phase relationships"""

    def __init__(self, num_harmonics: int, eps: float = 1e-6):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_harmonics))
        self.shift = nn.Parameter(torch.zeros(num_harmonics))

    def forward(self, wave: Wave) -> Wave:
        # Normalize amplitudes (energy normalization)
        total_energy = torch.sqrt(torch.sum(wave.amplitudes ** 2, dim=-1, keepdim=True) + self.eps)
        norm_amps = wave.amplitudes / total_energy

        # Scale and shift
        scaled_amps = norm_amps * self.scale + self.shift

        # Frequency normalization (keep in valid range)
        norm_freqs = torch.sigmoid(wave.frequencies) * 20.0 + 0.1

        # Phase wrapping
        norm_phases = torch.fmod(wave.phases, 2 * np.pi)

        return Wave(norm_freqs, scaled_amps, norm_phases)


class FourierMixing(nn.Module):
    """Mix sequences using Fourier-domain operations"""

    def __init__(self, num_harmonics: int):
        super().__init__()
        self.num_harmonics = num_harmonics

        # Learnable frequency-domain filters
        self.freq_filter_r = nn.Parameter(torch.ones(1, 1, num_harmonics))
        self.freq_filter_i = nn.Parameter(torch.zeros(1, 1, num_harmonics))

    def forward(self, wave: Wave) -> Wave:
        B, L, H = wave.frequencies.shape

        # Convert to time-domain signal
        t = torch.linspace(0, 1, L, device=wave.frequencies.device).view(1, L, 1)
        signal = wave.amplitudes * torch.cos(2 * np.pi * wave.frequencies * t + wave.phases)

        # FFT along sequence dimension
        fft_signal = torch.fft.rfft(signal, dim=1, norm='ortho')

        # Apply frequency-domain filter
        filter_complex = torch.complex(self.freq_filter_r, self.freq_filter_i)
        fft_filtered = fft_signal * filter_complex[:, :fft_signal.size(1), :]

        # Back to time domain
        filtered_signal = torch.fft.irfft(fft_filtered, n=L, dim=1, norm='ortho')

        # Extract new amplitudes (RMS over time window)
        new_amps = torch.sqrt(torch.mean(filtered_signal ** 2, dim=1, keepdim=True) + 1e-6)
        new_amps = new_amps.expand(-1, L, -1)

        # Phases shift from filtering
        phase_shift = torch.angle(filter_complex).expand(B, L, -1)
        new_phases = wave.phases + phase_shift

        return Wave(wave.frequencies, new_amps, new_phases)


# Training utilities
class WaveRegularization(nn.Module):
    """
    Regularization terms specific to wave representations
    """

    def __init__(self, harmonic_penalty: float = 0.01, phase_penalty: float = 0.01):
        super().__init__()
        self.harmonic_penalty = harmonic_penalty
        self.phase_penalty = phase_penalty

    def forward(self, wave: Wave) -> torch.Tensor:
        """
        Compute regularization loss for waves
        """
        # Encourage harmonic relationships (integer frequency ratios)
        freq_sorted, _ = torch.sort(wave.frequencies, dim=-1)
        freq_ratios = freq_sorted[..., 1:] / (freq_sorted[..., :-1] + 1e-6)
        harmonic_loss = torch.abs(freq_ratios - torch.round(freq_ratios)).mean()

        # Encourage phase coherence within tokens
        phase_var = torch.var(wave.phases, dim=-1).mean()

        # Combine regularization terms
        reg_loss = (self.harmonic_penalty * harmonic_loss +
                    self.phase_penalty * phase_var)

        return reg_loss


class SpectralSynthesis(nn.Module):
    """Synthesize discrete tokens from continuous wave representation"""

    def __init__(self, num_harmonics: int, vocab_size: int, synthesis_depth: int = 3):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.vocab_size = vocab_size

        # Learn vocab-specific frequency patterns
        self.vocab_frequencies = nn.Parameter(torch.randn(vocab_size, num_harmonics))
        self.vocab_phases = nn.Parameter(torch.zeros(vocab_size, num_harmonics))

        # Synthesis network
        hidden_dim = num_harmonics * 4
        self.synthesis_layers = nn.ModuleList()

        for i in range(synthesis_depth):
            if i == 0:
                self.synthesis_layers.append(
                    nn.Linear(num_harmonics * 3, hidden_dim)
                )
            else:
                self.synthesis_layers.append(
                    nn.Linear(hidden_dim, hidden_dim)
                )

        # Final projection from hidden_dim to vocab_size
        self.synthesis_to_vocab = nn.Linear(hidden_dim, vocab_size)

        # Residual connections for each layer
        self.residual_weights = nn.Parameter(torch.ones(synthesis_depth) * 0.5)

    def forward(self, wave_repr: torch.Tensor) -> torch.Tensor:
        """
        Convert wave representation to vocabulary logits
        Args:
            wave_repr: [B, L, num_harmonics * 3] concatenated wave representation
        Returns:
            logits: [B, L, vocab_size]
        """
        B, L, _ = wave_repr.shape
        H = self.num_harmonics

        # Split wave representation
        freqs = wave_repr[..., :H]
        amps = wave_repr[..., H:2 * H]
        phases = wave_repr[..., 2 * H:]

        # Deep synthesis
        x = wave_repr
        for i, layer in enumerate(self.synthesis_layers):
            residual = x if i > 0 else None
            x = layer(x)
            x = F.gelu(x)
            if residual is not None and x.shape == residual.shape:
                x = x + self.residual_weights[i] * residual

        # Compute similarity to vocabulary patterns
        # Frequency matching (how close are frequencies to vocab patterns)
        freq_similarity = -torch.cdist(freqs, self.vocab_frequencies, p=2)

        # Phase coherence (how aligned are phases)
        phase_diff = phases.unsqueeze(2) - self.vocab_phases.unsqueeze(0).unsqueeze(0)
        phase_coherence = torch.cos(phase_diff).mean(dim=-1)

        # Amplitude weighting
        amp_energy = amps.sum(dim=-1, keepdim=True)

        # Combine similarities
        pattern_match = freq_similarity + 0.5 * phase_coherence
        pattern_match = pattern_match * torch.log1p(amp_energy)

        # Project synthesis result to vocab (using the proper projection layer)
        synthesis_proj = self.synthesis_to_vocab(x)

        # Combine pattern matching and synthesis
        logits = pattern_match + 0.1 * synthesis_proj

        return logits


class WaveInterferenceDecoder(nn.Module):
    """Decode using wave interference patterns"""

    def __init__(
            self,
            num_harmonics: int,
            vocab_size: int,
            num_heads: int = 8
    ):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.vocab_size = vocab_size
        self.num_heads = num_heads

        # Interference patterns for each vocabulary item
        self.vocab_patterns = nn.Parameter(
            torch.randn(vocab_size, num_heads, num_harmonics // num_heads)
        )

        # Phase relationships for constructive/destructive interference
        self.interference_phases = nn.Parameter(
            torch.zeros(num_heads, num_harmonics // num_heads)
        )

        # Output projection
        self.output_projection = nn.Linear(num_heads, 1)

    def forward(self, freqs: torch.Tensor, amps: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        """
        Decode through interference patterns
        """
        B, L, H = freqs.shape
        harmonics_per_head = H // self.num_heads

        # Reshape for multi-head processing
        freqs = freqs.view(B, L, self.num_heads, harmonics_per_head)
        amps = amps.view(B, L, self.num_heads, harmonics_per_head)
        phases = phases.view(B, L, self.num_heads, harmonics_per_head)

        # Compute interference with each vocabulary pattern
        interference_scores = []

        for head in range(self.num_heads):
            # Get head-specific components
            f_h = freqs[:, :, head, :]  # [B, L, harmonics_per_head]
            a_h = amps[:, :, head, :]
            p_h = phases[:, :, head, :]

            # Vocabulary patterns for this head
            vocab_f = self.vocab_patterns[:, head, :]  # [vocab_size, harmonics_per_head]
            inter_p = self.interference_phases[head, :]  # [harmonics_per_head]

            # Compute interference (constructive when phases align)
            phase_alignment = torch.cos(
                p_h.unsqueeze(2) - inter_p.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )  # [B, L, 1, harmonics_per_head]

            # Frequency matching
            freq_match = torch.exp(
                -torch.abs(f_h.unsqueeze(2) - vocab_f.unsqueeze(0).unsqueeze(0))
            )  # [B, L, vocab_size, harmonics_per_head]

            # Amplitude-weighted interference
            interference = (a_h.unsqueeze(2) * freq_match * phase_alignment).sum(dim=-1)
            # [B, L, vocab_size]

            interference_scores.append(interference)

        # Stack and combine heads
        interference_scores = torch.stack(interference_scores, dim=-1)  # [B, L, vocab_size, num_heads]

        # Project to final scores
        logits = self.output_projection(interference_scores).squeeze(-1)  # [B, L, vocab_size]

        return logits


class HarmonicReconstructor(nn.Module):
    """Reconstruct discrete tokens from harmonic patterns"""

    def __init__(self, num_harmonics: int, vocab_size: int, num_basis: int = 128):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.vocab_size = vocab_size
        self.num_basis = num_basis

        # Learnable harmonic basis functions
        self.harmonic_basis = nn.Parameter(torch.randn(num_basis, num_harmonics))
        self.basis_to_vocab = nn.Linear(num_basis, vocab_size)

        # Phase-sensitive decoding
        self.phase_encoder = nn.Linear(num_harmonics, num_basis)

    def forward(self, freqs: torch.Tensor, amps: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct vocabulary items from harmonic decomposition
        """
        B, L, H = freqs.shape

        # Project waves onto harmonic basis
        # Weight by frequency similarity
        freq_similarity = torch.exp(-torch.cdist(freqs, self.harmonic_basis, p=1))  # [B, L, num_basis]

        # Phase-sensitive projection
        phase_features = torch.sin(self.phase_encoder(phases))  # [B, L, num_basis]

        # Amplitude-weighted combination
        amp_energy = amps.mean(dim=-1, keepdim=True)  # [B, L, 1]

        # Combine all factors
        basis_coefficients = freq_similarity * torch.sigmoid(phase_features) * amp_energy

        # Project to vocabulary
        logits = self.basis_to_vocab(basis_coefficients)

        return logits


class WaveAttention(nn.Module):
    """Attention mechanism for wave representations"""

    def __init__(self, num_harmonics: int, num_heads: int = 8):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.num_heads = num_heads
        self.head_dim = num_harmonics // num_heads

        # Separate projections for each wave component
        self.q_freq = nn.Linear(num_harmonics, num_harmonics)
        self.k_freq = nn.Linear(num_harmonics, num_harmonics)
        self.v_amp = nn.Linear(num_harmonics, num_harmonics)

        self.out_proj = nn.Linear(num_harmonics, num_harmonics)

    def forward(self, wave: 'Wave') -> 'Wave':
        B, L, H = wave.frequencies.shape

        # Compute queries and keys from frequencies (semantic similarity)
        Q = self.q_freq(wave.frequencies).view(B, L, self.num_heads, self.head_dim)
        K = self.k_freq(wave.frequencies).view(B, L, self.num_heads, self.head_dim)

        # Values from amplitudes (energy transfer)
        V = self.v_amp(wave.amplitudes).view(B, L, self.num_heads, self.head_dim)

        # Scaled dot-product attention
        scores = torch.einsum('blhd,bmhd->blmh', Q, K) / np.sqrt(self.head_dim)

        # Causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=scores.device), diagonal=1).bool()
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(-1), -1e9)

        attn_weights = F.softmax(scores, dim=2)

        # Apply attention to values
        attn_output = torch.einsum('blmh,bmhd->blhd', attn_weights, V)
        attn_output = attn_output.reshape(B, L, H)

        # Project and update amplitudes
        new_amps = self.out_proj(attn_output)

        # Phases are modulated by attention weights
        avg_attn = attn_weights.mean(dim=-1).mean(dim=1)  # [B, L]
        phase_modulation = (avg_attn - 0.5) * 0.1 * np.pi
        new_phases = wave.phases + phase_modulation.unsqueeze(-1)

        return Wave(wave.frequencies, new_amps, new_phases)


class WaveRefinementLayer(nn.Module):
    """Refine wave representation before decoding"""

    def __init__(self, num_harmonics: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # Self-attention in wave space
        self.wave_attention = WaveAttention(num_harmonics, num_heads)

        # Frequency-domain processing
        self.freq_processor = nn.Sequential(
            nn.Linear(num_harmonics, num_harmonics * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_harmonics * 2, num_harmonics)
        )

        # Amplitude modulation
        self.amp_modulator = nn.Sequential(
            nn.Linear(num_harmonics, num_harmonics),
            nn.Sigmoid()
        )

        # Phase correction
        self.phase_corrector = nn.Linear(num_harmonics, num_harmonics)

        # Layer norm
        self.norm = WaveNormalization(num_harmonics)

    def forward(self, wave: 'Wave') -> 'Wave':
        # Store for residual
        wave_residual = wave

        # Wave attention
        wave = self.wave_attention(wave)

        # Process each component
        new_freqs = wave.frequencies + 0.1 * self.freq_processor(wave.frequencies)
        new_amps = wave.amplitudes * self.amp_modulator(wave.amplitudes)
        phase_correction = self.phase_corrector(wave.frequencies)
        new_phases = wave.phases + 0.1 * torch.tanh(phase_correction) * np.pi

        # Create new wave
        wave = Wave(new_freqs, new_amps, new_phases)

        # Residual and norm
        wave = Wave(
            wave.frequencies + wave_residual.frequencies,
            wave.amplitudes + wave_residual.amplitudes,
            wave.phases + 0.1 * wave_residual.phases
        )

        wave = self.norm(wave)

        return wave