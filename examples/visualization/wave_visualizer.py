import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path


class WaveVisualizer:
    def __init__(self, save_dir="wave_viz"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def plot_heatmap(self, tensor: torch.Tensor, title: str, filename: str):
        """
        tensor: [seq_len, harmonics] (choose one sample from batch before calling)
        """
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            tensor.cpu().detach().numpy(),
            cmap="magma",
            cbar=True
        )
        plt.title(title)
        plt.xlabel("Harmonics")
        plt.ylabel("Sequence Position")
        plt.tight_layout()
        plt.savefig(self.save_dir / filename)
        plt.close()

    def plot_distribution(self, tensor: torch.Tensor, title: str, filename: str):
        """
        Flattened histogram of values.
        """
        values = tensor.detach().cpu().numpy().flatten()
        plt.figure(figsize=(6, 4))
        sns.histplot(values, bins=50, kde=True)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(self.save_dir / filename)
        plt.close()

    def visualize_wave(self, wave, token_idx: int = 0):
        """
        wave: Wave object (frequencies, amplitudes, phases)
        token_idx: which token trajectory to plot
        """
        freqs = wave.frequencies[0]
        amps = wave.amplitudes[0]
        phs = wave.phases[0]

        # Heatmaps
        self.plot_heatmap(freqs, "Frequencies Heatmap", "frequencies.png")
        self.plot_heatmap(amps, "Amplitudes Heatmap", "amplitudes.png")
        self.plot_heatmap(phs, "Phases Heatmap", "phases.png")

        # Distributions
        self.plot_distribution(freqs, "Frequency Distribution", "freq_dist.png")
        self.plot_distribution(amps, "Amplitude Distribution", "amp_dist.png")
        self.plot_distribution(phs, "Phase Distribution", "phase_dist.png")

        # Token trajectory plot
        freqs_tok = freqs[token_idx].detach().cpu().numpy()
        amps_tok = amps[token_idx].detach().cpu().numpy()
        phs_tok = phs[token_idx].detach().cpu().numpy()

        plt.figure(figsize=(6, 6))
        plt.plot(freqs_tok, label="freq")
        plt.plot(amps_tok, label="amp")
        plt.plot(phs_tok, label="phase")
        plt.legend()
        plt.title(f"Token {token_idx} Harmonics")
        plt.tight_layout()
        plt.savefig(self.save_dir / f"token_{token_idx}_trajectory.png")
        plt.close()
