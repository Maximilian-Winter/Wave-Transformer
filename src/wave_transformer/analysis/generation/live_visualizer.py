"""
Live visualization of Wave Transformer generation process.

This module provides real-time visualization capabilities for autoregressive
generation, showing wave evolution as tokens are generated step-by-step.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
from pathlib import Path


class LiveGenerationVisualizer:
    """
    Visualize wave evolution during autoregressive text generation.

    Supports both real-time interactive visualization and post-hoc animation
    creation. Tracks wave statistics and token generation simultaneously.

    Example:
        >>> visualizer = LiveGenerationVisualizer(model, tokenizer)
        >>> output_ids, waves = visualizer.generate_with_visualization(
        ...     prompt="Once upon a time",
        ...     max_length=50,
        ...     temperature=0.8
        ... )
        >>> visualizer.create_animation("generation.mp4")
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Optional[object] = None,
        figsize: Tuple[int, int] = (16, 10),
        update_interval: int = 1,
        device: Optional[torch.device] = None
    ):
        """
        Initialize live generation visualizer.

        Args:
            model: Wave Transformer model with forward(encoder_input, return_encoder_outputs=True)
            tokenizer: Tokenizer with encode/decode methods (optional, for display)
            figsize: Figure size for visualization
            update_interval: Update plot every N tokens (1=every token)
            device: Device for generation (defaults to model's device)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.figsize = figsize
        self.update_interval = update_interval
        self.device = device if device is not None else next(model.parameters()).device

        # Storage for generation history
        self.token_history: List[int] = []
        self.wave_history: List = []  # List of Wave objects
        self.logits_history: List[torch.Tensor] = []
        self.probability_history: List[torch.Tensor] = []

        # Matplotlib figure and axes
        self.fig = None
        self.axes = None
        self.is_interactive = False

    def setup_figure(self, interactive: bool = True) -> Tuple[plt.Figure, np.ndarray]:
        """
        Initialize matplotlib figure with subplots for live visualization.

        Args:
            interactive: If True, use plt.ion() for live updates

        Returns:
            Tuple of (figure, axes array)
        """
        if interactive:
            plt.ion()
            self.is_interactive = True
        else:
            plt.ioff()
            self.is_interactive = False

        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        fig.suptitle('Wave Transformer - Live Generation', fontsize=14, fontweight='bold')

        # Configure subplots
        # Row 1: Wave heatmaps (freq, amp, phase)
        axes[0, 0].set_title('Frequencies')
        axes[0, 0].set_xlabel('Sequence Position')
        axes[0, 0].set_ylabel('Harmonic Index')

        axes[0, 1].set_title('Amplitudes')
        axes[0, 1].set_xlabel('Sequence Position')
        axes[0, 1].set_ylabel('Harmonic Index')

        axes[0, 2].set_title('Phases')
        axes[0, 2].set_xlabel('Sequence Position')
        axes[0, 2].set_ylabel('Harmonic Index')

        # Row 2: Statistics over time
        axes[1, 0].set_title('Total Energy per Position')
        axes[1, 0].set_xlabel('Sequence Position')
        axes[1, 0].set_ylabel('Energy')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].set_title('Token Probability')
        axes[1, 1].set_xlabel('Generation Step')
        axes[1, 1].set_ylabel('Max Probability')
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].set_title('Spectral Centroid')
        axes[1, 2].set_xlabel('Sequence Position')
        axes[1, 2].set_ylabel('Centroid (Hz)')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        self.fig = fig
        self.axes = axes

        return fig, axes

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample next token from logits with temperature and optional top-k/top-p.

        Args:
            logits: Logits tensor [batch_size, vocab_size]
            temperature: Sampling temperature (higher=more random)
            top_k: If set, only sample from top k tokens
            top_p: If set, nucleus sampling threshold

        Returns:
            Tuple of (sampled_token_id [batch_size, 1], probabilities [batch_size, vocab_size])
        """
        # Apply temperature
        logits = logits / temperature

        # Top-k filtering
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Top-p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove[..., 0] = False

            # Scatter back to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)

        # Sample
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token, probs

    def _update_plot(self, step: int, wave, text_so_far: str = ""):
        """
        Update live plot with latest wave and statistics.

        Args:
            step: Current generation step
            wave: Latest Wave object from encoder
            text_so_far: Generated text for display
        """
        if self.fig is None or self.axes is None:
            return

        # Extract wave components [batch=1, seq_len, num_harmonics]
        freqs = wave.frequencies[0].detach().cpu().numpy()  # [S, H]
        amps = wave.amplitudes[0].detach().cpu().numpy()
        phases = wave.phases[0].detach().cpu().numpy()

        seq_len = freqs.shape[0]

        # Update heatmaps
        self.axes[0, 0].clear()
        self.axes[0, 0].imshow(freqs.T, aspect='auto', cmap='viridis', interpolation='nearest')
        self.axes[0, 0].set_title('Frequencies')
        self.axes[0, 0].set_xlabel('Sequence Position')
        self.axes[0, 0].set_ylabel('Harmonic')

        self.axes[0, 1].clear()
        self.axes[0, 1].imshow(amps.T, aspect='auto', cmap='plasma', interpolation='nearest')
        self.axes[0, 1].set_title('Amplitudes')
        self.axes[0, 1].set_xlabel('Sequence Position')
        self.axes[0, 1].set_ylabel('Harmonic')

        self.axes[0, 2].clear()
        self.axes[0, 2].imshow(phases.T, aspect='auto', cmap='twilight', interpolation='nearest')
        self.axes[0, 2].set_title('Phases')
        self.axes[0, 2].set_xlabel('Sequence Position')
        self.axes[0, 2].set_ylabel('Harmonic')

        # Update energy plot
        energy_per_pos = (amps ** 2).sum(axis=1)  # [S]
        self.axes[1, 0].clear()
        self.axes[1, 0].plot(energy_per_pos, color='purple', linewidth=2)
        self.axes[1, 0].set_title('Total Energy per Position')
        self.axes[1, 0].set_xlabel('Sequence Position')
        self.axes[1, 0].set_ylabel('Energy')
        self.axes[1, 0].grid(True, alpha=0.3)

        # Update probability plot
        if len(self.probability_history) > 0:
            max_probs = [p.max().item() for p in self.probability_history]
            self.axes[1, 1].clear()
            self.axes[1, 1].plot(max_probs, marker='o', markersize=4, color='green')
            self.axes[1, 1].set_title('Token Probability')
            self.axes[1, 1].set_xlabel('Generation Step')
            self.axes[1, 1].set_ylabel('Max Probability')
            self.axes[1, 1].set_ylim([0, 1.05])
            self.axes[1, 1].grid(True, alpha=0.3)

        # Update spectral centroid
        eps = 1e-8
        centroid = (freqs * amps).sum(axis=1) / (amps.sum(axis=1) + eps)
        self.axes[1, 2].clear()
        self.axes[1, 2].plot(centroid, color='orange', linewidth=2)
        self.axes[1, 2].set_title('Spectral Centroid')
        self.axes[1, 2].set_xlabel('Sequence Position')
        self.axes[1, 2].set_ylabel('Centroid (Hz)')
        self.axes[1, 2].grid(True, alpha=0.3)

        # Update title with progress
        title = f'Wave Transformer - Generation Step {step}'
        if text_so_far:
            title += f'\nText: "{text_so_far}"'
        self.fig.suptitle(title, fontsize=12, fontweight='bold')

        if self.is_interactive:
            plt.pause(0.01)  # Allow GUI to update

    @torch.no_grad()
    def generate_with_visualization(
        self,
        prompt: Optional[str] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        interactive: bool = True,
        show_text: bool = True
    ) -> Tuple[torch.Tensor, List]:
        """
        Generate tokens autoregressively with live wave visualization.

        Args:
            prompt: Text prompt (requires tokenizer)
            prompt_ids: Token IDs as tensor [1, prompt_len] (alternative to prompt)
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            interactive: Enable live plot updates
            show_text: Show decoded text in plot (requires tokenizer)

        Returns:
            Tuple of (generated_ids [1, seq_len], wave_history list)
        """
        # Setup
        self.model.eval()
        self.token_history.clear()
        self.wave_history.clear()
        self.logits_history.clear()
        self.probability_history.clear()

        # Initialize prompt
        if prompt_ids is not None:
            current_ids = prompt_ids.to(self.device)
        elif prompt is not None and self.tokenizer is not None:
            tokenized =  self.tokenizer.encode(prompt)
            current_ids = torch.tensor(tokenized.ids, dtype=torch.long, device=self.device).unsqueeze(0)
        else:
            raise ValueError("Either prompt or prompt_ids must be provided")

        # Setup figure
        self.setup_figure(interactive=interactive)

        # Generation loop
        for step in range(max_length):
            # Forward pass with encoder wave return
            logits, encoder_wave = self.model(
                current_ids,
                return_encoder_outputs=True
            )

            # Get logits for last position
            next_logits = logits[:, -1, :]  # [1, vocab_size]

            # Sample next token
            next_token, probs = self._sample_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            # Store history
            self.token_history.append(next_token.item())
            self.wave_history.append(encoder_wave)
            self.logits_history.append(next_logits.cpu())
            self.probability_history.append(probs[0].cpu())

            # Update visualization every update_interval steps
            if (step + 1) % self.update_interval == 0 or step == max_length - 1:
                text_so_far = ""
                if show_text and self.tokenizer is not None:
                    try:
                        text_so_far = self.tokenizer.decode(current_ids[0])
                        # Truncate if too long for display
                        if len(text_so_far) > 100:
                            text_so_far = text_so_far[:97] + "..."
                    except:
                        text_so_far = f"{len(self.token_history)} tokens generated"
                else:
                    text_so_far = f"{len(self.token_history)} tokens generated"

                self._update_plot(step + 1, encoder_wave, text_so_far)

            # Append token to sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)

            # Check for EOS token (if tokenizer available)
            if self.tokenizer is not None and hasattr(self.tokenizer, 'eos_token_id'):
                if next_token.item() == self.tokenizer.eos_token_id:
                    print(f"EOS token reached at step {step + 1}")
                    break

        if interactive:
            plt.ioff()
            plt.show()

        return current_ids, self.wave_history

    def create_animation(
        self,
        save_path: str,
        fps: int = 2,
        dpi: int = 100,
        format: str = 'mp4'
    ) -> None:
        """
        Create MP4 or GIF animation from stored wave history.

        Args:
            save_path: Path to save animation (e.g., 'generation.mp4')
            fps: Frames per second
            dpi: Resolution
            format: 'mp4' or 'gif'
        """
        if len(self.wave_history) == 0:
            raise ValueError("No generation history available. Run generate_with_visualization first.")

        # Create new figure for animation
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)

        def animate(frame_idx: int):
            """Animation update function."""
            wave = self.wave_history[frame_idx]

            # Clear all axes
            for ax_row in axes:
                for ax in ax_row:
                    ax.clear()

            # Extract components
            freqs = wave.frequencies[0].detach().cpu().numpy()
            amps = wave.amplitudes[0].detach().cpu().numpy()
            phases = wave.phases[0].detach().cpu().numpy()

            # Plot heatmaps
            axes[0, 0].imshow(freqs.T, aspect='auto', cmap='viridis', interpolation='nearest')
            axes[0, 0].set_title('Frequencies')
            axes[0, 0].set_xlabel('Position')
            axes[0, 0].set_ylabel('Harmonic')

            axes[0, 1].imshow(amps.T, aspect='auto', cmap='plasma', interpolation='nearest')
            axes[0, 1].set_title('Amplitudes')
            axes[0, 1].set_xlabel('Position')
            axes[0, 1].set_ylabel('Harmonic')

            axes[0, 2].imshow(phases.T, aspect='auto', cmap='twilight', interpolation='nearest')
            axes[0, 2].set_title('Phases')
            axes[0, 2].set_xlabel('Position')
            axes[0, 2].set_ylabel('Harmonic')

            # Plot statistics
            energy = (amps ** 2).sum(axis=1)
            axes[1, 0].plot(energy, color='purple', linewidth=2)
            axes[1, 0].set_title('Total Energy')
            axes[1, 0].set_xlabel('Position')
            axes[1, 0].set_ylabel('Energy')
            axes[1, 0].grid(True, alpha=0.3)

            # Probability history up to current frame
            if frame_idx < len(self.probability_history):
                max_probs = [p.max().item() for p in self.probability_history[:frame_idx+1]]
                axes[1, 1].plot(max_probs, marker='o', markersize=4, color='green')
                axes[1, 1].set_title('Token Probability')
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].set_ylabel('Max Prob')
                axes[1, 1].set_ylim([0, 1.05])
                axes[1, 1].grid(True, alpha=0.3)

            # Spectral centroid
            eps = 1e-8
            centroid = (freqs * amps).sum(axis=1) / (amps.sum(axis=1) + eps)
            axes[1, 2].plot(centroid, color='orange', linewidth=2)
            axes[1, 2].set_title('Spectral Centroid')
            axes[1, 2].set_xlabel('Position')
            axes[1, 2].set_ylabel('Centroid')
            axes[1, 2].grid(True, alpha=0.3)

            fig.suptitle(f'Generation Step {frame_idx + 1}/{len(self.wave_history)}',
                        fontsize=14, fontweight='bold')

            plt.tight_layout()

        # Create animation
        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=len(self.wave_history),
            interval=1000/fps,
            blit=False
        )

        # Save
        save_path = Path(save_path)
        if format == 'mp4':
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(str(save_path), writer=writer, dpi=dpi)
        elif format == 'gif':
            anim.save(str(save_path), writer='pillow', fps=fps, dpi=dpi)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'mp4' or 'gif'")

        plt.close(fig)
        print(f"Animation saved to {save_path}")

    def get_generation_summary(self) -> Dict:
        """
        Get summary statistics of the generation process.

        Returns:
            Dictionary with generation statistics
        """
        if len(self.wave_history) == 0:
            return {}

        # Compute statistics across generation
        energies = []
        centroids = []
        max_probs = []

        for wave, probs in zip(self.wave_history, self.probability_history):
            amps = wave.amplitudes[0].detach().cpu().numpy()
            freqs = wave.frequencies[0].detach().cpu().numpy()

            energy = (amps ** 2).sum()
            centroid = (freqs * amps).sum() / (amps.sum() + 1e-8)

            energies.append(energy)
            centroids.append(centroid)
            max_probs.append(probs.max().item())

        return {
            'num_tokens_generated': len(self.token_history),
            'energy': {
                'mean': np.mean(energies),
                'std': np.std(energies),
                'min': np.min(energies),
                'max': np.max(energies),
            },
            'spectral_centroid': {
                'mean': np.mean(centroids),
                'std': np.std(centroids),
                'min': np.min(centroids),
                'max': np.max(centroids),
            },
            'probability': {
                'mean': np.mean(max_probs),
                'std': np.std(max_probs),
                'min': np.min(max_probs),
                'max': np.max(max_probs),
            },
            'tokens': self.token_history,
        }
