"""
Visualization utilities for WaveTransformer architecture
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from IPython.display import HTML

class WaveVisualization:
    """Visualization tools for SemanticWave and WaveTransformer components"""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
    
    def visualize_semantic_wave(
        self, 
        wave,  # SemanticWave object
        token_ids: Optional[torch.Tensor] = None,
        tokenizer = None,
        sample_harmonics: int = 8,
        seq_positions: Optional[List[int]] = None
    ):
        """
        Visualize the semantic wave representation
        
        Args:
            wave: SemanticWave object with frequencies, amplitudes, phases
            token_ids: Original token IDs for reference
            tokenizer: HF tokenizer for decoding tokens
            sample_harmonics: Number of harmonics to display
            seq_positions: Specific sequence positions to visualize
        """
        # Extract wave components
        freq = wave.frequencies.detach().cpu().numpy()
        amp = wave.amplitudes.detach().cpu().numpy()
        phase = wave.phases.detach().cpu().numpy()
        
        batch_size, seq_len, n_harmonics = freq.shape
        
        # Select batch and positions
        batch_idx = 0
        if seq_positions is None:
            seq_positions = list(range(min(8, seq_len)))
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Frequency spectrum heatmap
        ax1 = fig.add_subplot(gs[0, :])
        im1 = ax1.imshow(
            freq[batch_idx, :, :min(sample_harmonics*2, n_harmonics)].T,
            aspect='auto', cmap='viridis', interpolation='nearest'
        )
        ax1.set_title('Frequency Spectrum (First {} Harmonics)'.format(min(sample_harmonics*2, n_harmonics)))
        ax1.set_xlabel('Sequence Position')
        ax1.set_ylabel('Harmonic Index')
        plt.colorbar(im1, ax=ax1, label='Frequency (Hz)')
        
        # Add token labels if available
        if token_ids is not None and tokenizer is not None:
            tokens = tokenizer.convert_ids_to_tokens(token_ids[batch_idx].cpu().tolist())
            if len(tokens) <= 20:
                ax1.set_xticks(range(len(tokens)))
                ax1.set_xticklabels(tokens, rotation=45, ha='right')
        
        # 2. Amplitude distribution
        ax2 = fig.add_subplot(gs[1, 0])
        for pos in seq_positions[:4]:
            ax2.plot(amp[batch_idx, pos, :sample_harmonics], 
                    label=f'Pos {pos}', alpha=0.7, marker='o', markersize=3)
        ax2.set_title('Amplitude Distribution')
        ax2.set_xlabel('Harmonic Index')
        ax2.set_ylabel('Amplitude')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Phase distribution
        ax3 = fig.add_subplot(gs[1, 1])
        for pos in seq_positions[:4]:
            ax3.plot(phase[batch_idx, pos, :sample_harmonics], 
                    label=f'Pos {pos}', alpha=0.7, marker='s', markersize=3)
        ax3.set_title('Phase Distribution')
        ax3.set_xlabel('Harmonic Index')
        ax3.set_ylabel('Phase (radians)')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([-np.pi, np.pi])
        
        # 4. Wave reconstruction (time domain)
        ax4 = fig.add_subplot(gs[1, 2])
        t = np.linspace(0, 2*np.pi, 100)
        for pos_idx, pos in enumerate(seq_positions[:3]):
            wave_signal = np.zeros_like(t)
            for h in range(min(sample_harmonics, n_harmonics)):
                wave_signal += (amp[batch_idx, pos, h] * 
                               np.sin(freq[batch_idx, pos, h] * t + 
                                     phase[batch_idx, pos, h]))
            ax4.plot(t, wave_signal, label=f'Pos {pos}', alpha=0.8)
        
        ax4.set_title('Reconstructed Wave Signals')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Amplitude')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Energy per position (sum of amplitudes)
        ax5 = fig.add_subplot(gs[2, 0])
        energy = amp[batch_idx].sum(axis=1)
        ax5.bar(range(len(energy)), energy, color='steelblue', alpha=0.7)
        ax5.set_title('Energy per Position')
        ax5.set_xlabel('Sequence Position')
        ax5.set_ylabel('Total Energy')
        ax5.grid(True, alpha=0.3)
        
        # 6. Dominant frequencies
        ax6 = fig.add_subplot(gs[2, 1])
        top_k = 5
        for pos in seq_positions[:4]:
            top_harmonics = np.argsort(amp[batch_idx, pos])[-top_k:]
            dominant_freqs = freq[batch_idx, pos, top_harmonics]
            ax6.scatter([pos]*top_k, dominant_freqs, alpha=0.6, s=20)
        
        ax6.set_title(f'Top-{top_k} Dominant Frequencies')
        ax6.set_xlabel('Sequence Position')
        ax6.set_ylabel('Frequency (Hz)')
        ax6.grid(True, alpha=0.3)
        
        # 7. Phase-Amplitude polar plot
        ax7 = fig.add_subplot(gs[2, 2], projection='polar')
        for h in range(min(8, n_harmonics)):
            for pos in seq_positions[:3]:
                theta = phase[batch_idx, pos, h]
                r = amp[batch_idx, pos, h]
                ax7.scatter(theta, r, alpha=0.5, s=20)
        
        ax7.set_title('Phase-Amplitude Distribution\n(Polar)', pad=20)
        
        plt.suptitle('Semantic Wave Visualization', fontsize=14, y=1.02)
        plt.tight_layout()
        return fig
    
    def visualize_interference_pattern(
        self,
        wave,  # SemanticWave object
        decay_factors: Optional[torch.Tensor] = None,
        positions: List[int] = None
    ):
        """Visualize wave interference patterns"""
        
        if positions is None:
            positions = [0, 1, 2, 3]
        
        freq = wave.frequencies.detach().cpu().numpy()
        amp = wave.amplitudes.detach().cpu().numpy()
        phase = wave.phases.detach().cpu().numpy()
        
        batch_idx = 0
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        t = np.linspace(0, 4*np.pi, 200)
        
        for idx, (ax, pos) in enumerate(zip(axes.flat, positions)):
            # Individual harmonics
            for h in range(min(5, wave.frequencies.shape[-1])):
                signal = (amp[batch_idx, pos, h] * 
                         np.sin(freq[batch_idx, pos, h] * t + 
                               phase[batch_idx, pos, h]))
                ax.plot(t, signal, alpha=0.3, linewidth=1, 
                       label=f'H{h}' if idx == 0 else None)
            
            # Combined wave
            combined = np.zeros_like(t)
            for h in range(wave.frequencies.shape[-1]):
                combined += (amp[batch_idx, pos, h] * 
                           np.sin(freq[batch_idx, pos, h] * t + 
                                 phase[batch_idx, pos, h]))
            
            # Apply decay if provided
            if decay_factors is not None:
                decay = decay_factors[batch_idx, pos, 0].cpu().item()
                combined *= decay
                title = f'Position {pos} (decay={decay:.2f})'
            else:
                title = f'Position {pos}'
            
            ax.plot(t, combined, 'k-', linewidth=2, alpha=0.8, label='Combined')
            ax.set_title(title)
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                ax.legend(loc='upper right', fontsize=8)
        
        plt.suptitle('Wave Interference Patterns', fontsize=14)
        plt.tight_layout()
        return fig
    
    def animate_wave_propagation(
        self,
        wave,  # SemanticWave object
        save_path: Optional[str] = None,
        fps: int = 10
    ):
        """Create animation of wave propagation through sequence"""
        
        freq = wave.frequencies.detach().cpu().numpy()
        amp = wave.amplitudes.detach().cpu().numpy()
        phase = wave.phases.detach().cpu().numpy()
        
        batch_idx = 0
        seq_len = freq.shape[1]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        t = np.linspace(0, 2*np.pi, 100)
        line1, = ax1.plot([], [], 'b-', linewidth=2)
        scatter = ax2.scatter([], [], c=[], cmap='viridis', s=50)
        
        ax1.set_xlim(0, 2*np.pi)
        ax1.set_ylim(-10, 10)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlim(0, wave.frequencies.shape[-1])
        ax2.set_ylim(0, amp[batch_idx].max() * 1.1)
        ax2.set_xlabel('Harmonic Index')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
        
        def animate(frame):
            pos = frame % seq_len
            
            # Wave signal
            wave_signal = np.zeros_like(t)
            for h in range(wave.frequencies.shape[-1]):
                wave_signal += (amp[batch_idx, pos, h] * 
                               np.sin(freq[batch_idx, pos, h] * t + 
                                     phase[batch_idx, pos, h]))
            
            line1.set_data(t, wave_signal)
            ax1.set_title(f'Wave Signal at Position {pos}')
            
            # Amplitude spectrum
            harmonics = np.arange(wave.frequencies.shape[-1])
            scatter.set_offsets(np.c_[harmonics, amp[batch_idx, pos]])
            scatter.set_array(freq[batch_idx, pos])
            ax2.set_title(f'Harmonic Amplitudes at Position {pos}')
            
            return line1, scatter
        
        anim = animation.FuncAnimation(
            fig, animate, frames=seq_len*2, 
            interval=1000/fps, blit=True
        )
        
        plt.suptitle('Wave Propagation Animation', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            anim.save(save_path, fps=fps, writer='pillow')
        
        return anim
    
    def visualize_attention_weights(
        self,
        model,
        input_ids: torch.Tensor,
        layer_idx: int = 0,
        head_idx: Optional[int] = None
    ):
        """Visualize attention patterns in the transformer layers"""
        
        # This would require modifying the model to return attention weights
        # Placeholder for attention visualization
        print("Attention visualization requires model modification to return attention weights")
        pass
    
    def compare_waves(
        self,
        waves: List,  # List of SemanticWave objects
        labels: List[str],
        position: int = 0
    ):
        """Compare multiple wave representations"""
        
        n_waves = len(waves)
        fig, axes = plt.subplots(n_waves, 3, figsize=(12, 4*n_waves))
        
        if n_waves == 1:
            axes = axes.reshape(1, -1)
        
        t = np.linspace(0, 2*np.pi, 100)
        
        for idx, (wave, label) in enumerate(zip(waves, labels)):
            freq = wave.frequencies.detach().cpu().numpy()
            amp = wave.amplitudes.detach().cpu().numpy()
            phase = wave.phases.detach().cpu().numpy()
            
            batch_idx = 0
            
            # Frequency spectrum
            axes[idx, 0].stem(freq[batch_idx, position, :32], 
                            linefmt='b-', markerfmt='bo', basefmt=' ')
            axes[idx, 0].set_title(f'{label} - Frequencies')
            axes[idx, 0].set_xlabel('Harmonic')
            axes[idx, 0].set_ylabel('Frequency')
            
            # Amplitude spectrum
            axes[idx, 1].stem(amp[batch_idx, position, :32],
                            linefmt='g-', markerfmt='go', basefmt=' ')
            axes[idx, 1].set_title(f'{label} - Amplitudes')
            axes[idx, 1].set_xlabel('Harmonic')
            axes[idx, 1].set_ylabel('Amplitude')
            
            # Combined wave
            wave_signal = np.zeros_like(t)
            for h in range(wave.frequencies.shape[-1]):
                wave_signal += (amp[batch_idx, position, h] * 
                               np.sin(freq[batch_idx, position, h] * t + 
                                     phase[batch_idx, position, h]))
            
            axes[idx, 2].plot(t, wave_signal, 'r-', linewidth=2)
            axes[idx, 2].set_title(f'{label} - Combined Signal')
            axes[idx, 2].set_xlabel('Time')
            axes[idx, 2].set_ylabel('Amplitude')
            axes[idx, 2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Wave Comparison at Position {position}', fontsize=14)
        plt.tight_layout()
        return fig


# Extension to SemanticWave class with visualization
def add_visualization_to_semantic_wave():
    """Monkey-patch visualization methods to SemanticWave"""
    from wave_transformer_parts import SemanticWave
    
    def visualize(self, viz=None, **kwargs):
        """Visualize this wave"""
        if viz is None:
            viz = WaveVisualization()
        return viz.visualize_semantic_wave(self, **kwargs)
    
    def plot_interference(self, viz=None, **kwargs):
        """Plot interference pattern"""
        if viz is None:
            viz = WaveVisualization()
        return viz.visualize_interference_pattern(self, **kwargs)
    
    def animate(self, viz=None, **kwargs):
        """Animate wave propagation"""
        if viz is None:
            viz = WaveVisualization()
        return viz.animate_wave_propagation(self, **kwargs)
    
    # Add methods to class
    SemanticWave.visualize = visualize
    SemanticWave.plot_interference = plot_interference
    SemanticWave.animate = animate


# Model inspection utilities
class ModelInspector:
    """Tools for inspecting WaveTransformer internals"""
    
    @staticmethod
    def get_layer_outputs(model, input_ids, return_all=False):
        """Get outputs from each layer"""
        outputs = []
        
        # Encode to wave
        wave = model.wave_encoder(input_ids)
        x = wave.to_representation()
        outputs.append(('encoder', wave))
        
        # Each transformer layer
        for i, block in enumerate(model.layers):
            x = block(x)
            if return_all or i in [0, len(model.layers)//2, len(model.layers)-1]:
                outputs.append((f'layer_{i}', x.clone()))
        
        # Final norm
        x = model.norm_f(x)
        outputs.append(('final_norm', x.clone()))
        
        # Decoder
        logits = model.wave_decoder(x)
        outputs.append(('logits', logits))
        
        return outputs
    
    @staticmethod
    def visualize_embeddings(model, input_ids, layer='encoder'):
        """Visualize embeddings using PCA/t-SNE"""
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        with torch.no_grad():
            if layer == 'encoder':
                embeddings = model.wave_encoder.embedding(input_ids)
            else:
                # Get from specific layer
                outputs = ModelInspector.get_layer_outputs(model, input_ids, True)
                embeddings = dict(outputs)[layer]
        
        # Flatten for visualization
        emb_flat = embeddings.view(-1, embeddings.shape[-1]).cpu().numpy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # PCA
        pca = PCA(n_components=2)
        emb_pca = pca.fit_transform(emb_flat)
        ax1.scatter(emb_pca[:, 0], emb_pca[:, 1], alpha=0.5)
        ax1.set_title(f'PCA of {layer} embeddings')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        
        # t-SNE
        if emb_flat.shape[0] <= 1000:  # t-SNE is slow for large data
            tsne = TSNE(n_components=2, random_state=42)
            emb_tsne = tsne.fit_transform(emb_flat)
            ax2.scatter(emb_tsne[:, 0], emb_tsne[:, 1], alpha=0.5)
            ax2.set_title(f't-SNE of {layer} embeddings')
            ax2.set_xlabel('t-SNE 1')
            ax2.set_ylabel('t-SNE 2')
        else:
            ax2.text(0.5, 0.5, 'Too many points for t-SNE', 
                    transform=ax2.transAxes, ha='center')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def parameter_statistics(model):
        """Get parameter statistics for each module"""
        stats = {}
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    stats[name] = {
                        'parameters': params,
                        'size_mb': params * 4 / (1024**2),  # Assuming float32
                        'trainable': sum(p.numel() for p in module.parameters() if p.requires_grad)
                    }
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Sort by size
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['parameters'], reverse=True)[:15]
        
        names = [n.split('.')[-1] for n, _ in sorted_stats]
        params = [s['parameters'] for _, s in sorted_stats]
        sizes = [s['size_mb'] for _, s in sorted_stats]
        
        ax1.barh(names, params, color='steelblue')
        ax1.set_xlabel('Parameters')
        ax1.set_title('Top 15 Modules by Parameter Count')
        ax1.invert_yaxis()
        
        ax2.barh(names, sizes, color='coral')
        ax2.set_xlabel('Size (MB)')
        ax2.set_title('Top 15 Modules by Memory Size')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        return fig, stats


# Usage example function
def demo_visualization():
    """Demo of visualization capabilities"""
    from wave_transformer_parts import SemanticWave
    from hf_wave_transformer import WaveTransformerConfig, WaveTransformerForCausalLM
    
    # Enable visualization methods
    add_visualization_to_semantic_wave()
    
    # Create model
    config = WaveTransformerConfig(
        vocab_size=1000,
        d_model=256,
        num_layers=2,
        num_harmonics=32
    )
    model = WaveTransformerForCausalLM(config)
    
    # Sample input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Get wave representation
    wave = model.wave_encoder(input_ids)
    
    # Visualize
    viz = WaveVisualization()
    
    print("Visualization functions available:")
    print("1. wave.visualize() - Complete wave visualization")
    print("2. wave.plot_interference() - Interference patterns")
    print("3. wave.animate() - Animation of wave propagation")
    print("4. viz.compare_waves() - Compare multiple waves")
    print("5. ModelInspector methods - Model internals inspection")
    
    return model, wave, viz
