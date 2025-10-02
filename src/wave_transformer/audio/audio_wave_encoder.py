"""
AudioToWave: The Sacred Bridge Between Sound and Meaning
Dancing between raw audio and semantic wave representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math
from typing import Tuple

from wave_transformer.core.transformer import Wave, RMSNorm, MultiQueryFlashAttention


class AudioToWave(nn.Module):
    """
    Transform raw audio waveforms into Wave representations
    Learning to extract meaningful harmonics from the chaos of sound!
    """

    def __init__(
            self,
            num_harmonics: int = 64,
            sample_rate: int = 16000,
            n_fft: int = 1024,
            hop_length: int = 256,
            n_mels: int = 128,
            learnable_filterbank: bool = True,
            use_raw_waveform: bool = False,
            d_model: int = 512,
            num_heads: int = 8,
            num_layers: int = 3,
            dropout: float = 0.1,
            freq_range: Tuple[float, float] = (20.0, 8000.0),  # Hz
    ):
        super().__init__()

        self.num_harmonics = num_harmonics
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.use_raw_waveform = use_raw_waveform

        # APPROACH 1: Learnable Filterbank (The Dakini's Choice!)
        if learnable_filterbank:
            self.filterbank = LearnableFilterbank(
                num_filters=num_harmonics,
                sample_rate=sample_rate,
                freq_range=freq_range,
                n_fft=n_fft
            )
        else:
            # APPROACH 2: Fixed Mel-filterbank (Traditional but reliable)
            self.mel_scale = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=freq_range[0],
                f_max=freq_range[1]
            )
            # Project mel bands to harmonics
            self.mel_to_harmonics = nn.Linear(n_mels, num_harmonics)

        self.learnable_filterbank = learnable_filterbank

        # APPROACH 3: Direct waveform processing (The Wild Path!)
        if use_raw_waveform:
            self.wave_encoder = WaveformEncoder(
                d_model=d_model,
                num_harmonics=num_harmonics,
                kernel_sizes=[3, 5, 7, 11, 17, 31]  # Multi-scale kernels
            )

        # Semantic processing layers - transform acoustic to semantic
        self.semantic_encoder = nn.ModuleList([
            SemanticTransformBlock(
                num_harmonics * 3,  # freq + amp + phase
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Learn semantic frequency mapping
        self.freq_transform = nn.Sequential(
            nn.Linear(num_harmonics, num_harmonics * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_harmonics * 2, num_harmonics),
            nn.Sigmoid()  # Output 0-1 for frequency scaling
        )

        # Learn semantic amplitude weighting
        self.amp_transform = nn.Sequential(
            nn.Linear(num_harmonics, num_harmonics * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_harmonics * 2, num_harmonics),
            nn.Softplus()  # Always positive amplitudes
        )

        # Learn semantic phase relationships
        self.phase_transform = nn.Sequential(
            nn.Linear(num_harmonics, num_harmonics * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_harmonics * 2, num_harmonics),
            nn.Tanh()  # Output -1 to 1, scale to -π to π
        )

        # Temporal modeling - audio has strong temporal dependencies!
        self.temporal_attention = MultiQueryFlashAttention(
            d_model=num_harmonics * 3,
            n_heads_q=num_heads,
            n_heads_kv=num_heads // 2,  # MQA for efficiency
            dropout_p=dropout
        )

        self.norm = RMSNorm(num_harmonics * 3)

    def forward(
            self,
            audio: torch.Tensor,
            return_raw_spectrum: bool = False
    ) -> Wave:
        """
        Transform audio to Wave

        Args:
            audio: Raw waveform (batch, samples) or (batch, channels, samples)
            return_raw_spectrum: Also return the intermediate spectrum

        Returns:
            Wave with learned semantic harmonics
        """

        # Handle stereo -> mono if needed
        if audio.dim() == 3 and audio.size(1) > 1:
            audio = audio.mean(dim=1)  # Average channels
        elif audio.dim() == 3:
            audio = audio.squeeze(1)

        # Extract base features based on approach
        if self.use_raw_waveform:
            # Wild path: Direct waveform processing
            features = self.wave_encoder(audio)
            raw_spectrum = None
        else:
            # Get spectral representation
            if self.learnable_filterbank:
                # Learnable filterbank - THE WAY!
                spectrum = torch.stft(
                    audio,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    return_complex=True,
                    window=torch.hann_window(self.n_fft, device=audio.device)
                )
                # Apply learnable filters
                features = self.filterbank(spectrum)  # (batch, time, harmonics)
                raw_spectrum = spectrum if return_raw_spectrum else None
            else:
                # Traditional mel-spectrogram
                mel_spec = self.mel_scale(audio)  # (batch, n_mels, time)
                mel_spec = mel_spec.transpose(1, 2)  # (batch, time, n_mels)

                # Project to harmonic space
                features = self.mel_to_harmonics(mel_spec)  # (batch, time, harmonics)
                raw_spectrum = mel_spec if return_raw_spectrum else None

        # Split into magnitude and phase
        if torch.is_complex(features):
            magnitudes = torch.abs(features)
            phases = torch.angle(features)
        else:
            # For real features, create synthetic phases
            magnitudes = F.relu(features)  # Ensure positive
            # Generate phases through learned transformation
            phases = self.phase_transform(features) * math.pi

        # Initial wave representation
        init_freq = torch.sigmoid(magnitudes) * 20.0 + 0.1  # Scale to reasonable frequency range
        init_amp = magnitudes
        init_phase = phases

        # Combine for semantic processing
        combined = torch.cat([init_freq, init_amp, init_phase], dim=-1)

        # Apply temporal attention - understand context!
        combined = combined + self.temporal_attention(combined, causal=False)
        combined = self.norm(combined)

        # Semantic transformation blocks
        for block in self.semantic_encoder:
            combined = block(combined)

        # Split back into components
        freq_feat, amp_feat, phase_feat = combined.chunk(3, dim=-1)

        # Final semantic transformations
        semantic_freq = self.freq_transform(freq_feat) * 20.0 + 0.1  # 0.1 to 20.1
        semantic_amp = self.amp_transform(amp_feat)
        semantic_phase = self.phase_transform(phase_feat) * math.pi

        wave = Wave(
            frequencies=semantic_freq,
            amplitudes=semantic_amp,
            phases=semantic_phase
        )

        if return_raw_spectrum:
            return wave, raw_spectrum
        return wave


class LearnableFilterbank(nn.Module):
    """
    Learnable frequency decomposition - Let the model decide what frequencies matter!
    The Dakini's gift of adaptive perception!
    """

    def __init__(
            self,
            num_filters: int = 64,
            sample_rate: int = 16000,
            freq_range: Tuple[float, float] = (20.0, 8000.0),
            n_fft: int = 1024,
            init_mel: bool = True  # Initialize with mel-scale frequencies
    ):
        super().__init__()

        self.num_filters = num_filters
        self.sample_rate = sample_rate
        self.n_fft = n_fft

        # Frequency bin centers (learnable!)
        if init_mel:
            # Initialize with mel-scale spacing
            mel_min = 2595 * math.log10(1 + freq_range[0] / 700)
            mel_max = 2595 * math.log10(1 + freq_range[1] / 700)
            mel_points = torch.linspace(mel_min, mel_max, num_filters)
            freq_points = 700 * (10 ** (mel_points / 2595) - 1)
        else:
            # Linear initialization
            freq_points = torch.linspace(freq_range[0], freq_range[1], num_filters)

        # Convert to FFT bins
        freq_bins = freq_points * n_fft / sample_rate
        self.center_freqs = nn.Parameter(freq_bins)

        # Learnable bandwidths for each filter
        init_bandwidth = torch.ones(num_filters) * (n_fft / num_filters / 4)
        self.bandwidths = nn.Parameter(init_bandwidth)

        # Learnable filter shapes (asymmetric!)
        self.filter_shapes = nn.Parameter(torch.ones(num_filters, 2))  # Left/right slopes

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable filterbank to spectrum

        Args:
            spectrum: Complex STFT (batch, freq_bins, time)

        Returns:
            Filtered outputs (batch, time, num_filters) - complex values!
        """
        batch_size, freq_bins, time_steps = spectrum.shape
        device = spectrum.device

        # Create frequency axis
        freqs = torch.arange(freq_bins, device=device).unsqueeze(0)  # (1, freq_bins)

        # Build filters
        filters = []
        for i in range(self.num_filters):
            center = self.center_freqs[i]
            bandwidth = F.softplus(self.bandwidths[i]) + 1e-3  # Ensure positive
            left_slope, right_slope = F.softplus(self.filter_shapes[i]) + 0.1

            # Asymmetric triangular filter (learnable shape!)
            left_edge = center - bandwidth * left_slope
            right_edge = center + bandwidth * right_slope

            # Create filter
            filter_response = torch.zeros(freq_bins, device=device)

            # Rising edge
            rise_mask = (freqs >= left_edge) & (freqs <= center)
            filter_response = torch.where(
                rise_mask.squeeze(),
                (freqs.squeeze() - left_edge) / (center - left_edge + 1e-8),
                filter_response
            )

            # Falling edge
            fall_mask = (freqs > center) & (freqs <= right_edge)
            filter_response = torch.where(
                fall_mask.squeeze(),
                1.0 - (freqs.squeeze() - center) / (right_edge - center + 1e-8),
                filter_response
            )

            filters.append(filter_response)

        # Stack filters
        filterbank = torch.stack(filters, dim=0)  # (num_filters, freq_bins)

        # Apply filters - preserve complex values!
        spectrum_flat = spectrum.reshape(batch_size, freq_bins, -1)  # Flatten time
        filterbank = filterbank.to(spectrum_flat.dtype)
        filtered = torch.matmul(filterbank, spectrum_flat)  # (num_filters, batch*time)
        filtered = filtered.reshape(self.num_filters, batch_size, time_steps)
        filtered = filtered.permute(1, 2, 0)  # (batch, time, num_filters)

        return filtered


class WaveformEncoder(nn.Module):
    """
    Direct waveform processing - The Wild Path!
    For when you want to learn EVERYTHING from scratch!
    """

    def __init__(
            self,
            d_model: int = 512,
            num_harmonics: int = 64,
            kernel_sizes: list = [3, 5, 7, 11, 17, 31]
    ):
        super().__init__()

        # Multi-scale convolutions
        self.convs = nn.ModuleList([
            nn.Conv1d(1, d_model // len(kernel_sizes), kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])

        # Combine multi-scale features
        self.combine = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(d_model, num_harmonics * 3, kernel_size=1)  # freq, amp, phase
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Process raw waveform directly

        Args:
            waveform: (batch, samples)

        Returns:
            Features (batch, time, harmonics * 3)
        """
        x = waveform.unsqueeze(1)  # (batch, 1, samples)

        # Multi-scale processing
        multi_scale = []
        for conv in self.convs:
            multi_scale.append(conv(x))

        # Concatenate scales
        x = torch.cat(multi_scale, dim=1)  # (batch, d_model, time)

        # Combine and project
        x = self.combine(x)  # (batch, harmonics * 3, time)
        x = x.transpose(1, 2)  # (batch, time, harmonics * 3)

        return x


class SemanticTransformBlock(nn.Module):
    """
    Transform acoustic patterns into semantic patterns
    The alchemy of meaning-making!
    """

    def __init__(
            self,
            d_model: int,
            num_heads: int = 8,
            dropout: float = 0.1
    ):
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # Self-attention for finding patterns
        self.attention = MultiQueryFlashAttention(
            d_model=d_model,
            n_heads_q=num_heads,
            n_heads_kv=num_heads // 2,
            dropout_p=dropout
        )

        # FFN for semantic transformation
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention for pattern finding
        x = x + self.attention(self.norm1(x), causal=False)

        # FFN for semantic transformation
        x = x + self.ffn(self.norm2(x))

        return x


# Test the sacred creation!
if __name__ == "__main__":
    print("🔥 MANIFESTING THE AUDIO-TO-WAVE BRIDGE! 🔥")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create the transformer of sound
    audio_encoder = AudioToWave(
        num_harmonics=64,
        sample_rate=16000,
        learnable_filterbank=True,
        num_layers=3
    ).to(device)

    # Test with random audio
    batch_size = 2
    audio_length = 16000  # 1 second
    fake_audio = torch.randn(batch_size, audio_length).to(device)

    # Transform to waves!
    semantic_waves = audio_encoder(fake_audio)

    print(f"\n💀 WAVE MANIFESTATION COMPLETE! 💀")
    print(f"Frequencies shape: {semantic_waves.frequencies.shape}")
    print(f"Amplitudes shape: {semantic_waves.amplitudes.shape}")
    print(f"Phases shape: {semantic_waves.phases.shape}")

    # Test conversion to representation
    representation = semantic_waves.to_representation()
    print(f"\nCombined representation: {representation.shape}")
    print(f"Ready to feed into WaveTransformer! 🌊")

    # Parameter count
    params = sum(p.numel() for p in audio_encoder.parameters())
    print(f"\nTotal parameters: {params:,}")
    print("\n✨ THE BRIDGE IS BUILT! AUDIO AND SEMANTICS DANCE AS ONE! ✨")