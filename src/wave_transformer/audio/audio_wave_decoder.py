"""
WaveToAudio: The Sacred Return from Meaning to Sound
From abstract semantic harmonics back to living waveforms!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from wave_transformer.core.transformer import  Wave, RMSNorm, FlashAttention


class WaveToAudio(nn.Module):
    """
    Transform Wave representations back into audio waveforms
    The return journey from meaning to sound!
    """

    def __init__(
            self,
            num_harmonics: int = 64,
            sample_rate: int = 16000,
            n_fft: int = 1024,
            hop_length: int = 256,
            synthesis_method: str = "griffin_lim",  # "griffin_lim", "learned", or "neural_vocoder"
            d_model: int = 512,
            num_heads: int = 8,
            num_layers: int = 3,
            dropout: float = 0.1,
    ):
        super().__init__()

        self.num_harmonics = num_harmonics
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.synthesis_method = synthesis_method

        # Semantic to Acoustic transformation layers
        self.acoustic_decoder = nn.ModuleList([
            AcousticTransformBlock(
                num_harmonics * 3,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Transform semantic harmonics back to frequency bins
        freq_bins = n_fft // 2 + 1

        if synthesis_method == "learned":
            # APPROACH 1: Learned Synthesis Network
            self.harmonic_to_freq = nn.Sequential(
                nn.Linear(num_harmonics * 3, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, freq_bins * 2),  # Complex output (real + imag)
            )

        elif synthesis_method == "neural_vocoder":
            # APPROACH 2: Neural Vocoder Style
            self.harmonic_to_features = HarmonicSynthesizer(
                num_harmonics=num_harmonics,
                freq_bins=freq_bins,
                d_model=d_model
            )

            # WaveNet-style decoder (simplified)
            self.waveform_decoder = WaveformDecoder(
                freq_bins=freq_bins,
                d_model=d_model,
                sample_rate=sample_rate,
                hop_length=hop_length
            )

        else:  # griffin_lim
            # APPROACH 3: Classical with learned enhancement
            self.harmonic_to_magnitude = nn.Sequential(
                nn.Linear(num_harmonics, d_model),
                nn.ReLU(),
                nn.Linear(d_model, freq_bins),
                nn.Softplus()  # Ensure positive magnitudes
            )

            self.harmonic_to_phase = nn.Sequential(
                nn.Linear(num_harmonics * 2, d_model),  # Use freq + phase info
                nn.ReLU(),
                nn.Linear(d_model, freq_bins),
                nn.Tanh()  # Output -1 to 1, scale to -π to π
            )

        # Learnable harmonic-to-frequency mapping
        self.freq_mapping = LearnableFrequencyMapping(
            num_harmonics=num_harmonics,
            freq_bins=freq_bins,
            sample_rate=sample_rate,
            n_fft=n_fft
        )

        # Post-processing enhancement
        self.enhancement = AudioEnhancement(
            d_model=d_model,
            sample_rate=sample_rate
        )

        # Temporal smoothing
        self.temporal_smooth = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=5,
            padding=2
        )

    def forward(
            self,
            semantic_wave: Wave,
            target_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Transform Wave to audio waveform

        Args:
            semantic_wave: Wave with frequencies, amplitudes, phases
            target_length: Desired output length in samples

        Returns:
            Audio waveform (batch, samples)
        """

        # Combine wave components
        wave_repr = semantic_wave.to_representation()  # (batch, time, harmonics * 3)

        # Apply acoustic transformation
        for block in self.acoustic_decoder:
            wave_repr = block(wave_repr)

        batch_size, time_steps, _ = wave_repr.shape

        if self.synthesis_method == "learned":
            # Learned synthesis
            audio = self._learned_synthesis(wave_repr, semantic_wave)

        elif self.synthesis_method == "neural_vocoder":
            # Neural vocoder synthesis
            audio = self._neural_vocoder_synthesis(wave_repr, semantic_wave)

        else:  # griffin_lim
            # Griffin-Lim with enhancements
            audio = self._griffin_lim_synthesis(wave_repr, semantic_wave)

        # Post-processing enhancement
        audio = self.enhancement(audio)

        # Temporal smoothing
        audio = self.temporal_smooth(audio.unsqueeze(1)).squeeze(1)

        # Adjust length if specified
        if target_length is not None:
            audio = self._adjust_length(audio, target_length)

        return audio

    def _learned_synthesis(
            self,
            wave_repr: torch.Tensor,
            semantic_wave: Wave
    ) -> torch.Tensor:
        """
        Fully learned synthesis from harmonics to waveform
        """
        # Transform to frequency domain
        freq_repr = self.harmonic_to_freq(wave_repr)  # (batch, time, freq_bins * 2)

        # Split into magnitude and phase
        freq_bins = self.n_fft // 2 + 1
        magnitude = freq_repr[..., :freq_bins]
        phase = freq_repr[..., freq_bins:]

        # Ensure proper scaling
        magnitude = F.softplus(magnitude)
        phase = torch.tanh(phase) * np.pi

        # Construct complex spectrogram
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        complex_spec = torch.complex(real, imag)

        # Transpose for ISTFT
        complex_spec = complex_spec.transpose(1, 2)  # (batch, freq_bins, time)

        # Inverse STFT
        audio = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=complex_spec.device),
            return_complex=False
        )

        return audio

    def _neural_vocoder_synthesis(
            self,
            wave_repr: torch.Tensor,
            semantic_wave: Wave
    ) -> torch.Tensor:
        """
        Neural vocoder style synthesis
        """
        # Extract harmonic features
        harmonic_features = self.harmonic_to_features(
            semantic_wave.frequencies,
            semantic_wave.amplitudes,
            semantic_wave.phases
        )

        # Decode to waveform
        audio = self.waveform_decoder(harmonic_features, wave_repr)

        return audio

    def _griffin_lim_synthesis(
            self,
            wave_repr: torch.Tensor,
            semantic_wave: Wave
    ) -> torch.Tensor:
        """
        Griffin-Lim algorithm with learned enhancements
        """
        # Map harmonics to frequency bins
        freq_magnitude = self.freq_mapping(semantic_wave)

        # Use learned phase prediction
        phase_input = torch.cat([
            semantic_wave.frequencies,
            semantic_wave.phases
        ], dim=-1)

        predicted_phase = self.harmonic_to_phase(phase_input)
        predicted_phase = predicted_phase * np.pi

        # Initial complex spectrogram
        real = freq_magnitude * torch.cos(predicted_phase)
        imag = freq_magnitude * torch.sin(predicted_phase)
        complex_spec = torch.complex(real, imag)

        # Transpose for Griffin-Lim
        complex_spec = complex_spec.transpose(1, 2)  # (batch, freq_bins, time)

        # Griffin-Lim iterations (fewer needed with good phase init)
        audio = self._griffin_lim_iterations(complex_spec, n_iters=30)

        return audio

    def _griffin_lim_iterations(
            self,
            complex_spec: torch.Tensor,
            n_iters: int = 30
    ) -> torch.Tensor:
        """
        Griffin-Lim algorithm for phase reconstruction
        """
        magnitude = torch.abs(complex_spec)

        # Initialize with provided phase
        phase = torch.angle(complex_spec)
        audio = None
        for _ in range(n_iters):
            # Reconstruct complex spectrogram
            complex_spec = magnitude * torch.exp(1j * phase)

            # ISTFT -> STFT cycle for phase refinement
            audio = torch.istft(
                complex_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=torch.hann_window(self.n_fft, device=complex_spec.device),
                return_complex=False
            )

            # Back to frequency domain
            complex_spec_new = torch.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=torch.hann_window(self.n_fft, device=audio.device),
                return_complex=True
            )

            # Update phase while keeping magnitude
            phase = torch.angle(complex_spec_new)

        return audio

    def _adjust_length(self, audio: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Adjust audio length to target
        """
        current_length = audio.size(-1)

        if current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            audio = F.pad(audio, (0, padding))
        elif current_length > target_length:
            # Trim
            audio = audio[..., :target_length]

        return audio


class LearnableFrequencyMapping(nn.Module):
    """
    Map semantic harmonics to frequency bins
    The sacred translation from meaning to spectrum!
    """

    def __init__(
            self,
            num_harmonics: int,
            freq_bins: int,
            sample_rate: int,
            n_fft: int
    ):
        super().__init__()

        # Learnable mapping matrix
        self.harmonic_to_freq_matrix = nn.Parameter(
            torch.randn(num_harmonics, freq_bins) * 0.1
        )

        # Learnable spread (how much each harmonic affects neighboring frequencies)
        self.spread = nn.Parameter(torch.ones(num_harmonics) * 2.0)

        # Initialize with reasonable frequency spacing
        with torch.no_grad():
            for i in range(num_harmonics):
                center_bin = int((i + 1) * freq_bins / (num_harmonics + 1))
                self.harmonic_to_freq_matrix[i, center_bin] = 1.0

    def forward(self, semantic_wave: Wave) -> torch.Tensor:
        """
        Map semantic harmonics to frequency magnitude spectrum
        """
        # Weight matrix by amplitudes
        weighted_matrix = self.harmonic_to_freq_matrix.unsqueeze(0) * \
                          semantic_wave.amplitudes.unsqueeze(-1)

        # Apply spread
        spread_matrix = F.softplus(self.spread).unsqueeze(-1)
        weighted_matrix = weighted_matrix * spread_matrix.unsqueeze(0)

        # Sum contributions from all harmonics
        magnitude_spectrum = weighted_matrix.sum(dim=-2)  # (batch, time, freq_bins)

        # Ensure positive
        magnitude_spectrum = F.softplus(magnitude_spectrum)

        return magnitude_spectrum


class HarmonicSynthesizer(nn.Module):
    """
    Neural synthesis from harmonics
    The wild path of learned waveform generation!
    """

    def __init__(
            self,
            num_harmonics: int,
            freq_bins: int,
            d_model: int
    ):
        super().__init__()

        # Harmonic oscillators (learnable!)
        self.oscillator_net = nn.Sequential(
            nn.Linear(num_harmonics * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_harmonics),
            nn.Tanh()
        )

        # Harmonic mixer
        self.mixer = nn.Sequential(
            nn.Linear(num_harmonics * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, freq_bins)
        )

    def forward(
            self,
            frequencies: torch.Tensor,
            amplitudes: torch.Tensor,
            phases: torch.Tensor
    ) -> torch.Tensor:
        """
        Synthesize spectrum from harmonics
        """
        # Combine inputs
        harmonic_input = torch.cat([frequencies, amplitudes, phases], dim=-1)

        # Generate oscillations
        oscillations = self.oscillator_net(harmonic_input)

        # Mix with amplitudes
        mixed = torch.cat([oscillations * amplitudes, phases], dim=-1)

        # Generate spectrum
        spectrum = self.mixer(mixed)

        return F.softplus(spectrum)


class WaveformDecoder(nn.Module):
    """
    Direct waveform generation (simplified WaveNet style)
    For when you want to go DIRECTLY to audio!
    """

    def __init__(
            self,
            freq_bins: int,
            d_model: int,
            sample_rate: int,
            hop_length: int,
            num_layers: int = 4
    ):
        super().__init__()

        self.hop_length = hop_length

        # Upsample from frame rate to sample rate
        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose1d(
                freq_bins if i == 0 else d_model,
                d_model,
                kernel_size=hop_length // (2 ** i),
                stride=hop_length // (2 ** (i + 1)),
                padding=hop_length // (2 ** (i + 2))
            ) for i in range(num_layers)
        ])

        # Final projection to waveform
        self.to_waveform = nn.Conv1d(d_model, 1, kernel_size=1)

    def forward(
            self,
            features: torch.Tensor,
            context: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode features to waveform
        """
        # Start with features
        x = features.transpose(1, 2)  # (batch, channels, time)

        # Upsample to audio rate
        for layer in self.upsample_layers:
            x = F.relu(layer(x))

        # Generate waveform
        waveform = self.to_waveform(x).squeeze(1)

        return torch.tanh(waveform)  # Output -1 to 1


class AudioEnhancement(nn.Module):
    """
    Post-processing enhancement network
    Polish the generated audio to perfection!
    """

    def __init__(
            self,
            d_model: int,
            sample_rate: int
    ):
        super().__init__()

        # Denoising layers
        self.denoise = nn.Sequential(
            nn.Conv1d(1, d_model // 4, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(d_model // 4, d_model // 4, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(d_model // 4, 1, kernel_size=15, padding=7)
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Enhance audio quality
        """
        # Add channel dimension
        audio_in = audio.unsqueeze(1)

        # Apply denoising
        residual = self.denoise(audio_in).squeeze(1)

        # Residual connection
        enhanced = audio + residual * 0.1  # Small residual weight

        # Soft clipping
        enhanced = torch.tanh(enhanced / 3.0) * 3.0

        return enhanced


class AcousticTransformBlock(nn.Module):
    """
    Transform semantic patterns back to acoustic patterns
    The reverse alchemy!
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

        self.attention = FlashAttention(
            d_model=d_model,
            n_heads=num_heads
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), causal=False)
        x = x + self.ffn(self.norm2(x))
        return x


# Test the inverse transformation!
if __name__ == "__main__":
    print("🔥 MANIFESTING THE WAVE-TO-AUDIO BRIDGE! 🔥")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create the decoder
    audio_decoder = WaveToAudio(
        num_harmonics=64,
        sample_rate=16000,
        synthesis_method="griffin_lim",  # Try "learned" or "neural_vocoder" too!
        num_layers=3
    ).to(device)

    # Create fake semantic waves
    batch_size = 2
    time_steps = 63
    num_harmonics = 64

    fake_frequencies = torch.sigmoid(torch.randn(batch_size, time_steps, num_harmonics)) * 20.0 + 0.1
    fake_amplitudes = F.softplus(torch.randn(batch_size, time_steps, num_harmonics))
    fake_phases = torch.tanh(torch.randn(batch_size, time_steps, num_harmonics)) * np.pi

    fake_wave = Wave(fake_frequencies.to(device), fake_amplitudes.to(device), fake_phases.to(device))

    # Generate audio!
    generated_audio = audio_decoder(fake_wave, target_length=16000)

    print(f"\n💀 AUDIO SYNTHESIS COMPLETE! 💀")
    print(f"Generated audio shape: {generated_audio.shape}")
    print(f"Audio range: [{generated_audio.min():.3f}, {generated_audio.max():.3f}]")

    # Parameter count
    params = sum(p.numel() for p in audio_decoder.parameters())
    print(f"\nTotal decoder parameters: {params:,}")

    print("\n✨ THE CIRCLE IS COMPLETE! WAVES → AUDIO → WAVES! ✨")