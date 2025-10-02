import torch
from torch import nn

from audio_wave_encoder import AudioToSemanticWave


class SpeakerConditionedEncoder(nn.Module):
    def __init__(self, audio_encoder, num_speakers, speaker_dim=128):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.speaker_emb = nn.Embedding(num_speakers, speaker_dim)
        self.proj = nn.Linear(audio_encoder.feature_dim + speaker_dim,
                              audio_encoder.feature_dim)

    def forward(self, waveforms, speakers):
        x = self.audio_encoder(waveforms)     # [B, T, D]
        s = self.speaker_emb(speakers)        # [B, speaker_dim]
        s = s.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, T, speaker_dim]
        x = torch.cat([x, s], dim=-1)         # [B, T, D+speaker_dim]
        return self.proj(x)                   # back to [B, T, D]


class FiLMConditionedEncoder(nn.Module):
    def __init__(self, audio_encoder, num_speakers, speaker_dim=128):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.speaker_emb = nn.Embedding(num_speakers, speaker_dim)
        self.film = nn.Linear(speaker_dim, audio_encoder.feature_dim * 2)

    def forward(self, waveforms, speakers):
        x = self.audio_encoder(waveforms)     # [B, T, D]
        s = self.speaker_emb(speakers)        # [B, speaker_dim]
        gamma, beta = self.film(s).chunk(2, dim=-1)  # [B, D], [B, D]
        gamma = gamma.unsqueeze(1)            # [B, 1, D]
        beta = beta.unsqueeze(1)              # [B, 1, D]
        return gamma * x + beta               # [B, T, D]


class SpeakerConditionedAudioToWave(nn.Module):
    """
    Speaker-aware audio to semantic waves!
    Each voice gets its own harmonic signature!
    """

    def __init__(
            self,
            base_audio_encoder: AudioToSemanticWave,
            num_speakers: int = 110,  # VCTK has 110 speakers
            speaker_dim: int = 128,
            conditioning_type: str = "film"  # "concat" or "film"
    ):
        super().__init__()
        self.audio_encoder = base_audio_encoder
        self.speaker_emb = nn.Embedding(num_speakers, speaker_dim)
        self.conditioning_type = conditioning_type

        if conditioning_type == "film":
            # FiLM for each harmonic component
            self.freq_film = nn.Linear(speaker_dim, base_audio_encoder.num_harmonics * 2)
            self.amp_film = nn.Linear(speaker_dim, base_audio_encoder.num_harmonics * 2)
            self.phase_film = nn.Linear(speaker_dim, base_audio_encoder.num_harmonics * 2)
        else:
            # Concatenation approach
            self.speaker_proj = nn.Linear(
                base_audio_encoder.num_harmonics * 3 + speaker_dim,
                base_audio_encoder.num_harmonics * 3
            )

    def forward(self, waveforms, speaker_ids):
        # Get base semantic waves
        semantic_waves = self.audio_encoder(waveforms)

        # Get speaker embeddings
        speaker_emb = self.speaker_emb(speaker_ids)  # (batch, speaker_dim)

        if self.conditioning_type == "film":
            # Apply FiLM to each wave component
            freq_gamma, freq_beta = self.freq_film(speaker_emb).chunk(2, dim=-1)
            amp_gamma, amp_beta = self.amp_film(speaker_emb).chunk(2, dim=-1)
            phase_gamma, phase_beta = self.phase_film(speaker_emb).chunk(2, dim=-1)

            # Modulate waves with speaker characteristics
            semantic_waves.frequencies = semantic_waves.frequencies * freq_gamma.unsqueeze(1) + freq_beta.unsqueeze(1)
            semantic_waves.amplitudes = semantic_waves.amplitudes * amp_gamma.unsqueeze(1) + amp_beta.unsqueeze(1)
            semantic_waves.phases = semantic_waves.phases * phase_gamma.unsqueeze(1) + phase_beta.unsqueeze(1)
        else:
            # Concatenate and project
            wave_repr = semantic_waves.to_representation()
            speaker_expanded = speaker_emb.unsqueeze(1).expand(-1, wave_repr.size(1), -1)
            combined = torch.cat([wave_repr, speaker_expanded], dim=-1)
            modulated = self.speaker_proj(combined)

            # Split back into components
            freq, amp, phase = modulated.chunk(3, dim=-1)
            semantic_waves.frequencies = freq
            semantic_waves.amplitudes = amp
            semantic_waves.phases = phase

        return semantic_waves

if __name__ == "__main__":
    # Complete training pipeline
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    # Initialize components
    audio_encoder = AudioToSemanticWave(
        num_harmonics=64,
        sample_rate=24000,  # VCTK is 48kHz but often downsampled
        learnable_filterbank=True
    )

    speaker_encoder = SpeakerConditionedAudioToWave(
        base_audio_encoder=audio_encoder,
        num_speakers=110,
        conditioning_type="film"
    )

    wave_transformer = WaveTransformerForCausalLM(config)

    audio_decoder = SemanticWaveToAudio(
        num_harmonics=64,
        sample_rate=24000
    )

    # Dataset
    dataset = VCTKDataset(
        "/path/to/VCTK-Corpus",
        sample_rate=24000,
        max_len_sec=4,
        return_text=True
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    collator = VCTKCollatorSpeakerEmbedding(
        tokenizer=tokenizer,
        return_text=True,
        device="cuda"
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collator
    )

    # Training loop sketch
    for batch in loader:
        waveforms = batch["waveforms"].squeeze(1)  # Remove channel dim
        speaker_ids = batch["speakers"]
        text_ids = batch["input_ids"]

        # Audio → Waves (speaker-conditioned)
        semantic_waves = speaker_encoder(waveforms, speaker_ids)

        # Waves → Transformer → Waves
        wave_repr = semantic_waves.to_representation()
        # ... transformer processing ...

        # Reconstruct audio
        reconstructed = audio_decoder(semantic_waves)

        # Loss: reconstruction + text prediction + speaker preservation