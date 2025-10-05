"""
Multi-Resolution Wave Pyramid (MRWP)

Implements multi-scale wave processing through frequency band decomposition.
The pyramid processes different frequency ranges (low, mid, high) with specialized
transformers, enabling the model to learn scale-specific patterns.

Key Components:
- FrequencyBandDecomposer: Splits wave representation into frequency bands
- BandProcessor: Specialized transformer for each frequency band
- CrossBandAttention: Information exchange between frequency bands
- AdaptiveFusion: Learned weighting to combine band outputs
- MultiResolutionWavePyramid: Main module orchestrating the pyramid

Architecture:
    Input Wave (H=64)
         |
    Decompose into 3 bands:
         ├─ Low  [0-20%]:  13 harmonics → 8-layer transformer
         ├─ Mid  [20-60%]: 26 harmonics → 6-layer transformer
         └─ High [60-100%]: 26 harmonics → 4-layer transformer
         |
    Cross-band attention (bidirectional)
         |
    Adaptive fusion
         |
    Output Wave (H=64)

References:
- "Feature Pyramid Networks for Object Detection" (Lin et al., 2017)
- "Multi-Scale Vision Transformers" (Fan et al., 2021)
- Wavelet decomposition and multi-resolution analysis
"""

import math
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyBandDecomposer(nn.Module):
    """
    Decomposes wave representation into multiple frequency bands.

    The decomposition uses soft boundaries (sigmoid transitions) to avoid
    sharp cutoffs that could harm gradient flow.

    Args:
        num_harmonics: Total number of harmonics in input
        band_boundaries: List of boundary points as fractions (e.g., [0.2, 0.6] for 3 bands)
        transition_width: Width of sigmoid transition between bands

    Example:
        >>> decomposer = FrequencyBandDecomposer(
        ...     num_harmonics=64,
        ...     band_boundaries=[0.2, 0.6],  # Creates low/mid/high bands
        ... )
        >>> wave_repr = torch.randn(2, 128, 192)  # (B, S, 3*H)
        >>> bands = decomposer(wave_repr)  # List of 3 tensors
        >>> for i, band in enumerate(bands):
        ...     print(f"Band {i}: {band.shape}")
    """

    def __init__(
        self,
        num_harmonics: int,
        band_boundaries: List[float] = [0.2, 0.6],
        transition_width: float = 0.05,
    ):
        super().__init__()

        self.num_harmonics = num_harmonics
        self.band_boundaries = sorted(band_boundaries)
        self.num_bands = len(band_boundaries) + 1
        self.transition_width = transition_width

        # Compute harmonic indices for each band
        self.band_ranges = self._compute_band_ranges()

        # Create soft band masks
        self.register_buffer('band_masks', self._create_band_masks())

    def _compute_band_ranges(self) -> List[Tuple[int, int]]:
        """Compute start/end harmonic indices for each band."""
        ranges = []
        boundaries_idx = [0] + [int(b * self.num_harmonics) for b in self.band_boundaries] + [self.num_harmonics]

        for i in range(len(boundaries_idx) - 1):
            start = boundaries_idx[i]
            end = boundaries_idx[i + 1]
            ranges.append((start, end))

        return ranges

    def _create_band_masks(self) -> torch.Tensor:
        """
        Create soft masks for each frequency band.

        Returns:
            masks: (num_bands, num_harmonics) - soft masks in [0, 1]
        """
        harmonic_indices = torch.arange(self.num_harmonics, dtype=torch.float32)
        masks = []

        for band_idx in range(self.num_bands):
            start, end = self.band_ranges[band_idx]

            # Create sigmoid transitions at boundaries
            # Mask is 1 in [start, end], with smooth transitions
            width = self.transition_width * self.num_harmonics

            # Left edge: sigmoid from 0 to 1 around start
            left_transition = torch.sigmoid((harmonic_indices - start) / width)

            # Right edge: sigmoid from 1 to 0 around end
            right_transition = torch.sigmoid((end - harmonic_indices) / width)

            # Combine: both transitions must be high
            mask = left_transition * right_transition

            masks.append(mask)

        return torch.stack(masks)  # (num_bands, num_harmonics)

    def decompose_components(
        self,
        component: torch.Tensor,
        band_masks: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Decompose a single component (freqs/amps/phases) into bands.

        Args:
            component: (B, S, H)
            band_masks: (num_bands, H)

        Returns:
            band_components: List of (B, S, H_band) tensors
        """
        B, S, H = component.shape
        band_components = []

        for band_idx in range(self.num_bands):
            mask = band_masks[band_idx].to(component.device)  # (H,)

            # Apply mask
            masked = component * mask.unsqueeze(0).unsqueeze(0)  # (B, S, H)

            # Extract only the relevant harmonics (non-zero mask regions)
            # For efficiency, we keep the full dimension but zero out others
            # Alternatively, could slice to save memory
            band_components.append(masked)

        return band_components

    def forward(
        self,
        wave_repr: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Decompose wave representation into frequency bands.

        Args:
            wave_repr: (B, S, 3*H) - concatenated [freqs, amps, phases]

        Returns:
            band_reprs: List of (B, S, 3*H) band representations
        """
        B, S, _ = wave_repr.shape

        # Split into components
        freqs, amps, phases = wave_repr.chunk(3, dim=-1)  # Each (B, S, H)

        # Decompose each component
        freqs_bands = self.decompose_components(freqs, self.band_masks)
        amps_bands = self.decompose_components(amps, self.band_masks)
        phases_bands = self.decompose_components(phases, self.band_masks)

        # Recombine into band representations
        band_reprs = []
        for i in range(self.num_bands):
            band_repr = torch.cat([
                freqs_bands[i],
                amps_bands[i],
                phases_bands[i],
            ], dim=-1)  # (B, S, 3*H)
            band_reprs.append(band_repr)

        return band_reprs


class BandProcessor(nn.Module):
    """
    Specialized transformer processor for a single frequency band.

    Each band has its own transformer with configurable depth,
    allowing different complexity for different frequency ranges.

    Args:
        d_model: Model dimension (3 * num_harmonics)
        num_layers: Number of transformer layers for this band
        n_heads_q: Number of query heads
        n_heads_kv: Number of key/value heads
        d_ff_multi: FFN dimension multiplier
        dropout: Dropout probability
        use_yarn: Whether to use YaRN positional encoding
        use_flash: Whether to use Flash Attention

    Example:
        >>> processor = BandProcessor(
        ...     d_model=192,
        ...     num_layers=6,
        ...     n_heads_q=8,
        ...     n_heads_kv=4,
        ... )
        >>> band_repr = torch.randn(2, 128, 192)
        >>> processed = processor(band_repr, causal=True)
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        n_heads_q: int,
        n_heads_kv: int,
        d_ff_multi: int = 4,
        dropout: float = 0.1,
        use_yarn: bool = True,
        use_flash: bool = True,
    ):
        super().__init__()

        # Import here to avoid circular dependency
        from wave_transformer.core.transformer import ParallelBlock, RMSNorm

        self.d_model = d_model
        self.num_layers = num_layers

        # Transformer layers
        self.layers = nn.ModuleList([
            ParallelBlock(
                d_model=d_model,
                n_heads=n_heads_q,
                n_heads_kv=n_heads_kv,
                d_ff=d_model * d_ff_multi,
                dropout=dropout,
                use_yarn=use_yarn,
                use_flash=use_flash,
            )
            for _ in range(num_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process band through transformer layers.

        Args:
            x: (B, S, d_model) - band representation
            causal: Whether to use causal masking
            attention_mask: Optional attention mask

        Returns:
            output: (B, S, d_model) - processed representation
        """
        for layer in self.layers:
            x = layer(x, causal=causal, attention_mask=attention_mask)

        return self.norm(x)


class CrossBandAttention(nn.Module):
    """
    Cross-attention between different frequency bands.

    Allows information exchange between bands, enabling the model to
    capture cross-scale interactions (e.g., how low frequencies modulate
    high frequencies).

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
        bidirectional: Whether to attend both ways between bands

    Example:
        >>> cross_attn = CrossBandAttention(d_model=192, n_heads=8)
        >>> low_band = torch.randn(2, 128, 192)
        >>> high_band = torch.randn(2, 128, 192)
        >>> low_updated, high_updated = cross_attn(low_band, high_band)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.dropout_p = dropout
        self.bidirectional = bidirectional

        # Cross-attention: query from one band, key/value from another
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        if bidirectional:
            # Reverse direction projections
            self.q_proj_rev = nn.Linear(d_model, d_model, bias=False)
            self.k_proj_rev = nn.Linear(d_model, d_model, bias=False)
            self.v_proj_rev = nn.Linear(d_model, d_model, bias=False)
            self.out_proj_rev = nn.Linear(d_model, d_model, bias=False)

    def attend(
        self,
        q_src: torch.Tensor,
        kv_src: torch.Tensor,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        out_proj: nn.Linear,
    ) -> torch.Tensor:
        """
        Perform cross-attention from q_src to kv_src.

        Args:
            q_src: Query source (B, S, D)
            kv_src: Key/Value source (B, S, D)
            q_proj, k_proj, v_proj, out_proj: Projection layers

        Returns:
            output: (B, S, D) - attended representation
        """
        B, S, D = q_src.shape

        # Project
        q = q_proj(q_src).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = k_proj(kv_src).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = v_proj(kv_src).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)

        if self.training and self.dropout_p > 0:
            attn = F.dropout(attn, p=self.dropout_p)

        out = torch.matmul(attn, v)  # (B, n_heads, S, d_head)
        out = out.transpose(1, 2).reshape(B, S, D)

        return out_proj(out)

    def forward(
        self,
        band_a: torch.Tensor,
        band_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention between two bands.

        Args:
            band_a: (B, S, D) - first band
            band_b: (B, S, D) - second band

        Returns:
            band_a_updated: (B, S, D) - band_a + attention from band_b
            band_b_updated: (B, S, D) - band_b + attention from band_a (if bidirectional)
        """
        # band_a attends to band_b
        attn_a_to_b = self.attend(
            band_a, band_b,
            self.q_proj, self.k_proj, self.v_proj, self.out_proj
        )
        band_a_updated = band_a + attn_a_to_b

        if self.bidirectional:
            # band_b attends to band_a
            attn_b_to_a = self.attend(
                band_b, band_a,
                self.q_proj_rev, self.k_proj_rev, self.v_proj_rev, self.out_proj_rev
            )
            band_b_updated = band_b + attn_b_to_a
        else:
            band_b_updated = band_b

        return band_a_updated, band_b_updated


class AdaptiveFusion(nn.Module):
    """
    Learned fusion of multiple frequency bands back into a unified representation.

    Uses a gating mechanism to determine the contribution of each band,
    allowing the model to adaptively weight different frequency ranges.

    Args:
        num_harmonics: Total number of harmonics in output
        num_bands: Number of input bands
        d_model: Model dimension (3 * num_harmonics)

    Example:
        >>> fusion = AdaptiveFusion(num_harmonics=64, num_bands=3)
        >>> bands = [torch.randn(2, 128, 192) for _ in range(3)]
        >>> fused = fusion(bands)
        >>> print(fused.shape)  # (2, 128, 192)
    """

    def __init__(
        self,
        num_harmonics: int,
        num_bands: int,
        d_model: int,
    ):
        super().__init__()

        self.num_harmonics = num_harmonics
        self.num_bands = num_bands
        self.d_model = d_model

        # Gating network: computes importance weights for each band
        self.gate = nn.Sequential(
            nn.Linear(d_model * num_bands, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_bands),
            nn.Softmax(dim=-1),  # Weights sum to 1
        )

        # Optional: learnable combination weights (in addition to gating)
        self.band_weights = nn.Parameter(torch.ones(num_bands) / num_bands)

    def forward(
        self,
        bands: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Fuse multiple band representations into unified output.

        Args:
            bands: List of (B, S, 3*H) band representations

        Returns:
            fused: (B, S, 3*H) - fused representation
        """
        B, S, D = bands[0].shape

        # Concatenate all bands
        all_bands = torch.cat(bands, dim=-1)  # (B, S, num_bands * D)

        # Compute adaptive weights via gating
        gate_weights = self.gate(all_bands)  # (B, S, num_bands)

        # Combine with learned base weights
        combined_weights = gate_weights * F.softmax(self.band_weights, dim=0)

        # Weighted sum of bands
        fused = sum(
            combined_weights[:, :, i:i+1] * bands[i]
            for i in range(self.num_bands)
        )

        return fused


class MultiResolutionWavePyramid(nn.Module):
    """
    Multi-Resolution Wave Pyramid for hierarchical frequency processing.

    This module decomposes the wave representation into multiple frequency bands,
    processes each band with a specialized transformer, enables cross-band
    information exchange, and fuses the results back into a unified representation.

    Architecture:
        1. Decompose into frequency bands (e.g., low/mid/high)
        2. Process each band with specialized transformer
        3. Cross-band attention for inter-scale communication
        4. Adaptive fusion to recombine bands

    Args:
        num_harmonics: Total number of harmonics
        band_boundaries: Frequency band boundaries as fractions (e.g., [0.2, 0.6])
        band_num_layers: Number of transformer layers per band (e.g., [8, 6, 4])
        n_heads_q: Number of query heads for transformers
        n_heads_kv: Number of key/value heads for transformers
        d_ff_multi: FFN dimension multiplier
        dropout: Dropout probability
        use_cross_band_attn: Whether to use cross-band attention
        use_yarn: Whether to use YaRN positional encoding
        use_flash: Whether to use Flash Attention

    Shape:
        - Input: (B, S, 3*H) - wave representation
        - Output: (B, S, 3*H) - processed wave representation

    Example:
        >>> pyramid = MultiResolutionWavePyramid(
        ...     num_harmonics=64,
        ...     band_boundaries=[0.2, 0.6],
        ...     band_num_layers=[8, 6, 4],
        ... )
        >>> wave_repr = torch.randn(2, 128, 192)
        >>> output = pyramid(wave_repr, causal=True)
    """

    def __init__(
        self,
        num_harmonics: int,
        band_boundaries: List[float] = [0.2, 0.6],
        band_num_layers: List[int] = [8, 6, 4],
        n_heads_q: int = 8,
        n_heads_kv: int = 4,
        d_ff_multi: int = 4,
        dropout: float = 0.1,
        use_cross_band_attn: bool = True,
        use_yarn: bool = True,
        use_flash: bool = True,
    ):
        super().__init__()

        self.num_harmonics = num_harmonics
        self.d_model = 3 * num_harmonics
        self.band_boundaries = band_boundaries
        self.num_bands = len(band_boundaries) + 1
        self.use_cross_band_attn = use_cross_band_attn

        assert len(band_num_layers) == self.num_bands, \
            f"band_num_layers ({len(band_num_layers)}) must match num_bands ({self.num_bands})"

        # 1. Band decomposer
        self.decomposer = FrequencyBandDecomposer(
            num_harmonics=num_harmonics,
            band_boundaries=band_boundaries,
        )

        # 2. Band processors (specialized transformers)
        self.band_processors = nn.ModuleList([
            BandProcessor(
                d_model=self.d_model,
                num_layers=band_num_layers[i],
                n_heads_q=n_heads_q,
                n_heads_kv=n_heads_kv,
                d_ff_multi=d_ff_multi,
                dropout=dropout,
                use_yarn=use_yarn,
                use_flash=use_flash,
            )
            for i in range(self.num_bands)
        ])

        # 3. Cross-band attention (optional)
        if use_cross_band_attn:
            # Create cross-attention for adjacent bands
            self.cross_band_attns = nn.ModuleList([
                CrossBandAttention(
                    d_model=self.d_model,
                    n_heads=n_heads_q,
                    dropout=dropout,
                    bidirectional=True,
                )
                for _ in range(self.num_bands - 1)
            ])

        # 4. Adaptive fusion
        self.fusion = AdaptiveFusion(
            num_harmonics=num_harmonics,
            num_bands=self.num_bands,
            d_model=self.d_model,
        )

    def forward(
        self,
        wave_repr: torch.Tensor,
        causal: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through multi-resolution pyramid.

        Args:
            wave_repr: (B, S, 3*H) - input wave representation
            causal: Whether to use causal masking
            attention_mask: Optional attention mask

        Returns:
            output: (B, S, 3*H) - processed wave representation
        """
        # 1. Decompose into frequency bands
        bands = self.decomposer(wave_repr)  # List of (B, S, 3*H)

        # 2. Process each band through its specialized transformer
        processed_bands = []
        for i, band in enumerate(bands):
            processed = self.band_processors[i](
                band,
                causal=causal,
                attention_mask=attention_mask,
            )
            processed_bands.append(processed)

        # 3. Cross-band attention (optional)
        if self.use_cross_band_attn:
            # Attend between adjacent bands (low↔mid, mid↔high)
            updated_bands = [processed_bands[0]]  # Start with lowest band

            for i in range(self.num_bands - 1):
                # Cross-attend between band i and band i+1
                band_low, band_high = self.cross_band_attns[i](
                    updated_bands[-1],  # Use updated version of lower band
                    processed_bands[i + 1],
                )

                # Update the lower band (already in list) and add higher band
                updated_bands[-1] = band_low
                updated_bands.append(band_high)

            processed_bands = updated_bands

        # 4. Fuse bands back into unified representation
        output = self.fusion(processed_bands)

        return output


# ==================== INTEGRATION EXAMPLE ====================

def example_usage():
    """
    Example of using Multi-Resolution Wave Pyramid.
    """
    # Setup
    batch_size = 2
    seq_len = 128
    num_harmonics = 64
    d_model = 3 * num_harmonics  # 192

    # Create pyramid
    pyramid = MultiResolutionWavePyramid(
        num_harmonics=num_harmonics,
        band_boundaries=[0.2, 0.6],  # 3 bands: [0-20%], [20-60%], [60-100%]
        band_num_layers=[8, 6, 4],   # Different depths per band
        n_heads_q=8,
        n_heads_kv=4,
        d_ff_multi=4,
        dropout=0.1,
        use_cross_band_attn=True,
        use_yarn=True,
        use_flash=True,
    )

    print("Multi-Resolution Wave Pyramid")
    print("=" * 60)
    print(f"Input: {num_harmonics} harmonics")
    print(f"Bands: {pyramid.num_bands}")
    print(f"  Low  [0-20%]:  ~13 harmonics → 8 layers")
    print(f"  Mid  [20-60%]: ~26 harmonics → 6 layers")
    print(f"  High [60-100%]: ~26 harmonics → 4 layers")
    print(f"Cross-band attention: {pyramid.use_cross_band_attn}")
    print("=" * 60)

    # Create input
    wave_repr = torch.randn(batch_size, seq_len, d_model)

    print(f"\nInput shape: {wave_repr.shape}")

    # Forward pass
    output = pyramid(wave_repr, causal=True)

    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in pyramid.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Break down by component
    decomposer_params = sum(p.numel() for p in pyramid.decomposer.parameters())
    processor_params = sum(sum(p.numel() for p in proc.parameters()) for proc in pyramid.band_processors)
    fusion_params = sum(p.numel() for p in pyramid.fusion.parameters())

    print(f"  Decomposer: {decomposer_params:,}")
    print(f"  Processors: {processor_params:,}")
    print(f"  Fusion: {fusion_params:,}")

    if pyramid.use_cross_band_attn:
        cross_attn_params = sum(sum(p.numel() for p in attn.parameters()) for attn in pyramid.cross_band_attns)
        print(f"  Cross-band attention: {cross_attn_params:,}")

    return pyramid, output


def integration_with_wave_transformer():
    """
    Example of integrating MRWP into WaveTransformer.
    """
    print("\n" + "="*60)
    print("INTEGRATION WITH WAVETRANSFORMER")
    print("="*60 + "\n")

    code = '''
# Option 1: Replace some transformer layers with pyramid layers

from wave_transformer.core.transformer import WaveTransformer, ParallelBlock
from wave_transformer.enhancements import MultiResolutionWavePyramid

class PyramidWaveTransformer(nn.Module):
    """
    WaveTransformer with multi-resolution pyramid layers.
    """
    def __init__(
        self,
        wave_encoder,
        wave_decoder,
        num_harmonics=64,
        num_standard_layers=4,
        num_pyramid_layers=2,
        **kwargs
    ):
        super().__init__()

        self.wave_encoder = wave_encoder
        self.wave_decoder = wave_decoder
        self.num_harmonics = num_harmonics
        self.d_model = 3 * num_harmonics

        # Early layers: standard transformers
        self.standard_layers = nn.ModuleList([
            ParallelBlock(
                d_model=self.d_model,
                n_heads=8,
                n_heads_kv=4,
                d_ff=self.d_model * 4,
                dropout=0.1,
            )
            for _ in range(num_standard_layers)
        ])

        # Middle layers: multi-resolution pyramids
        self.pyramid_layers = nn.ModuleList([
            MultiResolutionWavePyramid(
                num_harmonics=num_harmonics,
                band_boundaries=[0.2, 0.6],
                band_num_layers=[8, 6, 4],
                use_cross_band_attn=True,
            )
            for _ in range(num_pyramid_layers)
        ])

        # Late layers: standard transformers
        self.final_layers = nn.ModuleList([
            ParallelBlock(
                d_model=self.d_model,
                n_heads=8,
                n_heads_kv=4,
                d_ff=self.d_model * 4,
                dropout=0.1,
            )
            for _ in range(num_standard_layers)
        ])

        from wave_transformer.core.transformer import RMSNorm
        self.norm_f = RMSNorm(self.d_model)

    def forward(self, encoder_input, causal=True, attention_mask=None):
        # Encode to wave
        wave = self.wave_encoder(attention_mask=attention_mask, **encoder_input)
        x = wave.to_representation()

        # Standard layers
        for layer in self.standard_layers:
            x = layer(x, causal=causal, attention_mask=attention_mask)

        # Pyramid layers
        for pyramid in self.pyramid_layers:
            x = pyramid(x, causal=causal, attention_mask=attention_mask)

        # Final layers
        for layer in self.final_layers:
            x = layer(x, causal=causal, attention_mask=attention_mask)

        x = self.norm_f(x)

        # Decode
        output = self.wave_decoder(x, attention_mask=attention_mask)

        return output


# Option 2: Use pyramid as a standalone enhancement module

class WaveTransformerWithPyramid(WaveTransformer):
    """
    Standard WaveTransformer with optional pyramid processing.
    """
    def __init__(self, *args, use_pyramid=True, pyramid_interval=3, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_pyramid = use_pyramid
        self.pyramid_interval = pyramid_interval

        if use_pyramid:
            self.pyramid = MultiResolutionWavePyramid(
                num_harmonics=self.num_harmonics,
                band_boundaries=[0.2, 0.6],
                band_num_layers=[4, 3, 2],  # Lighter than full transformer
            )

    def forward(self, encoder_input, causal=True, attention_mask=None):
        wave = self.wave_encoder(attention_mask=attention_mask, **encoder_input)
        x = wave.to_representation()

        for i, block in enumerate(self.layers):
            x = block(x, causal=causal, attention_mask=attention_mask)

            # Apply pyramid every N layers
            if self.use_pyramid and (i + 1) % self.pyramid_interval == 0:
                x = self.pyramid(x, causal=causal, attention_mask=attention_mask)

        x = self.norm_f(x)
        output = self.wave_decoder(x, attention_mask=attention_mask)

        return output
    '''

    print(code)


if __name__ == "__main__":
    # Run examples
    example_usage()
    integration_with_wave_transformer()
