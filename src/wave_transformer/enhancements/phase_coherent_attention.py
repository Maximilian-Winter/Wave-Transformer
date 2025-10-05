"""
Phase-Coherent Cross-Attention (PCCA)

Implements phase-aware attention mechanism that explicitly models phase coherence
between harmonics. This enhancement modulates attention scores based on phase
relationships, allowing the model to capture wave interference patterns.

Key Components:
- PhaseCoherentAttention: Main attention module extending MultiQueryFlashAttention
- PhaseCoherenceComputer: Efficient phase coherence matrix computation
- Learnable alpha/beta blending of amplitude and phase attention scores
- Compatible with Flash Attention and standard attention

References:
- "Attention Is All You Need" (Vaswani et al., 2017)
- "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)
- Wave interference theory and phase coherence in signal processing
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseCoherenceComputer(nn.Module):
    """
    Efficiently computes phase coherence between query and key positions.

    Phase coherence measures how aligned the phases of two wave representations are.
    High coherence indicates constructive interference, low coherence indicates
    destructive interference.

    Args:
        num_harmonics: Number of harmonics in wave representation
        coherence_mode: 'cosine' (phase difference) or 'complex' (complex correlation)
        temperature: Temperature for coherence scores

    Example:
        >>> computer = PhaseCoherenceComputer(num_harmonics=64, coherence_mode='cosine')
        >>> phases_q = torch.randn(2, 8, 128, 64)  # (B, H, S_q, num_harmonics)
        >>> phases_k = torch.randn(2, 8, 128, 64)  # (B, H, S_k, num_harmonics)
        >>> coherence = computer(phases_q, phases_k)  # (B, H, S_q, S_k)
    """

    def __init__(
        self,
        num_harmonics: int,
        coherence_mode: str = 'cosine',
        temperature: float = 1.0,
    ):
        super().__init__()
        assert coherence_mode in ['cosine', 'complex'], \
            "coherence_mode must be 'cosine' or 'complex'"

        self.num_harmonics = num_harmonics
        self.coherence_mode = coherence_mode
        self.temperature = temperature

    def compute_cosine_coherence(
        self,
        phases_q: torch.Tensor,
        phases_k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute phase coherence using cosine of phase differences.

        Args:
            phases_q: (B, H, S_q, num_harmonics)
            phases_k: (B, H, S_k, num_harmonics)

        Returns:
            coherence: (B, H, S_q, S_k) - coherence scores in [-1, 1]
        """
        # Compute phase differences for all query-key pairs
        # Expand dimensions for broadcasting
        # phases_q: (B, H, S_q, 1, num_harmonics)
        # phases_k: (B, H, 1, S_k, num_harmonics)

        phase_diff = phases_q.unsqueeze(-2) - phases_k.unsqueeze(-3)  # (B, H, S_q, S_k, num_harmonics)

        # Compute coherence as mean cosine of phase differences
        # cos(0) = 1 (perfect coherence), cos(π) = -1 (anti-coherence)
        coherence_per_harmonic = torch.cos(phase_diff)  # (B, H, S_q, S_k, num_harmonics)

        # Average across harmonics
        coherence = coherence_per_harmonic.mean(dim=-1)  # (B, H, S_q, S_k)

        return coherence / self.temperature

    def compute_complex_coherence(
        self,
        phases_q: torch.Tensor,
        phases_k: torch.Tensor,
        amps_q: Optional[torch.Tensor] = None,
        amps_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute phase coherence using complex correlation.

        Args:
            phases_q: (B, H, S_q, num_harmonics)
            phases_k: (B, H, S_k, num_harmonics)
            amps_q: Optional (B, H, S_q, num_harmonics) - amplitude weighting
            amps_k: Optional (B, H, S_k, num_harmonics) - amplitude weighting

        Returns:
            coherence: (B, H, S_q, S_k) - coherence scores
        """
        # Convert phases to complex exponentials
        complex_q = torch.exp(1j * phases_q)  # (B, H, S_q, num_harmonics)
        complex_k = torch.exp(1j * phases_k)  # (B, H, S_k, num_harmonics)

        # Apply amplitude weighting if provided
        if amps_q is not None:
            complex_q = complex_q * amps_q
        if amps_k is not None:
            complex_k = complex_k * amps_k

        # Compute complex correlation
        # complex_q: (B, H, S_q, 1, num_harmonics)
        # complex_k*: (B, H, 1, S_k, num_harmonics)
        correlation = complex_q.unsqueeze(-2) * torch.conj(complex_k.unsqueeze(-3))
        # (B, H, S_q, S_k, num_harmonics)

        # Sum across harmonics and take real part (or magnitude)
        # For phase coherence, we use the magnitude of the sum
        coherence = torch.abs(correlation.sum(dim=-1))  # (B, H, S_q, S_k)

        # Normalize by number of harmonics
        coherence = coherence / self.num_harmonics

        return coherence / self.temperature

    def forward(
        self,
        phases_q: torch.Tensor,
        phases_k: torch.Tensor,
        amps_q: Optional[torch.Tensor] = None,
        amps_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute phase coherence matrix.

        Args:
            phases_q: (B, H, S_q, num_harmonics) - query phases
            phases_k: (B, H, S_k, num_harmonics) - key phases
            amps_q: Optional (B, H, S_q, num_harmonics) - query amplitudes
            amps_k: Optional (B, H, S_k, num_harmonics) - key amplitudes

        Returns:
            coherence: (B, H, S_q, S_k) - phase coherence matrix
        """
        if self.coherence_mode == 'cosine':
            return self.compute_cosine_coherence(phases_q, phases_k)
        else:  # 'complex'
            return self.compute_complex_coherence(phases_q, phases_k, amps_q, amps_k)


class PhaseCoherentAttention(nn.Module):
    """
    Multi-Query Flash Attention with phase coherence awareness.

    This module extends the standard MultiQueryFlashAttention by incorporating
    phase coherence into the attention mechanism. The final attention scores
    are a learned combination of:
    1. Standard attention scores (based on Q·K^T)
    2. Phase coherence scores (based on phase relationships)

    Args:
        d_model: Model dimension (should be 3 * num_harmonics for wave representation)
        num_harmonics: Number of harmonics in wave representation
        n_heads_q: Number of query heads
        n_heads_kv: Number of key/value heads (for Multi-Query Attention)
        dropout_p: Dropout probability
        use_yarn: Whether to use YaRN rotary embeddings
        max_seq_len: Maximum sequence length for positional encoding
        use_flash: Whether to use Flash Attention
        phase_coherence_mode: 'cosine' or 'complex'
        phase_temp: Temperature for phase coherence scores
        learnable_blend: Whether alpha/beta are learnable or fixed

    Shape:
        - Input: (B, S, 3*H) - wave representation
        - Output: (B, S, 3*H) - attended representation

    Example:
        >>> attn = PhaseCoherentAttention(
        ...     d_model=192,
        ...     num_harmonics=64,
        ...     n_heads_q=8,
        ...     n_heads_kv=4,
        ...     phase_coherence_mode='cosine',
        ... )
        >>> wave_repr = torch.randn(2, 128, 192)
        >>> output = attn(wave_repr, causal=True)
    """

    def __init__(
        self,
        d_model: int,
        num_harmonics: int,
        n_heads_q: int,
        n_heads_kv: int,
        dropout_p: float = 0.0,
        use_yarn: bool = True,
        max_seq_len: int = 512,
        use_flash: bool = True,
        phase_coherence_mode: str = 'cosine',
        phase_temp: float = 1.0,
        learnable_blend: bool = True,
    ):
        super().__init__()

        assert n_heads_q % n_heads_kv == 0
        assert d_model == 3 * num_harmonics, \
            f"d_model ({d_model}) must equal 3 * num_harmonics ({3 * num_harmonics})"

        self.d_model = d_model
        self.num_harmonics = num_harmonics
        self.n_heads_q = n_heads_q
        self.n_heads_kv = n_heads_kv
        self.d_head = d_model // n_heads_q
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.dropout_p = dropout_p
        self.use_flash = use_flash
        self.use_yarn = use_yarn
        self.phase_coherence_mode = phase_coherence_mode

        # Flash attention functions
        if self.use_flash:
            try:
                from flash_attn import flash_attn_func, flash_attn_varlen_func
                self.flash_attn_func = flash_attn_func
                self.flash_attn_varlen_func = flash_attn_varlen_func
            except ImportError:
                print("Warning: flash_attn not available, falling back to manual attention")
                self.use_flash = False

        # YaRN rotary embeddings
        if use_yarn:
            from wave_transformer.core.yarn import YaRNRotaryEmbedding
            self.yarn_rope = YaRNRotaryEmbedding(
                d_model=self.d_head,
                max_len=max_seq_len,
                original_max_len=max_seq_len
            )

        # Standard Q, K, V projections
        self.q_proj = nn.Linear(d_model, n_heads_q * self.d_head, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * n_heads_kv * self.d_head, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Phase coherence computer
        self.phase_coherence = PhaseCoherenceComputer(
            num_harmonics=num_harmonics,
            coherence_mode=phase_coherence_mode,
            temperature=phase_temp,
        )

        # Blending parameters: attn_final = alpha * attn_standard + beta * attn_phase
        if learnable_blend:
            self.alpha = nn.Parameter(torch.tensor(1.0))
            self.beta = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer('alpha', torch.tensor(1.0))
            self.register_buffer('beta', torch.tensor(0.5))

    def extract_wave_components(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract frequencies, amplitudes, and phases from wave representation.

        Args:
            x: (B, S, 3*H) - wave representation

        Returns:
            freqs: (B, S, H)
            amps: (B, S, H)
            phases: (B, S, H)
        """
        freqs, amps, phases = x.chunk(3, dim=-1)
        return freqs, amps, phases

    def compute_phase_coherent_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        phases_q: torch.Tensor,
        phases_k: torch.Tensor,
        amps_q: Optional[torch.Tensor] = None,
        amps_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention scores with phase coherence.

        Args:
            q: (B, S_q, n_heads_q, d_head) - queries
            k: (B, S_k, n_heads_kv, d_head) - keys
            phases_q: (B, S_q, num_harmonics) - query phases
            phases_k: (B, S_k, num_harmonics) - key phases
            amps_q: Optional (B, S_q, num_harmonics) - query amplitudes
            amps_k: Optional (B, S_k, num_harmonics) - key amplitudes

        Returns:
            combined_scores: (B, n_heads_q, S_q, S_k) - attention scores
        """
        B, S_q, _, _ = q.shape
        S_k = k.size(1)

        # Standard attention scores
        q = q.transpose(1, 2)  # (B, n_heads_q, S_q, d_head)
        k = k.transpose(1, 2)  # (B, n_heads_kv, S_k, d_head)

        # Repeat k/v for multi-query attention
        repeat_factor = self.n_heads_q // self.n_heads_kv
        if repeat_factor > 1:
            k = k.repeat_interleave(repeat_factor, dim=1)

        # Compute standard attention scores
        standard_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # (B, n_heads_q, S_q, S_k)

        # Prepare phases for coherence computation
        # We need to replicate phases for each head
        phases_q_expanded = phases_q.unsqueeze(1).expand(B, self.n_heads_q, S_q, self.num_harmonics)
        phases_k_expanded = phases_k.unsqueeze(1).expand(B, self.n_heads_q, S_k, self.num_harmonics)

        # Optionally expand amplitudes
        amps_q_expanded = None
        amps_k_expanded = None
        if amps_q is not None:
            amps_q_expanded = amps_q.unsqueeze(1).expand(B, self.n_heads_q, S_q, self.num_harmonics)
        if amps_k is not None:
            amps_k_expanded = amps_k.unsqueeze(1).expand(B, self.n_heads_q, S_k, self.num_harmonics)

        # Compute phase coherence scores
        phase_scores = self.phase_coherence(
            phases_q_expanded,
            phases_k_expanded,
            amps_q_expanded,
            amps_k_expanded,
        )  # (B, n_heads_q, S_q, S_k)

        # Combine scores with learned blending
        combined_scores = self.alpha * standard_scores + self.beta * phase_scores

        return combined_scores

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with phase-coherent attention.

        Args:
            x: (B, S, 3*H) - wave representation
            causal: Whether to use causal masking
            attention_mask: Optional attention mask (B, S)

        Returns:
            output: (B, S, 3*H) - attended representation
        """
        B, N, C = x.shape

        # Extract wave components for phase coherence
        freqs, amps, phases = self.extract_wave_components(x)

        # Standard Q, K, V projections
        q = self.q_proj(x).reshape(B, N, self.n_heads_q, self.d_head)
        kv = self.kv_proj(x).reshape(B, N, 2, self.n_heads_kv, self.d_head)
        k, v = kv.unbind(2)

        # Apply YaRN if enabled
        if self.use_yarn:
            q_flat = q.reshape(B * self.n_heads_q, N, self.d_head)
            k_flat = k.reshape(B * self.n_heads_kv, N, self.d_head)

            q_flat = self.yarn_rope(q_flat)
            k_flat = self.yarn_rope(k_flat)

            q = q_flat.reshape(B, N, self.n_heads_q, self.d_head)
            k = k_flat.reshape(B, N, self.n_heads_kv, self.d_head)

        # Choose attention path based on use_flash and attention_mask
        # Note: Flash Attention doesn't support custom attention scores,
        # so we fall back to manual attention when using phase coherence
        use_manual_attn = True  # Always use manual for phase coherence

        if use_manual_attn:
            # Manual attention with phase coherence
            # Compute phase-coherent attention scores
            scores = self.compute_phase_coherent_scores(
                q, k, phases, phases, amps, amps
            )  # (B, n_heads_q, S_q, S_k)

            # Apply causal mask
            if causal:
                causal_mask = torch.triu(
                    torch.ones(N, N, dtype=torch.bool, device=scores.device),
                    diagonal=1
                )
                scores.masked_fill_(causal_mask, float("-inf"))

            # Apply attention mask
            if attention_mask is not None:
                mask = attention_mask[:, None, None, :].to(dtype=scores.dtype)
                scores = scores.masked_fill(mask == 0, float("-inf"))

            # Softmax and dropout
            attn = F.softmax(scores, dim=-1)
            if self.training and self.dropout_p > 0:
                attn = F.dropout(attn, p=self.dropout_p)

            # Apply attention to values
            v = v.transpose(1, 2)  # (B, n_heads_kv, S, d_head)
            repeat_factor = self.n_heads_q // self.n_heads_kv
            if repeat_factor > 1:
                v = v.repeat_interleave(repeat_factor, dim=1)

            out = torch.matmul(attn, v)  # (B, n_heads_q, S, d_head)
            out = out.transpose(1, 2).reshape(B, N, -1)  # (B, S, d_model)

        # Output projection
        return self.out_proj(out)


# ==================== INTEGRATION EXAMPLE ====================

def example_usage():
    """
    Example of using PhaseCoherentAttention in WaveTransformer.
    """
    # Setup
    batch_size = 2
    seq_len = 128
    num_harmonics = 64
    d_model = 3 * num_harmonics  # 192

    # Create phase-coherent attention module
    attn = PhaseCoherentAttention(
        d_model=d_model,
        num_harmonics=num_harmonics,
        n_heads_q=8,
        n_heads_kv=4,
        dropout_p=0.1,
        use_yarn=True,
        use_flash=False,  # Can't use flash with custom attention scores
        phase_coherence_mode='cosine',
        phase_temp=1.0,
        learnable_blend=True,
    )

    # Create wave representation
    # In practice, this comes from WaveEncoder
    freqs = torch.randn(batch_size, seq_len, num_harmonics)
    amps = torch.abs(torch.randn(batch_size, seq_len, num_harmonics))
    phases = torch.randn(batch_size, seq_len, num_harmonics) * 2 * math.pi

    wave_repr = torch.cat([freqs, amps, phases], dim=-1)

    print(f"Input shape: {wave_repr.shape}")

    # Forward pass
    output = attn(wave_repr, causal=True)

    print(f"Output shape: {output.shape}")
    print(f"\nLearnable blend parameters:")
    print(f"  alpha (standard attention): {attn.alpha.item():.3f}")
    print(f"  beta (phase coherence): {attn.beta.item():.3f}")

    # Demonstrate phase coherence computation
    print(f"\n{'='*60}")
    print("Phase Coherence Example")
    print('='*60)

    phase_computer = PhaseCoherenceComputer(
        num_harmonics=num_harmonics,
        coherence_mode='cosine',
        temperature=1.0,
    )

    # Create simple test: same phases should have high coherence
    test_phases = torch.randn(1, 8, seq_len, num_harmonics)
    coherence_self = phase_computer(test_phases, test_phases)
    print(f"\nSelf-coherence (should be ~1.0): {coherence_self.mean().item():.3f}")

    # Different phases should have lower coherence
    test_phases_2 = torch.randn(1, 8, seq_len, num_harmonics)
    coherence_diff = phase_computer(test_phases, test_phases_2)
    print(f"Cross-coherence (should be ~0.0): {coherence_diff.mean().item():.3f}")

    return attn, output


def integration_with_parallel_block():
    """
    Example of replacing standard attention with PCCA in ParallelBlock.
    """
    print("\n" + "="*60)
    print("INTEGRATION WITH PARALLELBLOCK")
    print("="*60 + "\n")

    code = '''
# Modified ParallelBlock with Phase-Coherent Attention

from wave_transformer.core.transformer import RMSNorm, SwiGLU
from wave_transformer.enhancements import PhaseCoherentAttention

class PhaseCoherentParallelBlock(nn.Module):
    """
    ParallelBlock with phase-coherent attention.
    """
    def __init__(
        self,
        d_model,
        num_harmonics,
        n_heads,
        n_heads_kv,
        d_ff,
        dropout=0.0,
        use_yarn=True,
        phase_coherence_mode='cosine',
    ):
        super().__init__()

        self.norm = RMSNorm(d_model)

        # Use PhaseCoherentAttention instead of standard attention
        self.attn = PhaseCoherentAttention(
            d_model=d_model,
            num_harmonics=num_harmonics,
            n_heads_q=n_heads,
            n_heads_kv=n_heads_kv,
            dropout_p=dropout,
            use_yarn=use_yarn,
            use_flash=False,  # Can't use flash with custom scores
            phase_coherence_mode=phase_coherence_mode,
        )

        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal=True, attention_mask=None):
        # Single normalization, then parallel paths
        normalized = self.norm(x)
        attn_out = self.attn(normalized, causal, attention_mask)
        ffn_out = self.ffn(normalized)

        # Combine and add residual
        return x + self.dropout(attn_out + ffn_out)


# Use in WaveTransformer:
# Simply replace ParallelBlock with PhaseCoherentParallelBlock in the layer stack
    '''

    print(code)


if __name__ == "__main__":
    # Run examples
    example_usage()
    integration_with_parallel_block()
