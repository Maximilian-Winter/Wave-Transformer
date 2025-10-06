"""
Enhanced Neurological Attention with Masking
加入因果與注意遮蔽 - Adding causal and attention masking

順流而遮 - Masking that follows the natural flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DendriticIntegration(nn.Module):
    """
    Mimics dendritic temporal integration
    樹突時間整合 - Tree-branch temporal integration
    """

    def __init__(self, dim, integration_window=4):
        super().__init__()
        self.dim = dim
        self.window = integration_window

        # Temporal kernel - like dendritic calcium dynamics
        self.temporal_kernel = nn.Parameter(
            torch.exp(-torch.linspace(0, 3, integration_window))
        )

    def forward(self, x):
        """
        x: (batch, seq_len, dim)
        Integrate over temporal window
        """
        b, s, d = x.shape

        # Pad sequence for temporal integration
        padded = F.pad(x, (0, 0, self.window - 1, 0))

        # Create temporal windows
        windows = padded.unfold(1, self.window, 1)  # (b, s, d, window)

        # Apply temporal kernel (exponential decay)
        kernel = self.temporal_kernel.view(1, 1, 1, -1)
        integrated = (windows * kernel).sum(dim=-1)

        return integrated  # (b, s, d)


class SparseCompetition(nn.Module):
    """
    Lateral inhibition creates sparse activation
    側抑稀疏 - Lateral inhibition sparsity

    Enhanced with proper masking support
    """

    def __init__(self, top_k_ratio=0.1, temperature=1.0):
        super().__init__()
        self.top_k_ratio = top_k_ratio
        self.temperature = temperature

    def forward(self, attention_scores, attention_mask=None):
        """
        attention_scores: (batch, heads, seq_len, seq_len)
        attention_mask: (batch, 1, seq_len, seq_len) or (batch, 1, 1, seq_len)
                        True/1 = attend, False/0 = mask out

        順勢而為 - following the natural flow of relevance
        """
        b, h, s, _ = attention_scores.shape

        # Apply mask before competition (critical ordering)
        if attention_mask is not None:
            # Convert boolean mask to additive mask
            # True (1) -> 0.0 (no change), False (0) -> -inf (mask out)
            mask_value = torch.finfo(attention_scores.dtype).min
            attention_scores = attention_scores.masked_fill(
                ~attention_mask.bool(), mask_value
            )

        # Calculate k based on sequence length
        # Only compete among valid (non-masked) positions
        k = max(1, int(s * self.top_k_ratio))

        # Get top-k values and indices (masked positions will be -inf)
        topk_vals, topk_idx = torch.topk(
            attention_scores, k=k, dim=-1, largest=True
        )

        # Create sparse attention matrix
        sparse_attention = torch.full_like(
            attention_scores,
            torch.finfo(attention_scores.dtype).min
        )

        # Fill in top-k values
        sparse_attention.scatter_(
            dim=-1,
            index=topk_idx,
            src=topk_vals
        )

        # Competitive softmax - winner-take-more
        # Softmax naturally handles -inf values
        attention_probs = F.softmax(
            sparse_attention / self.temperature,
            dim=-1
        )

        return attention_probs


class OscillatoryModulation(nn.Module):
    """
    Simulates neural oscillations for attention coordination
    振盪協調 - Oscillatory coordination

    Fixed to handle flexible position_ids shapes
    """

    def __init__(self, dim, seq_len_estimate=512):
        super().__init__()
        self.dim = dim

        # Learnable phase and frequency offsets
        self.gamma_freq = nn.Parameter(torch.ones(dim) * 60)  # ~60 Hz
        self.theta_freq = nn.Parameter(torch.ones(1) * 6)  # ~6 Hz

        self.phase = nn.Parameter(torch.zeros(dim))

    def forward(self, x, timestep_positions):
        """
        x: (batch, seq_len, dim)
        timestep_positions: position encoding - flexible shapes:
            - (seq_len,)
            - (1, seq_len)
            - (batch, seq_len)

        順應形狀 - Adapting to shape naturally
        """
        b, s, d = x.shape

        # Normalize position_ids to (batch, seq_len, 1)
        # 形狀歸一 - Shape normalization
        if timestep_positions.dim() == 1:
            # (seq_len,) -> (1, seq_len, 1) -> broadcast to (batch, seq_len, 1)
            positions = timestep_positions.unsqueeze(0).unsqueeze(-1)
        elif timestep_positions.dim() == 2:
            # (1, seq_len) or (batch, seq_len) -> (batch, seq_len, 1)
            positions = timestep_positions.unsqueeze(-1)
        else:
            # Already (batch, seq_len, 1) or higher dim
            positions = timestep_positions

        # Ensure batch dimension matches
        if positions.size(0) == 1 and b > 1:
            positions = positions.expand(b, -1, -1)

        # Position-based phase
        # Gamma: fast local oscillation
        gamma_phase = (
                              2 * math.pi * self.gamma_freq.view(1, 1, -1) *
                              positions.float() / 100.0
                      ) + self.phase.view(1, 1, -1)

        # Theta: slow sequence oscillation
        theta_phase = (
                2 * math.pi * self.theta_freq *
                positions.float() / 100.0
        )

        # Modulation as phase coupling
        gamma_mod = torch.cos(gamma_phase)
        theta_mod = torch.cos(theta_phase)

        # Combined oscillatory gain
        osc_gain = 0.5 + 0.3 * gamma_mod + 0.2 * theta_mod

        return x * osc_gain


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class YaRNRotaryEmbedding(nn.Module):
    def __init__(
            self,
            d_model,
            max_len=2048,
            base=10000,
            scale=1.0,
            original_max_len=2048,
            beta_fast=32,
            beta_slow=1,
            mscale=1.0
    ):
        super().__init__()
        self.d_model = d_model

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))

        # YaRN scaling
        if scale > 1.0:
            wavelengths = 2 * math.pi / inv_freq
            smooth_factor = (wavelengths - beta_fast) / (beta_slow - beta_fast)
            smooth_factor = torch.clamp(smooth_factor, 0.0, 1.0)

            low_freq_factor = original_max_len / max_len
            scale_factors = (1 - smooth_factor) * 1.0 + smooth_factor * low_freq_factor
            inv_freq = inv_freq / scale_factors

            if mscale != 1.0:
                self.mscale = mscale * math.sqrt(1 + math.log(scale) / math.log(original_max_len))
            else:
                self.mscale = 1.0
        else:
            self.mscale = 1.0

        self.register_buffer('inv_freq', inv_freq)

        # Precompute for max length
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)

        self.register_buffer('cos', freqs.cos())
        self.register_buffer('sin', freqs.sin())

    def forward(self, q, k, position_ids=None):
        """
        Apply RoPE to Q and K tensors.
        q, k: (B, n_heads, seq_len, d_head)
        position_ids: Optional position indices (seq_len,) or (batch, seq_len)
        """
        seq_len = q.shape[2]

        if position_ids is not None:
            # Use custom positions
            if position_ids.dim() == 1:
                positions = position_ids
            else:
                positions = position_ids[0]  # Assume same positions for all batch items

            # Compute freqs on-the-fly for custom positions
            freqs = torch.outer(positions.float(), self.inv_freq)
            cos = freqs.cos()
            sin = freqs.sin()
        else:
            # Use precomputed values
            cos = self.cos[:seq_len]
            sin = self.sin[:seq_len]

        q_rot = self._apply_rotary(q, cos, sin)
        k_rot = self._apply_rotary(k, cos, sin)

        return q_rot, k_rot

    def _apply_rotary(self, x, cos, sin):
        """
        x: (B, n_heads, seq_len, d_head)
        cos, sin: (seq_len, d_head//2)
        """
        # Split into even and odd features
        x_even = x[..., 0::2]  # (B, n_heads, seq_len, d_head//2)
        x_odd = x[..., 1::2]  # (B, n_heads, seq_len, d_head//2)

        # Reshape cos/sin for broadcasting: (1, 1, seq_len, d_head//2)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Apply rotation
        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos

        return out * self.mscale


class NeurologicalAttention(nn.Module):
    """
    Neurologically-inspired attention with YaRN RoPE
    神經真意機 - Neural true-meaning mechanism
    """

    def __init__(
            self,
            dim,
            num_heads=8,
            integration_window=4,
            sparsity_ratio=0.1,
            dropout=0.1,
            is_causal=False,
            use_yarn=True,
            max_seq_len=512
    ):
        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.is_causal = is_causal
        self.use_yarn = use_yarn

        # Core neurological components
        self.dendritic_q = DendriticIntegration(dim, integration_window)
        self.dendritic_k = DendriticIntegration(dim, integration_window)
        self.dendritic_v = DendriticIntegration(dim, integration_window)

        self.oscillatory_mod = OscillatoryModulation(dim)
        self.sparse_compete = SparseCompetition(sparsity_ratio)

        # YaRN RoPE
        if use_yarn:
            self.yarn_rope = YaRNRotaryEmbedding(
                d_model=self.head_dim,
                max_len=max_seq_len,
                original_max_len=max_seq_len
            )

        # Projections
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Causal mask buffer
        self.register_buffer("causal_mask", None, persistent=False)

    def _get_causal_mask(self, seq_len, device, dtype):
        if self.causal_mask is None or self.causal_mask.size(-1) < seq_len:
            mask = torch.tril(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
            )
            self.causal_mask = mask
        return self.causal_mask[:seq_len, :seq_len]

    def forward(self, x, position_ids=None, attention_mask=None):
        """
        x: (batch, seq_len, dim)
        position_ids: Optional position indices
        attention_mask: Optional attention mask
        """
        b, s, d = x.shape

        # Create default position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(s, device=x.device, dtype=torch.long)

        # Project to Q, K, V
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = qkv

        # Apply dendritic temporal integration
        q = self.dendritic_q(q)
        k = self.dendritic_k(k)
        v = self.dendritic_v(v)

        # Apply oscillatory modulation
        q = self.oscillatory_mod(q, position_ids)
        k = self.oscillatory_mod(k, position_ids)

        # Reshape for multi-head: (B, seq_len, n_heads, d_head) -> (B, n_heads, seq_len, d_head)
        q = q.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply YaRN RoPE
        if self.use_yarn:
            q, k = self.yarn_rope(q, k, position_ids)

        # Compute attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale

        # Prepare combined mask
        combined_mask = None

        if self.is_causal:
            causal_mask = self._get_causal_mask(s, x.device, x.dtype)
            combined_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attn_mask = attn_mask.expand(b, 1, s, s)
            elif attention_mask.dim() == 3:
                attn_mask = attention_mask.unsqueeze(2)
            else:
                attn_mask = attention_mask

            if combined_mask is not None:
                combined_mask = combined_mask & attn_mask
            else:
                combined_mask = attn_mask

        # Apply sparse competition with combined mask
        attn_probs = self.sparse_compete(attn_scores, combined_mask)
        attn_probs = self.dropout(attn_probs)

        # Compute output
        out = attn_probs @ v
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        out = self.proj(out)

        return out


class NeurologicalTransformerBlock(nn.Module):
    """
    Drop-in replacement for standard transformer block
    簡潔落實 - Simple and grounded implementation
    """

    def __init__(
            self,
            dim,
            num_heads=8,
            mlp_ratio=4,
            is_causal=False,
            dropout=0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = NeurologicalAttention(
            dim,
            num_heads,
            is_causal=is_causal,
            dropout=dropout
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, position_ids=None, attention_mask=None):
        # Attention with residual
        attn_out = self.attn(
            self.norm1(x),
            position_ids=position_ids,
            attention_mask=attention_mask
        )
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x