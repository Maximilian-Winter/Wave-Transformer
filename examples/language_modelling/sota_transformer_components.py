"""
State-of-the-Art Transformer Components Collection
Cutting-edge architectural improvements from recent papers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from einops import rearrange
from matplotlib import pyplot as plt


# ============================================================================
# POSITIONAL ENCODINGS
# ============================================================================

class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE) from RoFormer paper
    Used in: LLaMA, PaLM, CodeGen
    Benefits: Better length extrapolation, no learned parameters
    """
    def __init__(self, dim, max_seq_len=5000, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Cache for cos and sin
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = -1

    def _get_cos_sin(self, seq_len, device):
        if seq_len != self._seq_len_cached:
            t = torch.arange(seq_len, device=device).float()
            freqs = torch.outer(t, self.inv_freq)
            # Concatenate to match dimension
            freqs = torch.cat([freqs, freqs], dim=-1)
            self._cos_cached = torch.cos(freqs).to(device)
            self._sin_cached = torch.sin(freqs).to(device)
            self._seq_len_cached = seq_len
        return self._cos_cached, self._sin_cached

    def forward(self, q, k):
        """Apply RoPE to queries and keys"""
        batch_size, seq_len, n_heads, d_head = q.shape
        cos, sin = self._get_cos_sin(seq_len, q.device)

        # Reshape cos and sin to match q, k dimensions
        cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, d_head)
        sin = sin.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, d_head)

        # Apply rotation
        q_rot = self._apply_rotation(q, cos, sin)
        k_rot = self._apply_rotation(k, cos, sin)

        return q_rot, k_rot

    def _apply_rotation(self, x, cos, sin):
        # Split x into two halves for rotation
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]

        # Rotate using complex number multiplication
        # (x1 + ix2) * (cos + isin) = (x1*cos - x2*sin) + i(x1*sin + x2*cos)
        rotated = torch.cat([
            x1 * cos[..., :x1.shape[-1]] - x2 * sin[..., :x2.shape[-1]],
            x1 * sin[..., :x1.shape[-1]] + x2 * cos[..., :x2.shape[-1]]
        ], dim=-1)

        return rotated


class ALiBi(nn.Module):
    """
    Attention with Linear Biases (ALiBi)
    From: "Train Short, Test Long" paper
    Benefits: No position embeddings needed, excellent length extrapolation
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

        # Geometric sequence of slopes
        slopes = torch.tensor([2 ** (-8 * i / n_heads) for i in range(n_heads)])
        self.register_buffer('slopes', slopes)

    def forward(self, attention_scores, seq_len):
        """Add ALiBi bias to attention scores"""
        # Create position bias matrix
        positions = torch.arange(seq_len, device=attention_scores.device)
        distances = positions.unsqueeze(0) - positions.unsqueeze(1)

        # Apply slopes
        alibi = distances.unsqueeze(0) * self.slopes.view(-1, 1, 1)

        return attention_scores + alibi


# ============================================================================
# FEED-FORWARD VARIANTS
# ============================================================================

class SwiGLU(nn.Module):
    """
    SwiGLU activation from PaLM/LLaMA
    Outperforms standard FFN in most benchmarks
    """
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        # Need 3 projections for gated activation
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU: x * SiLU(gate)
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class MoE(nn.Module):
    """
    Mixture of Experts layer from Switch Transformers/Mixtral
    Sparse routing to multiple expert FFNs
    """
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2, dropout=0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Router (gate)
        self.router = nn.Linear(d_model, num_experts)

        # Experts
        self.experts = nn.ModuleList([
            SwiGLU(d_model, d_ff, dropout) for _ in range(num_experts)
        ])

        # Load balancing loss weight
        self.load_balance_loss = 0.0

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        # Compute routing scores
        router_logits = self.router(x)  # (B, L, E)
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # Renormalize

        # Route tokens to experts
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_mask = topk_indices[..., i]  # (B, L)
            expert_weights = topk_probs[..., i].unsqueeze(-1)  # (B, L, 1)

            for expert_id in range(self.num_experts):
                mask = (expert_mask == expert_id).unsqueeze(-1)
                if mask.any():
                    expert_input = x * mask
                    expert_output = self.experts[expert_id](expert_input)
                    output += expert_output * expert_weights * mask

        # Compute load balancing loss (for training)
        if self.training:
            # Fraction of tokens routed to each expert
            tokens_per_expert = router_probs.sum(dim=[0, 1]) / (batch_size * seq_len)
            # Ideal uniform distribution
            ideal_load = 1.0 / self.num_experts
            # Auxiliary loss to balance load
            self.load_balance_loss = ((tokens_per_expert - ideal_load) ** 2).sum()

        return output


# ============================================================================
# NORMALIZATION VARIANTS
# ============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    Used in: LLaMA, T5
    Benefits: Simpler, faster than LayerNorm, often better performance
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / norm


class GroupNorm(nn.Module):
    """
    Group Normalization for transformers
    Benefits: More stable training, works well with small batch sizes
    """
    def __init__(self, num_groups, d_model, eps=1e-5):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, d_model, eps=eps)

    def forward(self, x):
        # Reshape for GroupNorm (expects channel-first)
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.gn(x)
        return x.transpose(1, 2)  # (B, L, D)


# ============================================================================
# ATTENTION VARIANTS
# ============================================================================

class FlashAttentionWithRoPE(nn.Module):
    """
    Flash Attention combined with RoPE
    The modern standard for large language models
    """
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RoPE(self.d_head)  # RoPE operates on head dimension

        try:
            from flash_attn import flash_attn_qkvpacked_func
            self.use_flash = True
        except:
            self.use_flash = False

    def forward(self, x, causal=True):
        B, L, D = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)

        # Apply RoPE to q and k
        q, k = self.rope(q, k)

        if self.use_flash:
            try:
                # Recombine for packed flash attention
                qkv = torch.stack([q, k, v], dim=2)
                from flash_attn import flash_attn_qkvpacked_func
                out = flash_attn_qkvpacked_func(
                    qkv.half(),
                    causal=causal,
                    softmax_scale=1.0 / math.sqrt(self.d_head)
                ).to(qkv.dtype)
            except:
                self.use_flash = False

        if not self.use_flash:
            # Fallback to standard attention
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
            if causal:
                mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
                scores.masked_fill_(mask, float('-inf'))

            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v).transpose(1, 2)

        out = out.reshape(B, L, D)
        return self.out_proj(out)


class SlidingWindowAttention(nn.Module):
    """
    Local sliding window attention from Longformer/Mistral
    Efficient for long sequences
    """
    def __init__(self, d_model, n_heads, window_size=256, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)

        # Use Flash Attention with window if available
        try:
            from flash_attn import flash_attn_func
            q, k, v = q.half(), k.half(), v.half()
            out = flash_attn_func(
                q, k, v,
                window_size=(self.window_size, 0),  # Look back window_size tokens
                causal=True,
                softmax_scale=1.0 / math.sqrt(self.d_head)
            ).to(x.dtype)
        except:
            # Manual sliding window implementation
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

            # Create sliding window mask
            mask = torch.ones(L, L, device=x.device, dtype=torch.bool)
            for i in range(L):
                start = max(0, i - self.window_size + 1)
                mask[i, start:i+1] = False
            scores.masked_fill_(mask, float('-inf'))

            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v).transpose(1, 2)

        out = out.reshape(B, L, D)
        return self.out_proj(out)


# ============================================================================
# ADVANCED BLOCK ARCHITECTURES
# ============================================================================

class ParallelBlock(nn.Module):
    """
    Parallel attention and FFN from GPT-J/PaLM
    Reduces latency by computing attention and FFN in parallel
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.attn = FlashAttentionWithRoPE(d_model, n_heads, dropout)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal=True):
        # Single normalization, then parallel paths
        normalized = self.norm(x)
        attn_out = self.attn(normalized, causal)
        ffn_out = self.ffn(normalized)

        # Combine and add residual
        return x + self.dropout(attn_out + ffn_out)


class SandwichBlock(nn.Module):
    """
    Sandwich-style normalization from CogView
    Pre-norm + Post-norm for better stability
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        # Pre-norms
        self.norm1_pre = RMSNorm(d_model)
        self.norm2_pre = RMSNorm(d_model)
        # Post-norms
        self.norm1_post = RMSNorm(d_model)
        self.norm2_post = RMSNorm(d_model)

        self.attn = FlashAttentionWithRoPE(d_model, n_heads, dropout)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal=True):
        # Attention with sandwich norm
        attn_out = self.attn(self.norm1_pre(x), causal)
        x = x + self.dropout(self.norm1_post(attn_out))

        # FFN with sandwich norm
        ffn_out = self.ffn(self.norm2_pre(x))
        x = x + self.dropout(self.norm2_post(ffn_out))

        return x


class SkipConnection(nn.Module):
    """
    Learnable skip connections from ReZero/SkipInit
    Improves gradient flow in deep networks
    """
    def __init__(self, d_model, init_value=0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((d_model,), init_value))

    def forward(self, x, residual):
        return residual + self.alpha * x


# ============================================================================
# MODERN TRANSFORMER ARCHITECTURE
# ============================================================================

class ModernTransformerBlock(nn.Module):
    """
    State-of-the-art transformer block combining best practices
    """
    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        dropout=0.0,
        use_moe=False,
        num_experts=8,
        use_parallel=True
    ):
        super().__init__()
        self.use_parallel = use_parallel

        if use_parallel:
            # Parallel architecture (faster)
            self.block = ParallelBlock(d_model, n_heads, d_ff, dropout)
        else:
            # Sequential architecture (traditional)
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
            self.attn = FlashAttentionWithRoPE(d_model, n_heads, dropout)

            if use_moe:
                self.ffn = MoE(d_model, d_ff, num_experts, top_k=2, dropout=dropout)
            else:
                self.ffn = SwiGLU(d_model, d_ff, dropout)

            self.dropout = nn.Dropout(dropout)

            # Learnable skip connections
            self.skip1 = SkipConnection(d_model, init_value=0.1)
            self.skip2 = SkipConnection(d_model, init_value=0.1)

    def forward(self, x, causal=True):
        if self.use_parallel:
            return self.block(x, causal)
        else:
            # Pre-norm with learnable skip
            attn_out = self.attn(self.norm1(x), causal)
            x = self.skip1(self.dropout(attn_out), x)

            ffn_out = self.ffn(self.norm2(x))
            x = self.skip2(self.dropout(ffn_out), x)

            return x


class ModernTransformer(nn.Module):
    """
    Complete modern transformer with all the bells and whistles
    """
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_layers=12,
        num_heads=8,
        d_ff=2048,
        dropout=0.1,
        max_seq_len=8192,
        use_moe=False,
        num_experts=8,
        use_parallel=True,
        tie_embeddings=True
    ):
        super().__init__()

        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # No position embeddings needed with RoPE!


        # Transformer blocks
        self.blocks = nn.ModuleList([
            ModernTransformerBlock(
                d_model, num_heads, d_ff, dropout,
                use_moe=(use_moe and i % 2 == 1),  # MoE every other layer
                num_experts=num_experts,
                use_parallel=use_parallel
            )
            for i in range(num_layers)
        ])

        # Final norm
        self.norm_f = RMSNorm(d_model)

        # Output projection
        if tie_embeddings:
            self.lm_head = lambda x: F.linear(x, self.token_emb.weight)
        else:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Modern weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, causal=True):
        # Embed tokens
        x = self.token_emb(input_ids)

        # Scale embeddings (important for stability)
        x = x * math.sqrt(self.token_emb.embedding_dim)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, causal)

        # Final normalization
        x = self.norm_f(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        return logits




class LearnableActivation(nn.Module):
    """
    A generic learnable activation.
    Uses a small hidden layer to map scalars -> scalars,
    essentially learning its own transfer curve.
    """

    def __init__(self, hidden_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.view(-1, 1)
        out = torch.tanh(self.fc1(x_flat))
        out = self.fc2(out)
        return out.view(shape)


class Bagua(nn.Module):
    """Eight Trigrams - Universal patterns of change"""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Eight fundamental gates
        self.qian = LearnableActivation()  # Heaven ☰
        self.kun = LearnableActivation()  # Earth ☷
        self.zhen = LearnableActivation()  # Thunder ☳
        self.xun = LearnableActivation()  # Wind ☴
        self.kan = LearnableActivation()  # Water ☵
        self.li = LearnableActivation()  # Fire ☲
        self.gen = LearnableActivation()  # Mountain ☶
        self.dui = LearnableActivation()  # Lake ☱

        # Learn weights for combining the eight gates
        self.bagua_weights = nn.Linear(d_model, 8)

    def visualize_activations(self, device="cpu", num_points=200, file_name="bagua_activations.png"):
        """
        Plot the learned transfer curves for each Bagua gate.
        """
        self.eval()
        x = torch.linspace(-3, 3, num_points, device=device).view(-1, 1)

        gates = {
            "Qian ☰ (Heaven)": self.qian,
            "Kun ☷ (Earth)": self.kun,
            "Zhen ☳ (Thunder)": self.zhen,
            "Xun ☴ (Wind)": self.xun,
            "Kan ☵ (Water)": self.kan,
            "Li ☲ (Fire)": self.li,
            "Gen ☶ (Mountain)": self.gen,
            "Dui ☱ (Lake)": self.dui,
        }

        plt.figure(figsize=(14, 8))
        for i, (name, gate) in enumerate(gates.items()):
            y = gate(x).detach().cpu().numpy()
            plt.subplot(2, 4, i + 1)
            plt.plot(x.cpu().numpy(), y, label=name, color="C{}".format(i))
            plt.title(name, fontsize=10)
            plt.grid(True, alpha=0.3)
        plt.suptitle("Learned Bagua Activation Transfer Curves", fontsize=16)
        plt.tight_layout()
        plt.show()
        plt.savefig(file_name)


    def forward(self, x):
        # Compute all gate outputs
        gates = [
            self.qian(x),
            self.kun(x),
            self.zhen(x),
            self.xun(x),
            self.kan(x),
            self.li(x),
            self.gen(x),
            self.dui(x),
        ]  # list of 8 tensors, each [batch, seq_len, d_model] (if x is that shape)

        # Stack along a new "gates" dimension: [batch, seq_len, d_model, 8]
        gates = torch.stack(gates, dim=-1)

        # Compute weights per feature: [batch, seq_len, 8]
        weights = self.bagua_weights(x)  # same shape as input but last dim=8
        weights = F.softmax(weights, dim=-1)

        # Expand weights to match gates: [batch, seq_len, d_model, 8]
        weights = weights.unsqueeze(2).expand_as(gates)

        # Weighted sum across the gates dimension
        out = (gates * weights).sum(dim=-1)  # [batch, seq_len, d_model]

        return out

class AdvancedMechanismDetector(nn.Module):
    """
    Enhanced mechanism detection based on multiple strategic principles:
    - Entropy changes (information theory)
    - Semantic shifts (meaning transitions)
    - Temporal patterns (timing opportunities)
    - Energy gradients (force differentials)
    """

    def __init__(self, d_model, num_mechanisms=12):
        super().__init__()
        self.d_model = d_model
        self.num_mechanisms = num_mechanisms

        # Multi-scale pattern detection
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(d_model, num_mechanisms, kernel_size=k, padding=k // 2)
            for k in [3, 5, 7]  # Different time scales
        ])

        # Entropy-based detection (information density changes)
        self.entropy_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_mechanisms)
        )

        # Semantic shift detection (meaning boundaries)
        self.semantic_detector = nn.LSTM(d_model, d_model // 2, batch_first=True)
        self.semantic_classifier = nn.Linear(d_model // 2, num_mechanisms)

        # Temporal rhythm detection (timing patterns)
        self.rhythm_detector = FlashAttentionWithRoPE(d_model, 4)
        self.rhythm_classifier = nn.Linear(d_model, num_mechanisms)

        # Final mechanism synthesis
        self.mechanism_synthesizer = nn.Sequential(
            nn.Linear(num_mechanisms * 4, num_mechanisms * 2),
            nn.ReLU(),
            nn.Linear(num_mechanisms * 2, num_mechanisms),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, seq, d_model = x.shape
        mechanisms = []

        # Multi-scale convolution detection
        x_conv = rearrange(x, 'b s d -> b d s')
        conv_mechanisms = []
        for conv in self.conv_layers:
            conv_out = conv(x_conv)
            conv_mechanisms.append(rearrange(conv_out, 'b m s -> b s m'))
        multi_scale = torch.stack(conv_mechanisms).mean(dim=0)
        mechanisms.append(multi_scale)

        # Entropy-based detection
        entropy_mechanisms = torch.sigmoid(self.entropy_detector(x))
        mechanisms.append(entropy_mechanisms)

        # Semantic shift detection
        semantic_hidden, _ = self.semantic_detector(x)
        semantic_mechanisms = torch.sigmoid(self.semantic_classifier(semantic_hidden))
        mechanisms.append(semantic_mechanisms)

        # Temporal rhythm detection
        rhythm_attn = self.rhythm_detector(x)
        rhythm_mechanisms = torch.sigmoid(self.rhythm_classifier(rhythm_attn))
        mechanisms.append(rhythm_mechanisms)

        # Synthesize all mechanism signals
        all_mechanisms = torch.cat(mechanisms, dim=-1)
        final_mechanisms = self.mechanism_synthesizer(all_mechanisms)

        return final_mechanisms


class FlowController(nn.Module):
    """
    Controls the flow of transformations based on mechanism detection
    Implements the "grabbing" of mechanisms at optimal moments
    """

    def __init__(self, d_model, num_mechanisms=8):
        super().__init__()
        self.d_model = d_model
        self.num_mechanisms = num_mechanisms

        # Decide which mechanisms to activate
        self.activation_threshold = nn.Parameter(torch.tensor(0.5))

        # Route information based on active mechanisms
        self.flow_router = nn.Linear(d_model + num_mechanisms, d_model)

    def forward(self, x, mechanism_strengths):

        # Determine which mechanisms are active (above threshold)
        active_mechanisms = (mechanism_strengths > self.activation_threshold).float()

        # Create gating signal based on active mechanisms
        gate_input = torch.cat([x, mechanism_strengths], dim=-1)
        flow_gate = torch.sigmoid(self.flow_router(gate_input))

        # Apply transformations only where mechanisms are active
        mechanism_gate = active_mechanisms.sum(dim=-1, keepdim=True) / self.num_mechanisms
        mechanism_gate = torch.clamp(mechanism_gate, 0, 1)

        # Combine original and transformed based on mechanism activity
        output = x + mechanism_gate * flow_gate

        return output

class ModernDaoTransformer(nn.Module):
    """
    Complete modern transformer with all the bells and whistles
    """
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_layers=12,
        num_heads=8,
        d_ff=2048,
        dropout=0.1,
        max_seq_len=8192,
        use_moe=False,
        num_experts=8,
        use_parallel=True,
        tie_embeddings=True
    ):
        super().__init__()

        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # No position embeddings needed with RoPE!

        self.mechanism_detector = AdvancedMechanismDetector(d_model, num_mechanisms=64)
        self.flow_controller = FlowController(d_model, num_mechanisms=64)
        self.norm_flow = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ModernTransformerBlock(
                d_model, num_heads, d_ff, dropout,
                use_moe=(use_moe and i % 2 == 1),  # MoE every other layer
                num_experts=num_experts,
                use_parallel=use_parallel,
            )
            for i in range(num_layers)
        ])

        # Final norm
        self.norm_f = RMSNorm(d_model)

        # Output projection
        if tie_embeddings:
            self.lm_head = lambda x: F.linear(x, self.token_emb.weight)
        else:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Modern weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, causal=True):
        # Embed tokens
        x = self.token_emb(input_ids)

        # Scale embeddings (important for stability)
        x = x * math.sqrt(self.token_emb.embedding_dim)

        mechanism_strengths = self.mechanism_detector(x)
        flow_output = self.flow_controller(x, mechanism_strengths)
        x = self.dropout(x + flow_output)
        x = self.norm_flow(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, causal)

        # Final normalization
        x = self.norm_f(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        return logits
# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*60)
    print("MODERN TRANSFORMER COMPONENTS SHOWCASE")
    print("="*60)

    # Test configuration
    batch_size = 2
    seq_len = 512
    vocab_size = 50000
    d_model = 512
    n_heads = 8
    d_ff = 2048

    # Create input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    print("\n1. Testing RoPE (Rotary Position Embeddings):")
    d_head = d_model // n_heads
    rope = RoPE(d_head).to(device)  # RoPE operates on head dimension
    q = k = torch.randn(batch_size, seq_len, n_heads, d_head).to(device)
    q_rot, k_rot = rope(q, k)
    print(f"   Input shape: {q.shape}, Output shape: {q_rot.shape}")

    print("\n2. Testing SwiGLU (Better FFN):")
    swiglu = SwiGLU(d_model, d_ff).to(device)
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    out = swiglu(x)
    print(f"   Input shape: {x.shape}, Output shape: {out.shape}")

    print("\n3. Testing MoE (Mixture of Experts):")
    moe = MoE(d_model, d_ff, num_experts=4, top_k=2).to(device)
    out = moe(x)
    print(f"   Input shape: {x.shape}, Output shape: {out.shape}")
    print(f"   Load balance loss: {moe.load_balance_loss:.4f}")

    print("\n4. Testing RMSNorm (Faster normalization):")
    rmsnorm = RMSNorm(d_model).to(device)
    out = rmsnorm(x)
    print(f"   Input shape: {x.shape}, Output shape: {out.shape}")

    print("\n5. Testing Flash Attention with RoPE:")
    attn = FlashAttentionWithRoPE(d_model, n_heads).to(device)
    out = attn(x, causal=True)
    print(f"   Input shape: {x.shape}, Output shape: {out.shape}")

    print("\n6. Testing Parallel Block (GPT-J style):")
    parallel = ParallelBlock(d_model, n_heads, d_ff).to(device)
    out = parallel(x, causal=True)
    print(f"   Input shape: {x.shape}, Output shape: {out.shape}")

    print("\n7. Testing Complete Modern Transformer:")
    model = ModernTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=4,
        n_heads=n_heads,
        d_ff=d_ff,
        use_moe=True,
        use_parallel=True
    ).to(device)

    logits = model(input_ids, causal=True)
    print(f"   Input shape: {input_ids.shape}, Output shape: {logits.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    print("\n" + "="*60)
    print("KEY IMPROVEMENTS OVER STANDARD TRANSFORMER:")
    print("="*60)
    print("""
    1. RoPE instead of learned/sinusoidal positions (better extrapolation)
    2. SwiGLU instead of ReLU/GELU FFN (better performance)
    3. RMSNorm instead of LayerNorm (faster, often better)
    4. Flash Attention (2-4x faster, less memory)
    5. MoE for scaling (more capacity, same compute)
    6. Parallel blocks (reduced latency)
    7. Learnable skip connections (better gradient flow)
    8. Tied embeddings (parameter efficiency)
    """)