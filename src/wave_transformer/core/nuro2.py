import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def create_padding_mask(lengths, max_length=None):
    """Create padding mask from sequence lengths"""
    if isinstance(lengths, list):
        lengths = torch.tensor(lengths)

    if max_length is None:
        max_length = lengths.max().item()

    batch_size = len(lengths)
    mask = torch.arange(max_length).expand(batch_size, max_length) < lengths.unsqueeze(1)
    return mask


def create_causal_mask(seq_length):
    """Create causal (lower triangular) mask"""
    return torch.tril(torch.ones(seq_length, seq_length, dtype=torch.bool))


class StableInhibitoryActivation(nn.Module):
    """Stabilized activation that can excite and inhibit with gentler gradients"""

    def __init__(self, threshold=0.0, sharpness=2.0, temperature=1.0):
        super().__init__()
        self.threshold = threshold
        self.sharpness = sharpness  # Reduced from 10.0 to 2.0
        self.temperature = temperature

    def forward(self, x, mask=None):
        # Gentler sigmoid gating with temperature control
        gate = torch.sigmoid(self.sharpness * (x - self.threshold) / self.temperature)

        # Smooth gating instead of hard multiplication
        output = x * gate

        # Apply mask with less extreme values
        if mask is not None:
            output = output.masked_fill(~mask, -1e4)  # Less extreme than -1e9

        return output


class LocalAttentionCluster(nn.Module):
    """Stabilized local attention with better gradient flow"""

    def __init__(self, d_model, cluster_size=8, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.cluster_size = cluster_size
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)  # Output projection

        self.inhibitory = StableInhibitoryActivation()
        self.dropout = nn.Dropout(0.1)

        # Initialize weights for stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better stability"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(module.weight, gain=1 / math.sqrt(2))

    def create_cluster_mask(self, seq_len, causal_mask=None, padding_mask=None):
        """Create attention mask for clustered attention"""
        device = causal_mask.device if causal_mask is not None else (
            padding_mask.device if padding_mask is not None else torch.device('cpu'))

        if seq_len <= self.cluster_size:
            if causal_mask is not None and padding_mask is not None:
                return causal_mask & padding_mask
            elif causal_mask is not None:
                return causal_mask
            elif padding_mask is not None:
                return padding_mask
            else:
                return None

        cluster_mask = None
        if causal_mask is not None or padding_mask is not None:
            cluster_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)

            if causal_mask is not None:
                cluster_mask = cluster_mask & causal_mask

            if padding_mask is not None:
                seq_mask = padding_mask[0] if len(padding_mask.shape) > 1 else padding_mask
                cluster_mask = cluster_mask & seq_mask.unsqueeze(0) & seq_mask.unsqueeze(1)

        return cluster_mask

    def forward(self, x, causal_mask=None, padding_mask=None):
        B, L, D = x.shape
        original_length = L

        attention_mask = self.create_cluster_mask(L, causal_mask, padding_mask)

        if L <= self.cluster_size:
            q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(0).unsqueeze(0).expand(B, self.n_heads, -1, -1)
                attention_weights = self.inhibitory(scores, mask_expanded)
            else:
                attention_weights = self.inhibitory(scores)

            # Add stability through dropout and normalization
            attention_weights = self.dropout(attention_weights)
            attended = torch.matmul(attention_weights, v)
            attended = attended.transpose(1, 2).contiguous().view(B, L, D)
            return self.o_proj(attended)

        # Clustering logic with stability improvements
        n_clusters = L // self.cluster_size
        remainder = L % self.cluster_size
        padded_input = x

        if remainder > 0:
            pad_len = self.cluster_size - remainder
            padded_input = F.pad(x, (0, 0, 0, pad_len))
            L = padded_input.shape[1]
            n_clusters = L // self.cluster_size

        clustered = padded_input.view(B, n_clusters, self.cluster_size, D)

        q = self.q_proj(clustered).view(B, n_clusters, self.cluster_size, self.n_heads, self.head_dim)
        k = self.k_proj(clustered).view(B, n_clusters, self.cluster_size, self.n_heads, self.head_dim)
        v = self.v_proj(clustered).view(B, n_clusters, self.cluster_size, self.n_heads, self.head_dim)

        q = q.transpose(2, 3)
        k = k.transpose(2, 3)
        v = v.transpose(2, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            padded_mask = attention_mask
            if remainder > 0:
                padded_mask = F.pad(attention_mask, (0, pad_len, 0, pad_len), value=False)

            cluster_masks = []
            for i in range(n_clusters):
                start_idx = i * self.cluster_size
                end_idx = start_idx + self.cluster_size
                cluster_mask = padded_mask[start_idx:end_idx, start_idx:end_idx]
                cluster_masks.append(cluster_mask)

            cluster_mask_tensor = torch.stack(cluster_masks, dim=0)
            cluster_mask_expanded = cluster_mask_tensor.unsqueeze(0).unsqueeze(2).expand(B, -1, self.n_heads, -1, -1)

            attention_weights = self.inhibitory(scores, cluster_mask_expanded)
        else:
            attention_weights = self.inhibitory(scores)

        attention_weights = self.dropout(attention_weights)
        attended = torch.matmul(attention_weights, v)

        attended = attended.transpose(2, 3).contiguous()
        attended = attended.view(B, n_clusters, self.cluster_size, D)
        attended = attended.view(B, L, D)
        attended = attended[:, :original_length]

        return self.o_proj(attended)


class SmoothImportanceGate(nn.Module):
    """Smoother importance gating with learnable threshold"""

    def __init__(self, d_model, gate_threshold=0.5, temperature=2.0):
        super().__init__()
        self.importance_scorer = nn.Linear(d_model, 1)
        # Make threshold learnable for better adaptation
        self.threshold = nn.Parameter(torch.tensor(gate_threshold))
        self.temperature = temperature

        # Initialize for stability
        nn.init.xavier_uniform_(self.importance_scorer.weight)
        nn.init.zeros_(self.importance_scorer.bias)

    def forward(self, x, padding_mask=None):
        # Compute importance scores
        importance = torch.sigmoid(self.importance_scorer(x)).squeeze(-1)

        # Mask out padding tokens
        if padding_mask is not None:
            importance = importance * padding_mask.float()

        # Smooth gating instead of hard threshold
        gates = torch.sigmoid((importance - self.threshold) * self.temperature)

        # Apply smooth gates
        gated_output = x * gates.unsqueeze(-1)

        return gated_output, importance


class StabilizedHierarchicalAttention(nn.Module):
    """Hierarchical attention with improved stability"""

    def __init__(self, d_model, n_levels=3, base_cluster_size=8):
        super().__init__()
        self.n_levels = n_levels
        self.base_cluster_size = base_cluster_size

        cluster_sizes = [max(2, base_cluster_size // (2 ** i)) for i in range(n_levels)]

        self.local_attention = nn.ModuleList([
            LocalAttentionCluster(d_model, cluster_sizes[i])
            for i in range(n_levels)
        ])

        self.importance_gates = nn.ModuleList([
            SmoothImportanceGate(d_model)
            for _ in range(n_levels - 1)
        ])

        self.level_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_levels)
        ])

        # Projection layers with proper initialization
        self.upsample_projections = nn.ModuleList([
            nn.Linear(d_model, d_model)
            for _ in range(n_levels - 1)
        ])

        # Learnable scaling factors for residual connections
        self.level_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.5)  # Start with 0.5 for stability
            for _ in range(n_levels)
        ])

        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights for stability"""
        for proj in self.upsample_projections:
            nn.init.xavier_uniform_(proj.weight, gain=1 / math.sqrt(2))
            nn.init.zeros_(proj.bias)

    def create_causal_mask(self, seq_len, device):
        return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

    def downsample_mask(self, mask):
        if mask is None:
            return None

        B, L = mask.shape
        if L % 2 == 1:
            mask = F.pad(mask, (0, 1), value=False)
            L += 1

        downsampled = mask.view(B, L // 2, 2).any(dim=2)
        return downsampled

    def upsample_mask(self, mask, target_length):
        if mask is None:
            return None

        B, L = mask.shape
        if L == target_length:
            return mask

        repeat_factor = target_length // L
        remainder = target_length % L

        if repeat_factor > 0:
            upsampled = mask.repeat_interleave(repeat_factor, dim=1)
            if remainder > 0:
                extra = mask[:, :remainder]
                upsampled = torch.cat([upsampled, extra], dim=1)
        else:
            upsampled = mask[:, :target_length]

        return upsampled

    def forward(self, x, causal=True, padding_mask=None):
        level_outputs = []
        current = x
        current_padding_mask = padding_mask
        original_length = x.shape[1]
        device = x.device

        causal_mask = self.create_causal_mask(original_length, device) if causal else None
        current_causal_mask = causal_mask

        # Downward pass with scaled residuals
        for level in range(self.n_levels):
            attended = self.local_attention[level](
                current,
                causal_mask=current_causal_mask,
                padding_mask=current_padding_mask
            )

            # Scaled residual connection for stability
            scale = torch.sigmoid(self.level_scales[level])  # Ensure 0 < scale < 1
            attended = self.level_norms[level](current + scale * attended)

            level_outputs.append(attended)

            if level < self.n_levels - 1:
                gated, importance = self.importance_gates[level](attended, current_padding_mask)

                B, L, D = gated.shape
                if L % 2 == 1:
                    gated = F.pad(gated, (0, 0, 0, 1))
                    L += 1

                current = gated.view(B, L // 2, 2, D).mean(dim=2)
                current_padding_mask = self.downsample_mask(current_padding_mask)

                if current_causal_mask is not None:
                    new_seq_len = current.shape[1]
                    current_causal_mask = self.create_causal_mask(new_seq_len, device)

        # Upward pass with careful combination
        output = level_outputs[-1]

        for level in range(self.n_levels - 2, -1, -1):
            target_length = level_outputs[level].shape[1]

            if output.shape[1] != target_length:
                repeat_factor = target_length // output.shape[1]
                remainder = target_length % output.shape[1]

                if repeat_factor > 0:
                    upsampled = output.repeat_interleave(repeat_factor, dim=1)
                    if remainder > 0:
                        extra = output[:, :remainder]
                        upsampled = torch.cat([upsampled, extra], dim=1)
                else:
                    upsampled = output[:, :target_length]

                output = self.upsample_projections[level](upsampled)

            # Weighted combination instead of direct addition
            alpha = 0.7  # Weight for current level
            output = alpha * level_outputs[level] + (1 - alpha) * output

        if output.shape[1] != original_length:
            output = output[:, :original_length]

        if padding_mask is not None:
            output = output * padding_mask.unsqueeze(-1).float()

        return output


class StabilizedNeurologicalTransformerBlock(nn.Module):
    """Stabilized transformer block with gradient clipping and better initialization"""

    def __init__(self, d_model, d_ff=None, n_levels=3, base_cluster_size=8, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.attention = StabilizedHierarchicalAttention(d_model, n_levels, base_cluster_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable scaling for residuals
        self.attn_scale = nn.Parameter(torch.ones(1) * 0.5)
        self.ff_scale = nn.Parameter(torch.ones(1) * 0.5)

        self._init_weights()

    def _init_weights(self):
        """Proper weight initialization for stability"""
        for module in self.feed_forward:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1 / math.sqrt(2))
                nn.init.zeros_(module.bias)

    def forward(self, x, causal=True, padding_mask=None):
        # Attention with scaled residual
        attended = self.attention(x, causal=causal, padding_mask=padding_mask)
        attn_scale = torch.sigmoid(self.attn_scale)
        x = self.norm1(x + attn_scale * self.dropout(attended))

        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1).float()

        # Feed forward with scaled residual
        ff_out = self.feed_forward(x)
        ff_scale = torch.sigmoid(self.ff_scale)
        x = self.norm2(x + ff_scale * ff_out)

        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1).float()

        return x


# Debugging utilities
class LossMonitor:
    """Monitor for detecting training instabilities"""

    def __init__(self, window_size=10, patience=5):
        self.losses = []
        self.window_size = window_size
        self.patience = patience
        self.oscillation_count = 0

    def add_loss(self, loss):
        self.losses.append(loss)

        if len(self.losses) >= self.window_size:
            # Check for oscillations in recent window
            recent_losses = self.losses[-self.window_size:]

            # Simple oscillation detection: alternating up/down trends
            trends = []
            for i in range(1, len(recent_losses)):
                if recent_losses[i] > recent_losses[i - 1]:
                    trends.append(1)  # up
                else:
                    trends.append(-1)  # down

            # Count trend changes
            changes = sum(1 for i in range(1, len(trends)) if trends[i] != trends[i - 1])

            if changes >= len(trends) * 0.7:  # More than 70% changes = oscillation
                self.oscillation_count += 1
                if self.oscillation_count >= self.patience:
                    return "OSCILLATING"
            else:
                self.oscillation_count = 0

        return "STABLE"


def get_recommended_hyperparams(d_model, seq_len):
    """Get recommended hyperparameters based on model size"""
    base_lr = 1e-4 * math.sqrt(512 / d_model)  # Scale with model size

    if seq_len <= 32:
        n_levels = 2
        cluster_size = 8
    elif seq_len <= 128:
        n_levels = 3
        cluster_size = 8
    else:
        n_levels = 4
        cluster_size = 16

    return {
        'learning_rate': base_lr,
        'n_levels': n_levels,
        'base_cluster_size': cluster_size,
        'warmup_steps': 1000,
        'weight_decay': 0.1,
        'gradient_clip': 1.0
    }


# Test function with stability monitoring
def stability():
    """Test the stabilized attention mechanism"""
    print("🧠 Testing Stabilized Neurological Attention")
    print("=" * 50)

    d_model = 256
    seq_len = 256
    batch_size = 32

    # Get recommended hyperparameters
    hyperparams = get_recommended_hyperparams(d_model, seq_len)
    print(f"📋 Recommended hyperparameters for d_model={d_model}, seq_len={seq_len}:")
    for key, value in hyperparams.items():
        print(f"   {key}: {value}")

    # Test basic functionality
    x = torch.randn(batch_size, seq_len, d_model)
    model = StabilizedNeurologicalTransformerBlock(
        d_model,
        n_levels=hyperparams['n_levels'],
        base_cluster_size=hyperparams['base_cluster_size']
    )

    print(f"\n🧪 Testing forward pass...")
    try:
        output = model(x, causal=True)
        print(f"✓ Input: {x.shape} → Output: {output.shape}")

        # Check for NaN or extreme values
        if torch.isnan(output).any():
            print("⚠️  WARNING: NaN values detected in output")
        elif output.abs().max() > 1e3:
            print(f"⚠️  WARNING: Large values detected: max={output.abs().max():.2e}")
        else:
            print("✓ Output values look stable")

    except Exception as e:
        print(f"✗ Error: {e}")

    # Test gradient flow
    print(f"\n🌊 Testing gradient flow...")
    try:
        loss = output.mean()
        loss.backward()

        total_grad_norm = 0
        param_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm ** 2
                param_count += 1

        total_grad_norm = math.sqrt(total_grad_norm)
        print(f"✓ Total gradient norm: {total_grad_norm:.4f}")

        if total_grad_norm > 100:
            print("⚠️  WARNING: Large gradients detected - consider gradient clipping")
        elif total_grad_norm < 1e-6:
            print("⚠️  WARNING: Very small gradients - may indicate vanishing gradients")
        else:
            print("✓ Gradient magnitudes look healthy")

    except Exception as e:
        print(f"✗ Gradient test error: {e}")

    print(f"\n📊 Model info:")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print(f"\n💡 Stability improvements:")
    print(f"   • Reduced InhibitoryActivation sharpness: 10.0 → 2.0")
    print(f"   • Added learnable residual scaling")
    print(f"   • Smooth importance gating vs hard thresholding")
    print(f"   • Better weight initialization")
    print(f"   • Output projections for attention")
    print(f"   • Dropout for regularization")

    print(f"\n🎯 Training tips:")
    print(f"   • Use gradient clipping (max_norm=1.0)")
    print(f"   • Start with lower learning rate: {hyperparams['learning_rate']:.2e}")
    print(f"   • Use warmup: {hyperparams['warmup_steps']} steps")
    print(f"   • Monitor for oscillations with LossMonitor")


if __name__ == "__main__":
    stability()