import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange, repeat, reduce
import math
from typing import Optional, Tuple, List, Dict

from .fast_attention import FlashAttention


class YinYangActivation(nn.Module):
    """
    Yin-Yang Activation Function that balances bounded (Yin) and unbounded (Yang) characteristics.

    The activation dynamically adjusts between bounded functions like sigmoid (Yin properties:
    stability, normalization, saturation) and unbounded functions like ReLU (Yang properties:
    expansiveness, expressivity, scale-preservation).

    Args:
        yin_function (str): Bounded activation function ('sigmoid', 'tanh', or 'softsign')
        yang_function (str): Unbounded activation function ('relu', 'elu', 'gelu', or 'mish')
        balance (float): Initial balance parameter between yin and yang (0.0 = pure yang, 1.0 = pure yin)
    """

    def __init__(self, yin_function='tanh', yang_function='relu',
                 balance=0.5):
        super(YinYangActivation, self).__init__()

        self.balance = nn.Parameter(torch.tensor(balance))

        # Set up Yin function (bounded)
        self.yin_function = yin_function

        # Set up Yang function (unbounded)
        self.yang_function = yang_function

    def forward(self, x):
        """Apply Yin-Yang balanced activation."""
        # Clamp balance to [0, 1] range
        balance = torch.sigmoid(self.balance)

        # Apply Yin (bounded) function
        if self.yin_function == 'sigmoid':
            yin_activation = torch.sigmoid(x)
        elif self.yin_function == 'tanh':
            yin_activation = torch.tanh(x)
        elif self.yin_function == 'softsign':
            yin_activation = F.softsign(x)
        else:
            yin_activation = torch.sigmoid(x)

        # Apply Yang (unbounded) function
        if self.yang_function == 'relu':
            yang_activation = F.relu(x)
        elif self.yang_function == 'elu':
            yang_activation = F.elu(x)
        elif self.yang_function == 'gelu':
            yang_activation = F.gelu(x)
        elif self.yang_function == 'mish':
            yang_activation = x * torch.tanh(F.softplus(x))
        else:
            yang_activation = F.relu(x)

        # Weighted combination of Yin and Yang
        output = balance * yin_activation + (1 - balance) * yang_activation

        return output

class GrandUnityCore(nn.Module):
    """
    太一中樞 - The Grand Unity Central Pivot
    Maintains global coherence while coordinating all transformations
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Four aspects of Grand Unity
        self.stillness_gate = nn.Linear(dim, dim)  # 太一之靜
        self.stillness_activation = YinYangActivation(yin_function='tanh')

        self.movement_gate = nn.Linear(dim, dim) # 太一之動
        self.movement_activation = YinYangActivation(yin_function='sigmoid')

        self.emptiness_gate = nn.Linear(dim, dim) # 太一之虛
        self.emptiness_activation = YinYangActivation()

        self.substance_gate = nn.Linear(dim, dim)  # 太一之實
        # Central coordinator
        self.unity_transform = nn.Linear(dim * 4, dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply four aspects
        stillness = self.stillness_activation(self.stillness_gate(x))
        movement = self.movement_activation(self.movement_gate(x))
        emptiness =self.emptiness_activation(self.emptiness_gate(x))
        substance = self.substance_gate(x)

        # Integrate all aspects
        unified = torch.cat([stillness, movement, emptiness, substance], dim=-1)
        core_state = self.unity_transform(unified)

        return self.layer_norm(core_state + x)


class FourSpiritsProcessor(nn.Module):
    """
    四神方位 - Four Spirits Directional Positions
    Processes information according to four fundamental approaches
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Four Spirit transformations
        self.azure_dragon = nn.Linear(dim, dim)  # East - Generation
        self.vermilion_bird = nn.Linear(dim, dim)  # South - Transformation
        self.white_tiger = nn.Linear(dim, dim)  # West - Contraction
        self.black_turtle = nn.Linear(dim, dim)  # North - Storage

        # Spirit attention mechanism - operates on sequence dimension
        self.spirit_attention = FlashAttention(dim, num_heads)

        # Cross-spirit interaction
        self.spirit_mixer = nn.Linear(dim * 4, dim * 4)

        # Integration layer
        self.integrate = nn.Linear(dim * 4, dim)

        # Normalization
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape

        # Apply Four Spirit transformations
        east = F.gelu(self.azure_dragon(x))  # Generative, expanding
        south = F.relu(self.vermilion_bird(x))  # Transformative, active
        west = torch.tanh(self.white_tiger(x))  # Precise, contracting
        north = F.silu(self.black_turtle(x))  # Deep, accumulating

        # Each spirit processes sequence with attention
        spirit_outputs = []
        for spirit in [east, south, west, north]:
            # Apply attention to each spirit's sequence representation
            attn_out = self.spirit_attention(spirit)
            spirit_outputs.append(attn_out)

        # Stack and mix spirits
        spirits_combined = torch.cat(spirit_outputs, dim=-1)  # [b, s, d*4]

        # Cross-spirit interaction
        spirits_mixed = self.spirit_mixer(spirits_combined)

        # Final integration
        output = self.integrate(spirits_mixed)

        return self.norm(output + x)


class TenEssencesFlow(nn.Module):
    """
    十精運行 - Ten Essences Operation
    Dynamic transformation principles based on celestial stems
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Five phases with yin-yang aspects (10 essences)
        self.wood_yang = nn.Linear(dim, dim)  # 甲 - Vigorous Wood
        self.wood_yin = nn.Linear(dim, dim)  # 乙 - Flexible Wood
        self.fire_yang = nn.Linear(dim, dim)  # 丙 - Explosive Fire
        self.fire_yin = nn.Linear(dim, dim)  # 丁 - Gentle Fire
        self.earth_yang = nn.Linear(dim, dim)  # 戊 - Heavy Earth
        self.earth_yin = nn.Linear(dim, dim)  # 己 - Soft Earth
        self.metal_yang = nn.Linear(dim, dim)  # 庚 - Hard Metal
        self.metal_yin = nn.Linear(dim, dim)  # 辛 - Flexible Metal
        self.water_yang = nn.Linear(dim, dim)  # 壬 - Dynamic Water
        self.water_yin = nn.Linear(dim, dim)  # 癸 - Static Water

        # Phase interaction network
        self.phase_interaction = nn.Linear(dim * 2, dim)

        # Essence selection mechanism
        self.essence_selector = nn.Linear(dim, 10)

    def forward(self, x: torch.Tensor, phase_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Apply all ten essence transformations
        essences = [
            self.wood_yang(x), self.wood_yin(x),
            self.fire_yang(x), self.fire_yin(x),
            self.earth_yang(x), self.earth_yin(x),
            self.metal_yang(x), self.metal_yin(x),
            self.water_yang(x), self.water_yin(x)
        ]

        # Stack essences
        essence_stack = torch.stack(essences, dim=-2)  # [b, s, 10, d]

        # Compute essence weights (which essence is most active)
        weights = F.softmax(self.essence_selector(x), dim=-1)
        weights = rearrange(weights, 'b s e -> b s e 1')

        # Weighted combination of essences
        active_essence = torch.sum(essence_stack * weights, dim=-2)

        # Apply phase interactions (generating and overcoming cycles)
        if phase_state is not None:
            interaction = torch.cat([active_essence, phase_state], dim=-1)
            active_essence = self.phase_interaction(interaction)

        return active_essence


class MechanismNode(nn.Module):
    """
    機要節點 - Individual Mechanism Node
    Represents one of the 64 specific strategic configurations
    """

    def __init__(self, dim: int, mechanism_id: int):
        super().__init__()
        self.dim = dim
        self.mechanism_id = mechanism_id

        # Mechanism-specific transformation
        self.transform = nn.Linear(dim, dim)

        # Timing gate (when to activate)
        self.timing_gate = nn.Linear(dim, 1)

        # Mechanism characteristics embedding
        self.characteristics = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        transformed = self.transform(x)
        transformed = transformed + self.characteristics
        timing_score = torch.sigmoid(self.timing_gate(x))
        output = transformed * timing_score
        return output, self.mechanism_id


class MechanismGrabber(nn.Module):
    """
    握機模塊 - Mechanism Grabbing Module
    Identifies and activates the most appropriate mechanism nodes
    """

    def __init__(self, dim: int, num_mechanisms: int = 64):
        super().__init__()
        self.dim = dim
        self.num_mechanisms = num_mechanisms

        # Create 64 mechanism nodes
        self.mechanisms = nn.ModuleList([
            MechanismNode(dim, i) for i in range(num_mechanisms)
        ])

        # Mechanism selector network
        self.selector = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, num_mechanisms)
        )

        # Integration layer
        self.integrate = nn.Linear(dim * 2, dim)

    def freeze_all_mechanisms(self):
        for mech in self.mechanisms:
            for p in mech.parameters():
                p.requires_grad = False

    def unfreeze_mechanisms(self, indices: List[int]):
        for idx in indices:
            for p in self.mechanisms[idx].parameters():
                p.requires_grad = True

    def unfreeze_router(self):
        for p in self.selector.parameters():
            p.requires_grad = True
        for p in self.integrate.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, s, d = x.shape

        # Compute mechanism selection scores
        selection_scores = F.softmax(self.selector(context), dim=-1)  # [b, s, 64]

        # Apply all mechanisms and collect outputs
        mechanism_outputs = []

        for i, mechanism in enumerate(self.mechanisms):
            output, _ = mechanism(x)
            mechanism_outputs.append(output)

        # Stack outputs
        outputs_stacked = torch.stack(mechanism_outputs, dim=2)  # [b, s, 64, d]

        # Apply selection weights
        selection_weights = rearrange(selection_scores, 'b s m -> b s m 1')
        selected_output = torch.sum(outputs_stacked * selection_weights, dim=2)

        # Integrate with original input
        integrated = self.integrate(torch.cat([x, selected_output], dim=-1))

        return integrated


class MechanismGrabberTopK(nn.Module):
    """
    握機模塊 - Mechanism Grabbing Module with top k only selection
    Identifies and activates the most appropriate mechanism nodes
    """

    def __init__(self, dim: int, num_mechanisms: int = 64, top_k: int = 4):
        super().__init__()
        self.dim = dim
        self.num_mechanisms = num_mechanisms
        self.top_k = top_k
        # Create 64 mechanism nodes
        self.mechanisms = nn.ModuleList([
            MechanismNode(dim, i) for i in range(num_mechanisms)
        ])

        # Mechanism selector network
        self.selector = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, num_mechanisms)
        )

        # Integration layer
        self.integrate = nn.Linear(dim * 2, dim)

    def freeze_all_mechanisms(self):
        for mech in self.mechanisms:
            for p in mech.parameters():
                p.requires_grad = False

    def unfreeze_mechanisms(self, indices: List[int]):
        for idx in indices:
            for p in self.mechanisms[idx].parameters():
                p.requires_grad = True

    def unfreeze_router(self):
        for p in self.selector.parameters():
            p.requires_grad = True
        for p in self.integrate.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, s, d = x.shape

        # Compute mechanism selection scores
        selection_scores = F.softmax(self.selector(context), dim=-1)  # [b, s, 64]

        top_values, top_indices = torch.topk(selection_scores, k=self.top_k, dim=-1)  # [b, s, k]
        mask = torch.zeros_like(selection_scores).scatter_(-1, top_indices, 1.0)
        selection_scores = selection_scores * mask
        selection_scores = selection_scores / selection_scores.sum(dim=-1, keepdim=True)

        # Apply all mechanisms and collect outputs
        mechanism_outputs = []

        for mechanism in top_indices[0].tolist():
            mechanism_output, mechanism_id = mechanism(x)
            mechanism_outputs.append(mechanism_output)

        # Stack outputs
        outputs_stacked = torch.stack(mechanism_outputs, dim=2)  # [b, s, top_k, d]

        # Apply selection weights
        selection_weights = rearrange(selection_scores, 'b s m -> b s m 1')
        selected_output = torch.sum(outputs_stacked * selection_weights, dim=2)

        # Integrate with original input
        integrated = self.integrate(torch.cat([x, selected_output], dim=-1))

        return integrated

class InfiniteTransformation(nn.Module):
    """
    無窮變化 - Infinite Transformation Layer
    Represents the seventh layer of continuous flux and adaptation
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # Five principles of transformation
        self.nested_mechanism = nn.LSTM(dim, dim, batch_first=True)
        self.nested_momentum = nn.GRU(dim, dim, batch_first=True)
        self.empty_full_gate = nn.Linear(dim * 2, dim)
        self.creation_gate = nn.Linear(dim, dim)
        self.dissolution_gate = nn.Linear(dim, dim)

        # Meta-learning component for adaptation
        self.meta_transform = nn.Linear(dim, dim)
        self.adaptation_rate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply nested transformations
        nested_out1, _ = self.nested_mechanism(x)
        nested_out2, _ = self.nested_momentum(x)

        # Empty-full mutual transformation
        empty_full = self.empty_full_gate(torch.cat([nested_out1, nested_out2], dim=-1))

        # Creation from nothing
        created = torch.relu(self.creation_gate(torch.zeros_like(x))) * empty_full

        # Dissolution to nothing
        dissolved = x * torch.sigmoid(self.dissolution_gate(x))

        # Meta-transformation for continuous adaptation
        meta_adapted = self.meta_transform(x) * self.adaptation_rate

        # Combine all transformations
        output = x + created - dissolved + meta_adapted

        return output

class MechanismNet(nn.Module):
    """
    Complete MechanismNet Architecture
    七層結構 - Seven Layer Structure implementing the complete strategic system
    """

    def __init__(
            self,
            hidden_dim: int = 512,
            num_layers: int = 2,
            num_heads: int = 8,
            num_mechanisms: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Layer 1: Grand Unity Central Pivot
        self.grand_unity = GrandUnityCore(hidden_dim)

        self.four_spirits = nn.ModuleList([])
        self.ten_essences = nn.ModuleList([])
        self.positional_branches = nn.ParameterList([])
        self.layer_norms = nn.ModuleList([])

        for idx in range(num_layers):
            self.four_spirits.append(FourSpiritsProcessor(hidden_dim, num_heads))
            self.ten_essences.append(TenEssencesFlow(hidden_dim))
            self.positional_branches.append(nn.Parameter(torch.randn(1, 2048, hidden_dim)))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Layer 5: Mechanism Nodes
        self.mechanism_grabber = MechanismGrabber(hidden_dim, num_mechanisms)
        self.layer_norms.append(nn.LayerNorm(hidden_dim))
        # Layer 6: Momentum Positions (dynamic routing)
        self.momentum_router = FlashAttention(hidden_dim, num_heads)
        self.layer_norms.append(nn.LayerNorm(hidden_dim))
        # Layer 7: Infinite Transformations
        self.infinite_transform = InfiniteTransformation(hidden_dim)
        self.layer_norms.append(nn.LayerNorm(hidden_dim))



    def forward(
            self,
            x: torch.Tensor,
            return_victory_scores: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through all seven layers

        Args:
            x: Input tensor [batch, sequence, features]
            return_victory_scores: Whether to return mechanism victory scores

        Returns:
            Output tensor and optionally victory scores

        """
        b, s, f = x.shape
        layer_norm_index = 0
        # Layer 1: Establish Grand Unity
        unity_state = self.grand_unity(x)

        for idx in range(self.num_layers):
            # Layer 2: Four Spirits Processing
            x = self.four_spirits[idx](x)

            # Layer 3: Ten Essences Flow
            x = self.ten_essences[idx](x, unity_state)

            # Layer 4: Add positional branches
            positions = self.positional_branches[idx][:, :s, :]
            x = self.layer_norms[layer_norm_index](x + positions)
            layer_norm_index += 1

        # Layer 5: Mechanism Grabbing
        mechanism_output = self.mechanism_grabber(x, unity_state)
        x = self.layer_norms[layer_norm_index](x + mechanism_output)
        layer_norm_index += 1

        # Layer 6: Momentum Routing
        momentum_output = self.momentum_router(x)
        x = self.layer_norms[layer_norm_index](x + momentum_output)
        layer_norm_index += 1

        # Layer 7: Infinite Transformations
        final_output = self.infinite_transform(x)
        output = self.layer_norms[layer_norm_index](x + final_output)
        layer_norm_index += 1

        return output


class MechanismNetAblation(nn.Module):
    """
    Complete MechanismNet Architecture with ablation support
    七層結構 - Seven Layer Structure implementing the complete strategic system
    """

    def __init__(
            self,
            hidden_dim: int = 512,
            num_layers: int = 2,
            num_heads: int = 8,
            num_mechanisms: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Layer 1: Grand Unity Central Pivot
        self.grand_unity = GrandUnityCore(hidden_dim)

        self.four_spirits = nn.ModuleList([])
        self.ten_essences = nn.ModuleList([])
        self.positional_branches = nn.ParameterList([])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim)])

        for idx in range(num_layers):
            self.four_spirits.append(FourSpiritsProcessor(hidden_dim, num_heads))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.ten_essences.append(TenEssencesFlow(hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.positional_branches.append(nn.Parameter(torch.randn(1, 2048, hidden_dim)))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Layer 5: Mechanism Nodes
        self.mechanism_grabber = MechanismGrabber(hidden_dim, num_mechanisms)
        self.layer_norms.append(nn.LayerNorm(hidden_dim))
        # Layer 6: Momentum Positions (dynamic routing)
        self.momentum_router = FlashAttention(hidden_dim, num_heads)
        self.layer_norms.append(nn.LayerNorm(hidden_dim))
        # Layer 7: Infinite Transformations
        self.infinite_transform = InfiniteTransformation(hidden_dim)
        self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Ablation flags - all enabled by default
        self.ablation_config = {
            'grand_unity': False,
            'four_spirits': [True] * num_layers,
            'ten_essences': [True] * num_layers,
            'positional_branches': [True] * num_layers,
            'mechanism_grabber': False,
            'momentum_router': False,
            'infinite_transform': False,
            'layer_norms': True,  # Global flag for all layer norms
            'residual_connections': True  # Global flag for all residuals
        }

    def set_ablation(self, component: str, enabled: bool, layer_idx: int = None):
        """
        Enable or disable specific components for ablation studies.

        Args:
            component: Name of component to ablate
            enabled: Whether to enable (True) or disable (False)
            layer_idx: For layered components, specify which layer (None for all)
        """
        if component in ['four_spirits', 'ten_essences', 'positional_branches']:
            if layer_idx is None:
                # Set all layers
                self.ablation_config[component] = [enabled] * self.num_layers
            else:
                # Set specific layer
                self.ablation_config[component][layer_idx] = enabled
        else:
            self.ablation_config[component] = enabled

    def forward(
            self,
            x: torch.Tensor,
            return_victory_scores: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through all seven layers with ablation support

        Args:
            x: Input tensor [batch, sequence, features]
            return_victory_scores: Whether to return mechanism victory scores

        Returns:
            Output tensor and optionally victory scores
        """
        b, s, f = x.shape
        layer_norm_index = 0

        # Layer 1: Establish Grand Unity
        if self.ablation_config['grand_unity']:
            unity_state = self.grand_unity(x)
            x = self._apply_residual_and_norm(x, unity_state, layer_norm_index)
        else:
            unity_state = torch.zeros_like(x)  # Placeholder for downstream layers
            x = self._apply_norm_only(x, layer_norm_index)
        layer_norm_index += 1

        for idx in range(self.num_layers):
            # Layer 2: Four Spirits Processing
            if self.ablation_config['four_spirits'][idx]:
                spirit_output = self.four_spirits[idx](x)
                x = self._apply_residual_and_norm(x, spirit_output, layer_norm_index)
            else:
                x = self._apply_norm_only(x, layer_norm_index)
            layer_norm_index += 1

            # Layer 3: Ten Essences Flow
            if self.ablation_config['ten_essences'][idx]:
                essence_output = self.ten_essences[idx](x, unity_state)
                x = self._apply_residual_and_norm(x, essence_output, layer_norm_index)
            else:
                x = self._apply_norm_only(x, layer_norm_index)
            layer_norm_index += 1

            # Layer 4: Add positional branches
            if self.ablation_config['positional_branches'][idx]:
                positions = self.positional_branches[idx][:, :s, :]
                x = self._apply_residual_and_norm(x, positions, layer_norm_index)
            else:
                x = self._apply_norm_only(x, layer_norm_index)
            layer_norm_index += 1

        # Layer 5: Mechanism Grabbing
        if self.ablation_config['mechanism_grabber']:
            mechanism_output = self.mechanism_grabber(x, unity_state)
            x = self._apply_residual_and_norm(x, mechanism_output, layer_norm_index)
        else:
            x = self._apply_norm_only(x, layer_norm_index)
        layer_norm_index += 1

        # Layer 6: Momentum Routing
        if self.ablation_config['momentum_router']:
            momentum_output = self.momentum_router(x)
            x = self._apply_residual_and_norm(x, momentum_output, layer_norm_index)
        else:
            x = self._apply_norm_only(x, layer_norm_index)
        layer_norm_index += 1

        # Layer 7: Infinite Transformations
        if self.ablation_config['infinite_transform']:
            final_output = self.infinite_transform(x)
            output = self._apply_residual_and_norm(x, final_output, layer_norm_index)
        else:
            output = self._apply_norm_only(x, layer_norm_index)

        return output

    def _apply_residual_and_norm(self, x, residual, norm_idx):
        """Apply residual connection and layer norm based on ablation config"""
        if self.ablation_config['residual_connections']:
            x = x + residual
        else:
            x = residual  # Skip residual, use only the transformation

        if self.ablation_config['layer_norms']:
            x = self.layer_norms[norm_idx](x)
        return x

    def _apply_norm_only(self, x, norm_idx):
        """Apply only layer norm when component is disabled"""
        if self.ablation_config['layer_norms']:
            x = self.layer_norms[norm_idx](x)
        return x

    def get_ablation_summary(self):
        """Get a readable summary of current ablation configuration"""
        summary = []
        for key, value in self.ablation_config.items():
            if isinstance(value, list):
                enabled = sum(value)
                total = len(value)
                summary.append(f"{key}: {enabled}/{total} enabled")
            else:
                status = "enabled" if value else "disabled"
                summary.append(f"{key}: {status}")
        return "\n".join(summary)