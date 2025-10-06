import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
import math
from typing import Optional, Tuple, List



from .fast_attention import FlashAttention


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
        self.movement_gate = nn.Linear(dim, dim)  # 太一之動
        self.emptiness_gate = nn.Linear(dim, dim)  # 太一之虛
        self.substance_gate = nn.Linear(dim, dim)  # 太一之實

        # Central coordinator
        self.unity_transform = nn.Linear(dim * 4, dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Apply four aspects
        stillness = torch.tanh(self.stillness_gate(x))
        movement = torch.sigmoid(self.movement_gate(x))
        emptiness = F.relu(self.emptiness_gate(x))
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

        # Spirit attention mechanism
        self.spirit_attention = FlashAttention(dim, num_heads)

        # Integration layer
        self.integrate = nn.Linear(dim * 4, dim)

        # Governor head for final state
        self.governor_head = nn.Linear(dim, dim * 2)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b, s, d = x.shape

        # Apply Four Spirit transformations
        east = F.gelu(self.azure_dragon(x))
        south = F.relu(self.vermilion_bird(x))
        west = torch.tanh(self.white_tiger(x))
        north = F.silu(self.black_turtle(x))

        # Process through attention
        spirits = torch.stack([east, south, west, north], dim=1)
        spirits_flat = rearrange(spirits, 'b n s d -> (b s) n d')
        attn_out= self.spirit_attention(spirits_flat)
        attn_out = rearrange(attn_out, '(b s) n d -> b n s d', b=b, s=s)

        # Integrate spirit outputs to get the main processed data stream
        combined = rearrange(attn_out, 'b n s d -> b s (n d)')
        processed_output = self.integrate(combined)

        # Generate governance gates from the processed output
        gates = self.governor_head(processed_output)

        # Split the gates for different modules and apply sigmoid to scale between 0 and 1
        essence_gate, mechanism_gate = torch.chunk(gates, 2, dim=-1)
        essence_gate = torch.sigmoid(essence_gate)
        mechanism_gate = torch.sigmoid(mechanism_gate)

        # Return both the processed data and the governance gates
        return processed_output, (essence_gate, mechanism_gate)


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

        # Victory point detector
        self.victory_detector = nn.Linear(dim, 1)

        # Mechanism characteristics embedding
        self.characteristics = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Apply mechanism-specific transformation
        transformed = self.transform(x)

        # Add mechanism characteristics
        transformed = transformed + self.characteristics

        # Compute timing activation
        timing_score = torch.sigmoid(self.timing_gate(x))

        # Detect victory points
        victory_score = torch.sigmoid(self.victory_detector(transformed))

        # Apply timing gate
        output = transformed * timing_score

        return output, victory_score


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

        self.overcoming_dynamics = StrategicOvercomingMatrix(num_mechanisms, use_celestial_init=True)

        # Mechanism selector network
        self.selector = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, num_mechanisms)
        )

        # Integration layer
        self.integrate = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, s, d = x.shape


        selector_input = torch.cat([x, context], dim=-1)
        selection_scores = F.softmax(self.selector(selector_input), dim=-1)  # [b, s, 64]


        # Apply all mechanisms and collect outputs
        mechanism_outputs = []
        victory_scores = []

        for i, mechanism in enumerate(self.mechanisms):
            output, victory = mechanism(x)
            mechanism_outputs.append(output)
            victory_scores.append(victory)

        # Stack outputs
        outputs_stacked = torch.stack(mechanism_outputs, dim=2)  # [b, s, 64, d]
        victories_stacked = torch.stack(victory_scores, dim=2)  # [b, s, 64, 1]

        modulated_outputs = self.overcoming_dynamics(outputs_stacked, selection_scores)

        # Apply selection weights to the *modulated* outputs
        selection_weights = rearrange(selection_scores, 'b s m -> b s m 1')
        selected_output = torch.sum(modulated_outputs * selection_weights, dim=2)

        # Compute overall victory potential (this remains based on raw scores)
        victory_potential = torch.sum(victories_stacked.squeeze(-1) * selection_scores, dim=-1)

        # Integrate with original input
        integrated = self.integrate(torch.cat([x, selected_output], dim=-1))

        return integrated, victory_potential


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

    def forward(self, x: torch.Tensor, hidden_state: Optional[Tuple] = None) -> torch.Tensor:
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class BatchedMechanismGrabber(nn.Module):
    """
    握機模塊 - Batched Mechanism Grabbing Module
    Efficient batched version of MechanismGrabber.
    - Applies all mechanism transforms in parallel.
    - Computes timing + victory scores with shared einsums.
    """

    def __init__(self, dim: int, num_mechanisms: int = 64):
        super().__init__()
        self.dim = dim
        self.num_mechanisms = num_mechanisms

        # Mechanism-specific transforms (like a bank of Linear layers)
        self.transform_weights = nn.Parameter(torch.randn(num_mechanisms, dim, dim))
        self.transform_bias = nn.Parameter(torch.zeros(num_mechanisms, dim))

        # Timing gates
        self.timing_gates = nn.Parameter(torch.randn(num_mechanisms, dim))
        self.timing_bias = nn.Parameter(torch.zeros(num_mechanisms))

        # Victory detectors
        self.victory_detectors = nn.Parameter(torch.randn(num_mechanisms, dim))
        self.victory_bias = nn.Parameter(torch.zeros(num_mechanisms))

        # Overcoming dynamics
        self.overcoming_dynamics = StrategicOvercomingMatrix(num_mechanisms, use_celestial_init=True)

        # Selector network (router)
        self.selector = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, num_mechanisms)
        )

        # Integration layer
        self.integrate = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        """
        Args:
            x: [b, s, d]
            context: [b, s, d]
        Returns:
            integrated: [b, s, d]
            victory_potential: [b, s]
        """
        b, s, d = x.shape

        # === Mechanism selection ===
        selector_input = torch.cat([x, context], dim=-1)  # [b, s, 2d]
        selection_scores = F.softmax(self.selector(selector_input), dim=-1)  # [b, s, m]

        # === Expand x for mechanism transforms ===
        x_expanded = x.unsqueeze(2)  # [b, s, 1, d]

        # === Apply mechanism-specific transforms ===
        # einsum: [b, s, 1, d] x [m, d, d] -> [b, s, m, d]
        transformed = torch.einsum("bsnd,mde->bsme", x_expanded, self.transform_weights)
        transformed = transformed + self.transform_bias.unsqueeze(0).unsqueeze(0)  # add bias

        # === Timing gate ===
        # einsum: [b, s, 1, d] x [m, d] -> [b, s, m]
        timing_scores = torch.einsum("bsnd,md->bsm", x_expanded, self.timing_gates)
        timing_scores = torch.sigmoid(timing_scores + self.timing_bias)  # [b, s, m]

        # === Victory detector ===
        victory_scores = torch.einsum("bsmd,md->bsm", transformed, self.victory_detectors)
        victory_scores = torch.sigmoid(victory_scores + self.victory_bias)  # [b, s, m]

        # Apply timing gate
        outputs_stacked = transformed * timing_scores.unsqueeze(-1)  # [b, s, m, d]

        # === Overcoming dynamics ===
        modulated_outputs = self.overcoming_dynamics(outputs_stacked, selection_scores)

        # === Weighted combination of mechanism outputs ===
        selection_weights = selection_scores.unsqueeze(-1)  # [b, s, m, 1]
        selected_output = torch.sum(modulated_outputs * selection_weights, dim=2)  # [b, s, d]

        # === Victory potential (weighted sum of victories) ===
        victory_potential = torch.sum(victory_scores * selection_scores, dim=-1)  # [b, s]

        # === Integrate ===
        integrated = self.integrate(torch.cat([x, selected_output], dim=-1))  # [b, s, d]

        return integrated, victory_potential



class MechanismNet(nn.Module):
    """
    Complete MechanismNet Architecture
    七層結構 - Seven Layer Structure implementing the complete strategic system
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 512,
            num_heads: int = 8,
            num_layers: int = 2,
            num_mechanisms: int = 64,
            output_dim: Optional[int] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = 7
        self.output_dim = output_dim or input_dim

        if input_dim != hidden_dim:
            # Input projection
            self.input_projection = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_projection = None
        # Layer 1: Grand Unity Central Pivot
        self.grand_unity = GrandUnityCore(hidden_dim)

        # Layer 2: Four Spirits Directional Positions
        self.four_spirits = FourSpiritsProcessor(hidden_dim, num_heads)

        # Layer 3: Ten Essences Operation
        self.ten_essences = TenEssencesFlow(hidden_dim)

        self.celestial_cycles = CelestialCycleEncoding(hidden_dim)

        # Layer 5: Mechanism Nodes
        self.mechanism_grabber = BatchedMechanismGrabber(hidden_dim, num_mechanisms)

        # Layer 6: Momentum Positions (dynamic routing)
        self.momentum_router = FlashAttention(hidden_dim, num_heads)

        # Layer 7: Infinite Transformations
        self.infinite_transform = InfiniteTransformation(hidden_dim)

        if output_dim != hidden_dim:
            # Output projection
            self.output_projection = nn.Linear(hidden_dim, self.output_dim)
        else:
            self.output_projection = None
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(7)
        ])

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            return_victory_scores: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through all seven layers

        Args:
            x: Input tensor [batch, sequence, features]
            mask: Optional attention mask
            return_victory_scores: Whether to return mechanism victory scores

        Returns:
            Output tensor and optionally victory scores
        """
        b, s, f = x.shape

        if self.hidden_dim != self.input_dim:
            # Input projection
            x = self.input_projection(x)

        # Layer 1: Establish Grand Unity
        unity_state = self.grand_unity(x)
        x = self.layer_norms[0](x + unity_state)

        # Layer 2: Four Spirits now returns gates
        spirit_output, (essence_gate, mechanism_gate) = self.four_spirits(x)
        x = self.layer_norms[1](x + spirit_output)

        # Layer 3: Ten Essences, now governed by a spirit gate
        essence_output = self.ten_essences(x, phase_state=unity_state)
        # Apply the gate: if the dominant spirit is the Black Turtle (defensive),
        # it might learn to suppress the output of the generative essences.
        x = self.layer_norms[2](x + (essence_output * essence_gate))

        # Layer 4: Positional Branches
        x = self.celestial_cycles(x)
        x = self.layer_norms[3](x)  # Apply LayerNorm after the positional addition

        # Layer 5: Mechanism Grabbing, also governed by a spirit gate
        mechanism_output, victory_scores = self.mechanism_grabber(x, context=unity_state)
        # Apply the gate: if the dominant spirit is the Vermilion Bird (swift attack),
        # it might learn to amplify the output of aggressive mechanisms.
        x = self.layer_norms[4](x + (mechanism_output * mechanism_gate))

        # Layer 6: Momentum Routing
        momentum_output = self.momentum_router(x)
        x = self.layer_norms[5](x + momentum_output)

        # Layer 7: Infinite Transformations
        final_output = self.infinite_transform(x)
        x = self.layer_norms[6](x + final_output)

        if self.hidden_dim != self.output_dim:
            # Output projection
            output = self.output_projection(x)
        else:
            output = x
        if return_victory_scores:
            return output, victory_scores
        return output

class StrategicOvercomingMatrix(nn.Module):
    """
    勝負機理圖 - Chart of Victory-Defeat Mechanisms.
    This module learns the dynamic relationships of support (相生) and
    suppression (相克) between all strategic mechanisms.
    """
    def __init__(self, num_mechanisms: int, use_celestial_init: bool = True):
        super().__init__()
        self.num_mechanisms = num_mechanisms

        # This matrix represents the learned influence of each mechanism upon every other.
        # A positive value in overcoming_matrix[i, j] means mechanism i supports mechanism j.
        # A negative value means mechanism i suppresses mechanism j.
        self.overcoming_matrix = nn.Parameter(torch.randn(num_mechanisms, num_mechanisms))
        if use_celestial_init:
            # Initialize the parameter with the wisdom of the Five Phases
            initial_weights = initialize_overcoming_matrix(num_mechanisms)
            self.overcoming_matrix.data = initial_weights
        else:
            # Default random initialization if celestial wisdom is not used
            nn.init.normal_(self.overcoming_matrix, mean=0, std=0.02)


    def forward(self, mechanism_outputs: torch.Tensor, selection_scores: torch.Tensor) -> torch.Tensor:
        """
        Modulates mechanism outputs based on their interactions.

        Args:
            mechanism_outputs (torch.Tensor): Raw outputs from all MechanismNodes.
                                              Shape: [b, s, num_mechanisms, d]
            selection_scores (torch.Tensor): The activation probability for each mechanism.
                                             Shape: [b, s, num_mechanisms]
        Returns:
            torch.Tensor: The modulated mechanism outputs. Shape: [b, s, num_mechanisms, d]
        """
        # The `selection_scores` tell us which mechanisms are currently active.
        # We use these active mechanisms to calculate a net influence on every other mechanism.

        # Reshape scores for matrix multiplication: [b, s, 1, num_mechanisms]
        active_influencers = selection_scores.unsqueeze(2)

        # Calculate the net influence received by each mechanism.
        # This is the core of the "overcoming and generating" cycle.
        # [b, s, 1, num_mechanisms] @ [num_mechanisms, num_mechanisms] -> [b, s, 1, num_mechanisms]
        net_influence = torch.matmul(active_influencers, self.overcoming_matrix)

        # Squeeze and apply a non-linearity (tanh) to keep the influence factors bounded.
        # This prevents runaway amplification or suppression.
        gating_factors = torch.tanh(net_influence.squeeze(2)) # Shape: [b, s, num_mechanisms]

        # The gating factors now represent the net support/suppression for each mechanism.
        # We add 1 to shift the modulation range from [-1, 1] to [0, 2].
        # 0 means fully suppressed, 1 means no change, 2 means fully supported.
        modulation_gates = (1 + gating_factors).unsqueeze(-1) # Shape: [b, s, num_mechanisms, 1]

        # Apply the gates to modulate the original mechanism outputs.
        modulated_outputs = mechanism_outputs * modulation_gates

        return modulated_outputs


import torch


def initialize_overcoming_matrix(num_mechanisms: int) -> torch.Tensor:
    """
    Initializes the Strategic Overcoming Matrix based on the principles of the Five Phases.
    """
    # Define the 64 mechanisms in order. This MUST match the order in MechanismGrabber.
    # I have inferred this order from your provided charts.
    mechanism_names = [
        '甲子', '甲寅', '甲辰', '甲午', '甲申', '甲戌',  # Wood A (6)
        '乙丑', '乙卯', '乙巳', '乙未', '乙酉', '乙亥',  # Wood B (6)
        '丙子', '丙寅', '丙辰', '丙午', '丙申', '丙戌',  # Fire A (6)
        '丁丑', '丁卯', '丁巳', '丁未', '丁酉', '丁亥',  # Fire B (6)
        '戊子', '戊寅', '戊辰', '戊午', '戊申', '戊戌',  # Earth A (6)
        '己丑', '己卯', '己巳', '己未', '己酉', '己亥',  # Earth B (6)
        '庚子', '庚寅', '庚辰', '庚午', '庚申', '庚戌',  # Metal A (6)
        '辛丑', '辛卯', '辛巳', '辛未', '辛酉', '辛亥',  # Metal B (6)
        '壬子', '壬寅', '壬辰', '壬午', '壬申', '壬戌',  # Water A (6)
        '癸丑', '癸卯', '癸巳', '癸未', '癸酉', '癸亥',  # Water B (6)
        '虛無', '混沌', '無極', '歸元'  # Special (4) -> Total 64
    ]

    # 1. Map Mechanisms to Phases (0:Wood, 1:Fire, 2:Earth, 3:Metal, 4:Water)
    phase_map = torch.zeros(num_mechanisms, dtype=torch.long)
    stem_to_phase = {'甲': 0, '乙': 0, '丙': 1, '丁': 1, '戊': 2, '己': 2,
                     '庚': 3, '辛': 3, '壬': 4, '癸': 4}
    for i, name in enumerate(mechanism_names):
        if i >= num_mechanisms:
            break
        stem = name[0]
        if stem in stem_to_phase:
            phase_map[i] = stem_to_phase[stem]
        else:  # Special mechanisms are assigned to Earth (center/balance)
            phase_map[i] = 2

    # 2. Define Interaction Weights
    weights = {
        'self': 0.1,
        'same_phase': 0.4,
        'sheng': 0.8,  # Generation
        'child_drains': -0.2,  # Child drains Mother
        'ke': -1.0,  # Conquest
        'wu': -0.3,  # Insult (Reverse Conquest)
    }

    # 3. Construct the Matrix
    matrix = torch.zeros(num_mechanisms, num_mechanisms)

    # Define phase cycles
    sheng_cycle = {0: 1, 1: 2, 2: 3, 3: 4, 4: 0}  # Wood->Fire, etc.
    ke_cycle = {0: 2, 1: 3, 2: 4, 3: 0,
                4: 1}  # Wood->Earth, etc. (Note: My charts use a different Ke cycle, let me correct it)
    # Correct Ke Cycle from the chart: 木克土, 土克水, 水克火, 火克金, 金克木
    ke_cycle = {0: 2, 2: 4, 4: 1, 1: 3, 3: 0}  # Wood->Earth, Earth->Water, etc.

    for i in range(num_mechanisms):
        for j in range(num_mechanisms):
            phase_i = phase_map[i].item()
            phase_j = phase_map[j].item()

            if i == j:
                matrix[i, j] = weights['self']
            elif phase_i == phase_j:
                matrix[i, j] = weights['same_phase']
            else:
                # Sheng (Generation) Cycle
                if sheng_cycle.get(phase_i) == phase_j:
                    matrix[i, j] += weights['sheng']  # Mother generates Child
                    matrix[j, i] += weights['child_drains']  # Child drains Mother

                # Ke (Conquest) Cycle
                if ke_cycle.get(phase_i) == phase_j:
                    matrix[i, j] += weights['ke']  # Conqueror suppresses Conquered
                    matrix[j, i] += weights['wu']  # Conquered insults Conqueror

    return matrix


class CelestialCycleEncoding(nn.Module):
    """
    天輪地支位 - Celestial Cycle Positional Encoding
    Replaces the static positional branches with a dynamic, infinitely scalable
    encoding based on the cycles of the 12 Terrestrial Branches.
    """

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Dimension must be even for Yin-Yang wheel concatenation.")

        half_dim = dim // 2

        # The 12 stations for each wheel, corresponding to the Terrestrial Branches.
        # Each wheel controls half of the total dimension.
        self.yang_wheel = nn.Embedding(12, half_dim)  # 陽輪
        self.yin_wheel = nn.Embedding(12, half_dim)  # 陰輪

        # The Grand Cycle encoding, to distinguish between cycles (e.g., pos 3 vs pos 15)
        # This is the standard sinusoidal encoding from "Attention Is All You Need".
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('grand_cycle_pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, sequence, features]
        """
        b, s, d = x.shape

        # Create position indices for the sequence: [0, 1, 2, ..., s-1]
        positions = torch.arange(s, device=x.device)

        # Calculate indices for the celestial wheels
        yang_indices = positions % 12
        yin_indices = (positions + 6) % 12  # Phase-shifted by 6 stations (opposition)

        # Look up the embeddings for each wheel
        yang_embedding = self.yang_wheel(yang_indices)  # [s, half_dim]
        yin_embedding = self.yin_wheel(yin_indices)  # [s, half_dim]

        # Concatenate the Yin and Yang embeddings to form the full cyclical encoding
        cyclical_encoding = torch.cat([yang_embedding, yin_embedding], dim=-1)  # [s, dim]

        # Get the Grand Cycle encoding for the current sequence length
        grand_cycle_encoding = self.grand_cycle_pe[:s, :]  # [s, dim]

        # The final positional signal is the sum of the cyclical texture and the grand cycle context.
        # We add this to the input tensor `x`.
        # The unsqueeze(0) broadcasts the positional signal across the batch dimension.
        positional_signal = cyclical_encoding + grand_cycle_encoding
        return x + positional_signal.unsqueeze(0)


# Specialized variants for different strategic scenarios

class TemporalMechanismNet(MechanismNet):
    """
    Variant optimized for time-series and sequential decision making
    Emphasizes the flow of Ten Essences and momentum tracking
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add temporal tracking components
        self.temporal_memory = nn.LSTM(self.hidden_dim, self.hidden_dim,
                                       num_layers=3, batch_first=True)
        self.phase_tracker = nn.Linear(self.hidden_dim, 5)  # Track five phases

    def forward(self, x, mask=None, hidden=None, return_victory_scores=False):
        # Apply temporal memory
        if hidden is None:
            x_temporal, hidden = self.temporal_memory(x)
        else:
            x_temporal, hidden = self.temporal_memory(x, hidden)

        # Track phase states
        phase_states = F.softmax(self.phase_tracker(x_temporal), dim=-1)

        # Continue with standard forward pass
        output = super().forward(x_temporal, mask, return_victory_scores)

        if return_victory_scores:
            output, scores = output
            return output, scores, hidden, phase_states
        return output, hidden, phase_states


class StrategicTransformer(nn.Module):
    """
    Transformer variant using MechanismNet principles
    Replaces standard attention with mechanism-grabbing
    """

    def __init__(
            self,
            dim: int,
            depth: int = 12,
            heads: int = 8,
            mlp_dim: int = 2048,
            num_mechanisms: int = 64
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MechanismGrabber(dim, num_mechanisms),
                FourSpiritsProcessor(dim, heads),
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Linear(mlp_dim, dim)
                )
            ]))

        self.grand_unity = GrandUnityCore(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        # Establish Grand Unity state
        unity_state = self.grand_unity(x)

        for grabber, spirits, ff in self.layers:
            # Mechanism attention
            x_grab, _ = grabber(x, unity_state)
            x = x + x_grab

            # Four spirits processing
            x = x + spirits(x, mask)

            # Feedforward
            x = x + ff(x)

        return self.norm(x)


# Utility functions for working with MechanismNet

def create_mechanism_mask(seq_len: int, mechanism_pattern: str = "natural") -> torch.Tensor:
    """
    Create attention masks based on strategic patterns

    Args:
        seq_len: Sequence length
        mechanism_pattern: One of "natural", "spiral", "explosive", "contractive"
    """
    mask = torch.ones(seq_len, seq_len)

    if mechanism_pattern == "natural":
        # Natural flow - each position attends to previous positions
        mask = torch.tril(mask)
    elif mechanism_pattern == "spiral":
        # Spiral pattern - increasing radius of attention
        for i in range(seq_len):
            radius = int(math.sqrt(i + 1) * 2)
            start = max(0, i - radius)
            end = min(seq_len, i + radius + 1)
            mask[i, start:end] = 1
    elif mechanism_pattern == "explosive":
        # Explosive pattern - sudden expansion of attention
        threshold = seq_len // 2
        mask[:threshold] = torch.tril(mask[:threshold])
        mask[threshold:] = 1
    elif mechanism_pattern == "contractive":
        # Contractive pattern - narrowing attention
        for i in range(seq_len):
            width = max(1, seq_len - i)
            start = max(0, i - width // 2)
            end = min(seq_len, i + width // 2 + 1)
            mask[i, start:end] = 1

    return mask


def apply_five_phases_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        phase_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Apply loss calculation based on five phases principle
    Different phases emphasize different aspects of the loss
    """
    base_loss = F.mse_loss(predictions, targets, reduction='none')

    if phase_weights is None:
        # Default five phases weights
        phase_weights = torch.tensor([
            1.0,  # Wood - growth, allow more variation
            0.7,  # Fire - transformation, moderate strictness
            1.2,  # Earth - stability, higher penalty for deviation
            1.5,  # Metal - precision, highest penalty
            0.8  # Water - flow, adaptive penalty
        ])

    # Apply phase-based weighting
    weighted_loss = base_loss * phase_weights.view(1, 1, -1)

    return weighted_loss.mean()