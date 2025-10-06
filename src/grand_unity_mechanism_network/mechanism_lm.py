import numpy as np
import torch
import torch.nn as nn
import math
import random
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from .mechanism_network import MechanismNet, MechanismNetAblation


class LMWithMechanismNet(nn.Module):
    def __init__(self, vocab_size,  d_model, num_layers=3, num_heads=8, num_mechanisms=64, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.mechanism_net = MechanismNetAblation(
            #input_dim=d_model,
            hidden_dim=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            num_mechanisms=num_mechanisms
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = d_model

    def forward(self, src_tokens):  # src_mask for attention inside MechanismNet

        x = self.embedding(src_tokens) * math.sqrt(self.embed_dim)  # [batch_size, seq_len, embed_dim]
        x = self.dropout(x)

        output = self.mechanism_net(x)  # output is [batch, seq, hidden_dim]

        logits = self.fc_out(output)  # [batch, seq, vocab_size]
        return logits, None, None



class MechanismNetAnalyzer:
    def __init__(self, model: LMWithMechanismNet, device="cpu"):
        self.model = model.to(device)
        self.device = device

    def analyze_batch(self, tokens: torch.Tensor):
        """
        Run a batch through LMWithMechanismNet and visualize inner workings.
        tokens: [batch, seq]
        """
        tokens = tokens.to(self.device)
        with torch.no_grad():
            embeddings = self.model.embedding(tokens) * math.sqrt(self.model.embed_dim)
            embeddings = self.model.dropout(embeddings)

            # Access the mechanism net
            net = self.model.mechanism_net

            # Layer 1: Grand Unity Core
            self._plot_grand_unity(net.grand_unity, embeddings)

            # One pass through Unity for context
            unity_state = net.grand_unity(embeddings)

            # Layers 2–4: Four Spirits + Ten Essences + Positional Branches
            x = embeddings
            for idx in range(net.num_layers):
                self._plot_four_spirits(net.four_spirits[idx], x)
                self._plot_ten_essences(net.ten_essences[idx], x)

                # add positional branch for completeness
                positions = net.positional_branches[idx][:, :x.size(1), :]
                x = x + positions

            # Layer 5: Mechanism Grabber
            self._plot_mechanism_grabber(net.mechanism_grabber, x, unity_state)

            # Layer 7: Infinite Transformation
            self._plot_infinite_transformation(net.infinite_transform, x)

    # === Visualization helpers ===

    def _plot_grand_unity(self, core, x):
        with torch.no_grad():
            move = torch.sigmoid(core.movement_gate(x)).mean().item()
            empty = F.relu(core.emptiness_gate(x)).mean().item()

        plt.bar(["Movement", "Emptiness"], [move, empty])
        plt.title("Grand Unity Gates")
        plt.show()

    def _plot_four_spirits(self, processor, x):
        with torch.no_grad():
            east = F.gelu(processor.azure_dragon(x)).mean().item()
            south = F.relu(processor.vermilion_bird(x)).mean().item()
            west = torch.tanh(processor.white_tiger(x)).mean().item()
            north = F.silu(processor.black_turtle(x)).mean().item()
        values = [east, south, west, north]
        labels = ["Azure Dragon", "Vermilion Bird", "White Tiger", "Black Turtle"]
        angles = np.linspace(0, 2*np.pi, len(values), endpoint=False).tolist()
        values += values[:1]; angles += angles[:1]
        fig, ax = plt.subplots(subplot_kw={"polar": True})
        ax.plot(angles, values, "o-", linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
        plt.title("Four Spirits Balance")
        plt.show()

    def _plot_ten_essences(self, flow, x):
        with torch.no_grad():
            weights = F.softmax(flow.essence_selector(x), dim=-1)
            avg_w = weights.mean(dim=(0,1)).cpu().numpy()
        labels = ["Wood+", "Wood-", "Fire+", "Fire-", "Earth+", "Earth-", "Metal+", "Metal-", "Water+", "Water-"]
        plt.bar(labels, avg_w)
        plt.xticks(rotation=45)
        plt.title("Ten Essences Weights")
        plt.show()

    def _plot_mechanism_grabber(self, grabber, x, context):
        with torch.no_grad():
            scores = F.softmax(grabber.selector(context), dim=-1)
            avg_scores = scores.mean(dim=(0,1)).cpu().numpy()
        plt.figure(figsize=(12,4))
        plt.bar(range(len(avg_scores)), avg_scores)
        plt.title("Mechanism Selection Distribution")
        plt.xlabel("Mechanism ID")
        plt.ylabel("Avg Probability")
        plt.show()

    def _plot_infinite_transformation(self, layer, x):
        with torch.no_grad():
            n1, _ = layer.nested_mechanism(x)
            n2, _ = layer.nested_momentum(x)
            empty_full = layer.empty_full_gate(torch.cat([n1, n2], dim=-1))
            created = torch.relu(layer.creation_gate(torch.zeros_like(x))) * empty_full
            dissolved = x * torch.sigmoid(layer.dissolution_gate(x))
            meta = layer.meta_transform(x) * layer.adaptation_rate

            norms = {
                "Nested LSTM": n1.norm(dim=-1).mean().item(),
                "Nested GRU": n2.norm(dim=-1).mean().item(),
                "Creation": created.norm(dim=-1).mean().item(),
                "Dissolution": dissolved.norm(dim=-1).mean().item(),
                "Meta": meta.norm(dim=-1).mean().item(),
            }
        plt.bar(norms.keys(), norms.values())
        plt.title("Infinite Transformation Contributions")
        plt.xticks(rotation=30)
        plt.show()
