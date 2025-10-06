import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Create input range
x = torch.linspace(-5, 5, 200)

# Define activation functions
activations = {
    'ReLU': nn.ReLU(),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'LeakyReLU': nn.LeakyReLU(0.1),
    'ELU': nn.ELU(),
    'Softplus': nn.Softplus(),
    'GELU': nn.GELU(),
    'SiLU (Swish)': nn.SiLU()
}

# Create subplots
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

# Plot each activation function
for idx, (name, activation) in enumerate(activations.items()):
    y = activation(x)
    axes[idx].plot(x.numpy(), y.numpy(), linewidth=2)
    axes[idx].set_title(name, fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].axhline(y=0, color='k', linewidth=0.5)
    axes[idx].axvline(x=0, color='k', linewidth=0.5)
    axes[idx].set_xlabel('Input')
    axes[idx].set_ylabel('Output')

plt.tight_layout()
plt.savefig('activation_functions.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization saved to activation_functions.png")