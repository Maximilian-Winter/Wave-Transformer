import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model

        # Precompute inverse frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos and sin for all positions
        t = torch.arange(max_len).unsqueeze(1).float()
        freqs = t * inv_freq.unsqueeze(0)

        cos = freqs.cos()
        sin = freqs.sin()

        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape

        # Get precomputed values for current sequence length
        cos = self.cos[:seq_len].unsqueeze(0)  # (1, seq_len, d_model//2)
        sin = self.sin[:seq_len].unsqueeze(0)  # (1, seq_len, d_model//2)

        # Split features
        x1 = x[..., :d_model // 2]
        x2 = x[..., d_model // 2:]

        # Apply rotation
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        return rotated


class HashEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_hashes=2):
        super().__init__()
        self.num_hashes = num_hashes
        self.embedding_dim = embedding_dim

        # Multiple smaller embedding tables
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings // num_hashes, embedding_dim)
            for _ in range(num_hashes)
        ])

    def forward(self, input_ids):
        # Simple hash functions
        hash_values = [
            input_ids % len(self.embeddings[i].weight)
            for i in range(self.num_hashes)
        ]

        # Combine embeddings
        embeddings = [self.embeddings[i](hash_values[i])
                      for i in range(self.num_hashes)]

        return sum(embeddings) / self.num_hashes


class CharCNNEmbedding(nn.Module):
    def __init__(self, vocab_size, char_embed_dim=16,
                 filters=[32, 64, 128], kernel_sizes=[3, 4, 5]):
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, char_embed_dim)

        self.convs = nn.ModuleList([
            nn.Conv1d(char_embed_dim, n_filters, kernel_size)
            for n_filters, kernel_size in zip(filters, kernel_sizes)
        ])

        self.output_dim = sum(filters)

    def forward(self, char_ids):
        # char_ids shape: (batch, seq_len, max_word_len)
        batch_size, seq_len, max_word_len = char_ids.shape

        # Flatten batch and sequence dimensions
        char_ids = char_ids.view(-1, max_word_len)

        x = self.char_embed(char_ids)
        x = x.transpose(1, 2)  # (batch*seq, embed_dim, max_word_len)

        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)

        output = torch.cat(conv_outputs, dim=1)
        return output.view(batch_size, seq_len, -1)


class SubwordEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_subwords=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.max_subwords = max_subwords

    def forward(self, subword_ids, subword_mask=None):
        # subword_ids shape: (batch, seq_len, max_subwords)
        embeddings = self.embedding(subword_ids)

        if subword_mask is not None:
            # Mask padding subwords
            embeddings = embeddings * subword_mask.unsqueeze(-1)
            # Average over valid subwords
            lengths = subword_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            embeddings = embeddings.sum(dim=2) / lengths
        else:
            embeddings = embeddings.mean(dim=2)

        return embeddings


class HybridEmbedding(nn.Module):
    def __init__(self, word_vocab_size, char_vocab_size,
                 word_embed_dim=200, char_embed_dim=100):
        super().__init__()
        self.word_embed = nn.Embedding(word_vocab_size, word_embed_dim)
        self.char_embed = CharCNNEmbedding(char_vocab_size)
        self.projection = nn.Linear(word_embed_dim + self.char_embed.output_dim,
                                    word_embed_dim)

    def forward(self, word_ids, char_ids):
        word_emb = self.word_embed(word_ids)
        char_emb = self.char_embed(char_ids)

        combined = torch.cat([word_emb, char_emb], dim=-1)
        return self.projection(combined)