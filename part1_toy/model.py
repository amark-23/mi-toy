"""Attention-only transformer for the induction task."""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    """Model sizes, kept in one place."""

    vocab_size: int = 100  # Number of distinct tokens
    n_ctx: int = 40        # Max sequence length (2 * half_len = 2 * 20)
    n_layers: int = 2      # Attention layers; induction needs two that compose
    n_heads: int = 4       # Parallel heads per layer; their roles are learned in training
    d_head: int = 16       # Size of each head's query/key/value/output vector
    d_model: int = 64      # Residual stream width (n_heads * d_head = 4 * 16 by convention)
    # No MLP size: the model is attention-only, which keeps it simple to interpret.


class Embed(nn.Module):
    """Token embedding: maps each token id to a learned vector."""

    def __init__(self, cfg: Config):
        """Create the embedding table W_E.

        W_E has shape (vocab_size, d_model), one row per token. It is
        initialized from a normal distribution (mean 0, std 0.02).

        Args:
            cfg: Model configuration.
        """
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(cfg.vocab_size, cfg.d_model) * 0.02)

    def forward(self, tokens):
        """Look up the embedding of each token.

        Args:
            tokens: Integer tensor of shape (batch, pos).

        Returns:
            Tensor of shape (batch, pos, d_model).
        """
        return self.W_E[tokens]


class PosEmbed(nn.Module):
    """Positional embedding: a learned vector per position in the sequence."""

    def __init__(self, cfg: Config):
        """Create the position table W_pos.

        W_pos has shape (n_ctx, d_model), one row per position. Its output is
        added to the token embedding, so the same token gets a different
        vector at different positions.

        Args:
            cfg: Model configuration.
        """
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(cfg.n_ctx, cfg.d_model) * 0.02)

    def forward(self, tokens):
        """Return the position vectors for the length of the input.

        Args:
            tokens: Integer tensor of shape (batch, pos); only its length is used.

        Returns:
            Tensor of shape (pos, d_model), broadcast over the batch when added.
        """
        seq_len = tokens.shape[1]
        return self.W_pos[:seq_len]


class Attention(nn.Module):
    """Multi-head attention (parameters only; forward pass not written yet).

    Each position looks at earlier positions and pulls information from them.
        Query (Q):  what am I looking for? (from the current position)
        Key (K):    what do I contain? (from each earlier position)
        Value (V):  what do I hand over if selected?
        Output (O): maps each head's result back to the residual stream.

    Steps:
        1. Compare the query with each earlier position's key (match scores).
        2. Turn the scores into weights with softmax.
        3. Take the weighted sum of the values and add it to the residual stream.

    Weights are stored per head: W_Q, W_K, W_V have shape
    (n_heads, d_model, d_head) and W_O has shape (n_heads, d_head, d_model).
    Bias terms are omitted: each head is then exactly four matrices, which
    keeps the analysis cleaner (modern models such as Llama also drop them).
    """

    def __init__(self, cfg: Config):
        """Create the per-head query, key, value and output matrices.

        Args:
            cfg: Model configuration.
        """
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head) * 0.02)
        self.W_K = nn.Parameter(torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head) * 0.02)
        self.W_V = nn.Parameter(torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head) * 0.02)
        self.W_O = nn.Parameter(torch.randn(cfg.n_heads, cfg.d_head, cfg.d_model) * 0.02)

    def forward(self, x):
        """Compute multi-head causal attention.

        Args:
            x: Residual stream, shape (batch, pos, d_model).

        Returns:
            Attention output to add to the residual stream, shape (batch, pos, d_model).
        """
        # Per-head projections; each has shape (batch, head, pos, d_head)
        q = torch.einsum("bpd,hde->bhpe", x, self.W_Q)
        k = torch.einsum("bpd,hde->bhpe", x, self.W_K)
        v = torch.einsum("bpd,hde->bhpe", x, self.W_V)

        # Match scores between each query position and each key position
        scores = torch.einsum("bhqe,bhke->bhqk", q, k) / self.cfg.d_head ** 0.5

        # Causal mask: a position may only attend to itself and earlier positions
        seq_len = x.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

        # Attention pattern: weights over earlier positions, each row sums to 1
        pattern = scores.softmax(dim=-1)

        # Weighted sum of values per head, shape (batch, head, pos, d_head)
        z = torch.einsum("bhqk,bhke->bhqe", pattern, v)

        # Map each head's result to d_model and sum over heads, shape (batch, pos, d_model)
        return torch.einsum("bhqe,hed->bqd", z, self.W_O)


class Unembed(nn.Module):
    """Unembedding: maps the final residual stream to one score per token."""

    def __init__(self, cfg: Config):
        """Create the unembedding matrix W_U.

        W_U has shape (d_model, vocab_size), initialized from a normal
        distribution (mean 0, std 0.02).

        Args:
            cfg: Model configuration.
        """
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(cfg.d_model, cfg.vocab_size) * 0.02)

    def forward(self, x):
        """Compute the logits for every position.

        Args:
            x: Final residual stream, shape (batch, pos, d_model).

        Returns:
            Logits, shape (batch, pos, vocab_size).
        """
        return x @ self.W_U


class Transformer(nn.Module):
    """Attention-only transformer: embeddings, attention layers, unembedding."""

    def __init__(self, cfg: Config):
        """Create all components from the config.

        Args:
            cfg: Model configuration.
        """
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.layers = nn.ModuleList([Attention(cfg) for _ in range(cfg.n_layers)])
        self.unembed = Unembed(cfg)

    def forward(self, tokens):
        """Compute next-token logits for every position.

        Args:
            tokens: Integer tensor of shape (batch, pos).

        Returns:
            Logits, shape (batch, pos, vocab_size).
        """
        # Start of the residual stream: token vector plus position vector
        x = self.embed(tokens) + self.pos_embed(tokens)
        # Each layer reads the stream and adds its output to it
        for layer in self.layers:
            x = x + layer(x)
        return self.unembed(x)