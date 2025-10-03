# attention.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

Tensor = torch.Tensor

# TODO: Make sure the weights are updated and correlated with the type of attention.

class BaseAttention(nn.Module):
    """
    Contract for attention blocks used by CascadeRNN.

    Input:
        hidden_states: (B, N, H) where
            B = batch size,
            N = num_automata,
            H = hidden_per_automaton

    Output:
        aggregated_values: (B, N, H)  # same hidden dimension
    """
    def __init__(self, num_automata: int, hidden_dim: int):
        super().__init__()
        self.num_automata = num_automata
        self.hidden_dim = hidden_dim
        self.last_weights: Optional[Tensor] = None  # (B, N, N) from last call

    def forward(self, hidden_states: Tensor) -> Tensor:
        raise NotImplementedError

    def get_last_weights(self) -> Optional[Tensor]:
        return self.last_weights


class StaticPositionalAttention(BaseAttention):
    """
    Attention is a learned, *static* routing
    (no dependence on inputs), built from positional encodings.
    """
    def __init__(self,
                 num_automata: int,
                 hidden_dim: int,
                 attn_dim: int = 64,
                 temperature: float = 1.0):
        super().__init__(num_automata, hidden_dim)
        self.attn_dim = attn_dim
        self.temperature = float(temperature)

        # Learned positional codes, then map to K/Q
        self.positional = nn.Parameter(torch.randn(num_automata, attn_dim))
        self.key_proj   = nn.Linear(attn_dim, attn_dim)
        self.query_proj = nn.Linear(attn_dim, attn_dim)

        # V and output map between hidden <-> attn channels
        self.value_proj = nn.Linear(hidden_dim, attn_dim)
        self.output_proj = nn.Linear(attn_dim, hidden_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        B, N, H = hidden_states.shape
        assert N == self.num_automata and H == self.hidden_dim

        # Static K/Q
        keys    = self.key_proj(self.positional)    # (N, A)
        queries = self.query_proj(self.positional)  # (N, A)

        # Values from current hidden
        values = self.value_proj(hidden_states)     # (B, N, A)

        # Scores (broadcast to batch)
        scores = (queries @ keys.transpose(0, 1))   # (N, N)
        scores = scores / (math.sqrt(self.attn_dim) * self.temperature)
        weights = F.softmax(scores, dim=-1)         # (N, N)
        weights_b = weights.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)

        # Aggregate and project back
        aggregated = torch.einsum('bij,bja->bia', weights_b, values)  # (B, N, A)
        out = self.output_proj(aggregated)                            # (B, N, H)

        # cache batch weights for logging
        self.last_weights = weights_b.detach()
        return out
    
class PastOnlyStaticPositionalAttention(BaseAttention):
    """
    Static (input-independent) attention from positional encodings,
    with a strict lower-triangular mask: row i may attend only to j < i.

    Args:
      num_automata: N
      hidden_dim:   H
      attn_dim:     A (internal attention channel)
      temperature:  softmax temperature (>1 smoother)
      allow_self_row0: if True, allow only (0->0) so row 0 is valid
      soft_bias_boost: if not None, add +boost to all allowed past entries (j<i)
                       to *encourage* but not force past attention.
    """
    def __init__(self,
                 num_automata: int,
                 hidden_dim: int,
                 attn_dim: int = 64,
                 temperature: float = 1.0,
                 allow_self_row0: bool = True,
                 soft_bias_boost: Optional[float] = None):
        super().__init__(num_automata, hidden_dim)
        self.attn_dim = attn_dim
        self.temperature = float(temperature)
        self.allow_self_row0 = bool(allow_self_row0)

        # Learned positional encodings → K/Q
        self.positional = nn.Parameter(torch.randn(num_automata, attn_dim))
        self.key_proj   = nn.Linear(attn_dim, attn_dim)
        self.query_proj = nn.Linear(attn_dim, attn_dim)

        # V and output map
        self.value_proj  = nn.Linear(hidden_dim, attn_dim)
        self.output_proj = nn.Linear(attn_dim, hidden_dim)

        # --- Strict lower-triangular mask M (N,N) ---
        # M[i,j] = 0 if j < i;  M[i,j] = -inf otherwise
        N = num_automata
        mask = torch.zeros(N, N)
        upper = torch.triu(torch.ones(N, N), diagonal=0).bool()  # j >= i
        mask = mask.masked_fill(upper, float('-inf'))
        if self.allow_self_row0:
            mask[0, 0] = 0.0  # keep row 0 valid
        self.register_buffer("mask", mask)

        # Optional soft bias (+boost on allowed past positions)
        if soft_bias_boost is not None:
            bias = torch.zeros(N, N)
            tril = torch.tril(torch.ones(N, N), diagonal=-1).bool()
            bias[tril] = float(soft_bias_boost)
            if self.allow_self_row0:
                bias[0, 0] = 0.0
            self.register_buffer("bias", bias)
        else:
            self.register_buffer("bias", None)

    def forward(self, hidden_states: Tensor) -> Tensor:
        B, N, H = hidden_states.shape
        assert N == self.num_automata and H == self.hidden_dim

        # Static K/Q from positional encodings
        K = self.key_proj(self.positional)     # (N, A)
        Q = self.query_proj(self.positional)   # (N, A)

        # Values from current hidden
        V = self.value_proj(hidden_states)     # (B, N, A)

        # Scores (broadcast to batch)
        scores = (Q @ K.transpose(0, 1))       # (N, N)
        scores = scores / (math.sqrt(self.attn_dim) * self.temperature)

        # Apply strict past-only mask and optional soft bias
        scores = scores + self.mask.to(dtype=scores.dtype)
        if self.bias is not None:
            scores = scores + self.bias.to(dtype=scores.dtype)

        # Softmax → weights
        weights = F.softmax(scores, dim=-1)    # (N, N)

        # Safety: if allow_self_row0=False, row 0 was all -inf → NaNs; fix to one-hot on itself
        if not self.allow_self_row0 and (torch.isnan(weights[0]).any() or torch.isinf(scores[0]).all()):
            weights[0, :].fill_(0.0)
            weights[0, 0] = 1.0

        # Broadcast over batch and aggregate
        weights_b = weights.unsqueeze(0).expand(B, -1, -1)      # (B, N, N)
        aggregated = torch.einsum('bij,bja->bia', weights_b, V) # (B, N, A)
        out = self.output_proj(aggregated)                      # (B, N, H)

        # Cache for logging
        self.last_weights = weights_b.detach()
        return out


class ContentAttention(BaseAttention):
    """
    Content-based attention: K/Q derived from hidden states, so attention is dynamic.

    Optional:
      - mask (N,N): add large negative where attention is disallowed
      - temperature: softmax temperature ( >1 smoother, <1 sharper )
      - dropout on attention weights
    """
    def __init__(self,
                 num_automata: int,
                 hidden_dim: int,
                 attn_dim: int = 64,
                 temperature: float = 1.0,
                 dropout_p: float = 0.1,
                 mask: Optional[Tensor] = None):
        super().__init__(num_automata, hidden_dim)
        self.attn_dim = attn_dim
        self.temperature = float(temperature)
        self.dropout = nn.Dropout(dropout_p)

        self.key_from_h   = nn.Linear(hidden_dim, attn_dim)
        self.query_from_h = nn.Linear(hidden_dim, attn_dim)

        self.value_proj  = nn.Linear(hidden_dim, attn_dim)
        self.output_proj = nn.Linear(attn_dim, hidden_dim)

        if mask is not None:
            # Expect (N,N) on the right device later; register as buffer
            self.register_buffer("mask", mask)
        else:
            self.mask = None

    def forward(self, hidden_states: Tensor) -> Tensor:
        B, N, H = hidden_states.shape
        assert N == self.num_automata and H == self.hidden_dim

        keys    = self.key_from_h(hidden_states)     # (B, N, A)
        queries = self.query_from_h(hidden_states)   # (B, N, A)

        # (optional) stabilize magnitudes to reduce saturation
        keys    = F.normalize(keys, p=2, dim=-1)
        queries = F.normalize(queries, p=2, dim=-1)

        # Scores and temperature
        scores = torch.einsum('bia,bja->bij', queries, keys)  # (B, N, N)
        scores = scores / (math.sqrt(self.attn_dim) * self.temperature)

        # Structural mask if provided (broadcast over batch)
        if self.mask is not None:
            # Ensure mask is on the same device/dtype for addition
            scores = scores + self.mask.to(scores.dtype)

        weights = F.softmax(scores, dim=-1)          # (B, N, N)
        weights = self.dropout(weights)

        values = self.value_proj(hidden_states)      # (B, N, A)
        aggregated = torch.einsum('bij,bja->bia', weights, values)  # (B, N, A)
        out = self.output_proj(aggregated)                              # (B, N, H)

        self.last_weights = weights.detach()
        return out


def predecessor_self_mask(N: int, device=None) -> Tensor:
    """
    Build a (N,N) mask that allows only self and predecessor i-1,
    and disallows others by adding a large negative value.
    """
    m = torch.full((N, N), -1e9, device=device)
    m.fill_diagonal_(0.0)
    idx = torch.arange(N, device=device)
    m[idx, torch.clamp(idx - 1, min=0)] = 0.0
    return m

class PastOnlyContentAttention(BaseAttention):
    """
    Dynamic (content-based) attention with a strict lower-triangular mask:
        row i may attend only to columns j < i   (no self, no future).
    Row 0 has no valid past; we make it well-defined by forcing a one-hot
    self-weight (can be changed by allow_self_row0=True).

    Options:
      - attn_dim:      internal attention channel dimension A
      - temperature:   softmax temperature (>1 smoother, <1 sharper)
      - dropout_p:     dropout on attention weights
      - allow_self_row0: if True, allow (0->0) as the only valid entry in row 0
      - soft_bias_boost: if not None, add +boost to all allowed (strictly past) entries
                         to encourage (not force) mass on the past

    Shapes:
      hidden_states: (B, N, H) -> returns aggregated: (B, N, H)
    """

    def __init__(self,
                 num_automata: int,
                 hidden_dim: int,
                 attn_dim: int = 64,
                 temperature: float = 2.0,
                 dropout_p: float = 0.1,
                 allow_self_row0: bool = True,
                 soft_bias_boost: Optional[float] = None):
        super().__init__(num_automata, hidden_dim)
        self.attn_dim = attn_dim
        self.temperature = float(temperature)
        self.dropout = nn.Dropout(dropout_p)
        self.allow_self_row0 = bool(allow_self_row0)
        self.soft_bias_boost = soft_bias_boost

        # Projections
        self.key_from_h   = nn.Linear(hidden_dim, attn_dim)
        self.query_from_h = nn.Linear(hidden_dim, attn_dim)
        self.value_proj   = nn.Linear(hidden_dim, attn_dim)
        self.output_proj  = nn.Linear(attn_dim,  hidden_dim)

        # --- Build strict lower-triangular mask M (N,N) ---
        # M[i,j] = 0   if j < i
        # M[i,j] = -inf otherwise
        N = num_automata
        mask = torch.zeros(N, N)
        upper = torch.triu(torch.ones(N, N), diagonal=0).bool()  # j >= i
        mask = mask.masked_fill(upper, float('-inf'))
        if self.allow_self_row0:
            mask[0, 0] = 0.0  # keep row 0 valid
        self.register_buffer("mask", mask)  # will move with .to(device)

        # Optional soft bias B: +boost on strictly lower triangle (j < i), 0 elsewhere
        if soft_bias_boost is not None:
            bias = torch.zeros(N, N)
            tril = torch.tril(torch.ones(N, N), diagonal=-1).bool()
            bias[tril] = float(soft_bias_boost)
            if self.allow_self_row0:
                bias[0, 0] = 0.0
            self.register_buffer("bias", bias)
        else:
            self.register_buffer("bias", None)

    def forward(self, hidden_states: Tensor) -> Tensor:
        B, N, H = hidden_states.shape
        assert N == self.num_automata and H == self.hidden_dim

        # K/Q from current hidden (content-based); L2-normalize to reduce saturation
        K = F.normalize(self.key_from_h(hidden_states),   p=2, dim=-1)  # (B, N, A)
        Q = F.normalize(self.query_from_h(hidden_states), p=2, dim=-1)  # (B, N, A)

        # Scores with temperature
        scores = torch.einsum('bia,bja->bij', Q, K)  # (B, N, N)
        scores = scores / (math.sqrt(self.attn_dim) * self.temperature)

        # Apply strict past-only mask (broadcast over batch)
        scores = scores + self.mask.to(dtype=scores.dtype)  # (B, N, N)

        # Optional soft bias toward allowed past positions
        if self.bias is not None:
            scores = scores + self.bias.to(dtype=scores.dtype)

        # Softmax → weights
        weights = F.softmax(scores, dim=-1)  # (B, N, N)

        # Safety for the strict case where row 0 had no valid positions
        # (only if allow_self_row0=False): force row 0 to one-hot on itself
        if not self.allow_self_row0:
            # Detect rows that became NaN (sum of exp was zero)
            row0_bad = torch.isnan(weights[:, 0, :]).any(dim=-1)
            if row0_bad.any():
                weights[row0_bad, 0, :] = 0.0
                weights[row0_bad, 0, 0] = 1.0

        weights = self.dropout(weights)

        # Aggregate values and project back to hidden
        V = self.value_proj(hidden_states)                 # (B, N, A)
        aggregated = torch.einsum('bij,bja->bia', weights, V)  # (B, N, A)
        out = self.output_proj(aggregated)                 # (B, N, H)

        # Cache batch weights for external logging
        self.last_weights = weights.detach()
        return out