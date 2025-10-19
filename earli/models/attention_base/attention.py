import logging as log
from typing import Callable, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# from rl4co.utils import get_pylogger
#
# log = get_pylogger(__name__)


def scaled_dot_product_attention_simple(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, decay_factor=None,
        ):
    """Simple Scaled Dot-Product Attention in PyTorch without Flash Attention"""
    # Check for causal and attn_mask conflict
    if is_causal and attn_mask is not None:
        raise ValueError("Cannot set both is_causal and attn_mask")

    # Calculate scaled dot product
    scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)

    # Apply the provided attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores.masked_fill_(~attn_mask, float("-inf"))
        else:
            scores += attn_mask

    if decay_factor is not None:
        scores = scores * decay_factor

    # Apply causal mask
    if is_causal:
        s, l_ = scores.size(-2), scores.size(-1)
        mask = torch.triu(torch.ones((s, l_), device=scores.device), diagonal=1)
        scores.masked_fill_(mask.bool(), float("-inf"))

    # Softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)

    # Apply dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # Compute the weighted sum of values
    return torch.matmul(attn_weights, v)


try:
    from torch.nn.functional import scaled_dot_product_attention
except ImportError:
    log.warning(
            "torch.nn.functional.scaled_dot_product_attention not found. Make sure you are using PyTorch >= 2.0.0."
            "Alternatively, install Flash Attention https://github.com/HazyResearch/flash-attention ."
            "Using custom implementation of scaled_dot_product_attention without Flash Attention. "
            )
    scaled_dot_product_attention = scaled_dot_product_attention_simple


class MultiHeadAttention(nn.Module):
    """PyTorch native implementation of Flash Multi-Head Attention with automatic mixed precision support.
    Uses PyTorch's native `scaled_dot_product_attention` implementation, available from 2.0

    Note:
        If `scaled_dot_product_attention` is not available, use custom implementation of `scaled_dot_product_attention` without Flash Attention.

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        bias: whether to use bias
        attention_dropout: dropout rate for attention weights
        causal: whether to apply causal mask to attention scores
        device: torch device
        dtype: torch dtype
        sdpa_fn: scaled dot product attention function (SDPA)
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            bias: bool = True,
            attention_dropout: float = 0.0,
            causal: bool = False,
            device: str = None,
            dtype: torch.dtype = None,
            sdpa_fn: Optional[Callable] = None,
            config=None
            ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.attention_dropout = attention_dropout

        # Default to `scaled_dot_product_attention` if `sdpa_fn` is not provided
        if sdpa_fn is None:
            sdpa_fn = scaled_dot_product_attention
        self.sdpa_fn = sdpa_fn
        if config['attention_model']['reweight_by_distance']:
            self.sdpa_fn = scaled_dot_product_attention_simple
            if config['attention_model']['reweight_function'] == 'exponential':
                self.distance_decay = nn.Parameter(torch.randn(1))
                decay_fn = lambda x: torch.exp(-self.distance_decay.abs() * x)
            elif config['attention_model']['reweight_function'] == 'poly':
                self.distance_decay = nn.Parameter(torch.randn(self.config['attention_model']['poly_decay_order']))

                def decay_fn(x):
                    x_powers = (x ** torch.arange(config['attention_model']['poly_decay_order'])).abs()
                    return torch.dot(self.distance_decay * x_powers)
            self.distance_decay_fn = decay_fn
            self.register_parameter('decay_params', self.distance_decay)
        else:
            self.distance_decay_fn = None

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert (
                self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, x, mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        # Project query, key, value
        q, k, v = rearrange(
                self.Wqkv(x), "b s (three h d) -> three b h s d", three=3, h=self.num_heads
                ).unbind(dim=0)

        # Scaled dot product attention
        out = self.sdpa_fn(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.attention_dropout,
                )
        return self.out_proj(rearrange(out, "b h s d -> b s (h d)"))


class LastLayerAttention(nn.Module):
    def __init__(self, num_heads: int, sdp_fn=scaled_dot_product_attention):
        super(LastLayerAttention, self).__init__()
        self.num_heads = num_heads
        self.sdp_fn = sdp_fn

    def _inner_mha(self, query, key, value, mask):
        q = self._make_heads(query)
        k = self._make_heads(key)
        v = self._make_heads(value)
        n_batch, n_heads, n_nodes, n_features = k.shape
        mask = mask.view(n_batch, 1, 1, n_nodes)
        heads = self.sdp_fn(q, k, v, attn_mask=mask)
        return rearrange(heads, "... h n g -> ... n (h g)", h=self.num_heads)

    def _make_heads(self, v):
        return rearrange(v, "... g (h s) -> ... h g s", h=self.num_heads)


class SimpleSetAttention(LastLayerAttention):
    """Calculate beam state given query, key and value and logit key.

    Perform the following:
        1. Apply cross attention to get the heads
        2. Project heads to get glimpse

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        linear_bias: whether to use bias in linear projection
        sdp_fn: scaled dot product attention function (SDPA)
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            linear_bias: bool = False,
            sdp_fn=scaled_dot_product_attention,
            ):
        super(SimpleSetAttention, self).__init__(num_heads, sdp_fn)
        self.Wkv = nn.Linear(2 * embed_dim, 2 * embed_dim)
        self.q_mlp = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask):
        q = self.q_mlp(query)
        kv = torch.cat([key, value], dim=-1)
        k, v = self.Wkv(kv).split(kv.size(-1) // 2, dim=-1)
        heads = self._inner_mha(q, k, v, mask.to(bool))
        glimpse = query + heads
        return glimpse


class LogitAttention(LastLayerAttention):
    """Calculate logits given query, key and value and logit key.

    Note:
        With Flash Attention, masking is not supported

    Perform the following:
        1. Apply cross attention to get the heads
        2. Project heads to get glimpse
        3. Compute attention score between glimpse and logit key
        4. Normalize and mask

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        tanh_clipping: tanh clipping value
        mask_inner: whether to mask inner attention
        mask_logits: whether to mask logits
        normalize: whether to normalize logits
        softmax_temp: softmax temperature
        linear_bias: whether to use bias in linear projection
        sdp_fn: scaled dot product attention function (SDPA)
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            tanh_clipping: float = 10.0,
            mask_inner: bool = True,
            mask_logits: bool = True,
            normalize: bool = True,
            softmax_temp: float = 1.0,
            linear_bias: bool = False,
            sdp_fn=scaled_dot_product_attention,
            ):
        super(LogitAttention, self).__init__(num_heads, sdp_fn)
        self.mask_logits = mask_logits
        self.mask_inner = mask_inner
        self.tanh_clipping = tanh_clipping
        self.normalize = normalize
        self.softmax_temp = softmax_temp
        # Projection - query, key, value already include projections
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=linear_bias)

    def forward(self, query, key, value, logit_key, mask, softmax_temp=None):
        # Compute inner multi-head attention with no projections.
        heads = self._inner_mha(query, key, value, mask.to(bool))
        glimpse = self.project_out(heads)

        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # bmm is slightly faster than einsum and matmul
        logits = (
                torch.bmm(glimpse, logit_key.squeeze(1).transpose(-2, -1))
                / math.sqrt(glimpse.size(-1))
        ).squeeze(1)

        return logits
