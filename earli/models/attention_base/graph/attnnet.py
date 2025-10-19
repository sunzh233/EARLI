from typing import Callable, Optional

import torch.nn as nn
from torch import Tensor

from ..attention import MultiHeadAttention
from ..ops import Normalization


# from rl4co.utils.pylogger import get_pylogger
#
# log = get_pylogger(__name__)


class MultiHeadAttentionLayer(nn.Module):
    """Multi-Head Attention Layer with normalization and feed-forward layer

    Args:
        num_heads: number of heads in the MHA
        embed_dim: dimension of the embeddings
        feed_forward_hidden: dimension of the hidden layer in the feed-forward layer
        normalization: type of normalization to use (batch, layer, none)
        sdpa_fn: scaled dot product attention function (SDPA)
    """

    def __init__(
            self,
            num_heads: int,
            embed_dim: int,
            feed_forward_hidden: int = 512,
            normalization: Optional[str] = "batch",
            sdpa_fn: Optional[Callable] = None,
            config=None,
            ):
        super(MultiHeadAttentionLayer, self).__init__()
        self.attention_layer = MultiHeadAttention(embed_dim, num_heads, sdpa_fn=sdpa_fn, config=config)
        self.normalization_1 = Normalization(embed_dim, normalization)
        self.normalization_2 = Normalization(embed_dim, normalization)
        if feed_forward_hidden > 0:
            self.mlp = nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim),
                    )
        else:
            self.mlp = nn.Linear(embed_dim, embed_dim)

    def forward(self, x_in, mask=None, vector_mask=None):
        x = x_in + self.attention_layer(x_in, mask=mask)
        if vector_mask is not None and vector_mask is not Ellipsis:
            x[~vector_mask] = 0
        x = self.normalization_1(x) # todo: normalization is incorrect for batchnorm with masked items, expect covariant shift
        x = x + self.mlp(x)
        if vector_mask is not None and vector_mask is not Ellipsis:
            x[~vector_mask] = 0
        x = self.normalization_2(x)
        if vector_mask is not None and vector_mask is not Ellipsis:
            x[~vector_mask] = 0
        return x


class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network to encode embeddings with a series of MHA layers consisting of a MHA layer,
    normalization, feed-forward layer, and normalization. Similar to Transformer encoder, as used in Kool et al. (2019).

    Args:
        num_heads: number of heads in the MHA
        embedding_dim: dimension of the embeddings
        num_layers: number of MHA layers
        normalization: type of normalization to use (batch, layer, none)
        feed_forward_hidden: dimension of the hidden layer in the feed-forward layer
        sdpa_fn: scaled dot product attention function (SDPA)
    """

    def __init__(
            self,
            num_heads: int,
            embedding_dim: int,
            num_layers: int,
            normalization: str = "batch",
            feed_forward_hidden: int = 512,
            sdpa_fn: Optional[Callable] = None,
            config = None,
            ):
        super(GraphAttentionNetwork, self).__init__()
        self.layers = nn.ModuleList([MultiHeadAttentionLayer(
                            num_heads,
                            embedding_dim,
                            feed_forward_hidden=feed_forward_hidden,
                            normalization=normalization,
                            sdpa_fn=sdpa_fn, config=config
                            ) for _ in range(num_layers)])

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, vector_mask=None) -> Tensor:
        """Forward pass of the encoder

        Args:
            x: [batch_size, graph_size, embed_dim] initial embeddings to process
            mask: [batch_size, graph_size, graph_size] mask for the input embeddings. Unused for now.
        """
        # assert mask is None, "Mask not yet supported!"
        for i, layer in enumerate(self.layers):
            x = layer(x, mask=mask, vector_mask=vector_mask)
        return x
