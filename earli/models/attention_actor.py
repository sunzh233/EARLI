import logging as log
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor

from .attention_base.decoder import AutoregressiveDecoder, PrecomputedCache
from .attention_base.encoder import GraphAttentionEncoder


# from rl4co.utils.ops import select_start_nodes
# from rl4co.utils.pylogger import get_pylogger

# log = get_pylogger(__name__)


class ActorAttentionModel(nn.Module):
    """Base Auto-regressive policy for NCO construction methods.
    The policy performs the following steps:
        1. Encode the environment initial state into node embeddings
        2. Decode (autoregressively) to construct the solution to the NCO problem
    Based on the policy from Kool et al. (2019) and extended for common use on multiple models in RL4CO.

    Note:
        We recommend to provide the decoding method as a keyword argument to the
        decoder during actual testing. The `{phase}_decode_type` arguments are only
        meant to be used during the main training loop. You may have a look at the
        evaluation scripts for examples.

    Args:
        env_name: Name of the environment used to initialize embeddings
        encoder: Encoder module. Can be passed by sub-classes.
        decoder: Decoder module. Can be passed by sub-classes.
        init_embedding: Model to use for the initial embedding. If None, use the default embedding for the environment
        context_embedding: Model to use for the context embedding. If None, use the default embedding for the environment
        dynamic_embedding: Model to use for the dynamic embedding. If None, use the default embedding for the environment
        # select_start_nodes_fn: Function to select the start nodes for multi-start decoding
        embedding_dim: Dimension of the node embeddings
        num_encoder_layers: Number of layers in the encoder
        num_heads: Number of heads in the attention layers
        normalization: Normalization type in the attention layers
        mask_inner: Whether to mask the inner diagonal in the attention layers
        use_graph_context: Whether to use the initial graph context to modify the query
        sdpa_fn: Scaled dot product function to use for the attention
        train_decode_type: Type of decoding during training
        val_decode_type: Type of decoding during validation
        test_decode_type: Type of decoding during testing
        **unused_kw: Unused keyword arguments
    """

    def __init__(
            self,
            config=None,
            sampler=None,
            encoder: nn.Module = None,
            decoder: nn.Module = None,
            init_embedding: nn.Module = None,
            context_embedding: nn.Module = None,
            dynamic_embedding: nn.Module = None,
            # select_start_nodes_fn: Callable = select_start_nodes,
            embedding_dim: int = 128,
            num_heads: int = 8,
            mask_inner: bool = True,
            use_graph_context: bool = True,
            sdpa_fn: Optional[Callable] = None,
            linear_bias: bool = False,
            train_decode_type: str = "sampling",
            val_decode_type: str = "greedy",
            test_decode_type: str = "greedy",
            train_actor = True,
            **unused_kw,
            ):

        super(ActorAttentionModel, self).__init__()
        self.config = config
        if len(unused_kw) > 0:
            log.warning(f"Unused kwargs: {unused_kw}")
        normalization = config['attention_model']['layer_normalization']
        num_encoder_layers = config['attention_model']['n_attention_layers_actor']
        num_head_encoder_layers = config['attention_model']['num_head_encoder_layers']
        self.env_name = config['problem_setup']['env']
        self.train_actor = train_actor

        self.encoder = GraphAttentionEncoder(
                env_name=self.env_name,
                num_heads=num_heads,
                embedding_dim=embedding_dim,
                num_layers=num_encoder_layers,
                normalization=normalization,
                init_embedding=init_embedding,
                sdpa_fn=sdpa_fn,
                eight_rounding=config['model']['eight_rounding'],
                config=config,
                )

        self.decoder = AutoregressiveDecoder(
                env_name=self.env_name,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                use_graph_context=use_graph_context,
                # select_start_nodes_fn=select_start_nodes_fn,
                mask_inner=mask_inner,
                context_embedding=context_embedding,
                dynamic_embedding=dynamic_embedding,
                config=config,
                )

        self.use_graph_context = use_graph_context
        if config['model']['agg_type'] == 'sum':
            self.agg_fn = torch.sum
        elif config['model']['agg_type'] == 'mean':
            self.agg_fn = torch.mean
        elif config['model']['agg_type'] == 'max':
            self.agg_fn = torch.max
        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(
                embedding_dim, 3 * embedding_dim, bias=linear_bias
                )
        self.project_fixed_context = nn.Linear(
                embedding_dim, embedding_dim, bias=linear_bias
                )

        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

    def disable_non_head_layers(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        trainable_params = self.decoder.disable_non_head_layers()
        return trainable_params

    def forward(
            self,
            state: TensorDict,
            env: Union[str] = None,
            phase: str = "train",
            return_actions: bool = False,
            return_entropy: bool = False,
            return_init_embeds: bool = False,
            **decoder_kwargs,
            ) -> dict:
        """Forward pass of the policy.

        Args:
            state: TensorDict containing the environment state
            env: Environment to use for decoding
            phase: Phase of the algorithm (train, val, test)
            return_actions: Whether to return the actions
            return_entropy: Whether to return the entropy
            decoder_kwargs: Keyword arguments for the decoder. See :class:`rl4co.models.zoo.common.autoregressive.decoder.AutoregressiveDecoder`

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally the actions and entropy
        """

        # ENCODER: get embeddings from initial state
        inference_mode= torch.is_inference_mode_enabled() or not self.train_actor
        visible_nodes= state['visible_nodes'].to(bool)
        with torch.inference_mode(mode=inference_mode):
            embeddings = self.encoder(state, mask=state['attention_matrix'].to(bool), vector_mask=visible_nodes)

            # the depot column in the visible matrix is always the visible_nodes, as the depot is always visible
            cached_embeds = self._precompute_cache(embeddings, visible_nodes=visible_nodes)

            log_p, head_encoding = self.decoder.get_log_p(cached_embeds, state,inference_mode=inference_mode)
        if self.config['attention_model']['separate_head_and_action_model']:
            head_encoding = self.head_encoder(state, mask=state['attention_matrix'].to(bool), vector_mask=visible_nodes)
            if self.config['model']['agg_type'] == 'sum':
                head_encoding[~visible_nodes] = 0  # Set the masked nodes to 0
                head_encoding = self.project_fixed_context(head_encoding.sum(dim=1))
            elif self.config['model']['agg_type'] == 'mean':
                head_encoding[~visible_nodes] = 0  # Set the masked nodes to 0
                head_encoding = self.project_fixed_context(head_encoding.sum(dim=1)) / head_encoding.sum(dim=-1, keepdim=True)
            elif self.config['model']['agg_type'] == 'max':
                head_encoding[~visible_nodes] = head_encoding.min()
                head_encoding = self.project_fixed_context(head_encoding.max(dim=1).values)
        return log_p, head_encoding


    def _precompute_cache(
            self,
            embeddings: Tensor, visible_nodes
            ):
        """Compute the cached embeddings for the attention

        Args:
            embeddings: Precomputed embeddings for the nodes
            td: TensorDict containing the environment state.
            This one is not used in this class. However, passing Tensordict can be useful in child classes.
        """

        # The projection of the node embeddings for the attention is calculated once up front
        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key_fixed,
            ) = self.project_node_embeddings(embeddings).chunk(3, dim=-1)

        # Optionally disable the graph context from the initial embedding as done in POMO
        if self.use_graph_context:
            if self.config['model']['agg_type'] == 'sum':
                site_embeddings = embeddings.clone()  # Apply the mask
                site_embeddings[~visible_nodes] = 0  # Set the masked nodes to 0
                graph_context = self.project_fixed_context(site_embeddings.sum(dim=1))
            elif self.config['model']['agg_type'] == 'mean':
                site_embeddings = embeddings.clone()  # Apply the mask
                site_embeddings[~visible_nodes] = 0  # Set the masked nodes to 0
                graph_context = self.project_fixed_context(site_embeddings.sum(dim=1)) / visible_nodes.sum(dim=-1, keepdim=True)
            elif self.config['model']['agg_type'] == 'max':
                site_embeddings = embeddings.clone()
                site_embeddings[~visible_nodes] = site_embeddings.min()
                graph_context = self.project_fixed_context(site_embeddings.max(dim=1).values)
            else:
                raise NotImplementedError
        else:
            graph_context = 0

        # Organize in a dataclass for easy access
        cached_embeds = PrecomputedCache(
                node_embeddings=embeddings,
                graph_context=graph_context,
                glimpse_key=glimpse_key_fixed,
                glimpse_val=glimpse_val_fixed,
                logit_key=logit_key_fixed,
                )

        return cached_embeds

    # def evaluate_action(
    #         self,
    #         td: TensorDict,
    #         action: Tensor,
    #         env: Union[str] = None,
    #         ) -> Tuple[Tensor, Tensor]:
    #     """Evaluate the action probability and entropy under the current policy
    #
    #     Args:
    #         td: TensorDict containing the current state
    #         action: Action to evaluate
    #         env: Environment to evaluate the action in.
    #     """
    #     embeddings, _ = self.encoder(td)
    #     ll, entropy = self.decoder.evaluate_action(td, embeddings, action, env)
    #     return ll, entropy
