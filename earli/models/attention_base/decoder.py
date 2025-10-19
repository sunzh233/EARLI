from typing import Tuple, Union

import torch
import torch.nn as nn
from dataclasses import dataclass
from tensordict import TensorDict
from torch import Tensor

from .attention import LogitAttention, SimpleSetAttention
from .env_embeddings.context import gather_by_index


# from rl4co.utils.ops import batchify, get_num_starts, select_start_nodes, unbatchify
# from rl4co.utils.pylogger import get_pylogger

# log = get_pylogger(__name__)


@dataclass
class PrecomputedCache:
    node_embeddings: Tensor
    graph_context: Union[Tensor, float]
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor


class AutoregressiveDecoder(nn.Module):
    """Auto-regressive decoder for constructing solutions for combinatorial optimization problems.
    Given the environment state and the embeddings, compute the logits and sample actions autoregressively until
    all the environments in the batch have reached a terminal state.
    We additionally include support for multi-starts as it is more efficient to do so in the decoder as we can
    natively perform the attention computation.

    Note:
        There are major differences between this decoding and most RL problems. The most important one is
        that reward is not defined for partial solutions, hence we have to wait for the environment to reach a terminal
        state before we can compute the reward with `env.get_reward()`.

    Warning:
        We suppose environments in the `done` state are still available for sampling. This is because in NCO we need to
        wait for all the environments to reach a terminal state before we can stop the decoding process. This is in
        contrast with the TorchRL framework (at the moment) where the `env.rollout` function automatically resets.
        You may follow tighter integration with TorchRL here: https://github.com/kaist-silab/rl4co/issues/72.

    Args:
        env_name: environment name to solve
        embedding_dim: Dimension of the embeddings
        num_heads: Number of heads for the attention
        use_graph_context: Whether to use the initial graph context to modify the query
        select_start_nodes_fn: Function to select the start nodes for multi-start decoding
        linear_bias: Whether to use a bias in the linear projection of the embeddings
        context_embedding: Module to compute the context embedding. If None, the default is used
        dynamic_embedding: Module to compute the dynamic embedding. If None, the default is used
    """

    def __init__(
            self,
            env_name: str,
            embedding_dim: int,
            num_heads: int,
            use_graph_context: bool = True,
            # select_start_nodes_fn: callable = select_start_nodes,
            linear_bias: bool = False,
            context_embedding: nn.Module = None,
            dynamic_embedding: nn.Module = None,
            config=None,
            **logit_attn_kwargs,
            ):
        super().__init__()
        self.env_name = env_name
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.eight_rounding = config['model']['eight_rounding']
        self.config = config
        self.separate_head_and_action_model = config['attention_model']['separate_head_and_action_model']
        assert embedding_dim % num_heads == 0
        self.context_and_head_embedding = nn.Linear(2 * embedding_dim + (8 if self.eight_rounding else 3),
                                                        2 * embedding_dim, bias=linear_bias)
        # MHA
        self.logit_attention = LogitAttention(
                embedding_dim, num_heads, **logit_attn_kwargs
                )
        if not config['attention_model']['use_basic_head_encoding']:
            self.simple_set_attention = SimpleSetAttention(embedding_dim, num_heads)

    def disable_non_head_layers(self):
        for param in self.logit_attention.parameters():
            param.requires_grad = False
        self.context_and_head_embedding.requires_grad_(False)
        if not self.config['attention_model']['use_basic_head_encoding']:
            trainable_params = self.simple_set_attention.parameters()
        else:
            trainable_params = None
        return trainable_params

    def forward(
            self,
            td: TensorDict,
            embeddings: Tensor,
            env: str = None,
            decode_type: str = "sampling",
            num_starts: int = None,
            softmax_temp: float = None,
            calc_reward: bool = True,
            ) -> Tuple[Tensor, Tensor, TensorDict]:
        """Forward pass of the decoder
        Given the environment state and the pre-computed embeddings, compute the logits and sample actions

        Args:
            td: Input TensorDict containing the environment state
            embeddings: Precomputed embeddings for the nodes
            env: Environment to use for decoding. If None, the environment is instantiated from `env_name`. Note that
                it is more efficient to pass an already instantiated environment each time for fine-grained control
            decode_type: Type of decoding to use. Can be one of:
                - "sampling": sample from the logits
                - "greedy": take the argmax of the logits
                - "multistart_sampling": sample as sampling, but with multi-start decoding
                - "multistart_greedy": sample as greedy, but with multi-start decoding
            num_starts: Number of multi-starts to use. If None, will be calculated from the action mask
            softmax_temp: Temperature for the softmax. If None, default softmax is used from the `LogitAttention` module
            calc_reward: Whether to calculate the reward for the decoded sequence

        Returns:
            outputs: Tensor of shape (batch_size, seq_len, num_nodes) containing the logits
            actions: Tensor of shape (batch_size, seq_len) containing the sampled actions
            td: TensorDict containing the environment state after decoding
        """
        # env_name = env
        # Instantiate environment if needed
        # if isinstance(env, str):
        #     env_name = self.env_name if env is None else env
        #     env = get_env(env_name)

        # Multi-start decoding. If num_starts is None, we use the number of actions in the action mask
        # if "multistart" in decode_type:
        #     if num_starts is None:
        #         num_starts = get_num_starts(td, env.name)
        # else:
        #     if num_starts is not None:
        #         if num_starts > 1:
        #             log.warn(
        #                 f"num_starts={num_starts} is ignored for decode_type={decode_type}"
        #             )
        #     num_starts = 0

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        cached_embeds = self._precompute_cache(embeddings, td=td)
        # Main decoding: loop until all sequences are done
        logits = self.get_log_p(cached_embeds, td, softmax_temp, num_starts)
        # Collect outputs
        # outputs = []
        # actions = []

        # Multi-start decoding: first action is chosen by ad-hoc node selection
        # if num_starts > 1 or "multistart" in decode_type:
        #     action = self.select_start_nodes_fn(td, env, num_starts=num_starts)
        #
        #     # Expand td to batch_size * num_starts
        #     td = batchify(td, num_starts)
        #
        #     td.set("action", action)
        #     td = env.step(td)["next"]
        #     log_p = torch.zeros_like(
        #         td["action_mask"], device=td.device
        #     )  # first log_p is 0, so p = log_p.exp() = 1
        #
        #     outputs.append(log_p)
        #     actions.append(action)

        # # Select the indices of the next nodes in the sequences, result (batch_size) long
        # action = decode_probs(log_p.exp(), mask, decode_type=decode_type)
        #
        # td.set("action", action)
        # td = env.step(td)["next"]
        #
        # # Collect output of step
        # outputs.append(log_p)
        # actions.append(action)

        # while not td["done"].all():
        #     log_p, mask = self._get_log_p(cached_embeds, td, softmax_temp, num_starts)
        #
        #     # Select the indices of the next nodes in the sequences, result (batch_size) long
        #     action = decode_probs(log_p.exp(), mask, decode_type=decode_type)
        #
        #     td.set("action", action)
        #     td = env.step(td)["next"]
        #
        #     # Collect output of step
        #     outputs.append(log_p)
        #     actions.append(action)
        #
        # assert (
        #     len(outputs) > 0
        # ), "No outputs were collected because all environments were done. Check your initial state"
        # outputs, actions = torch.stack(outputs, 1), torch.stack(actions, 1)
        # if calc_reward:
        #     td.set("reward", env.get_reward(td, actions))

        return logits

    def get_log_p(
            self,
            cached: PrecomputedCache,
            td: TensorDict,
            softmax_temp: float = None,
            num_starts: int = 0,
            inference_mode: bool = False,
            ):
        """Compute the log probabilities of the next actions given the current state

        Args:
            cache: Precomputed embeddings
            td: TensorDict with the current environment state
            softmax_temp: Temperature for the softmax
            num_starts: Number of starts for the multi-start decoding
        """
        with torch.inference_mode(mode=inference_mode):
            # Get precomputed (cached) embeddings
            node_embeds_cache, graph_context_cache = (
                cached.node_embeddings,
                cached.graph_context,
                )
            glimpse_k, glimpse_v, logit_k = (
                cached.glimpse_key,
                cached.glimpse_val,
                cached.logit_key,
                )  # [B, N, H]


            # they are constant for different nodes, we can just take the first one
            if self.eight_rounding:
                dynamic_features = torch.cat((td['capacity'][:, 0], td['remaining_demand'][:, 0], td['remaining_nodes'][:, 0],
                                              torch.zeros([len(td), 5], device='cuda')), dim=-1)  # padding to 8 to use TC
            else:
                dynamic_features = torch.cat((td['capacity'][:, 0], td['remaining_demand'][:, 0], td['remaining_nodes'][:, 0]),
                                             dim=-1)
            tensors = [gather_by_index(node_embeds_cache, td["head"].to(int)), graph_context_cache, dynamic_features]
            if tensors[0].ndim < tensors[1].ndim:
                tensors[0] = tensors[0].unsqueeze(0)
            step_context = torch.cat(tensors=tensors, dim=-1)


            # Get the mask
            mask = td["feasible_nodes"].to(bool)
            no_feasible_nodes = ~mask.any(dim=-1)
            mask[no_feasible_nodes, 0] = True  # mask must have at least one True per row, a hack to avoid nan
            step_context = self.context_and_head_embedding(step_context)
            glimpse_q, head_encoding_input = torch.split(step_context, self.embedding_dim, dim=-1)
            glimpse_q = (
                glimpse_q.unsqueeze(1) if glimpse_q.ndim == 2 else glimpse_q
            )  # add seq_len dim if not present

            # Compute logits
            log_p = self.logit_attention(glimpse_q, glimpse_k, glimpse_v, logit_k, mask, softmax_temp)
            if self.config['attention_model']['use_basic_head_encoding'] or self.separate_head_and_action_model:
                head_encoding = head_encoding_input
            else:
                head_encoding_input = head_encoding_input.unsqueeze(1) if head_encoding_input.ndim == 2 else head_encoding_input
                head_encoding = self.simple_set_attention(head_encoding_input, glimpse_k, glimpse_v, mask)

        return log_p, head_encoding  # , mask
