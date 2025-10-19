# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import torch.nn as nn
from ..vrp import build_attention_matrix
from .attention_base.graph.attnnet import GraphAttentionNetwork


class HeadAttentionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = GraphAttentionNetwork(num_heads=config['attention_model']['n_attention_heads'],
                                             embedding_dim=config['model']['embedding_dim'],
                                             num_layers=config['attention_model']['n_attention_layers_head_module'],
                                             normalization=config['attention_model']['layer_normalization'],config=config)
        self.project_to_score = nn.Linear(config['model']['embedding_dim'], 1)

    def forward(self, embedding, batch_shape, unmasked_heads):
        if unmasked_heads is Ellipsis:
            mask = None
        else:
            mask = build_attention_matrix(unmasked_heads).to(embedding.device)
        attention_output = self.encoder(embedding, mask=mask, vector_mask=unmasked_heads)
        scores = self.project_to_score(attention_output).squeeze(-1)
        if unmasked_heads is not Ellipsis:
            scores[~unmasked_heads] = 0
        return scores
