# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Utility functions for POMO-TW (Policy Optimization with Multiple Optima
for Time-Window problems).

Kept in a separate module so that they can be imported without triggering
the optional TensorBoard dependency pulled in by train.py.
"""

import numpy as np
import torch


def augment_coords_8fold(positions: torch.Tensor) -> torch.Tensor:
    """Generate 8 augmented versions of 2D node coordinates.

    Applies the 8 symmetries of the square (4 rotations × 2 reflections) to
    the normalised coordinate array.  All transformations preserve L2 distances,
    so the underlying distance matrix is the same for every augmentation –
    only the position *embedding* seen by the neural network changes.

    Args:
        positions: Float tensor of shape ``(n_problems, n_nodes, 2)`` with
                   coordinates already normalised to **[0, 1]**.

    Returns:
        augmented: Float tensor of shape ``(8 * n_problems, n_nodes, 2)``.
                   The first ``n_problems`` rows correspond to the identity
                   transform (original coordinates).
    """
    # Preserve device and dtype of input tensors
    device = positions.device
    dtype = positions.dtype
    x = positions[..., 0:1].to(device=device, dtype=dtype)   # (n, nodes, 1)
    y = positions[..., 1:2].to(device=device, dtype=dtype)   # (n, nodes, 1)
    # 8 distinct transformations: all combinations of (flip_x, flip_y, swap_xy)
    # Create ones tensor on same device/dtype to avoid device mismatches
    ones = x.new_ones(x.shape)
    variants = [
        torch.cat([x,         y        ], dim=-1),  # 0: identity
        torch.cat([ones - x,  y        ], dim=-1),  # 1: flip x
        torch.cat([x,         ones - y ], dim=-1),  # 2: flip y
        torch.cat([ones - x,  ones - y ], dim=-1),  # 3: flip both
        torch.cat([y,         x        ], dim=-1),  # 4: swap
        torch.cat([ones - y,  x        ], dim=-1),  # 5: swap + flip new-x
        torch.cat([y,         ones - x ], dim=-1),  # 6: swap + flip new-y
        torch.cat([ones - y,  ones - x ], dim=-1),  # 7: swap + flip both
    ]
    return torch.cat(variants, dim=0)   # (8*n, nodes, 2)


def augment_vrptw_dataset(data: dict, n_augments: int = 8) -> dict:
    """Augment a VRPTW / PDPTW dataset with *n_augments* coordinate transforms.

    Distance-invariant features (``distance_matrix``, ``demand``, ``capacity``,
    ``time_windows``, ``service_times``, ``pairs``) are tiled unchanged.
    Only node ``positions`` are transformed.

    Because L2 Euclidean distance is invariant under rotations and reflections,
    all *n_augments* versions of the same problem instance have identical
    distance matrices.  The neural-network encoder sees different coordinates
    (and thus different embeddings), which increases training diversity.

    Args:
        data: Dataset dictionary as loaded by
              :class:`earli.generate_data.ProblemLoader`.  Must contain
              ``positions`` ``(n, nodes, 2)`` and ``n_problems``.
        n_augments: Number of augmentations.  Must be 1, 2, 4, or 8.

    Returns:
        Augmented dataset dict with ``n_augments × n`` problems.  An extra
        key ``'pomo_group'`` (int tensor of shape ``(aug*n,)``) maps each
        augmented instance back to its original problem index.
    """
    assert n_augments in (1, 2, 4, 8), "n_augments must be 1, 2, 4, or 8"
    n = data['positions'].shape[0]
    pos = data['positions'].float()

    # Normalise to [0, 1] if coordinates appear to be in a larger range
    # (as determined by the pre-computed radius stored in the dataset).
    radius = float(data.get('radius', 1.0))
    if radius > 1.0:
        pos_norm = pos / (2.0 * radius) + 0.5   # map to [0, 1]
    else:
        pos_norm = pos

    all_variants = augment_coords_8fold(pos_norm)[:n_augments * n]  # (aug*n, nodes, 2)

    if radius > 1.0:
        # Restore to original coordinate scale
        all_variants = (all_variants - 0.5) * (2.0 * radius)

    aug_data: dict = {}
    for k, v in data.items():
        if k == 'positions':
            aug_data[k] = all_variants
        elif isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == n:
            aug_data[k] = v.repeat(n_augments, *([1] * (v.dim() - 1)))
        elif isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == n:
            aug_data[k] = np.tile(v, (n_augments, *([1] * (v.ndim - 1))))
        else:
            aug_data[k] = v

    aug_data['n_problems'] = n_augments * n
    # Record which original problem each augmented instance came from.
    # Keep pomo_group on same device as positions when possible.
    try:
        device = data['positions'].device
    except Exception:
        device = None
    if device is not None:
        aug_data['pomo_group'] = torch.arange(n, device=device).repeat(n_augments)
    else:
        aug_data['pomo_group'] = torch.arange(n).repeat(n_augments)
    return aug_data


# ---------------------------------------------------------------------------
# Aliases used inside train.py (_augment_* names kept for back-compat)
# ---------------------------------------------------------------------------

_augment_coords_8fold = augment_coords_8fold
_augment_vrptw_dataset = augment_vrptw_dataset
