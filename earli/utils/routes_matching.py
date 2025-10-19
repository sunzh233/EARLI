# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from scipy.optimize import linear_sum_assignment


def split_into_sequences(lst):
    sequences = []
    current_sequence = []
    for node in lst:
        if node == 0:
            if current_sequence:
                sequences.append(current_sequence)
                current_sequence = []
        else:
            current_sequence.append(node)
    if current_sequence:
        sequences.append(current_sequence)
    return sequences


def calculate_overlap(seq1, seq2):
    return len(set(seq1) & set(seq2))


def optimize_sequence_order(list1, list2):
    sequences1 = split_into_sequences(list1)
    sequences2 = split_into_sequences(list2)

    # Create cost matrix
    n = max(len(sequences1), len(sequences2))
    cost_matrix = np.zeros((n, n))
    for i in range(len(sequences1)):
        for j in range(len(sequences2)):
            cost_matrix[i][j] = -calculate_overlap(sequences1[i], sequences2[j])

    # Pad cost matrix if necessary
    if n > len(sequences1):
        cost_matrix = np.pad(cost_matrix, ((0, n - len(sequences1)), (0, 0)), mode='constant', constant_values=0)
    elif n > len(sequences2):
        cost_matrix = np.pad(cost_matrix, ((0, 0), (0, n - len(sequences2))), mode='constant', constant_values=0)

    # Apply Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Reorder sequences2
    new_sequences2 = [None] * len(sequences2)
    for i, j in zip(row_ind, col_ind):
        if i < len(sequences1) and j < len(sequences2):
            new_sequences2[i] = sequences2[j]

    # Fill in any remaining sequences
    remaining = set(range(len(sequences2))) - set(col_ind[:len(sequences2)])
    for i, seq in zip(range(len(new_sequences2)), remaining):
        if new_sequences2[i] is None:
            new_sequences2[i] = sequences2[seq]

    # Reconstruct the list
    result = [0]
    for seq in new_sequences2:
        result.extend(seq)
        result.append(0)

    return result


if __name__ == '__main__':
    # Example usage
    list1 = [0, 1, 2, 3, 0, 4, 5, 0]
    list2 = [0, 4, 5, 1, 0, 2, 3, 0]

    optimized_list2 = optimize_sequence_order(list1, list2)
    print(list1, list2, optimized_list2)
