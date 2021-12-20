# SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import numpy as np


def normalize_min_max(data, eps=1e-10):
    mn = data.min()
    mx = data.max()
    data_normalized = data - mn
    old_range = mx - mn + eps
    data_normalized /= old_range

    return data_normalized


def standardize(data, log=True):
    if log is True:
        data = np.log10(data)
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / std
    return data, mean, std


def standardize_inverse(data, mean=0, std=1, log=True):
    data = data * std + mean
    if log is True:
        data = 10 ** data
    return data