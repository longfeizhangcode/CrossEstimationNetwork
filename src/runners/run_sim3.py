#!/usr/bin/env python3
# =============================================================================
# Copyright (c) 2024 Longfei Zhang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================
# Author: Longfei Zhang
# Affil.: Collaborative Innovation Center of Assessment for
#         Basic Education Quality, Beijing Normal University
# E-mail: zhanglf@mail.bnu.edu.cn
# =============================================================================
"""Run Simulation Study 3."""

import os
import sys

# Add the "src" directory to sys.path, which makes sure the interpreter
# accesses the required modules appropriately.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import tensorflow as tf

from sims.sim3 import Simulation3

# Disable the use of GPU, use CPU only.
tf.config.experimental.set_visible_devices([], "GPU")

# Settings for Simulation Study 3.
n_persons_levels = [100, 500, 1000]
n_items_levels = [30, 60, 90]
n_persons_new = None  # The new persons will not be evaluated.
n_items_new = 30
n_reps = 30

# Settings for building CEN.
net_depth_levels = [1, 3]
linear_nonlinear = [True, False]

# Settings for training CEN.
epochs = 10000
batch_size = 500
early_stopping_threshold = 500 * 60
min_delta_levels = [0.001, 0.001]
patience_levels = [90, 30]
learning_rate = 0.0001

# Settings for saving results.
n_params_EvalTrain = 3  # Only 'a', 'b', and'p' are evaluated.
n_params_EvalTest = 2  # Only 'a', and'b' are evaluated.
n_metrics = 5

# Whether to visualize the layouts of CEN and its subnets.
show_model_layout = True
# Whether to show the details of the training process.
verbose = 2


simulation3 = Simulation3(
    n_persons_levels,
    n_items_levels,
    n_persons_new,
    n_items_new,
    n_reps,
    net_depth_levels,
    linear_nonlinear,
    epochs,
    batch_size,
    early_stopping_threshold,
    min_delta_levels,
    patience_levels,
    learning_rate,
    n_params_EvalTrain,
    n_params_EvalTest,
    n_metrics,
    show_model_layout,
    verbose,
)
simulation3.run()
