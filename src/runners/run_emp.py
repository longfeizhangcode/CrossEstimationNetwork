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
"""Run the empirical study."""
import os
import sys

import numpy as np

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

# Add the "src" directory to sys.path, which makes sure the interpreter
# accesses the required modules appropriately.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from models.cen import CEN
from utils import utils


path_emp_data = "./data/VIQT_scores.csv"
res_mat = np.loadtxt(path_emp_data, skiprows=1, dtype="int", delimiter=",")

cen = CEN(
    inp_size_person_net=res_mat.shape[1],
    inp_size_item_net=res_mat.shape[0],
    person_net_depth=1,
    item_net_depth=1,
    linear=False,
    show_model_layout=True,
)

cen.load_data(
    res_mat=res_mat,
    res_prob_mat=None,
)

optimizer = Adam(learning_rate=0.0001)
loss_func = BinaryCrossentropy()
early_stopping = EarlyStopping(
    monitor="loss",
    min_delta=0.001,
    patience=100,
    mode="min",
    restore_best_weights=True,
)

cen.train(
    optimizer=optimizer,
    loss_func=loss_func,
    epochs=10000,
    batch_size=500,
    early_stopping=early_stopping,
    verbose=2,
)

z_est = cen.person_net(res_mat).numpy()[:, 0]
a_est = cen.item_net(res_mat.transpose()).numpy()[:, 0]
b_est = cen.item_net(res_mat.transpose()).numpy()[:, 1]

path_emp_est = "./results/emp_study"
utils.make_dir(path_emp_est)
np.savetxt(os.path.join(path_emp_est, "z_est_cen.csv"), z_est, delimiter=",")
np.savetxt(os.path.join(path_emp_est, "a_est_cen.csv"), a_est, delimiter=",")
np.savetxt(os.path.join(path_emp_est, "b_est_cen.csv"), b_est, delimiter=",")
