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
"""Compute goodness-of-fit index (-2 * Log_Likelihood) for the EvalTrain task, 
Simulation Study 1.
"""
import itertools
import os
import sys
import time

import numpy as np


# Add the "src" directory to sys.path, which makes sure the interpreter
# accesses the required modules appropriately.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from models.cen import CEN
from utils import utils


start = time.time()

def compute_gof(binary, prob):
    """Compute goodness-of-fit index (-2LL).

    Args:
        binary (np.ndarray): The true binary response matrix.
        prob (np.ndarray): The predicted response probability matrix.

    Returns:
        float: The -2LL value of the model-data fit.
    """
    log_lik_correct = np.log(prob)
    log_lik_wrong = np.log(1 - prob)
    log_lik_1D = np.where(binary, log_lik_correct, log_lik_wrong)
    log_lik_sum = np.sum(log_lik_1D)
    gof = -2 * log_lik_sum

    return gof


# ======================================================================
# Settings for this analysis
# ======================================================================
n_persons_levels = [100, 500, 1000]
n_items_levels = [30, 60, 90]
n_reps = 100

net_depth_levels = [1, 3]
linear_nonlinear = [True, False]

table_gof_mean = np.zeros((1, 4))
table_gof_sd = np.zeros((1, 4))


# ======================================================================
# Main process of this analysis
# ======================================================================
for n_persons, n_items in itertools.product(n_persons_levels, n_items_levels):

    gof_all_reps = np.full(
        shape=(n_reps, 4),
        fill_value=np.nan,
    )

    for rep in range(n_reps):

        path_EvalTrain = (
            f"./data/n_persons_{n_persons}/n_items_{n_items}/rep_{rep}/EvalTrain"
        )

        res_mat = np.loadtxt(
            os.path.join(path_EvalTrain, "res_mat.csv"),
            dtype="int",
            delimiter=",",
        )
        res_mat_1D = res_mat.ravel()

        count = 0
        for net_depth, linear in itertools.product(net_depth_levels, linear_nonlinear):
            # Initialize an instance of CEN, the network weights saved in
            # Simulation Study 1 wold be loaded onto it later.
            cen = CEN(
                inp_size_person_net=n_items,
                inp_size_item_net=n_persons,
                person_net_depth=net_depth,
                item_net_depth=net_depth,
                linear=linear,
                show_model_layout=False,
            )

            # Restore the weights of the trained PN and IN in Simulation Study 1.
            path_models = f"./models/simulation_study1/n_persons_{n_persons}/n_items_{n_items}/rep_{rep}/net_depth_{net_depth}/linear_{linear}"
            cen.person_net.load_weights(os.path.join(path_models, "person_net/person_net"))
            cen.item_net.load_weights(os.path.join(path_models, "item_net/item_net"))

            cen.load_data(res_mat=res_mat, res_prob_mat=None)
            cen._get_X_person_net()
            cen._get_X_item_net()

            # Obtain the predicted probability matrix of correct responses,
            # tuned into a 1D array.
            res_prob_mat_cen_1D = cen.combined([cen.X_person_net, cen.X_item_net]).numpy()

            gof_cen = compute_gof(binary=res_mat_1D, prob=res_prob_mat_cen_1D)

            gof_all_reps[rep, count] = gof_cen

            count += 1

        print(f"n_persons: {n_persons}, n_items: {n_items}, rep: {rep}, end")

    table_gof_mean = np.vstack((table_gof_mean, np.nanmean(gof_all_reps, 0)))
    table_gof_sd = np.vstack((table_gof_sd, np.nanstd(gof_all_reps, 0)))


table_gof_mean = table_gof_mean[1:]
table_gof_sd = table_gof_sd[1:]

path_to_summary = (
    f"./results_summary/gof"
)
utils.make_dir(path_to_summary)

np.savetxt(
    os.path.join(path_to_summary, "table_gof_mean.csv"),
    table_gof_mean,
    delimiter=",",
)
np.savetxt(
    os.path.join(path_to_summary, "table_gof_sd.csv"),
    table_gof_sd,
    delimiter=",",
)

elapsed_in_seconds = time.time() - start
elapsed_formatted = utils.format_time_elapsed(elapsed_in_seconds)
print(f"Time spent: {elapsed_formatted}")