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
"""Python script for conducting Simulation Study 4 suggested by the 
esteemed reviewers of Behavioral Research Method.

Reviewer's Recommendations: "The authors consider a scenario where their
model is trained on a part on the data and then applied to another part
of the data. While this is a realistic scenario and an interesting
analysis, please also include a scenario where the distributions of the
model parameters slightly differ between these training and test
samples. I think this would represent another plausible scenario.
Consider, for instance, an application where we first train our model on
a sample, but apply it to a sample where the average test taker is
either slightly more able or slightly less able. Would your method still
lead to useful estimates? What if I applied my trained network to new
items which are slightly more difficult or show slightly higher
discrimination?"

Our Approach: Upon the reviewer's request for a supplementary simulation
study to assess the robustness of the CEN when new persons or items are
drawn from distributions distinct from those in the training dataset,
this script incorporates two main sections simulating the following
scenarios:
    1) In contrast to the original person parameters 'z' drawn from the
       standard normal distribution N(0, 1), the first section explores
       parameters 'z' drawn from a population of more "able" test
       takers, characterized by N(0.5, 1). 
    2) Deviating from the original item parameters 'a' drawn from the
       uniform distribution U(0.2, 2) and 'b' drawn from N(0, 1), the
       second section considers parameters 'a' drawn from a distribution
       of less "discriminating" items, characterized by U(0.2, 1), or
       'b' drawn from a distribution of more 'difficult' items,
       characterized by N(0.5, 1). This results in three specific
       sub-scenarios:
            2.1) 'a' drawn from the different distribution U(0.2, 1) and
                'b' drawn from the same distribution N(0, 1).
            2.2) 'a' drawn from the same distribution U(0.2, 2) and 'b'
                drawn from the different distribution N(0.5, 1).
            2.3) Both 'a' and 'b' drawn from different distributions
                U(0.2, 1) and N(0.5, 1), respectively.

        Note that items with smaller discrimination parameters can pose
        greater challenges for estimation than those with larger values,
        the parameters 'a' were thus drawn from a less "discriminating"
        distribution U(0.2, 1) to thoroughly assess the robustness of
        the CEN method.

The trained PN and IN obtained in Simulation Study 1 are leveraged in
this study to assess their adaptability to these new scenarios and
evaluate their performance on datasets featuring parameters from
different distributions.

Due to space constraints in the paper, we focus on the test setting with
n_persons=1000 and n_items=90.

Acknowledgment: We would like to express our sincere gratitude to the
reviewer for their valuable feedback and guidance, which has
significantly contributed to refining this research.
"""
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

# Add the "src" directory to sys.path, which makes sure the interpreter
# accesses the required modules appropriately.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from data.generator import ResponseData
from models.cen import CEN
from utils import utils

start = time.time()


def get_z_different(n_persons):
    """Generate the person parameters 'z' from N(0.5, 1)."""
    z = np.zeros(n_persons)
    i = 0
    while i < n_persons:
        z[i] = np.random.normal(loc=0.5, scale=1)
        if np.abs(z[i]) < 3:
            i += 1

    return z


def get_a_different(n_items):
    """Generate the item discrimination parameters 'a' from U(0.2, 1)."""
    a = np.random.uniform(0.2, 1, n_items)

    return a


def get_b_different(n_items):
    """Generate the item difficulty parameters 'b' from N(0.5, 1)."""
    b = np.zeros(n_items)
    j = 0
    while j < n_items:
        b[j] = np.random.normal(loc=0.5, scale=1)
        if np.abs(b[j]) < 3:
            j += 1

    return b


def generate_res_mat_new(
    scenario_name,
    n_persons,
    n_items,
    z_true,
    a_true,
    b_true,
):
    response_data_new = ResponseData(
        n_persons=n_persons,
        n_items=n_items,
        z=z_true,
        a=a_true,
        b=b_true,
    )
    res_mat_new = response_data_new.res_mat

    np.savetxt(
        os.path.join(path_to_res_mat_new, f"res_mat_new_{scenario_name}.csv"),
        res_mat_new,
        delimiter=",",
    )

    return res_mat_new


def run_DDP(
    scenario_name,
    cen,
    z_true_new,
    res_mat_new,
):
    """Run the supplementary study in the new dataset associated with a
    sample of persons drawn from a different distribution.

    The suffix 'DDP' indicates "the new Persons drawn for a Different
    Distribution".

    Args:
        scenario_name (str): Name of the scenarios, i.e., 'different_z',
            in this scenario.
        cen: The CEN class after the trained PN is loaded.
        z_true_new (numpy.ndarray): The parameters 'z' of new dataset.
        res_mat_new (numpy.ndarray): New response matrix associated with
            'z_true_new'.

    Return:
        numpy.ndarray: An array containing the correlation coefficients
        (Cor), bias (Bias), mean absolute error (MAE), mean squared
        error (MSE), and root mean squared error (RMSE) between the
        actual and estimated values of new 'z_s', with type 'float'.
    """

    # Estimate the new ability parameters using the trained PN.
    z_est_new = cen.param_est(param="z", res_mat=res_mat_new)
    np.savetxt(
        os.path.join(path_to_estimates, f"z_est_{scenario_name}.csv"),
        z_est_new,
        delimiter=",",
    )

    z_metrics_new = cen._compute_distance(z_true_new, z_est_new)

    return z_metrics_new


def run_DDI(
    scenario_name,
    cen,
    a_true_new,
    b_true_new,
    res_mat_new,
):
    """Run the supplementary study in the new dataset associated with a
    sample of items drawn from different distributions.

    The suffix 'DDI' indicates "the new Items drawn for a Different
    Distribution".

    Args:
        scenario_name (str): Name of the scenarios, e.g., "different_a",
            which stands for different distribution of the parameters
            'a'.
        cen: The CEN class after the trained IN is loaded.
        a_true_new (numpy.ndarray): The parameters 'a' of new dataset.
        b_true_new (numpy.ndarray): The parameters 'b' of new dataset.
        res_mat_new (numpy.ndarray): New response matrix associated with
            'a_true_new' and 'b_true_new'.
    Return:
        numpy.ndarray: An array containing the correlation coefficients
        (Cor), bias (Bias), mean absolute error (MAE), mean squared
        error (MSE), and root mean squared error (RMSE) between the
        actual and estimated values of new 'a_s' and 'b_s', with type
        'float'.
    """

    # Estimate the new difficulty and discrimination parameters using
    # the trained IN, with each column of the new response matrix
    # serving as the inputs.
    a_est_new = cen.param_est(param="a", res_mat=res_mat_new)
    b_est_new = cen.param_est(param="b", res_mat=res_mat_new)

    np.savetxt(
        os.path.join(path_to_estimates, f"a_est_{scenario_name}.csv"),
        a_est_new,
        delimiter=",",
    )
    np.savetxt(
        os.path.join(path_to_estimates, f"b_est_{scenario_name}.csv"),
        b_est_new,
        delimiter=",",
    )

    a_metrics_new = cen._compute_distance(a_true_new, a_est_new)
    b_metrics_new = cen._compute_distance(b_true_new, b_est_new)

    return np.vstack([a_metrics_new, b_metrics_new])


def adorn_metrics(metrics_mat):
    """Improve the appearance of the recovery metrics."""
    metrics_mat = pd.DataFrame(metrics_mat)
    metrics_mat.columns = ["Cor", "Bias", "MAE", "MSE", "RMSE"]
    metrics_mat.index = [
        "z_different_z",
        "a_different_a",
        "b_different_a",
        "a_different_b",
        "b_different_b",
        "a_different_ab",
        "b_different_ab",
    ]
    return metrics_mat


# ======================================================================
# Settings for the Supplementary Simulation Study
# ======================================================================

# Settings for the test.
n_persons = 1000
n_items = 90
n_reps = 100
n_new_persons = 100
n_new_items = 30

# Settings for saving results.
n_params_evaluated = 3
n_metrics = 5

# Settings for building CEN.
net_depth_levels = [1, 3]
linear_nonlinear = [False]

n_CEN_versions = len(net_depth_levels) * len(linear_nonlinear)

# Container for saving results in repetitions.
# (n_params_evaluated - 1) * 3 + 1 = 7, which indicates one row for
# Scenario 1, and two rows for each of the three sub-scenarios in
# Scenario 2.
metrics_reps = np.zeros(
    (n_reps, n_CEN_versions, (n_params_evaluated - 1) * 3 + 1, n_metrics)
)


# ======================================================================
# The main process of this study
# ======================================================================

for rep in range(n_reps):

    # Load the actual person and item parameters used for data
    # generation in Simulation Study 1, which would be used to generate
    # the new response matrices.
    path_EvalTrain = (
        f"./data/n_persons_{n_persons}/n_items_{n_items}/rep_{rep}/EvalTrain"
    )
    z_true = np.loadtxt(
        os.path.join(path_EvalTrain, "z_true.csv"),
        dtype="float",
        delimiter=",",
    )
    a_true = np.loadtxt(
        os.path.join(path_EvalTrain, "a_true.csv"),
        dtype="float",
        delimiter=",",
    )
    b_true = np.loadtxt(
        os.path.join(path_EvalTrain, "b_true.csv"),
        dtype="float",
        delimiter=",",
    )

    # Load parameters of the new persons and items (saved in the
    # 'EvalTest' folder), which were drawn from the distributions
    # consistent with those of the 'EvalTrain' task, Simulation Study 1.
    path_EvalTest = f"./data/n_persons_{n_persons}/n_items_{n_items}/rep_{rep}/EvalTest"

    res_mat_new_persons = np.loadtxt(
        os.path.join(path_EvalTest, "res_mat_new_persons.csv"),
        dtype="int",
        delimiter=",",
    )
    z_true_same = np.loadtxt(
        os.path.join(path_EvalTest, "z_true_new.csv"),
        dtype="float",
        delimiter=",",
    )
    res_mat_new_items = np.loadtxt(
        os.path.join(path_EvalTest, "res_mat_new_items.csv"),
        dtype="int",
        delimiter=",",
    )
    a_true_same = np.loadtxt(
        os.path.join(path_EvalTest, "a_true_new.csv"),
        dtype="float",
        delimiter=",",
    )
    b_true_same = np.loadtxt(
        os.path.join(path_EvalTest, "b_true_new.csv"),
        dtype="float",
        delimiter=",",
    )

    z_true_different = get_z_different(n_persons=n_new_persons)
    a_true_different = get_a_different(n_items=n_new_items)
    b_true_different = get_b_different(n_items=n_new_items)

    path_to_rep = (
        f"./results/suppl_sim_study/n_persons_{n_persons}/n_items_{n_items}/rep_{rep}"
    )

    path_to_params = os.path.join(path_to_rep, "params")
    path_to_res_mat_new = os.path.join(path_to_rep, "res_mat_new")
    utils.make_dir(path_to_params)
    utils.make_dir(path_to_res_mat_new)

    np.savetxt(
        os.path.join(path_to_params, f"z_true_same.csv"),
        z_true_same,
        delimiter=",",
    )
    np.savetxt(
        os.path.join(path_to_params, f"z_true_different.csv"),
        z_true_different,
        delimiter=",",
    )

    np.savetxt(
        os.path.join(path_to_params, f"a_true_same.csv"),
        a_true_same,
        delimiter=",",
    )
    np.savetxt(
        os.path.join(path_to_params, f"a_true_different.csv"),
        a_true_different,
        delimiter=",",
    )
    np.savetxt(
        os.path.join(path_to_params, f"b_true_same.csv"),
        b_true_same,
        delimiter=",",
    )
    np.savetxt(
        os.path.join(path_to_params, f"b_true_different.csv"),
        b_true_different,
        delimiter=",",
    )

    # Generate the new response matrices for all the conditions within
    # one repetition.
    res_mat_new_different_z = generate_res_mat_new(
        scenario_name="different_z",
        n_persons=n_new_persons,
        n_items=n_items,
        z_true=z_true_different,
        a_true=a_true,
        b_true=b_true,
    )

    res_mat_new_different_a = generate_res_mat_new(
        scenario_name="different_a",
        n_persons=n_persons,
        n_items=n_new_items,
        z_true=z_true,
        a_true=a_true_different,
        b_true=b_true_same,
    )

    res_mat_new_different_b = generate_res_mat_new(
        scenario_name="different_b",
        n_persons=n_persons,
        n_items=n_new_items,
        z_true=z_true,
        a_true=a_true_same,
        b_true=b_true_different,
    )

    res_mat_new_different_ab = generate_res_mat_new(
        scenario_name="different_ab",
        n_persons=n_persons,
        n_items=n_new_items,
        z_true=z_true,
        a_true=a_true_different,
        b_true=b_true_different,
    )

    CEN_version_indicator = 0
    for net_depth, linear in itertools.product(net_depth_levels, linear_nonlinear):
        # Initialize an instance of CEN, the network weights saved in Simulation
        # Study 1 wold be loaded onto it later.
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

        path_to_linear = os.path.join(
            path_to_rep, f"net_depth_{net_depth}/linear_{linear}"
        )

        path_to_estimates = os.path.join(path_to_linear, "estimates")
        path_to_metrics = os.path.join(path_to_linear, "metrics")
        utils.make_dir(path_to_estimates)
        utils.make_dir(path_to_metrics)

        # Get estimates in the "same" condition for drawing the mapping
        # plot.
        z_est_same_z = cen.param_est(param="z", res_mat=res_mat_new_persons)
        np.savetxt(
            os.path.join(path_to_estimates, f"z_est_same_z.csv"),
            z_est_same_z,
            delimiter=",",
        )

        a_est_same_ab = cen.param_est(param="a", res_mat=res_mat_new_items)
        b_est_same_ab = cen.param_est(param="b", res_mat=res_mat_new_items)
        np.savetxt(
            os.path.join(path_to_estimates, f"a_est_same_ab.csv"),
            a_est_same_ab,
            delimiter=",",
        )
        np.savetxt(
            os.path.join(path_to_estimates, f"b_est_same_ab.csv"),
            b_est_same_ab,
            delimiter=",",
        )

        # ------------------------------------------------------------------
        # Code snippet for exploring Scenario 1
        # ------------------------------------------------------------------
        metrics_different_z = run_DDP(
            scenario_name="different_z",
            cen=cen,
            z_true_new=z_true_different,
            res_mat_new=res_mat_new_different_z,
        )

        # ------------------------------------------------------------------
        # Code snippets for exploring Scenario 2.1, 2.2, and 2.3
        # ------------------------------------------------------------------

        # Different distribution of 'a' is considered (2.1).
        metrics_different_a = run_DDI(
            scenario_name="different_a",
            cen=cen,
            a_true_new=a_true_different,
            b_true_new=b_true_same,
            res_mat_new=res_mat_new_different_a,
        )

        # Different distribution of 'b' is considered (2.2).
        metrics_different_b = run_DDI(
            scenario_name="different_b",
            cen=cen,
            a_true_new=a_true_same,
            b_true_new=b_true_different,
            res_mat_new=res_mat_new_different_b,
        )

        # Different distributions of both 'a' and 'b' are considered (2.3).
        metrics_different_ab = run_DDI(
            scenario_name="different_ab",
            cen=cen,
            a_true_new=a_true_different,
            b_true_new=b_true_different,
            res_mat_new=res_mat_new_different_ab,
        )

        # Save the metrics of recovery of each repetition.
        metrics = np.vstack(
            [
                metrics_different_z,
                metrics_different_a,
                metrics_different_b,
                metrics_different_ab,
            ]
        )
        metrics = adorn_metrics(metrics)
        metrics.to_csv(os.path.join(path_to_metrics, "metrics.csv"))

        # revise here
        metrics_reps[
            rep,
            CEN_version_indicator,
        ] = metrics

        print(
            f"Supplementary study, current trial: n_persons: {n_persons}, n_items: {n_items}, rep: {rep}, net_depth: {net_depth}, linear: {linear}"
        )
        print("Recovery metrics of the current trial:")
        print(metrics)
        print("\n\n")

        CEN_version_indicator += 1


metrics_mean_versions = np.mean(metrics_reps, 0)
metrics_sem_versions = stats.sem(metrics_reps, 0)

CEN_version_indicator = 0
for net_depth, linear in itertools.product(net_depth_levels, linear_nonlinear):
    # Save the final results.
    path_to_summary = f"./results_summary/suppl_sim_study/n_persons_{n_persons}/n_items_{n_items}/net_depth_{net_depth}/linear_{linear}/summary"
    utils.make_dir(path_to_summary)

    metrics_mean = metrics_mean_versions[CEN_version_indicator]
    metrics_mean = adorn_metrics(metrics_mean)
    metrics_mean.to_csv(os.path.join(path_to_summary, "metrics_mean.csv"))

    metrics_sem = metrics_sem_versions[CEN_version_indicator]
    metrics_sem = adorn_metrics(metrics_sem)
    metrics_sem.to_csv(os.path.join(path_to_summary, "metrics_sem.csv"))

    CEN_version_indicator += 1


elapsed_in_seconds = time.time() - start
elapsed_formatted = utils.format_time_elapsed(elapsed_in_seconds)
print(f"Time spent: {elapsed_formatted}")
