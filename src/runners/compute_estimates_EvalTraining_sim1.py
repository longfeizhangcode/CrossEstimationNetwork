"""Get the CEN_1H_NL estimates in test setting of N=1000 and M=90, 
EvalTraining task, Simulation Study 1.

Additionally, the parameters of that setting would be copied into a new 
path where the estimates are situated.
"""

import os
import sys

import numpy as np

# Add the "src" directory to sys.path, which makes sure the interpreter
# accesses the required modules appropriately.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from models.cen import CEN
from utils import utils


# Settings for the test.
n_persons = 1000
n_items = 90
n_reps = 100

# Settings for building CEN.
linear = False
net_depth = 1

cen = CEN(
    inp_size_person_net=n_items,
    inp_size_item_net=n_persons,
    person_net_depth=net_depth,
    item_net_depth=net_depth,
    linear=linear,
    show_model_layout=False,
)

for rep in range(n_reps):
    # Restore the weights of the trained PN and IN in Simulation Study 1.
    path_models = f"./models/simulation_study1/n_persons_{n_persons}/n_items_{n_items}/rep_{rep}/net_depth_{net_depth}/linear_{linear}"
    cen.person_net.load_weights(os.path.join(path_models, "person_net/person_net"))
    cen.item_net.load_weights(os.path.join(path_models, "item_net/item_net"))

    path_EvalTrain = (
        f"./data/n_persons_{n_persons}/n_items_{n_items}/rep_{rep}/EvalTrain"
    )
    res_mat = np.loadtxt(
        os.path.join(path_EvalTrain, "res_mat.csv"),
        dtype="int",
        delimiter=",",
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

    z_est = cen.param_est(param="z", res_mat=res_mat)
    a_est = cen.param_est(param="a", res_mat=res_mat)
    b_est = cen.param_est(param="b", res_mat=res_mat)


    path_to_rep = (
        f"./results/EvalTrain_sim1_params_estimates/n_persons_{n_persons}/n_items_{n_items}/rep_{rep}/net_depth_{net_depth}/linear_{linear}"
    )

    utils.make_dir(path_to_rep)

    np.savetxt(
        os.path.join(path_to_rep, f"z_true.csv"),
        z_true,
        delimiter=",",
    )
    np.savetxt(
        os.path.join(path_to_rep, f"a_true.csv"),
        a_true,
        delimiter=",",
    )
    np.savetxt(
        os.path.join(path_to_rep, f"b_true.csv"),
        b_true,
        delimiter=",",
    )

    np.savetxt(
        os.path.join(path_to_rep, f"z_est.csv"),
        z_est,
        delimiter=",",
    )
    np.savetxt(
        os.path.join(path_to_rep, f"a_est.csv"),
        a_est,
        delimiter=",",
    )
    np.savetxt(
        os.path.join(path_to_rep, f"b_est.csv"),
        b_est,
        delimiter=",",
    )