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
"""Class for conducting Simulation Study 1.

Class:
    Simulation1: settings and implementing details of Simulation Study 1.
"""
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

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


class Simulation1:
    """The class of Simulation Study 1 designed to evaluate the
    performance of the standard CEN, which estimates both the person and
    item parameters.

    The study explores the impact of two factors: the number of persons
    (n_persons) and the number of items (n_items). Each test setting is
    repeated 'n_reps' times to minimize random error and to ensure the
    validity of the findings. Four versions of CEN are considered:
    'CEN_1H_L', 'CEN_1H_NL', 'CEN_3H_L', and 'CEN_3H_NL', where 'H'
    represents the number of hidden layers, and (N)L indicates Linearity
    or NonLinearity modeled in the subnets. In this study, the early
    stopping trick (implemented in the '_determine_early_stopping'
    method) is employed to halt the training process timely and
    appropriately.

    Attributes:
        n_persons_levels (list): A list that contains the different
            levels of the number of persons, with type 'int'.
        n_items_levels (list): A list that contains the different
            levels of the number of items, with type 'int'.
        n_persons_new (int): Number of the new persons in the EvalTest
            task.
        n_items_new (int): Number of the new items in the EvalTest task.
        n_reps (int): Number of repetitions under each test setting (a
            combination of n_person and n_item).
        learning_rate(float): Learning rate of the optimizer 'Adam'.
        epochs (int): Number of epochs for training CEN.
        batch_size (int): Batch size for training CEN.
        early_stopping_threshold (int): Cutoff point for choosing
            appropriate 'min_delta' and 'patience' values according to
            the test setting.
        min_delta_levels (list): A list including different 'min_delta'
            values for training CEN, with type 'float'.
        patience_levels (list): A list including different patience
            numbers for training CEN, with type 'float'.
        net_depth_levels (list): A list that contains different levels
            of network depth with type 'float'.
        linear_nonlinear (list):  A list that contains the booleans
            '[True, False]', with type 'float'.
        CEN_versions_names (list): The names of CEN versions in the
            order with regard to 'net_depth_levels' and
            'linear_nonlinear', with type 'str'.
        n_CEN_versions (int): Number of CEN versions in the study.
        n_params_EvalTrain (int): Number of parameters to be evaluated
            in the EvalTrain task.
        n_params_EvalTest (int): Number of parameters to be evaluated
            in the EvalTest task.
        n_metrics (int): Number of metrics for evaluating the
            performance of CEN.
        show_model_layout (bool): Whether to visualize the layouts of
            CEN and its subnets.
        verbose (int):  0, 1, or 2. Verbosity mode during training.
            0 = silent, 1 = progress bar, 2 = single line. verbose=2 is
            recommended when not running interactively.
    """

    def __init__(
        self,
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
    ):
        self.n_persons_levels = n_persons_levels
        self.n_items_levels = n_items_levels
        self.n_persons_new = n_persons_new
        self.n_items_new = n_items_new
        self.n_reps = n_reps

        self.net_depth_levels = net_depth_levels
        self.linear_nonlinear = linear_nonlinear

        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_threshold = early_stopping_threshold
        self.min_delta_levels = min_delta_levels
        self.patience_levels = patience_levels
        self.learning_rate = learning_rate

        self.n_params_EvalTrain = n_params_EvalTrain
        self.n_params_EvalTest = n_params_EvalTest
        self.n_metrics = n_metrics

        self.show_model_layout = show_model_layout
        self.verbose = verbose

        self.n_CEN_versions = len(net_depth_levels) * len(linear_nonlinear)
        self.CEN_versions_names = self._get_CEN_versions_names(
            net_depth_levels, linear_nonlinear
        )

    def run(self):
        """Run the simulation study.

        There are five loops within this method:
            1) n_persons_levels;
            2) n_items_levels;
            3) repetitions;
            4) net_depth_levels (number of hidden layers in subnets);
            5) linearity or nonlinearity modeled in subnets.

        The first two determine the number of test settings
        (combinations of n_persons and n_items). The last two determine
        the number of CEN versions. The third determines the number of
        implementation of CEN under each test setting.
        """
        start = time.time()

        (
            simulation_indicator_Upper_Case,
            simulation_indicator_lower_case,
        ) = self._get_simulation_indicator()

        for n_persons, n_items in itertools.product(
            self.n_persons_levels, self.n_items_levels
        ):
            (
                metrics_EvalTrain_reps,
                metrics_EvalTest_reps,
                n_cnvg_reps,
            ) = self._initialize_result_containers()

            for rep in range(self.n_reps):
                # Retrieve all the required data of a single repetition
                # under a test setting.
                (
                    res_mat,
                    res_prob_mat,
                    z_true,
                    a_true,
                    b_true,
                    res_mat_new_persons,
                    res_prob_mat_new_persons,
                    res_mat_new_items,
                    res_prob_mat_new_items,
                    z_true_new,
                    a_true_new,
                    b_true_new,
                ) = self._retrieve_data(n_persons, n_items, rep)

                # Iterate through all the versions of CEN.
                CEN_version_indicator = 0
                for net_depth, linear in itertools.product(
                    self.net_depth_levels, self.linear_nonlinear
                ):
                    # Configure the input size for the person net and
                    # item net of CEN. In Simulation Study 2, the input
                    # size of the item net should be set to 2, as the
                    # two item parameters 'a' and 'b' will be fed into
                    # the item net. In Simulation Study 3, the input
                    # size of the person net should be set to 1, as only
                    # the person parameter 'z' will be fed into the
                    # person net.
                    inp_size_person_net, inp_size_item_net = self._get_inp_size(
                        n_persons, n_items
                    )

                    # Selecting the appropriate class for each
                    # simulation study. Based on the simulation studies,
                    # you need to choose the appropriate class for each
                    # study:
                    # For Simulation Study 1, use the class 'CEN'.
                    # For Simulation Study 2, use the class 'CENz'.
                    # For Simulation Study 3, use the class 'CENab'.
                    # 'CEN' here would be written by 'CENz' and 'CENab'
                    # in the latter two studies.
                    CEN = self._get_CEN_class()

                    # Instantiate an object from the chosen class.
                    cen = CEN(
                        inp_size_person_net,
                        inp_size_item_net,
                        net_depth,
                        net_depth,
                        linear,
                        self.show_model_layout,
                    )

                    # Load data into the object.
                    cen.load_data(
                        res_mat,
                        res_prob_mat,
                        z_true,
                        a_true,
                        b_true,
                        res_mat_new_persons,
                        res_prob_mat_new_persons,
                        res_mat_new_items,
                        res_prob_mat_new_items,
                        z_true_new,
                        a_true_new,
                        b_true_new,
                    )

                    # Compile the CEN model.
                    optimizer = Adam(learning_rate=self.learning_rate)
                    loss_func = BinaryCrossentropy()
                    early_stopping = self._determine_early_stopping(n_persons, n_items)

                    # Train the CEN model.
                    cen.train(
                        optimizer,
                        loss_func,
                        self.epochs,
                        self.batch_size,
                        early_stopping,
                        self.verbose,
                    )

                    print("\n\n\n")
                    print(
                        f"{simulation_indicator_Upper_Case}, current trial: n_persons: {n_persons}, n_items: {n_items}, rep: {rep}, net_depth: {net_depth}, linear: {linear}, start"
                    )

                    # Create folder for saving the trained CEN model.
                    path_models = f"./models/{simulation_indicator_lower_case}/n_persons_{n_persons}/n_items_{n_items}/rep_{rep}/net_depth_{net_depth}/linear_{linear}"
                    utils.make_dir(path_models)
                    # cen.combined.save_weights(os.path.join(path_models, "combined/combined"))
                    cen.person_net.save_weights(
                        os.path.join(path_models, "person_net/person_net")
                    )
                    cen.item_net.save_weights(
                        os.path.join(path_models, "item_net/item_net")
                    )

                    # Create folder for saving the performance of the
                    # current CEN version within a repetition.
                    path_to_linear = f"./results/{simulation_indicator_lower_case}/n_persons_{n_persons}/n_items_{n_items}/rep_{rep}/net_depth_{net_depth}/linear_{linear}"
                    utils.make_dir(path_to_linear)

                    # Obtain the results of the current CEN version
                    # within a repetition.
                    metrics_EvalTrain = cen.evaluate_training_dataset()
                    metrics_EvalTest = cen.evaluate_test_dataset()
                    loss_values = cen.loss_values
                    n_cnvg = cen.n_cnvg

                    cen.print_results()

                    # Save the results of the current CEN version within
                    # a repetition.
                    metrics_EvalTrain_path = os.path.join(
                        path_to_linear, "metrics_EvalTrain.csv"
                    )
                    metrics_EvalTrain.to_csv(metrics_EvalTrain_path)
                    metrics_EvalTest_path = os.path.join(
                        path_to_linear, "metrics_EvalTest.csv"
                    )
                    metrics_EvalTest.to_csv(metrics_EvalTest_path)
                    loss_values_path = os.path.join(path_to_linear, "loss_values.csv")
                    np.savetxt(loss_values_path, loss_values, delimiter=",")

                    # Store the results of the current CEN version
                    # within a repetition to the containers for all the
                    # repetitions.
                    metrics_EvalTrain_reps[
                        rep, CEN_version_indicator
                    ] = metrics_EvalTrain
                    metrics_EvalTest_reps[rep, CEN_version_indicator] = metrics_EvalTest
                    n_cnvg_reps[rep, CEN_version_indicator] = n_cnvg

                    print("")
                    print(
                        f"{simulation_indicator_Upper_Case}, current trial: n_persons: {n_persons}, n_items: {n_items}, rep: {rep}, net_depth: {net_depth}, linear: {linear}, end"
                    )
                    print("\n\n\n")

                    CEN_version_indicator += 1

            # Create folder to save the performance of all the CEN
            # versions and all the repetitions under a test setting.
            path_to_summary = f"./results_summary/{simulation_indicator_lower_case}/n_persons_{n_persons}/n_items_{n_items}/summary"
            utils.make_dir(path_to_summary)

            self._save_metrics_summary(
                path_to_summary, "EvalTrain", metrics_EvalTrain_reps
            )
            self._save_metrics_summary(
                path_to_summary, "EvalTest", metrics_EvalTest_reps
            )
            self._save_n_cnvg_reps(n_cnvg_reps, path_to_summary)

        elapsed_in_seconds = time.time() - start
        elapsed_formatted = utils.format_time_elapsed(elapsed_in_seconds)
        print(f"Time spent in {simulation_indicator_Upper_Case}: {elapsed_formatted}")
        utils.save_to_file(
            f"./results/{simulation_indicator_lower_case}/time_spent.txt",
            f"Time spent: {elapsed_formatted}",
        )

    def _get_simulation_indicator(self):
        """Return the indicator for the present simulation study."""
        return "Simulation Study 1", "simulation_study1"

    def _get_inp_size(self, n_persons, n_items):
        """Determine the input sizes for the person net and item net.

        This method is necessary since 'inp_size_person_net' is not
        identical to 'n_items' in Simulation Study 2, and
        'inp_size_item_net' is not identical to 'n_persons' in
        Simulation Study 3.
        """
        inp_size_person_net = n_items
        inp_size_item_net = n_persons

        return inp_size_person_net, inp_size_item_net

    def _get_CEN_class(self):
        """Return the CEN class (choosing from 'CEN', 'CENz', or
        'CENab') for the present stimulation study.

        The 'CEN' class from models.cen is used for Simulation Study 1.
        """
        return CEN

    def _get_CEN_versions_names(self, net_depth_levels, linear_nonlinear):
        """Get the names of the CEN versions in the order with regard to
        'net_depth_levels' and 'linear_nonlinear'."""
        CEN_versions_names = []

        for net_depth, linear in itertools.product(net_depth_levels, linear_nonlinear):
            CEN_versions_names.append(f"CEN_{net_depth}h_{'n' if not linear else ''}l")

        return CEN_versions_names

    def _determine_early_stopping(self, n_persons, n_items):
        """Determine the early stopping rules for the training process."""
        if n_persons * n_items < self.early_stopping_threshold:
            min_delta = self.min_delta_levels[0]
            patience = self.patience_levels[0]
        else:
            min_delta = self.min_delta_levels[1]
            patience = self.patience_levels[1]

        return EarlyStopping(
            monitor="loss",
            min_delta=min_delta,
            patience=patience,
            mode="min",
            restore_best_weights=True,
        )

    def _initialize_result_containers(self):
        """Initialize the containers used for recording the results
        of all the repetitions under a test setting.

        The containers will not be saved to the disk, they are for
        provisional use to compute the summary of results across all the
        repetitions under a test setting.
        """
        metrics_EvalTrain_reps = np.zeros(
            (
                self.n_reps,
                self.n_CEN_versions,
                self.n_params_EvalTrain,
                self.n_metrics,
            )
        )
        metrics_EvalTest_reps = np.zeros(
            (
                self.n_reps,
                self.n_CEN_versions,
                self.n_params_EvalTest,
                self.n_metrics,
            )
        )
        n_cnvg_reps = np.zeros((self.n_reps, self.n_CEN_versions))

        return (
            metrics_EvalTrain_reps,
            metrics_EvalTest_reps,
            n_cnvg_reps,
        )

    def _retrieve_data(self, n_persons, n_items, rep):
        """Retrieve the data required to train and evaluate CEN within
        a single repetition."""
        path_EvalTrain = (
            f"./data/n_persons_{n_persons}/n_items_{n_items}/rep_{rep}/EvalTrain"
        )

        res_mat = np.loadtxt(
            os.path.join(path_EvalTrain, "res_mat.csv"),
            dtype="int",
            delimiter=",",
        )
        res_prob_mat = np.loadtxt(
            os.path.join(path_EvalTrain, "res_prob_mat.csv"),
            dtype="float",
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

        path_EvalTest = (
            f"./data/n_persons_{n_persons}/n_items_{n_items}/rep_{rep}/EvalTest"
        )

        res_mat_new_persons = np.loadtxt(
            os.path.join(path_EvalTest, "res_mat_new_persons.csv"),
            dtype="int",
            delimiter=",",
        )
        res_prob_mat_new_persons = np.loadtxt(
            os.path.join(path_EvalTest, "res_prob_mat_new_persons.csv"),
            dtype="float",
            delimiter=",",
        )
        z_true_new = np.loadtxt(
            os.path.join(path_EvalTest, "z_true_new.csv"),
            dtype="float",
            delimiter=",",
        )

        res_mat_new_items = np.loadtxt(
            os.path.join(path_EvalTest, "res_mat_new_items.csv"),
            dtype="int",
            delimiter=",",
        )
        res_prob_mat_new_items = np.loadtxt(
            os.path.join(path_EvalTest, "res_prob_mat_new_items.csv"),
            dtype="float",
            delimiter=",",
        )
        a_true_new = np.loadtxt(
            os.path.join(path_EvalTest, "a_true_new.csv"),
            dtype="float",
            delimiter=",",
        )
        b_true_new = np.loadtxt(
            os.path.join(path_EvalTest, "b_true_new.csv"),
            dtype="float",
            delimiter=",",
        )

        return (
            res_mat,
            res_prob_mat,
            z_true,
            a_true,
            b_true,
            res_mat_new_persons,
            res_prob_mat_new_persons,
            res_mat_new_items,
            res_prob_mat_new_items,
            z_true_new,
            a_true_new,
            b_true_new,
        )

    def _adorn_metrics_summary(self, which_task, metrics_summary):
        """Improve the appearance of the recovery metrics.

        Args:
            which_task (str): The indicator of the task: "EvalTrain" or
                "EvalTest".
            metrics_summary (numpy.ndarray): 'Mean' or 'SEM' of the
                metrics to be decorated.

        Returns:
            pandas.DataFrame: The adorned metrics summary.
        """
        metrics_summary = np.reshape(metrics_summary, (-1, self.n_metrics))
        metrics_summary = pd.DataFrame(metrics_summary)
        metrics_summary.columns = ["Cor", "Bias", "MAE", "MSE", "RMSE"]
        if which_task == "EvalTrain":
            metrics_summary.index = np.tile(["z", "a", "b", "p"], self.n_CEN_versions)
        elif which_task == "EvalTest":
            metrics_summary.index = np.tile(["z", "a", "b"], self.n_CEN_versions)

        return metrics_summary

    def _save_metrics_summary(self, path, which_task, metrics_reps):
        """Save the summary ('Mean' and 'Standard Error of the Mean') of
        the recovery metrics under a test setting.

        Args:
            path (str): The path of the directory where the summary is
                expected to be saved.
            which_task (str): The indicator of the task: "EvalTrain" or
                "EvalTest".
            metrics_reps (numpy.ndarray): The metrics of all the CEN
                versions and repetitions under a setting.
        """
        metrics_mean = np.mean(metrics_reps, 0)
        metrics_mean = self._adorn_metrics_summary(which_task, metrics_mean)
        metrics_mean_path = os.path.join(path, f"metrics_{which_task}_mean.csv")
        metrics_mean.to_csv(metrics_mean_path)

        metrics_sem = stats.sem(metrics_reps, 0)
        metrics_sem = self._adorn_metrics_summary(which_task, metrics_sem)
        metrics_sem_path = os.path.join(path, f"metrics_{which_task}_sem.csv")
        metrics_sem.to_csv(metrics_sem_path)

    def _save_n_cnvg_reps(self, n_cnvg_reps, path):
        """Save the number of epochs required for CEN convergence of all
        the repetitions under a test setting.

        Args:
            n_cnvg_reps (numpy.ndarray): Number of epochs required for
                converge of all the CEN versions and repetitions under
                a test setting.
            path (str): The path of the directory where 'n_cnvg_reps' is
                expected to be saved.
        """
        n_cnvg_reps = pd.DataFrame(n_cnvg_reps)
        n_cnvg_reps.columns = self.CEN_versions_names
        n_cnvg_reps_path = os.path.join(path, "n_cnvg_reps.csv")
        n_cnvg_reps.to_csv(n_cnvg_reps_path)
