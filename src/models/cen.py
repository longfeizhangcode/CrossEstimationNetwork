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
"""Class for implementing the standard CEN that estimates both the
person and item parameters of the 2PL-IRT model.

Class:
    CEN: building, training and evaluation of the standard CEN.
"""

import math
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Input,
    Lambda,
    concatenate,
)
from tensorflow.keras.metrics import binary_crossentropy
from tensorflow.keras.models import Model

# Add the "src" directory to sys.path, which makes sure the interpreter
# accesses the required modules appropriately.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils import utils


class CEN:
    """Cross estimation network for estimating both the person and item
    parameters of the 2PL-IRT model.

    This is the 'stand' CEN used in the Simulation Study 1 in our
    research. It can be inherited by the classes 'CENz' and 'CENab' used
    in the Simulation Study 2 and 3, respectively.

    Note that the input patterns of CEN resemble a cross shape '+',
    where the person net takes the row and the item net takes the column
    as inputs. To achieve this, the original response matrix is
    converted into two new matrices including the input patterns of the
    person network and item network, respectively. Details for this step
    can be found in the '_preprocess_data' method.

    Attributes:
        inp_size_person_net (int): Input size of the person net.
        inp_size_item_net (int): Input size of the item net.
        person_net_depth (int): Number of hidden layers in the
            person net, which takes the value of 1 or 3. Defaults to 1.
        item_net_depth (int): Number of hidden layers in the item net,
            taking the value of 1 or 3. Defaults to 1.
        linear (bool): Whether to model linearity or nonlinearity in the
            subnets.
        person_net (keras.engine.functional.Functional): The constructed
            person network.
        item_net (keras.engine.functional.Functional): The constructed
            item network.
        IRT_net(keras.engine.functional.Functional): The constructed IRT
            network. Note that it is just a node that implements the
            2PL-IRT formula.
        combined (keras.engine.functional.Functional): The constructed
            CEN.
        show_model_layout (bool): Whether to show the model layout of
            CEN.
        n_persons (int): Number of test takers.
        n_items (int): Number of test items.
        res_mat (numpy.ndarray): The actual response matrix for training
            CEN, with 'int' type.
        res_prob_mat (numpy.ndarray): The actual probability matrix of
            correct responses, with 'float' type.
        z_true (numpy.ndarray): The actual 'z_s' of all the test takers,
            with 'float' type. Defaults to None.
        a_true (numpy.ndarray): The actual 'a_s' of all the test items,
            with 'float' type. Defaults to None.
        b_true (numpy.ndarray): The actual 'b_s' of all the test items,
            with 'float' type. Defaults to None.
        X_person_net (numpy.ndarray): Input patterns of the person_net
            of CEN, with type 'float". It is obtained by preprocessing
            the res_mat.
        X_item_net (numpy.ndarray): Input patterns of the item net, with
            type 'float". It is obtained by preprocessing the res_mat.
        y_CEN (numpy.ndarray): The target of CEN, with type 'int'.
        optimizer (tf.keras.optimizers.Optimizer): Optimizer used for
            compiling CEN.
        loss_func (tf.keras.losses.Loss): Loss function used for
            compiling CEN.
        epochs (int): Number of epochs in the training process.
        batch_size (int): Batch size in the training process.
        early_stopping (int): Early stopping rules for training CEN.
        verbose (int):  0, 1, or 2. Verbosity mode during training.
            0 = silent, 1 = progress bar, 2 = single line. verbose=2 is
            recommended when not running interactively.
        loss_values (numpy.ndarray): Loss values yielded in the training
            process.
        n_cnvg (int): Number of epochs for CEN to converge in training.
        res_prob_mat_est_1D (numpy.ndarray): The estimates of 'p_s' 
            using CEN. It only has one dimension.
        z_est (numpy.ndarray): The estimates of 'z_s' using CEN.
        a_est (numpy.ndarray): The estimates of 'a_s' using CEN.
        b_est (numpy.ndarray): The estimates of 'b_s' using CEN.
        metrics_EvalTrain (pandas.DataFrame): Metrics indicating the
            performance of CEN in the EvalTrain task. The metrics
            pertain to the estimates of 'z', 'a', 'b', and 'p'.
        metrics_EvalTest (pandas.DataFrame): Metrics indicating the
            performance of CEN in the EvalTest task. The metrics pertain
            to the estimates of 'z', 'a', and 'b' (without 'p').
        n_persons_new (int): Number of new persons in the EvalTest task.
        n_items_new (int): Number of new items in the EvalTest task.
        bce_true (float): The loss value (binary cross-entropy)
            associated with the the parameters.
        bce_est (float): The loss value (binary cross-entropy)
            associated with the estimates.
    """

    def __init__(
        self,
        inp_size_person_net,
        inp_size_item_net,
        person_net_depth,
        item_net_depth,
        linear,
        show_model_layout,
    ):
        """
        Args:
            inp_size_person_net (int): Input size of the person net.
            inp_size_item_net (int): Input size of the item net.
            person_net_depth (int): Number of hidden layers in the
                person net, which can take the value of 1 or 3.
            item_net_depth (int): Number of hidden layers in the item
                net, which can take the value of 1 or 3.
            linear (bool): Whether to model linearity or nonlinearity
                in the person net and item net.
            show_model_layout (bool): Whether to show the model layout
                of CEN.
        """
        self.inp_size_person_net = inp_size_person_net
        self.inp_size_item_net = inp_size_item_net
        self.person_net_depth = person_net_depth
        self.item_net_depth = item_net_depth
        self.linear = linear
        self.show_model_layout = show_model_layout

        # Build the three components (person net, item net and IRT node)
        # of CEN.
        self.person_net = self._build_person_net()
        self.item_net = self._build_item_net()
        self.IRT_net = self._build_IRT_net()

        # Use the constructed person_net, item_net and IRT_net to
        # compose CEN.
        inp_person_net = Input(shape=(inp_size_person_net,), name="input_of_person_net")
        z = self.person_net(inp_person_net)

        inp_item_net = Input(shape=(inp_size_item_net,), name="input_of_item_net")
        ab = self.item_net(inp_item_net)

        prob = self.IRT_net(concatenate([z, ab]))

        self.combined = Model(
            [inp_person_net, inp_item_net], prob, name="cross_estimation_network"
        )

        if show_model_layout:
            self.combined.summary()
            self.person_net.summary()
            self.item_net.summary()
            self.IRT_net.summary()

    def _build_person_net(self):
        """Build the person network.

        The person network can have either one or three hidden layers. A
        single hidden layer is considered shallow, while three hidden
        layers are considered deep.

        Please note that for the present research, only one and three
        hidden layers are implemented in the construction of the person
        net. If a value other than 3 is specified for
        'inp_size_person_net', this method will create only one hidden
        layer for the person subnet.

        Returns:
            keras.engine.functional.Functional: The constructed person
            network.
        """
        inp_person_net = Input(
            shape=(self.inp_size_person_net,), name="input_of_person_net"
        )

        if self.linear:
            z = Dense(math.ceil(self.inp_size_person_net / 2), activation="linear")(
                inp_person_net
            )
            if self.person_net_depth == 3:
                z = Dense(math.ceil(self.inp_size_person_net / 4), activation="linear")(
                    z
                )
                z = Dense(math.ceil(self.inp_size_person_net / 8), activation="linear")(
                    z
                )
            z = Dense(1, activation="linear")(z)
            z = BatchNormalization(
                momentum=0, epsilon=0, center=False, scale=False, name="z"
            )(z)

        if not self.linear:
            z = Dense(math.ceil(self.inp_size_person_net / 2), activation="sigmoid")(
                inp_person_net
            )
            if self.person_net_depth == 3:
                z = Dense(
                    math.ceil(self.inp_size_person_net / 4), activation="sigmoid"
                )(z)
                z = Dense(
                    math.ceil(self.inp_size_person_net / 8), activation="sigmoid"
                )(z)
            z = Dense(1, activation="linear")(z)
            z = BatchNormalization(
                momentum=0, epsilon=0, center=False, scale=False, name="z"
            )(z)

        return Model(inp_person_net, z, name="person_net")

    def _build_item_net(self):
        """Build the item network.

        The item network can have either one or three hidden layers. A
        single hidden layer is considered shallow, while three hidden
        layers are considered deep.

        Please note that for the present research, only one and three
        hidden layers are implemented in the construction of the item
        net. If a value other than 3 is specified for
        'inp_size_item_net', this method will create only one hidden
        layer for this subnet.

        Returns:
            keras.engine.functional.Functional: The constructed item
            network.
        """
        inp_item_net = Input(shape=(self.inp_size_item_net,), name="input_of_item_net")

        if self.linear:
            ab = Dense(math.ceil(self.inp_size_item_net / 2), activation="linear")(
                inp_item_net
            )
            if self.item_net_depth == 3:
                ab = Dense(math.ceil(self.inp_size_item_net / 4), activation="linear")(
                    ab
                )
                ab = Dense(math.ceil(self.inp_size_item_net / 8), activation="linear")(
                    ab
                )
            a = Dense(1, activation="linear")(ab)
            a = Lambda(tf.abs, name="a")(a)
            b = Dense(1, activation="linear", name="b")(ab)

        if not self.linear:
            ab = Dense(math.ceil(self.inp_size_item_net / 2), activation="sigmoid")(
                inp_item_net
            )
            if self.item_net_depth == 3:
                ab = Dense(math.ceil(self.inp_size_item_net / 4), activation="sigmoid")(
                    ab
                )
                ab = Dense(math.ceil(self.inp_size_item_net / 8), activation="sigmoid")(
                    ab
                )
            a = Dense(1, activation="linear")(ab)
            a = Lambda(tf.abs, name="a")(a)
            b = Dense(1, activation="linear", name="b")(ab)

        return Model(inp_item_net, concatenate([a, b]), name="item_net")

    def _build_IRT_net(self):
        """Build the IRT network (node).

        Returns:
            keras.engine.functional.Functional: The constructed IRT
            network.
        """
        inp_IRT_net = Input(shape=(3,), name="input_of_IRT_net")
        prob = Lambda(self._compute_IRT, name="probability_of_correct_response")(
            inp_IRT_net
        )

        return Model(inp_IRT_net, prob, name="IRT_net")

    def _compute_IRT(self, tensor):
        """Compute the probability of a correct response in terms of
        the 2PL-IRT formula.

        This method is solely used within the TensorFlow neural network.

        Args:
            tensor (tf.Tensor): The input tensor where the first element
                represents the person parameter 'z', and the second
                and third elements represent the item discrimination
                parameter 'a' and difficulty parameter 'b', respectively.

        Returns:
            tf.Tensor: The probability of a correct response, with type
            'tf.float'.
        """
        x = tensor[:, 1] * (tensor[:, 0] - tensor[:, 2])
        p = 1 / (1 + tf.math.exp(-x))

        return p

    def load_data(
        self,
        res_mat,
        res_prob_mat,
        z_true=None,
        a_true=None,
        b_true=None,
        res_mat_new_persons=None,
        res_prob_mat_new_persons=None,
        res_mat_new_items=None,
        res_prob_mat_new_items=None,
        z_true_new=None,
        a_true_new=None,
        b_true_new=None,
    ):
        """Load the data into the CEN object.

        The loaded data will be used for training or evaluating CEN.

        Args:
            res_mat (numpy.ndarray): The response matrix for training
                CEN, with 'int' type.
            res_prob_mat (numpy.ndarray): The probability matrix of
                correct responses, with 'float' type.
            z_true (numpy.ndarray, optional): The actual 'z_s' of all
                the test takers, with 'float' type. Defaults to None.
            a_true (numpy.ndarray, optional): The actual 'a_s' of all
                the test items, with 'float' type. Defaults to None.
            b_true (numpy.ndarray, optional): The actual 'b_s' of all
                the test items, with 'float' type. Defaults to None.
            res_mat_new_persons (numpy.ndarray): The response matrix
                associated with new persons, with 'int' type.
            res_prob_mat_new_persons (numpy.ndarray): The probability
                matrix associated with new persons, with 'float' type.
            res_mat_new_items (numpy.ndarray): The response matrix
                associated with new items, with 'int' type.
            res_prob_mat_new_items (numpy.ndarray): The probability
                matrix associated with new items, with 'float' type.
            z_true_new (numpy.ndarray, optional): The actual 'z_s' of
                the new persons, with 'float' type. Defaults to None.
            a_true_new (numpy.ndarray, optional): The actual 'a_s' of
                the new items, with 'float' type. Defaults to None.
            b_true_new (numpy.ndarray, optional): The actual 'b_s' of
                the new items, with 'float' type. Defaults to None.
        """
        self.n_persons, self.n_items = res_mat.shape

        self.res_mat = res_mat
        self.res_prob_mat = res_prob_mat
        self.z_true = z_true
        self.a_true = a_true
        self.b_true = b_true

        self.res_mat_new_persons = res_mat_new_persons
        self.res_prob_mat_new_persons = res_prob_mat_new_persons
        self.res_mat_new_items = res_mat_new_items
        self.res_prob_mat_new_items = res_prob_mat_new_items
        self.z_true_new = z_true_new
        self.a_true_new = a_true_new
        self.b_true_new = b_true_new

    def train(
        self,
        optimizer,
        loss_func,
        epochs,
        batch_size,
        early_stopping,
        verbose,
    ):
        """Train the constructed CEN model.

        Args:
            optimizer (tf.keras.optimizers): Optimizer used for training
                CEN.
            loss_func (tf.keras.losses): Loss function used for
                training CEN.
            epochs (int): Number of epochs in the training process.
            batch_size (int): Batch size in the training process.
            early_stopping (int): Early stopping rules for training CEN.
            verbose (int):  0, 1, or 2. Verbosity mode during training.
                0 = silent, 1 = progress bar, 2 = single line. verbose=2
                is recommended when not running interactively.
        """
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.verbose = verbose

        # The original res_mat cannot be used directly for training, it
        # need to be preprocessed in accordance with the required cross
        # shape '+' of CEN.
        self._preprocess_data()

        # Compile the CEN model.
        self.combined.compile(optimizer=optimizer, loss=loss_func)

        # Mark the start time of the training process.
        start_train = time.time()

        # Train the CEN model with the specified details.
        history = self.combined.fit(
            x=[self.X_person_net, self.X_item_net],
            y=self.y_CEN,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[self.early_stopping],
            verbose=self.verbose,
        )
        self.history = history

        # Training end and compute the time elapsed in training.
        elapsed_in_seconds_train = time.time() - start_train
        self.elapsed_formatted_train = utils.format_time_elapsed(
            elapsed_in_seconds_train
        )

        self.loss_values = history.history["loss"]
        self.n_cnvg = len(self.loss_values)

    def _preprocess_data(self):
        """Preprocess the response matrix to match the input (feature)
        and target (label) data for training CEN.

        As the subnets of CEN require the row and column of a specific
        cross in the response matrix as inputs, a workaround is to
        convert the original response matrix into two new matrices.
        These matrices have the same number of rows as n_persons *
        n_items. One matrix ('X_person_net') includes the inputs of the
        person net while the other ('X_item_net') includes the inputs of
        the item net. The target data ('y_CEN') of CEN is obtained by
        flattening the response matrix to an one-dimensional array.

        Once all the three components are acquired, the data for
        training CEN can be represented as follows:
            input = [X_person_net, X_item_net], target = y_CEN
        """

        # Get the input patterns 'X_person_net' for the person net.
        self._get_X_person_net()

        # Get the input patterns 'X_item_net' for the item net.
        self._get_X_item_net()

        # Get the the target patterns (labels) 'y_CEN' for CEN.
        self._get_y_CEN()

    def _get_X_person_net(self):
        """Get the inputs of the person net for training CEN.

        The input of person net is the row of a cross shape '+' in the
        response matrix, i.e., the response pattern of a person.
        """
        X_person_net = np.repeat(self.res_mat, repeats=self.n_items, axis=0)
        self.X_person_net = X_person_net

    def _get_X_item_net(self):
        """Get the inputs of the item net for training CEN.

        The input of item net is the column of a cross shape '+' in the
        response matrix, i.e., the response pattern on a item.
        """
        X_item_net = self.res_mat.transpose()
        X_item_net = np.repeat(X_item_net[None, :], self.n_persons, axis=0)
        X_item_net = X_item_net.reshape((self.n_persons * self.n_items, self.n_persons))
        self.X_item_net = X_item_net

    def _get_y_CEN(self):
        """Get the target data (labels) for training CEN.

        The target are obtained by flattening the response matrix to
        an one-dimensional array.
        """
        self.y_CEN = self.res_mat.reshape(-1)

    def param_est(self, param, res_mat):
        """Estimate the parameters 'z', 'a' or 'b' using the CEN method.

        Args:
            param (str): Which parameter to be estimated? Choosing from 
                "z", "a" or "b".
            res_mat (numpy.ndarray): The response matrix.

        Returns:
            numpy.ndarray: Estimates of the parameters 'z', 'a' or 'b'.
        """
        if param == "z":
            estimates = self.person_net(res_mat).numpy()[:, 0]
            estimates = self._guard_z_est(estimates)

        if param == "a":
            estimates = self.item_net(res_mat.transpose()).numpy()[:, 0]
            estimates = self._guard_a_est(estimates)

        if param == "b":
            estimates = self.item_net(res_mat.transpose()).numpy()[:, 1]
            estimates = self._guard_b_est(estimates)

        return estimates
    
    def evaluate_training_dataset(self):
        """Evaluate the performance of CEN in the training dataset.

        The EvalTrain task primarily focused on evaluating the
        estimates of the parameters 'z', 'a', and 'b' for the original
        response matrix ('res_mat'). The estimates will be compared to
        the ground truth values.

        Returns:
            pandas.DataFrame: Metrics indicating the performance of CEN
            in the EvalTrain task. The metrics pertain to 'z',
            'a', 'b', and 'p' (probability of correct response).
        """
        self.z_est = self.param_est(param="z", res_mat=self.res_mat)
        self.a_est = self.param_est(param="a", res_mat=self.res_mat)
        self.b_est = self.param_est(param="b", res_mat=self.res_mat)

        z_metrics = self._compute_distance(self.z_true, self.z_est)
        a_metrics = self._compute_distance(self.a_true, self.a_est)
        b_metrics = self._compute_distance(self.b_true, self.b_est)

        self.res_prob_mat_est_1D = self.combined(
            [self.X_person_net, self.X_item_net]
        ).numpy()
        
        p_metrics = self._compute_distance(
            self.res_prob_mat.reshape(-1), self.res_prob_mat_est_1D
        )

        metrics_EvalTrain = np.vstack([z_metrics, a_metrics, b_metrics, p_metrics])
        metrics_EvalTrain = pd.DataFrame(metrics_EvalTrain)
        metrics_EvalTrain.columns = ["Cor", "Bias", "MAE", "MSE", "RMSE"]
        metrics_EvalTrain.index = ["z", "a", "b", "p"]
        self.metrics_EvalTrain = metrics_EvalTrain

        return metrics_EvalTrain

    def evaluate_test_dataset(self):
        """Evaluate the performance of CEN in the test dataset.

        In the EvalTest task, two new response matrices will be used:
            1) a new matrix obtained from the old items and the new
                persons ('self.res_mat_new_persons');
            2) a new matrix obtained from the old persons and the new
                items ('self.res_mat_new_items').
        The first is in line with the scenario that a group of new
        persons take part in the original test, and the second imitates
        the scenario that the a bunch of new items are answered by the
        original test takers. Therefore, in the first scenario, the
        person net derived of the trained CEN is used to predict the
        parameters of the new persons; in the second scenario, the item
        net derived of the trained CEN is used to predict the parameters
        of the new items.

        Returns:
            pandas.DataFrame: Metrics representing the performance of
            CEN in the EvalTest task. The metrics pertain to 'z',
            'a', and 'b'.
        """
        self.z_est_new = self.param_est(param="z", res_mat=self.res_mat_new_persons)
        self.a_est_new = self.param_est(param="a", res_mat=self.res_mat_new_items)
        self.b_est_new = self.param_est(param="b", res_mat=self.res_mat_new_items)

        z_metrics_new = self._compute_distance(self.z_true_new, self.z_est_new)
        a_metrics_new = self._compute_distance(self.a_true_new, self.a_est_new)
        b_metrics_new = self._compute_distance(self.b_true_new, self.b_est_new)

        metrics_EvalTest = np.vstack([z_metrics_new, a_metrics_new, b_metrics_new])
        metrics_EvalTest = pd.DataFrame(metrics_EvalTest)
        metrics_EvalTest.columns = ["Cor", "Bias", "MAE", "MSE", "RMSE"]
        metrics_EvalTest.index = ["z", "a", "b"]
        self.metrics_EvalTest = metrics_EvalTest

        return metrics_EvalTest

    def _compute_bce(self):
        """Compute the binary cross-entropy associated with the
        parameters and estimates.
        """
        self.bce_true = binary_crossentropy(
            y_true=self.res_mat.reshape(-1), y_pred=self.res_prob_mat.reshape(-1)
        ).numpy()
        self.bce_est = binary_crossentropy(
            y_true=self.res_mat.reshape(-1), y_pred=self.res_prob_mat_est_1D
        ).numpy()

    def print_results(self):
        """Print the results of training and evaluation for CEN.

        The results display the following information:
            1) Time spent for training,
            2) Number of epochs required for CEN to converge,
            3) Loss (measured by binary cross-entropy) associated with
               the parameters and estimates,
            4) Recovery metrics of the EvalTrain and EvalTest tasks.
        """
        self._compute_bce()

        print("")
        print("Results of CEN:")
        print(f"Training time: {self.elapsed_formatted_train}")
        print(f"Number to convergence: {self.n_cnvg}")
        print(
            "Loss associated with the parameters:",
            round(self.bce_true, 5),
        )
        print(
            "Loss associated with the estimates:",
            round(self.bce_est, 5),
        )
        print("++++++++++++++++++++++++++++++++++++")
        print("Evaluation in the Training Dataset:")
        print(self.metrics_EvalTrain.round(3))
        print("++++++++++++++++++++++++++++++++++++")
        print("Evaluation in the test Dataset:")
        print(self.metrics_EvalTest.round(3))
        print("++++++++++++++++++++++++++++++++++++")

    def _compute_distance(self, param_true, param_est):
        """Compute the distance between the actual and estimated values
        pertaining to the given parameter, which indicates the recovery
        of the estimates.

        Args:
            param_true (numpy.ndarray): The ground truth values of the
                parameter.
            param_est (numpy.ndarray): The estimates of the parameter.

        Returns:
            numpy.ndarray: An array containing the correlation
            coefficients (Cor), bias (Bias), mean absolute error (MAE),
            mean squared error (MSE), and root mean squared error (RMSE)
            between the actual and estimated values of the parameter,
            with type 'float'.
        """
        cor = np.corrcoef(param_true, param_est)[0, 1]
        bias = np.mean(param_true - param_est)
        mae = metrics.mean_absolute_error(param_true, param_est)
        mse = metrics.mean_squared_error(param_true, param_est)
        rmse = metrics.mean_squared_error(param_true, param_est) ** 0.5

        return np.array([cor, bias, mae, mse, rmse])

    def _guard_z_est(self, z_est):
        """Constrain the values of 'z' estimates within the range of
        (-6, 6).

        This function serves to prevent the estimates of the person
        parameters 'z' from venturing into rarely-encountered space.

        According to Baker (2001), the theoretical range of the values
        of 'z' is negative infinity to positive infinity, and practical
        considerations usually limit the range of values from -6 to +6.
        However, one should be aware that values beyond this range are
        possible.

        Args:
            z_est (numpy.ndarray): The estimated 'z' values.

        Returns:
            numpy.ndarray: The restricted 'z' values.
        """
        z_est[np.where(z_est > 6)] = 6
        z_est[np.where(z_est < -6)] = -6
        return z_est

    def _guard_a_est(self, a_est):
        """Constrain the values of 'a' estimates within the range of
        (-6, 6).

        This function serves to prevent the estimates of the item
        parameters 'a' from venturing into rarely-encountered space.

        Baker (2001) argued that the theoretical range of the values of
        'a' is negative infinity to positive infinity, but the usual
        range seen in practice is -6 to +6.

        Reference:
            Baker, F. B. (2001). The basics of item response theory.
                Maryland: ERIC Publications.

        Args:
            a_est (numpy.ndarray): The estimated 'a' values.

        Returns:
            numpy.ndarray: The restricted 'a' values.
        """
        a_est[np.where(a_est > 6)] = 6
        a_est[np.where(a_est < -6)] = -6
        return a_est

    def _guard_b_est(self, b_est):
        """Constrain the values of 'b' estimates within the range of
        (-6, 6).

        This function serves to prevent the estimates of the item
        parameters 'b' from venturing into rarely-encountered space.

        Baker (2001) suggested that the theoretical range of the values
        of 'b' is negative infinity to positive infinity, but the
        typical values have the range -6 to 6.

        Args:
            b_est (numpy.ndarray): The estimated 'b' values.

        Returns:
            numpy.ndarray: The restricted 'b' values.
        """
        b_est[np.where(b_est > 6)] = 6
        b_est[np.where(b_est < -6)] = -6
        return b_est
