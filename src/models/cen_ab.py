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
"""Class for implementing the CEN approach that estimates the item
parameters 'a' and 'b' with the response matrix and the person
parameters 'z' provided.

Class:
    CENab: building, training and evaluation of the CEN approach that
        solely estimates the item parameters 'a' and 'b'.
"""

import os
import sys

import numpy as np
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Lambda,
)

# Add the "src" directory to sys.path, which makes sure the interpreter
# accesses the required modules appropriately.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from models.cen import CEN


class CENab(CEN):
    """Cross estimation network that only estimates the item
    parameters 'a' and 'b' of the 2PL-IRT model.

    This class represents the CEN approach, which focuses on estimating
    the item parameters. CENab will be evaluated in Simulation Study 3.
    In CENab, the input for the item net remains the same as the
    standard CEN, which is the column of a specific cross '+'. However,
    the input for the person net includes the know the person parameter
    'z'. Correspondingly, only the performance pertaining to the
    estimation of item parameters 'a' and 'b' would be evaluated in the
    EvalTrain and EvalTest tasks.

    For more information, see the base class 'CEN'.
    """

    def _build_person_net(self):
        """See base class.

        A pseudo person net (where the input is the same as the output)
        is built, even though it is not actually required for CENab.
        This design choice is made to uphold consistency in the
        structure and organization across different CEN classes.
        """
        inp_person_net = Input(
            shape=(self.inp_size_person_net,), name="input_of_person_net"
        )
        z = Lambda(lambda x: x, name="z")(inp_person_net)

        return Model(inp_person_net, z, name="person_net")

    def _get_X_person_net(self):
        """See base class.

        Unlike the base class 'CEN', the inputs of person net includes
        person parameters 'z'.
        """
        X_person_net = np.repeat(self.z_true, repeats=self.n_items, axis=0)
        X_person_net = X_person_net[None, :]
        X_person_net = X_person_net.transpose()
        self.X_person_net = X_person_net

    def evaluate_training_dataset(self):
        """See base class.

        Only 'a', 'b', and'p' are evaluated here.
        """
        self.a_est = self.param_est(param="a", res_mat=self.res_mat)
        self.b_est = self.param_est(param="b", res_mat=self.res_mat)

        a_metrics = self._compute_distance(self.a_true, self.a_est)
        b_metrics = self._compute_distance(self.b_true, self.b_est)
        
        self.res_prob_mat_est_1D = self.combined(
            [self.X_person_net, self.X_item_net]
        ).numpy()
        
        p_metrics = self._compute_distance(
            self.res_prob_mat.reshape(-1), self.res_prob_mat_est_1D
        )

        metrics_EvalTrain = np.vstack([a_metrics, b_metrics, p_metrics])
        metrics_EvalTrain = pd.DataFrame(metrics_EvalTrain)
        metrics_EvalTrain.columns = ["Cor", "Bias", "MAE", "MSE", "RMSE"]
        metrics_EvalTrain.index = ["a", "b", "P"]
        self.metrics_EvalTrain = metrics_EvalTrain

        return metrics_EvalTrain

    def evaluate_test_dataset(self):
        """See base class.

        Only 'a' and 'b' is evaluated here.
        """
        self.a_est_new = self.param_est(param="a", res_mat=self.res_mat_new_items)
        self.b_est_new = self.param_est(param="b", res_mat=self.res_mat_new_items)

        a_metrics_new = self._compute_distance(self.a_true_new, self.a_est_new)
        b_metrics_new = self._compute_distance(self.b_true_new, self.b_est_new)

        metrics_EvalTest = np.vstack([a_metrics_new, b_metrics_new])
        metrics_EvalTest = pd.DataFrame(metrics_EvalTest)
        metrics_EvalTest.columns = ["Cor", "Bias", "MAE", "MSE", "RMSE"]
        metrics_EvalTest.index = ["a", "b"]
        self.metrics_EvalTest = metrics_EvalTest

        return metrics_EvalTest
