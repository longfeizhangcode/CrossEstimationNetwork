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
"""Class for implementing the CEN approach that estimates the person
parameters 'z' with the response matrix and the item parameters 'a' and
'b' provided.

Class:
    CENz: building, training and evaluation of the CEN approach that
        solely estimates the person parameter 'z'.
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


class CENz(CEN):
    """Cross estimation network that only estimates the person
    parameter 'z' of the 2PL-IRT model.

    This class represents the CENz approach, which focuses on estimating
    the person parameters. CENz will be evaluated in Simulation Study 2.
    In CENz, the input for the person net remains the same as the
    standard CEN, which is the row of a specific cross '+'. However, the
    input for the item net includes the two known item parameters 'a'
    and 'b'. Therefore, the EvalTrain and EvalTest tasks will
    specifically assess the performance of CENz in estimating the person
    parameter 'z'.

    For more information, see the base class 'CEN'.
    """

    def _build_item_net(self):
        """See base class.

        A pseudo item net (where the input is the same as the output) is
        built, even though it is not actually required for CENz. This
        design choice is made to uphold consistency in the structure and
        organization across different CEN classes.
        """
        inp_item_net = Input(shape=(self.inp_size_item_net,), name="input_of_item_net")
        ab = Lambda(lambda x: x, name="ab")(inp_item_net)

        return Model(inp_item_net, ab, name="item_net")

    def _get_X_item_net(self):
        """See base class.

        Unlike the base class 'CEN', inputs of the item net includes
        the 'a-b' pairs.
        """
        X_item_net = np.vstack([self.a_true, self.b_true]).transpose()
        X_item_net = np.repeat(X_item_net[None, :], self.n_persons, axis=0)
        X_item_net = X_item_net.reshape((self.n_persons * self.n_items, 2))
        self.X_item_net = X_item_net

    def evaluate_training_dataset(self):
        """See base class.

        Only 'z' and 'p' are evaluated here.
        """
        self.z_est = self.param_est(param="z", res_mat=self.res_mat)

        z_metrics = self._compute_distance(self.z_true, self.z_est)

        self.res_prob_mat_est_1D = self.combined(
            [self.X_person_net, self.X_item_net]
        ).numpy()

        p_metrics = self._compute_distance(
            self.res_prob_mat.reshape(-1), self.res_prob_mat_est_1D
        )

        metrics_EvalTrain = np.vstack([z_metrics, p_metrics])
        metrics_EvalTrain = pd.DataFrame(metrics_EvalTrain)
        metrics_EvalTrain.columns = ["Cor", "Bias", "MAE", "MSE", "RMSE"]
        metrics_EvalTrain.index = ["z", "p"]
        self.metrics_EvalTrain = metrics_EvalTrain

        return metrics_EvalTrain

    def evaluate_test_dataset(self):
        """See base class.

        Only 'z' is evaluated here.
        """
        self.z_est = self.param_est(param="z", res_mat=self.res_mat_new_persons)

        z_metrics_new = self._compute_distance(self.z_true_new, self.z_est_new) 
        
        metrics_EvalTest = np.vstack([z_metrics_new])
        metrics_EvalTest = pd.DataFrame(metrics_EvalTest)
        metrics_EvalTest.columns = ["Cor", "Bias", "MAE", "MSE", "RMSE"]
        metrics_EvalTest.index = ["z"]
        self.metrics_EvalTest = metrics_EvalTest

        return metrics_EvalTest
