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
"""Class for conducting Simulation Study 3.

Class:
    Simulation3: settings and implementing details of Simulation Study 3.
"""

import os
import sys

import numpy as np
import pandas as pd

# Add the "src" directory to sys.path, which makes sure the interpreter
# accesses the required modules appropriately.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from models.cen_ab import CENab
from sims.sim1 import Simulation1


class Simulation3(Simulation1):
    """The class of Simulation Study 3 designed to evaluate the
    performance of the CEN approach that only estimates the item
    parameter 'a' and 'b'.

    For more information, see the base class 'Simulation1'.
    """

    def _get_simulation_indicator(self):
        """See base class."""
        return "Simulation Study 3", "simulation_study3"

    def _get_inp_size(self, n_persons, n_items):
        """See base class.

        Here the 'inp_size_person_net' is set to 1, because in Simulation
        Study 3, the inputs of the person net only include 'z'.
        """
        inp_size_person_net = 1
        inp_size_item_net = n_persons

        return inp_size_person_net, inp_size_item_net

    def _get_CEN_class(self):
        """See base class.

        The 'CENab' class from models.cen_ab is used for Simulation Study 3.
        """
        return CENab

    def _adorn_metrics_summary(self, which_task, metrics_summary):
        """See base class."""
        metrics_summary = np.reshape(metrics_summary, (-1, self.n_metrics))
        metrics_summary = pd.DataFrame(metrics_summary)
        metrics_summary.columns = ["Cor", "Bias", "MAE", "MSE", "RMSE"]
        if which_task == "EvalTrain":
            metrics_summary.index = np.tile(["a", "b", "p"], self.n_CEN_versions)
        elif which_task == "EvalTest":
            metrics_summary.index = np.tile(["a", "b"], self.n_CEN_versions)

        return metrics_summary
