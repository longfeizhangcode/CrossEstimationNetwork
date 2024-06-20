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
"""Class for conducting Simulation Study 2.

Class:
    Simulation2: settings and implementing details of Simulation Study 2.
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

from models.cen_z import CENz
from sims.sim1 import Simulation1


class Simulation2(Simulation1):
    """The class of Simulation Study 2 designed to evaluate the
    performance of the CEN approach that solely estimates the person
    parameter 'z'.

    For more information, see the base class 'Simulation1'.
    """

    def _get_simulation_indicator(self):
        """See base class."""
        return "Simulation Study 2", "simulation_study2"

    def _get_inp_size(self, n_persons, n_items):
        """See base class.

        Here the 'inp_size_item_net' is set to 2, because in Simulation
        Study 2, the inputs of the item net only include the 'a-b' pairs.
        """
        inp_size_person_net = n_items
        inp_size_item_net = 2

        return inp_size_person_net, inp_size_item_net

    def _get_CEN_class(self):
        """See base class.

        The 'CENz' class from models.cen_z is used for Simulation Study 2.
        """
        return CENz

    def _adorn_metrics_summary(self, which_task, metrics_summary):
        """See base class."""
        metrics_summary = np.reshape(metrics_summary, (-1, self.n_metrics))
        metrics_summary = pd.DataFrame(metrics_summary)
        metrics_summary.columns = ["Cor", "Bias", "MAE", "MSE", "RMSE"]
        if which_task == "EvalTrain":
            metrics_summary.index = np.tile(["z", "p"], self.n_CEN_versions)
        elif which_task == "EvalTest":
            metrics_summary.index = np.tile("z", self.n_CEN_versions)

        return metrics_summary
