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
"""Visualization of the two subnets, PN and IN, in CEN.

This pertains specifically to CEN models obtained in the test setting of
N=1000 and M=90, EvalTraining task, Simulation Study 1.

For PN: 
    Given the input size of 90, in each repetition, there are a total of
    91 response patterns with the number of '1' ranging from 0 to 90.
    The input pattern of 't+1' is identical to that of 't', except for a
    '1' replacing a randomly chosen '0' in the latter pattern. The first
    pattern consists of all '0's, and the 91st pattern comprises all
    '1's. 

    These 91 patterns are then input into PN to obtain estimates for
    each pattern. After obtaining outputs of all repetitions, resulting
    in a numpy.ndarray of dimensions n_reps*91.

    Finally, the visualization of PN would be created: The x-axis would
    represent the numbers from '0' to '90', and the y-axis would display
    the estimates of all repetitions.
    
For IN:
    With an input size of 1000, there are 1001 input patterns. The
    generation and plotting procedure align with the previously
    described process.
"""
import os
import sys

import random
import numpy as np
import matplotlib.pyplot as plt

# Add the "src" directory to sys.path, which makes sure the interpreter
# accesses the required modules appropriately.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from models.cen import CEN
from utils import utils


class VisCEN:
    def __init__(
        self,
        n_persons,
        n_items,
        n_reps,
        net_depth,
        linear,
        dir,
        show_average_line,
        average_line_width,
        alpha,
    ):
        self.n_persons = n_persons
        self.n_items = n_items
        self.n_reps = n_reps
        self.net_depth = net_depth
        self.linear = linear
        self.show_average_line = show_average_line
        self.average_line_width = average_line_width
        self.alpha = alpha

        self.cen = CEN(
            inp_size_person_net=n_items,
            inp_size_item_net=n_persons,
            person_net_depth=net_depth,
            item_net_depth=net_depth,
            linear=linear,
            show_model_layout=False,
        )

        self.outputs_PN_reps = np.full(
            shape=(n_reps, n_items + 1),
            fill_value=np.nan,
        )
        self.outputs_IN_reps = np.full(
            shape=(n_reps, 2, n_persons + 1),
            fill_value=np.nan,
        )

        self.random_numbers_PN = self.generate_random_numbers(n_items)
        self.random_numbers_IN = self.generate_random_numbers(n_persons)

        self.inputs_PN = self.generate_inputs(self.random_numbers_PN)
        self.inputs_IN = self.generate_inputs(self.random_numbers_IN)

        self.z_est_mean, self.a_est_mean, self.b_est_mean = self.main_process()

        self.plot_CEN(dir=dir)

    def plot_CEN(self, dir):
        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(6, 10))
        plt.subplots_adjust(hspace=0.5)

        ax1.set_title("Person Net (output: z)")
        for rep in range(self.n_reps):
            ax1.plot(range(n_items+1), self.outputs_PN_reps[rep, :], color="#8da0cb", alpha=self.alpha)
        #ax1.set(xlabel="Number of '1's in the pattern")
        ax1.set_ylim(-3, 3)
        ax1.xaxis.set_ticks(np.arange(0, 100, 10))
        ax1.yaxis.set_ticks(np.arange(-3, 4, 1))

        ax2.set_title("Item Net (output: a)")
        for rep in range(self.n_reps):
            ax2.plot(range(n_persons+1), self.outputs_IN_reps[rep, 0, :], color="#fc8d62", alpha=self.alpha)
        #ax2.set(xlabel="Number of '1's in the pattern")
        ax2.set_ylim(0, 3)
        ax2.yaxis.set_ticks(np.arange(0, 4, 1))

        ax3.set_title("Item Net (output: b)")
        for rep in range(self.n_reps):
            ax3.plot(range(n_persons+1), self.outputs_IN_reps[rep, 1, :], color="#66c2a5", alpha=self.alpha)
        ax3.set(xlabel="Number of correct responses in the pattern")
        ax3.set_ylim(-3, 3)
        ax3.yaxis.set_ticks(np.arange(-3, 4, 1))

        if self.show_average_line:
            ax1.plot(range(n_items+1), self.z_est_mean, color="black", linewidth=self.average_line_width)
            ax2.plot(range(n_persons+1), self.a_est_mean, color="black", linewidth=self.average_line_width)
            ax3.plot(range(n_persons+1), self.b_est_mean, color="black", linewidth=self.average_line_width)

        if self.linear:
            linear = ""
        else:
            linear = "N"
        utils.make_dir(dir)
        fig.savefig(os.path.join(dir, f'vis_CEN_{self.net_depth}H_{linear}L.pdf'))

    def generate_random_numbers(self, length):
        random_numbers = random.sample(range(0, length), length)

        return random_numbers

    def generate_inputs(self, random_numbers):
        n_rows_inputs = len(random_numbers) + 1
        n_cols_inputs = len(random_numbers)
        inputs = np.zeros((n_rows_inputs, n_cols_inputs))

        for i in range(1, n_rows_inputs):
            inputs[i, random_numbers[i - 1]] = 1
            inputs[i,] = inputs[i - 1,] + inputs[i,]

        return inputs

    def main_process(
        self,
    ):
        for rep in range(n_reps):
            # Restore the weights of the trained PN and IN in Simulation Study 1.
            path_models = f"./models/simulation_study1/n_persons_{n_persons}/n_items_{n_items}/rep_{rep}/net_depth_{net_depth}/linear_{linear}"
            self.cen.person_net.load_weights(
                os.path.join(path_models, "person_net/person_net")
            )
            self.cen.item_net.load_weights(
                os.path.join(path_models, "item_net/item_net")
            )

            z_est = self.cen.person_net(self.inputs_PN).numpy()[:, 0]
            a_est = self.cen.item_net(self.inputs_IN).numpy()[:, 0]
            b_est = self.cen.item_net(self.inputs_IN).numpy()[:, 1]

            self.outputs_PN_reps[rep, :] = z_est
            self.outputs_IN_reps[rep, 0, :] = a_est
            self.outputs_IN_reps[rep, 1, :] = b_est

        z_est_mean = np.nanmean(self.outputs_PN_reps, 0)
        a_est_mean = np.nanmean(self.outputs_IN_reps, 0)[0]
        b_est_mean = np.nanmean(self.outputs_IN_reps, 0)[1]

        return z_est_mean, a_est_mean, b_est_mean


# Settings for the test.
n_persons = 1000
n_items = 90
n_reps = 100

# Settings for building CEN.
linear = False
net_depth = 1

# Settings for plotting.
show_average_line = False
average_line_width = 1
alpha = 0.4

vis_cen = VisCEN(
    n_persons=n_persons,
    n_items=n_items,
    n_reps=n_reps,
    net_depth=net_depth,
    linear=linear,
    dir="./figs",
    show_average_line = show_average_line,
    average_line_width=average_line_width,
    alpha=alpha,
)
