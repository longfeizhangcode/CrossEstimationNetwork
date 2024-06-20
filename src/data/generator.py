#!/usr/bin/env python3
# =============================================================================
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
"""Generate the data used to train and evaluate the cross estimation
network (CEN) in the three simulation studies of our research.

After running this script, the data sets (res_mat, res_prob_mat, z_true,
a_true, b_true) for each test setting (i.e., combination of the number
of persons and items) are obtained. Additionally, multiple sets of data
are generated under each test setting. The generated data sets are
stored in the 'data' folder (which will be created if it doesn't exist)
in the root of the project.

Classes:
    ResponseData: Response data of a single repetition under a test
        setting (characterized by n_persons and n_items).

    ResponseDataSets: Response data sets of all the repetitions and test
        settings.
"""

import itertools
import math
import os
import sys
import time

import numpy as np

# Add the "src" directory to sys.path, which makes sure the interpreter
# accesses the required modules appropriately.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils import utils


class ResponseData:
    """The response data obtained from a group of simulated individuals
    responding to a simulated test with a set of items.

    If the person parameters ('z') and item parameters ('a' and 'b') are
    not specified (i.e., set to None) during the initialization of the
    class, they will be generated automatically. On the other hand, if
    either the person or item parameters are specified, the other set of
    parameters will be generated.
    """

    def __init__(self, n_persons, n_items, z=None, a=None, b=None):
        """
        Args:
            n_persons (int): Number of test takers.
            n_items (int): Number of test items.
            z (numpy.ndarray, optional): Array including the ability
                parameters of all the persons, with 'float' type.
                Defaults to None.
            a (numpy.ndarray, optional): Array including the
                discrimination parameters of all the items, with 'float'
                type. Defaults to None.
            b (numpy.ndarray, optional): Array including the difficulty
                parameters of all the items, with 'float' type. Defaults
                to None.
        """
        self.n_persons = n_persons
        self.n_items = n_items

        z, a, b = self._get_parameters(n_persons, n_items, z, a, b)

        self.z = z
        self.a = a
        self.b = b

        self.res_mat, self.res_prob_mat = self._get_response_matrix(
            n_persons, n_items, z, a, b
        )

    def _get_response_matrix(self, n_persons, n_items, z, a, b):
        """Generate the response matrix and probability matrix.

        Caution: z, a, and b shall not be 'None' inside this method!
        since the response matrix cannot be obtained by using the
        2PL-IRT formula unless all the parameters are provided.

        Returns:
            A tuple (res_mat, res_prob_mat), where res_mat is the
            generated response matrix, and res_prob_mat is the
            probability matrix of correct responses.
        """
        res_prob_mat = np.zeros((n_persons, n_items))
        res_mat = np.zeros((n_persons, n_items))

        for i in range(n_persons):
            for j in range(n_items):
                res_prob_mat[i, j] = self._compute_IRT(z[i], a[j], b[j])
                res_mat[i, j] = self._get_response(res_prob_mat[i, j])

        return res_mat, res_prob_mat

    def _compute_IRT(self, z, a, b):
        """The 2PL-IRT model, which calculates the probability of a
        correct response for a person to an item.

        Args:
            z (float): The ability parameter of a person.
            a (float): The discrimination parameter of an item.
            b (float): The difficulty parameter of an item.

        Returns:
            float: Probability of a correct response.
        """
        x = a * (z - b)
        p = 1 / (1 + math.exp(-x))

        return p

    def _get_response(self, p):
        """Generate the response from the Bernoulli distribution given
        the probability of a correct response.

        Args:
            p (float): The probability of a correct response, produced
                by the 2PL-IRT model.

        Returns:
            int: A value of 0 or 1, where 0 represents an incorrect
            response and 1 represents a correct response.
        """
        random_number = np.random.rand()

        if p > random_number:
            response = 1
        else:
            response = 0

        return response

    def _get_parameters(self, n_persons, n_items, z, a, b):
        """Generate the person parameters 'z' and (or) item
        parameters 'a' and 'b'.

        Args:
            n_persons (int): Number of test takers.
            n_items (int): Number of test items.

        Returns:
            A tuple (z, a, b), each of which is a numpy.ndarray with
            type 'float'.
        """
        z = self._get_z(n_persons) if (z is None) else z
        a = self._get_a(n_items) if (a is None) else a
        b = self._get_b(n_items) if (b is None) else b

        return z, a, b

    def _get_z(self, n_persons):
        """Generate the person parameters 'z' from the standard normal
        distribution.

        Args:
            n_persons (int): Number of test takers

        Returns:
            numpy.ndarray: The person ability parameters, with type
            'float'.
        """
        z = np.zeros(n_persons)
        i = 0
        while i < n_persons:
            z[i] = np.random.randn()
            # z is restricted to the range of (-3, 3).
            if np.abs(z[i]) < 3:
                i += 1

        return z

    def _get_a(self, n_items):
        """Generate the item discrimination parameters 'a' from the
        Uniform(0.2, 2) distribution.

        Args:
            n_items (int): Number of test items.

        Returns:
            numpy.ndarray: The item discrimination parameters, with type
            'float'.
        """
        a = np.random.uniform(0.2, 2, n_items)

        return a

    def _get_b(self, n_items):
        """Generate the item difficulty parameters 'b' from the standard
        normal distribution.

        Args:
            n_items (int): Number of test items.

        Returns:
            numpy.ndarray: The item difficulty parameters, with type
            'float'.
        """
        b = np.zeros(n_items)
        j = 0
        while j < n_items:
            b[j] = np.random.randn()
            # b is restricted to the range of (-3, 3).
            if np.abs(b[j]) < 3:
                j += 1

        return b


class ResponseDataSets:
    """Response data sets used in the EvalTrain and EvalTest tasks.

    This class generates data sets that will be used in the three
    simulation studies. All of the data sets will be saved in the 'data'
    directory, which will be created if it does not exist. The 'data'
    directory is located in the root of the project.
    """

    def __init__(
        self, n_persons_levels, n_items_levels, n_persons_new, n_items_new, n_reps
    ):
        """
        Args:
            n_persons_levels (list of int): A list including different
                levels of number of test takers.
            n_items_levels (list of int): A list including different
                levels of number of test items.
            n_persons_new (int): Number of the new persons in the
                EvalTest task.
            n_items_new (int): Number of the new items in the EvalTest
                task.
            n_reps (int): The number of repetition, i.e., how many data
                sets will be created for a test setting?
        """
        self.n_persons_levels = n_persons_levels
        self.n_items_levels = n_items_levels
        self.n_persons_new = n_persons_new
        self.n_items_new = n_items_new
        self.n_reps = n_reps

        self._generate_response_data_sets()

    def _generate_response_data_sets(self):
        start = time.time()

        for n_persons, n_items, rep in itertools.product(
            n_persons_levels, n_items_levels, range(n_reps)
        ):
            # Generate data for the EvalTrain task.
            path_EvalTrain = f"./data/n_persons_{n_persons}/n_items_{n_items}/rep_{rep}/EvalTrain"
            utils.make_dir(path_EvalTrain)

            response_data = ResponseData(
                n_persons,
                n_items,
                z=None,
                a=None,
                b=None,
            )
            res_mat = response_data.res_mat
            res_prob_mat = response_data.res_prob_mat
            z_true = response_data.z
            a_true = response_data.a
            b_true = response_data.b

            np.savetxt(
                os.path.join(path_EvalTrain, "res_mat.csv"), res_mat, delimiter=",", fmt="%d"
            )
            np.savetxt(
                os.path.join(path_EvalTrain, "res_prob_mat.csv"), res_prob_mat, delimiter=","
            )
            np.savetxt(os.path.join(path_EvalTrain, "z_true.csv"), z_true, delimiter=",")
            np.savetxt(os.path.join(path_EvalTrain, "a_true.csv"), a_true, delimiter=",")
            np.savetxt(os.path.join(path_EvalTrain, "b_true.csv"), b_true, delimiter=",")

            # Generate data for the EvalTest task.
            path_EvalTest = f"./data/n_persons_{n_persons}/n_items_{n_items}/rep_{rep}/EvalTest"
            utils.make_dir(path_EvalTest)

            # Generate the new response matrix obtained from a group of
            # new test takers responding to the old items, which are
            # characterized by 'a_true', and 'b_true'.
            response_data_new_persons = ResponseData(
                self.n_persons_new,
                n_items,
                z=None,
                a=a_true,
                b=b_true,
            )
            res_mat_new_persons = response_data_new_persons.res_mat
            res_prob_mat_new_persons = response_data_new_persons.res_prob_mat
            z_true_new = response_data_new_persons.z

            np.savetxt(
                os.path.join(path_EvalTest, "res_mat_new_persons.csv"),
                res_mat_new_persons,
                delimiter=",",
                fmt="%d",
            )
            np.savetxt(
                os.path.join(path_EvalTest, "res_prob_mat_new_persons.csv"),
                res_prob_mat_new_persons,
                delimiter=",",
            )
            np.savetxt(
                os.path.join(path_EvalTest, "z_true_new.csv"), z_true_new, delimiter=","
            )

            # Generate the new response matrix obtained from a set of
            # new test items answered by the old persons, which are
            # characterized by 'z_true'.
            response_data_new_items = ResponseData(
                n_persons,
                self.n_items_new,
                z=z_true,
                a=None,
                b=None,
            )
            res_mat_new_items = response_data_new_items.res_mat
            res_prob_mat_new_items = response_data_new_items.res_prob_mat
            a_true_new = response_data_new_items.a
            b_true_new = response_data_new_items.b

            np.savetxt(
                os.path.join(path_EvalTest, "res_mat_new_items.csv"),
                res_mat_new_items,
                delimiter=",",
                fmt="%d",
            )
            np.savetxt(
                os.path.join(path_EvalTest, "res_prob_mat_new_items.csv"),
                res_prob_mat_new_items,
                delimiter=",",
            )
            np.savetxt(
                os.path.join(path_EvalTest, "a_true_new.csv"), a_true_new, delimiter=","
            )
            np.savetxt(
                os.path.join(path_EvalTest, "b_true_new.csv"), b_true_new, delimiter=","
            )

        elapsed_in_seconds = time.time() - start
        elapsed_formatted = utils.format_time_elapsed(elapsed_in_seconds)
        print(f"Time spent for data generation: {elapsed_formatted}")


if __name__ == "__main__":
    # Generate all the response data sets used in the three simulation
    # studies.
    n_persons_levels = [100, 500, 1000]
    n_items_levels = [30, 60, 90]
    n_persons_new = 100
    n_items_new = 30
    n_reps = 100

    ResponseDataSets(
        n_persons_levels,
        n_items_levels,
        n_persons_new,
        n_items_new,
        n_reps,
    )