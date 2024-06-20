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
"""
A collection of tools used in this project.

Functions:
    make_dir: Make a directory for saving files.
    save_to_file: Write contents to a file.
    format_time_elapsed: Format the elapsed time to 'hour:minute:second'.
"""

import os


def make_dir(path):
    """
    Make a directory for saving files.

    Args:
        path (str): The path of the directory to be created.

    Returns:
        bool: True if the directory was successfully created, False if
        it already exists.
    """
    is_exists = os.path.exists(path)

    if not is_exists:
        os.makedirs(path)
        # print(path + " built successfully")
        return True
    else:
        # print(path + " already exists")
        return False


def save_to_file(file_name, contents):
    """
    Write contents to a file.

    Args:
        file_name (str): The full path of the file.
        contents (str): The contents to be written into the file.
    """
    with open(file_name, "w") as f:
        f.write(contents)


def format_time_elapsed(elapsed_in_seconds):
    """
    Format the elapsed time (in seconds) to 'hour:minute:second'.

    Args:
        elapsed_in_seconds (float): The time duration in seconds.

    Returns:
        str: Formatted time duration in the format 'hour:minute:second'.
    """
    hour = int(elapsed_in_seconds / (60 * 60))
    minute = int((elapsed_in_seconds % (60 * 60)) / 60)
    second = elapsed_in_seconds % 60

    return "{}:{:>02}:{:>05.2f}".format(hour, minute, second)
