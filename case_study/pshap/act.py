# -*- coding: utf-8 -*-

"""
pshap.act
~~~~~~~~~

This module provides the implementation of various activation functions.
"""

import math


def relu(x):
    return max(x, 0)


def square(x):
    return x ** 2


def swish(x, beta=1):
    """
    y = x / [1 + exp(−beta * x)]

    Args:
        x: base value
        beta: beta value in equation y = x / [1 + exp(−beta * x)].
              Defaults to be 1.
    """
    return x / (1 + math.exp(-beta * x))


def mish(x):
    return x * math.tanh(math.log(1 + math.exp(x), math.e))
