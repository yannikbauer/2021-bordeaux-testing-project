import numpy as np


def logistic_map(x, r):
    y = r * x * (1 - x)
    return y

