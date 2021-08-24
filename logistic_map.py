import numpy as np


def logistic_map(x, r):
    y = r * x * (1 - x)
    return y


def iterate_f(x, r, it):
    y = logistic_map(x, r)
    if it == 1:
        return [y]
    else:
        return [y] + iterate_f(y, r, it-1)


def iterate_f2(x, r, it):

    l = []
    y = x
    for i in range(it):
        y = logistic_map(y, r)
        l.append(y)
    return l

