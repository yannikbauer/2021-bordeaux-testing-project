import numpy as np


def logistic_map(x, r):
    y = r * x * (1 - x)
    return y


def iterate_f(it, x, r):
    y = logistic_map(x, r)
    if it == 1:
        return [y]
    else:
        return [y] + iterate_f(it-1, y, r)


def iterate_f2(it, x, r):
    l = []
    y = x
    for i in range(it):
        y = logistic_map(y, r)
        l.append(y)
    return l

