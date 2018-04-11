# -*- coding: utf-8 -*-

import math
import pickle
import random
from typing import Any, List


class Tinn:

    def __init__(self, nips: int, nhid: int, nops: int) -> None:
        """Build a new Tinn object given:
        * number of inputs (nips),
        * number of hidden neurons for the hidden layer (nhid),
        * and number of outputs (nops)."""
        self.nips = nips  # number of inputs
        self.nhid = nhid
        self.nops = nops

        # biases, Tinn only supports one hidden layer so there are two biases
        self.b = [random.random() - 0.5 for _ in range(2)]
        # input to hidden layer
        self.x1 = [[float(0)] * nips for _ in range(nhid)]
        self.h = [float(0)] * nhid  # hidden layer
        self.x2 = [[random.random() - 0.5 for _ in range(nhid)]
                   for _ in range(nops)]  # hidden to output layer weights
        self.o = [float(0)] * nops  # output layer

    def save(self, path: str) -> None:
        """Saves the t to disk."""
        pickle.dump(self, open(path, 'wb'))


def xtload(path: str) -> Any:
    """Loads a new t from disk."""
    return pickle.load(open(path, 'rb'))


def xttrain(t: Tinn, in_: List[float], tg: List[float], rate: float) -> float:
    """Trains a Tinn (t) given:
    * an input (in_),
    * target output (tg), and
    * learning rate (rate).
    Returns error rate of the neural network."""
    fprop(t, in_)
    bprop(t, in_, tg, rate)
    return toterr(tg, t.o)


def xtpredict(t: Tinn, in_: List[float]) -> List[float]:
    """Returns an output prediction given an input."""
    fprop(t, in_)
    return t.o


def err(a: float, b: float) -> float:
    """Error function."""
    return 0.5 * (a - b) ** 2


def pderr(a: float, b: float) -> float:
    """Partial derivative of error function."""
    return a - b


def toterr(tg: List[float], o: List[float]) -> float:
    """Total error."""
    return sum([err(tg[i], o[i]) for i in range(len(o))])


def act(a: float) -> float:
    """Activation function."""
    return 1 / (1 + math.exp(-a))


def pdact(a: float) -> float:
    """Partial derivative of activation function."""
    return a * (1 - a)


def bprop(t: Tinn, in_: List[float], tg: List[float], rate: float) -> None:
    """Back propagation."""
    for i in range(t.nhid):
        s = float(0)
        # Calculate total error change with respect to output.
        for j in range(t.nops):
            ab = pderr(t.o[j], tg[j]) * pdact(t.o[j])
            s += ab * t.x2[j][i]
            # Correct weights in hidden to output layer.
            t.x2[j][i] -= rate * ab * t.h[i]
        # Correct weights in input to hidden layer.
        for j in range(t.nips):
            t.x1[i][j] -= rate * s * pdact(t.h[i]) * in_[j]


def fprop(t: Tinn, in_: List[float]) -> None:
    """Forward propagation."""
    # Calculate hidden layer neuron values.
    for i in range(t.nhid):
        s = t.b[0]  # start with bias
        for j in range(t.nips):
            s += in_[j] * t.x1[i][j]
        t.h[i] = act(s)
    # Calculate output layer neuron values.
    for i in range(t.nops):
        s = t.b[1]  # start with bias
        for j in range(t.nhid):
            s += t.h[j] * t.x2[i][j]
        t.o[i] = act(s)
