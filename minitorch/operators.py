"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, Any

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(a: float, b: float) -> float:
    """Multiplies two numbers"""
    return a * b


def id(a: Any) -> Any:
    """Returns the input unchanged"""
    return a


def add(a: float, b: float) -> float:
    """Adds two numbers"""
    return a + b


def neg(a: float) -> float:
    """Negates a number"""
    return -a


def lt(a: float, b: float) -> float:
    """Checks if a is less than b"""
    return a < b


def eq(a: float, b: float) -> float:
    """Checks if two numbers are equal"""
    return a == b


def max(a: float, b: float) -> float:
    """Returns the max of two numbers"""
    if lt(a, b):
        return b
    return a  # handles the less than or equal case


def is_close(a: float, b: float) -> float:
    """Checks if two numbers are close in value"""
    return (a - b) < 1e-8


def sigmoid(x: float) -> float:
    """Implements the sigmoid function"""
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def relu(x: float) -> float:
    """Implements the ReLU activation function"""
    return x if x > 0.0 else 0.0


def log(x: float) -> float:
    """Calcs natural log"""
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential function"""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal"""
    return 1 / x


def log_back(a: float, b: float) -> float:
    """Computes the derivative of log times a second arg"""
    return b / a


def inv_back(a: float, b: float) -> float:
    """Computes the derivative of reciprocal times a second arg"""
    return -(inv(a ** 2)) * b

def sigmoid_back(a : float, b : float) -> float:
    """Computes the derivative of sig(a) times a second arg"""
    return sigmoid(a) * (1 - sigmoid(a)) * b

def relu_back(a: float, b: float) -> float:
    """Computes the derivative of Relu times a second arg"""
    return 1 * b if a >= 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

def map(fn : Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Applies a given function fn to each element of an iterable"""
    def apply(container : Iterable[float]):
        ret = []
        for x in container:
            ret.append(fn(x))
        return ret
    return apply

def zipWith(fn : Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher order function that combines elements from two iterable using a given function"""
    def apply(ls1 : Iterable[float], ls2 : Iterable[float]):
        ret = []
        for val1, val2 in zip(ls1, ls2):
            ret.append(fn(val1, val2))
        return ret
    return apply

def reduce(fn : Callable[[float], float]) -> Callable[[Iterable[float]], float]:
    """Reduces an iterable to a single value using a given function"""
    def apply(ls : Iterable[float]):
        if len(ls) == 0:
            return 0
        ret = ls[0]
        for ind in range(1, len(ls)):
            ret = fn(ret, ls[ind])
        return ret
    return apply

def negList(ls : Iterable[float]) -> Iterable[float]:
    """negate all elements in a list"""
    neg_function = map(neg)
    return neg_function(ls)

def addLists(l1 : Iterable[float], l2 : Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith"""
    add_zipper = zipWith(add)
    return add_zipper(l1, l2)

def sum(ls : Iterable[float]) -> float:
    """Sum up all elements in list"""
    sum_reduce = reduce(add)
    return sum_reduce(ls)

def prod(ls : Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce"""
    prod_reduce = reduce(mul)
    return prod_reduce(ls)


# TODO: Implement for Task 0.3.
