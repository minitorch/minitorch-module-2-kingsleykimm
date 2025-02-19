from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple
from collections import defaultdict
from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals = list(vals)
    vals[arg] += epsilon
    plus_eps = f(*vals)
    vals[arg] -= 2 * epsilon
    minus_eps = f(*vals)
    return (plus_eps - minus_eps) /( 2 * epsilon)



variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = set()
    ret_li = []
    def recur(var):
        nonlocal ret_li
        if var.unique_id in visited:
            return
        for p in var.parents:
            if not p.is_constant():
                recur(p)
        visited.add(var.unique_id)
        ret_li = [var] + ret_li
        
    recur(variable)
    return ret_li
                



    # TODO: Implement for Task 1.4.



def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    queue = topological_sort(variable)
    print(queue)
    scalar_to_deriv = defaultdict(int)
    scalar_to_deriv[queue[0].unique_id] = deriv
    # chain_rule_outputs = right_most.chain_rule(deriv)
    # for inp, deriv in chain_rule_outputs:
    #     scalar_to_deriv[repr(inp)] += deriv
    # the right-most variable begins, by applying chain rule
    while queue:
        var = queue.pop(0)
        if not var.is_leaf():
            chain_rule_output = var.chain_rule(scalar_to_deriv[var.unique_id])
            print(chain_rule_output)
            for inp, derivative in chain_rule_output:
                if not inp.is_constant():
                    scalar_to_deriv[inp.unique_id] += derivative
        else:
            var.accumulate_derivative(scalar_to_deriv[var.unique_id])
            # var.derivative = scalar_to_deriv[repr(var)]
            # print(var.derivative)
            # print(scalar_to_deriv)
    # print(scalar_to_deriv)


    

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
