from functools import reduce
import json
import math
import operator
import sys

import matplotlib.pyplot as plt
import numpy as np


def fourie_term(n: int, xi: float, t: float) -> float:
    l = 0.5 * math.pi + math.pi * n
    return (
        4
        * (-1) ** n
        / (math.pi * (2 * n + 1))
        * math.exp(-(l**2) * t)
        * math.cos(l * xi)
    )


def fourie_series(N: int, xi: float, t: float) -> float:
    term = lambda n: fourie_term(n, xi, t)
    return reduce(operator.add, map(term, range(N)))


def phi(k: int, xi: float) -> float:
    return 1 - math.pow(xi, 2 * k)


def numeric_solution(alpha: np.matrix, lambda_: np.ndarray, c: np.ndarray):
    def f(t: float, xi: float):
        N, _ = alpha.shape
        return reduce(
            operator.add,
            (
                c[i] * math.exp(-lambda_[i] * t) * alpha[j, i] * phi(j + 1, xi)
                for i in range(N)
                for j in range(N)
            ),
        )

    return f


if __name__ == "__main__":
    payload = json.loads(next(sys.stdin))

    alpha = np.array(payload["alpha"])
    lambda_ = np.array(payload["lambda"])
    c = np.array(payload["c"])
    N, _ = alpha.shape

    print(alpha, lambda_, c)

    solution = numeric_solution(alpha, lambda_, c)

    fig, ax = plt.subplots(1, 1)

    t = 0.0
    x = np.linspace(0, 1, 1000)

    numeric_y = np.array(list(map(lambda x: solution(0.0, x), x)))
    ax.plot(x, numeric_y, label="Numeric")

    ax.plot(x, np.ones(x.shape), label="Initial")

    fourie_y = np.array(list(map(lambda x: fourie_series(N, x, t), x)))
    ax.plot(x, fourie_y, label="Fourie")

    ax.set_title(f"Solutions comparison when $t = {t}$")
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("$\\xi$")
    ax.set_ylabel("$\\Theta$")

    fig.savefig("images/test.png")
