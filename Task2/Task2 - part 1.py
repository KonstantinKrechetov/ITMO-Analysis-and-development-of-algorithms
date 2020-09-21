import numpy as np
import matplotlib.pyplot as plt
from math import sin, sqrt

counts = np.zeros(3)    # To count number of function calls


# Declairing considered functions
def f1(x):
    counts[0] += 1
    return x ** 3


def f2(x):
    counts[1] += 1
    return abs(x - 0.2)


def f3(x):
    counts[2] += 1
    return x * sin(1 / x)


# Declaring optimization methods
def brute_force(func, bounds, eps):
    iterations = 0
    x_best = bounds[0]
    func_best = func(x_best)
    for x in np.arange(bounds[0], bounds[1], eps):
        iterations += 1
        if func(x) < func_best:
            func_best = func(x)
            x_best = x
    return x_best, func_best, iterations


def dichotomy(func, bounds, eps):
    iterations = 0
    delta = eps / 2
    a = bounds[0]
    b = bounds[1]
    while b - a >= eps:
        iterations += 1
        x_1 = (a + b)/2 - delta/2
        x_2 = (a + b)/2 + delta/2
        if func(x_1) <= func(x_2):
            b = x_2
        else:
            a = x_1
    return (b + a)/2, func((b + a)/2), iterations


def golden_section(func, bounds, eps):
    iterations = 1
    a = bounds[0]
    b = bounds[1]
    x_1 = a + (3 - sqrt(5)) / 2 * (b - a)
    x_2 = b + (sqrt(5) - 3) / 2 * (b - a)
    func_x1 = func(x_1)
    func_x2 = func(x_2)
    while b - a >= eps:
        if func_x1 <= func_x2:
            b = x_2
            x_2 = x_1
            func_x2 = func_x1
            x_1 = a + (3 - sqrt(5)) / 2 * (b - a)
            func_x1 = func(x_1)
        else:
            a = x_1
            x_1 = x_2
            func_x1 = func_x2
            x_2 = b + (sqrt(5) - 3) / 2 * (b - a)
            func_x2 = func(x_2)
        iterations += 1
    return (b + a)/2, func((b + a)/2), iterations


eps = 0.001

methods = (brute_force, dichotomy, golden_section)
tested_funcs = ((f1, [0, 1]),
                (f2, [0, 1]),
                (f3, [0.01, 1]))
point_sizes = [100, 50, 35]
for ind, tested_func in enumerate(tested_funcs):
    counts_ = []
    for point_size, method in enumerate(methods):
        x, min_f, iters = method(tested_func[0], tested_func[1], eps)
        print(str(method.__name__) + ' called ' + str(tested_func[0].__name__) + ' ' + str(int(counts[ind])) + ' times')
        print(str(method.__name__) + ' x best: ' + str(x))
        print(str(method.__name__) + ' min_f(x): ' + str(min_f))
        counts_.append(counts[ind])
        counts[ind] = 0
        plt.scatter(x, min_f, label=str(method.__name__), s=point_sizes[point_size])
    xdata = np.linspace(tested_func[1][0], tested_func[1][1], 200)
    ydata = [tested_func[0](x) for x in xdata]
    plt.plot(xdata, ydata)
    plt.xlabel('x')
    plt.ylabel(str(tested_func[0].__name__) + '(x)')
    plt.legend()
    plt.savefig(str(tested_func[0].__name__))
    plt.show()

    plt.bar([method.__name__ for method in methods], counts_, width=0.4)
    plt.xlabel('Method')
    plt.ylabel('Function calls')
    plt.title('Function calls for ' + str(tested_func[0].__name__) + '(x)')
    plt.savefig(str(tested_func[0].__name__) + ' bars')
    plt.show()

