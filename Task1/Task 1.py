import numpy as np
from timeit import default_timer
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from decimal import Decimal


# Defining considered methods and functions
def f_const(v):
    return 0


def f_sum(v):
    return np.sum(v)


def f_prod(v):
    return np.product(v * v)


def polynom_direct(v, x=1.5):
    sum = Decimal(0)
    for k in range(len(v)):
        sum += Decimal(v[k]) * Decimal(x) ** Decimal(k)
    return sum


def polynom_horner(v, x=1.5):
    result = v[-1]
    for i in range(len(v) - 2, -1, -1):
        result = result * x + v[i]
    return result


def bubble_sort(v):
    sorted_v = v.copy()
    for i in range(len(sorted_v) - 1):
        for j in range(0, len(sorted_v) - i - 1):
            if sorted_v[j] > sorted_v[j + 1]:
                sorted_v[j], sorted_v[j + 1] = sorted_v[j + 1], sorted_v[j]
    return list(sorted_v)


def quicksort(v):
    return np.sort(v)


def timsort(v):
    return np.sort(v, kind='stable')


def matrices(v):
    size = len(v)
    A = np.random.rand(size, size) * 10 ** 3
    B = np.random.rand(size, size) * 10 ** 3
    C = np.dot(A, B)
    return C


# Defining approximation functions to evaluate their parameters
def approx_const(x, c):
    return [c] * len(x)


def approx_n(x, a, c):
    return np.multiply(a, x) + c


def approx_nlogn(x, a, c):
    logx = np.log2(x)
    return np.multiply(a, np.array(x) * logx) + c


def approx_n2(x, a, c):
    x2 = np.array([x[i] ** 2 for i in range(len(x))])
    return a * x2 + c


def approx_n3(x, a, c):
    x3 = np.array([x[i] ** 3 for i in range(len(x))])
    return a * x3 + c


# Matching complexity pairs
complexity = [
    (f_const, approx_const),
    (f_sum, approx_n),
    (f_prod, approx_n),
    (polynom_direct, approx_n),
    (polynom_horner, approx_n),
    (bubble_sort, approx_n2),
    (quicksort, approx_nlogn),
    (timsort, approx_nlogn),
    (matrices, approx_n3)
]
xdata = [i for i in range(1, 2001)]

for pair in complexity:
    exec_time_list = []
    for n in range(1, 2001):
        print('Iteration ' + str(n))
        exec_time = []
        for i in range(5):
            vec = np.random.uniform(1, 10 ** 3, n)  # Creating random vector of size n
            start = default_timer()
            ans = pair[0](vec)  # Call considered function
            exec_time.append(default_timer() - start)
        exec_time_list.append(exec_time)

    avg_exec_time = np.mean(exec_time_list, axis=1)  # Average exec. time of 5 iterations
    popt, pcov = curve_fit(pair[1], xdata, avg_exec_time)  # Fitting the theoretical curve
    # Creating and saving plots for each method
    plt.plot(xdata, avg_exec_time, label='Experimental', color='r')
    plt.plot(xdata, pair[1](xdata, *popt), label='Theoretical', color='b')
    plt.title(str(pair[0].__name__))
    plt.legend()
    plt.xlabel('n')
    plt.ylabel('Time, sec')
    plt.tight_layout()
    plt.savefig(str(pair[0].__name__))
    plt.show()
    plt.close()
