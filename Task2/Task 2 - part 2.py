import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import minimize


def f1(x, a, b):
    return a * x + b


def f2(x, a, b):
    return a/(1 + b * x)


def error_func(xdata, ydata):
    return np.sum([(xdata[i] - ydata[i]) * (xdata[i] - ydata[i]) for i in range(len(ydata))])


def error_func_f1(*args):
    try:
        a = args[0][0]
        b = args[0][1]
    except:
        a = args[0]
        b = args[1]
    xdata = [f1(x, a, b) for x in x_k]
    return np.sum([(xdata[i] - y_k[i]) * (xdata[i] - y_k[i]) for i in range(len(y_k))])


def error_func_f2(*args):
    try:
        a = args[0][0]
        b = args[0][1]
    except:
        a = args[0]
        b = args[1]
    xdata = [f2(x, a, b) for x in x_k]
    return np.sum([(xdata[i] - y_k[i]) * (xdata[i] - y_k[i]) for i in range(len(y_k))])


def exhaustive_search(func):
    a = np.linspace(0, 1, 1000)
    b = np.linspace(0, 1, 1000)
    a_best, b_best = None, None
    min_error = 10 ** 6
    for i in range(len(a)):
        for j in range(len(b)):
            err = func(a[i], b[j])
            if err < min_error:
                min_error = err
                a_best = a[i]
                b_best = b[j]
    return a_best, b_best, min_error


def gauss(func):
    a = 1/2
    b = 1/2
    a_grid = np.linspace(0, 1, 101)
    b_grid = np.linspace(0, 1, 101)
    min_f = func(a, b)
    step = 1
    while step > eps/2:
        a_best = a
        b_best = b
        for a_ in a_grid:
            err = func(a_, b)
            if err < min_f:
                min_f = err
                a_best = a_
        step = abs(a - a_best)
        a = a_best
        for b_ in b_grid:
            err = func(a, b_)
            if err < min_f:
                min_f = err
                b_best = b_
        step = min(step, abs(b - b_best))
        b = b_best
    return a, b, min_f


random.seed(1)
alpha = random.uniform(0, 1)
betta = random.uniform(0, 1)
noise = np.random.normal(0, 1, 101)
x_k = [k/100 for k in range(101)]
y_k = [alpha * x_k[k] + betta + noise[k] for k in range(len(x_k))]
eps = 0.001

res = minimize(error_func_f1, [0.5, 0.5],
               bounds=[(0, 1), (0, 1)], method='Nelder-Mead')    # Uses least-squares function
a, b, minerr = gauss(error_func_f1)
a2, b2, minerr2 = exhaustive_search(error_func_f1)


res_rat = minimize(error_func_f2, [0.5, 0.5],
               bounds=[(0, 1), (0, 1)], method='Nelder-Mead')    # Uses least-squares function
a_rat, b_rat, minerr_rat = gauss(error_func_f2)
a2_rat, b2_rat, minerr2_rat = exhaustive_search(error_func_f2)


plt.plot(x_k, y_k)
plt.plot(x_k, [f1(x, a, b) for x in x_k], label='gauss')
plt.plot(x_k, [f1(x, a2, b2) for x in x_k], label='exhaustive search')
plt.plot(x_k, [f1(x, res.x[0], res.x[1]) for x in x_k], label='Nelder-Mead')
plt.title('Linear approximation')
plt.legend()
plt.savefig('Linear approximation')
plt.show()


plt.plot(x_k, y_k)
plt.plot(x_k, [f2(x, a_rat, b_rat) for x in x_k], label='gauss')
plt.plot(x_k, [f2(x, a2_rat, b2_rat) for x in x_k], label='exhaustive search')
plt.plot(x_k, [f2(x, res_rat.x[0], res_rat.x[1]) for x in x_k], label='Nelder-Mead')
plt.title('Rational approximation')
plt.legend()
plt.savefig('Rational approximation')
plt.show()
