import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import minimize, least_squares,\
    differential_evolution, dual_annealing


# Declairing considered functions
def f(x):
    return 1/(x**2 - 3*x + 2)


def approx_f(x, a, b, c, d):
    return (a * x + b) / (x**2 + c*x + d)


def error_func(parms, func=approx_f):   # Least-square error for function f1
    a = parms[0]
    b = parms[1]
    c = parms[2]
    d = parms[3]
    approx_data = [func(x, a, b, c, d) for x in x_k]
    ans = np.sum([(approx_data[i] - y_k[i]) * (approx_data[i] - y_k[i]) for i in range(len(y_k))])
    return ans


def error_func_lm(parms, func=None):   # Least-square error for function f1
    a = parms[0]
    b = parms[1]
    c = parms[2]
    d = parms[3]
    approx_data = [func(x, a, b, c, d) for x in x_k]
    return [(approx_data[i] - y_k[i]) * (approx_data[i] - y_k[i]) for i in range(len(y_k))]


# Preparing data to be approximated
random.seed(1)
eps = 0.001
alpha = random.uniform(0, 1)
betta = random.uniform(0, 1)
noise = np.random.normal(0, 1, 1001)
x_k = [3 * k/1000 for k in range(1001)]
y_k = []
for k in range(len(x_k)):
    f_ = f(x_k[k])
    if f_ < -100:
        y_k.append(-100 + noise[k])
    elif -100 <= f_ <= 100:
        y_k.append(f_ + noise[k])
    else:
        y_k.append(100 + noise[k])


# Approximation of function
res = least_squares(error_func_lm, x0=[1, 1, 1, 1], method='lm', args=[approx_f])
res2 = minimize(error_func, x0=np.array([1, 1, 1, 1]), args=approx_f,
                method='Nelder-Mead', options={'disp': True})
bound = 4
res3 = differential_evolution(error_func, bounds=[(-bound, bound)] * 4,
                              args=[approx_f], disp=True, tol=1e-6, popsize=20)
res4 = dual_annealing(error_func, bounds=[(-bound, bound)] * 4,
                              args=[approx_f])


# Saving plot for linear approximation
approx_data = [error_func([0, 1, -3, 2]) for x in x_k]
res0 = np.sum([(approx_data[i] - y_k[i]) * (approx_data[i] - y_k[i]) for i in range(len(y_k))])
a, b, c, d = res.x
a2, b2, c2, d2 = res2.x
a3, b3, c3, d3 = res3.x
a4, b4, c4, d4 = res4.x


print(res0, res.cost, res2.fun, res3.fun, res4.fun, sep='\n')
plt.scatter(x_k, y_k, label='Approximated data', s=1, color='r')
plt.plot(x_k, [approx_f(x, a, b, c, d) for x in x_k],
         label='Levenberg-Marquardt', linewidth=4, alpha=0.5)
plt.plot(x_k, [approx_f(x, a2, b2, c2, d2) for x in x_k],
         label='Nelder-Mead', linewidth=3, alpha=0.5)
plt.plot(x_k, [approx_f(x, a3, b3, c3, d3) for x in x_k],
         label='Differential Evolution', linewidth=2, alpha=0.5)
plt.plot(x_k, [approx_f(x, a4, b4, c4, d4) for x in x_k],
         label='Dual Annealing', linewidth=1, alpha=0.5)
plt.legend()
plt.title('Approximation')
# plt.savefig('Approximation', dpi=300)
plt.show()
plt.close()


plt.bar(['Levenberg-Marquardt', 'Nelder-Mead', 'Differential evolution', 'Dual Annealing'],
        [res.nfev, res2.nfev, res3.nfev, res4.nfev], width=0.4)
plt.title('Number of function evaluations')
plt.savefig('Number of function evaluations', dpi=300)
plt.show()

