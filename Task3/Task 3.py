import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import minimize, least_squares


# Declairing considered functions
def f1(x, a, b):
    return a * x + b


def f2(x, a, b):
    return a/(1 + b * x)


def error_func(parms, func=None):   # Least-square error for function f1
    a = parms[0]
    b = parms[0]
    approx_data = [func(x, a, b) for x in x_k]
    return np.sum([(approx_data[i] - y_k[i]) * (approx_data[i] - y_k[i]) for i in range(len(y_k))])


def error_func_lm(parms, func=None):   # Least-square error for function f1
    a = parms[0]
    b = parms[0]
    approx_data = [func(x, a, b) for x in x_k]
    return [(approx_data[i] - y_k[i]) * (approx_data[i] - y_k[i]) for i in range(len(y_k))]


# Declairing optimization methods
def gradDescent(func):
    x = np.random.uniform(0, 1, 2)
    gamma = 1/10
    iter = 0
    func_eval = 0
    while gamma > eps:
        x_new = x - gamma * jac(x, func=func)
        if error_func(x_new, func=func) <= error_func(x, func=func):
            x = x_new
        func_eval += 2
        iter += 1
        gamma *= 0.85
        print(iter)
    return x[0], x[1]


# Jacobian
def jac(params, func=None):
    a = params[0]
    b = params[1]
    df_da = df_db = 0
    if func == f1:
        for i in range(len(x_k)):
            df_da += 2 * (a * x_k[i] + b - y_k[i]) * x_k[i]
            df_db += 2 * (a * x_k[i] + b - y_k[i])
    elif func == f2:
        for i in range(len(x_k)):
            df_da += 2 * (a * x_k[i] + b - y_k[i]) / (1 + b * x_k[i])
            df_db += 2 * (a * x_k[i] + b - y_k[i]) * (-a * x_k[i]) / (1 + b * x_k[i])**2
    return np.array([df_da, df_db])


random.seed(1)  # For repeating results
alpha = random.uniform(0, 1)
betta = random.uniform(0, 1)
noise = np.random.normal(0, 1, 101)
x_k = [k/100 for k in range(101)]
y_k = [alpha * x_k[k] + betta + noise[k] for k in range(len(x_k))]
eps = 0.001


# Linear approximation of function
# a, b = gradDescent(f1)
# print('Gradient descent error: ' + str(error_func([a, b], func=f1)))
# res = minimize(error_func, x0=[1/2, 1/2], jac=jac, args=f1, method='CG', options={'disp': True, 'maxiter': 1})
# res2 = minimize(error_func, x0=[1/2, 1/2], jac=jac, args=f1, method='Newton-CG', options={'disp': True})
# res3 = least_squares(error_func_lm, x0=[1/2, 1/2], method='lm', args=[f1])
#
# Saving plot for linear approximation
# plt.plot(x_k, y_k)
# plt.plot(x_k, [f1(x, a, b) for x in x_k], label='Gradient Descent')
# plt.plot(x_k, [f1(x, res.x[0], res.x[1]) for x in x_k], label='CG')
# plt.plot(x_k, [f1(x, res2.x[0], res2.x[1]) for x in x_k], label='Newton-CG')
# plt.plot(x_k, [f1(x, res3.x[0], res3.x[1]) for x in x_k], label='Levenberg-Marquardt')
# plt.title('Linear approximation')
# plt.legend()
# # plt.savefig('Linear approximation', dpi=300)
# plt.show()
# plt.close()


# Rational approximation of function
a, b = gradDescent(f2)
print('Gradient descent error: ' + str(error_func([a, b], func=f2)))
res = minimize(error_func, x0=[1/2, 1/2], jac=jac, args=f2, method='CG', options={'disp': True, 'maxiter': 1})
res2 = minimize(error_func, x0=[1/2, 1/2], jac=jac, args=f2, method='Newton-CG', options={'disp': True})
res3 = least_squares(error_func_lm, x0=[1/2, 1/2], method='lm', args=[f2])

# Saving plot for rational approximation
plt.plot(x_k, y_k)
plt.plot(x_k, [f2(x, a, b) for x in x_k], label='Gradient Descent')
plt.plot(x_k, [f2(x, res.x[0], res.x[1]) for x in x_k], label='CG')
plt.plot(x_k, [f2(x, res2.x[0], res2.x[1]) for x in x_k], label='Newton-CG')
plt.plot(x_k, [f2(x, res3.x[0], res3.x[1]) for x in x_k], label='Levenberg-Marquardt')
plt.title('Rational approximation')
plt.legend()
# plt.savefig('Rational approximation', dpi=300)
plt.show()


plt.bar(['Gradient Descent', 'CG', 'Newton-CG', 'Levenberg-Marquardt'],
        [100, res.nfev, res2.nfev, res3.nfev], width=0.4)
plt.title('Number of function evaluations')
# plt.savefig('Number of function evaluations for rational approximation', dpi=300)
plt.show()

