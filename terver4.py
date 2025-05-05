import numpy as np
import matplotlib.pyplot as plt

def get_best_coefs(x, y):
    n = x.size
    s_x = np.sum(x)
    s_y = np.sum(y)
    s_xy = np.sum(x * y)
    s_x2 = np.sum(x**2)
    
    a = (n*s_xy - s_x*s_y) / (n*s_x2 - s_x**2)
    b = (s_xy*s_x - s_y * s_x2) / (s_x*s_x - n*s_x2)
    return a, b

def get_r2(y, y_best):
    y_mean = np.mean(y)
    ss_res = np.sum((y - y_best)**2)
    ss_tot = np.sum((y - y_mean)**2)
    return 1 - (ss_res / ss_tot)

a = 2
b = 2
sigma = 1
n = 100
m = 20

X = np.arange(1, n+1)
y_true = a * X + b
y = y_true + np.random.normal(0, sigma, n)

a_star, b_star = get_best_coefs(X, y)
y_best = a_star * X + b_star

print(f"\nОцененные коэффициенты:\na* = {a_star:.3f}\nb* = {b_star:.3f}")

r2 = get_r2(y, y_best)
print(f"R²: {r2:.8f}")

X_new = np.arange(n+1, n+m+1)
y_new_true = a * X_new + b
y_new = y_new_true + np.random.normal(0, sigma, m)
y_pred = a_star * X_new + b_star 

plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Исходные данные')
plt.plot(X, y_best, color='red', label='Модель')
plt.scatter(X_new, y_new, color='green', marker='x', label='Доп. выборка')
plt.plot(X_new, y_pred, '--', color='orange', label='Прогноз')
plt.legend()
plt.title("Линейная регрессия с упорядоченными X")
plt.show()


t1 = 5
t2 = 15

X_random = np.random.uniform(t1, t2, n)
y_random_true = a * X_random + b
y_random = y_random_true + np.random.normal(0, sigma, n)

a_star_random, b_star_random = get_best_coefs(X_random, y_random)
y_best_random = a_star_random * X_random + b_star_random

print(f"\nОцененные коэффициенты (случайные X):\na* = {a_star_random:.3f}\nb* = {b_star_random:.3f}")
print(f"R² (случайные X): {get_r2(y_random, y_best_random):.8f}")

sorted_indices = np.argsort(X_random)
X_sorted = X_random[sorted_indices]
y_pred_sorted = y_best_random[sorted_indices]

plt.figure(figsize=(10, 6))
plt.scatter(X_random, y_random, label='Случайные данные')
plt.plot(X_sorted, y_pred_sorted, color='red', label='Модель')
plt.title("Линейная регрессия со случайными X")
plt.legend()
plt.show()