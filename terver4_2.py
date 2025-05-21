import numpy as np
from sklearn.metrics import r2_score

def generate_data(n, a1, a2, b, sigma, t1, t2, s1, s2):
    """Генерация данных для многомерной регрессии"""
    x1 = np.random.uniform(t1, t2, n)
    x2 = np.random.uniform(s1, s2, n)
    X = np.column_stack((x1, x2))
    y_true = a1 * x1 + a2 * x2 + b
    y = y_true + np.random.normal(0, sigma, n)
    return X, y, y_true

def fit_linear_regression(X, y):
    """Оценка коэффициентов регрессии с использованием МНК"""
    X_with_intercept = np.c_[X, np.ones(X.shape[0])]
    coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    return coefficients[:-1], coefficients[-1]

a1_true = 2.5
a2_true = -1.8
b_true = 3.0
sigma = 0.5
n = 100
m = 20

t1, t2 = 0, 10
s1, s2 = 5, 15

X_train, y_train, y_train_true = generate_data(n, a1_true, a2_true, b_true, sigma, t1, t2, s1, s2)

(a1_star, a2_star), b_star = fit_linear_regression(X_train, y_train)

y_pred = a1_star * X_train[:,0] + a2_star * X_train[:,1] + b_star
r2 = r2_score(y_train, y_pred)

X_test, y_test, y_test_true = generate_data(m, a1_true, a2_true, b_true, sigma, t1, t2, s1, s2)
y_test_pred = a1_star * X_test[:,0] + a2_star * X_test[:,1] + b_star

print(f"Оцененные коэффициенты:\na1* = {a1_star:.3f}\na2* = {a2_star:.3f}\nb* = {b_star:.3f}")
print(f"R²: {r2:.4f}")

mse = np.mean((y_test - y_test_pred)**2)
print(f"MSE на тестовой выборке: {mse:.4f}")