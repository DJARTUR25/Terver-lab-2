import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_data(n, a1, a2, b, sigma, t1, t2, s1, s2, seed=None):
    np.random.seed(seed)
    x1 = np.random.uniform(t1, t2, n)
    x2 = 10 - x1 
    X = np.column_stack((x1, x2))
    y_true = a1 * x1 + a2 * x2 + b
    y = y_true + np.random.normal(0, sigma, n)
    return X, y, y_true

def linear_regression(X, y):
    X_with_intercept = np.c_[X, np.ones(X.shape[0])]
    XTX = X_with_intercept.T @ X_with_intercept
    XTy = X_with_intercept.T @ y

    try:
        coefficients = np.linalg.solve(XTX, XTy)  # Метод Гаусса
    except np.linalg.LinAlgError:
        print("Матрица вырождена!")
        return np.zeros(X.shape[1]), 0.0
    return coefficients[:-1], coefficients[-1]

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

def manual_r2_score(a1_true, a2_true, sigma, t1, t2):
    return (a2_true**2 * (t2 - t1)**2) / (12*(sigma**2) + a2_true**2 * (t2-t1)**2)

# Параметры
a1_true = -1
a2_true = -1
b_true = 5
sigma = 2
n_train = 1400
n_test = 300

t1, t2 = 0, 5

# Генерация данных
X_train, y_train, _ = generate_data(n_train, a1_true, a2_true, b_true, sigma, t1, t2, 0, 0, seed=42)
X_test, y_test, _ = generate_data(n_test, a1_true, a2_true, b_true, sigma, t1, t2, 0, 0, seed=100)

# Обучение модели
a_coefs, b_star = linear_regression(X_train, y_train)
a1_star, a2_star = a_coefs

# Прогнозы
y_train_pred = a1_star * X_train[:,0] + a2_star * X_train[:,1] + b_star
y_test_pred = a1_star * X_test[:,0] + a2_star * X_test[:,1] + b_star

# Метрики
r2_train = r2_score(y_train, y_train_pred)
mse_test = np.mean((y_test - y_test_pred)**2)
r2_manual = manual_r2_score(a1_true, a2_true, sigma, t1, t2)

print(f"Истинные: a1 = {a1_true}, a2 = {a2_true}, b = {b_true}")
print(f"Оценки: a1* = {a1_star:.4f}, a2* = {a2_star:.4f}, b* = {b_star:.4f}")
print(f"R² (программа): {r2_train:.4f}, R² (формула): {r2_manual:.4f}, MSE: {mse_test:.4f}")

# Визуализация
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:,0], X_train[:,1], y_train, c='blue', alpha=0.5, label='Train')
ax.scatter(X_test[:,0], X_test[:,1], y_test, c='red', alpha=0.5, label='Test')

x1_grid, x2_grid = np.meshgrid(np.linspace(t1, t2, 10), np.linspace(10 - t2, 10 - t1, 10))
y_grid = a1_star * x1_grid + a2_star * x2_grid + b_star
ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.3, color='green')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
plt.legend()
plt.show()