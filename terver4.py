import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ================== Часть 1-4: Упорядоченные значения X ==================
# 1. Генерация исходных данных
a = float(input("Введите коэффициент a: "))
b = float(input("Введите коэффициент b: "))
sigma = float(input("Введите стандартное отклонение sigma: "))
n = int(input("Введите размер выборки n: "))
m = int(input("Введите размер доп. выборки m: "))

# Генерируем X = [1, 2, ..., n] и y с шумом
X = np.arange(1, n+1).reshape(-1, 1)
y_true = a * X + b
y = y_true + np.random.normal(0, sigma, size=(n, 1))

# 2. Обучение модели линейной регрессии
model = LinearRegression()
model.fit(X, y)
a_star = model.coef_[0][0]
b_star = model.intercept_[0]

print(f"\nОцененные коэффициенты:\na* = {a_star:.3f}\nb* = {b_star:.3f}")

# 3. Расчет коэффициента детерминации R²
r2 = model.score(X, y)
print(f"R²: {r2:.3f}")

# 4. Генерация дополнительной выборки и сравнение
X_new = np.arange(n+1, n+m+1).reshape(-1, 1)
y_new_true = a * X_new + b
y_new = y_new_true + np.random.normal(0, sigma, size=(m, 1))
y_pred = model.predict(X_new)

# Визуализация
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Исходные данные')
plt.plot(X, model.predict(X), color='red', label='Модель')
plt.scatter(X_new, y_new, color='green', marker='x', label='Доп. выборка')
plt.plot(X_new, y_pred, '--', color='orange', label='Прогноз')
plt.legend()
plt.title("Линейная регрессия с упорядоченными X")
plt.show()

# ================== Часть 5: Случайные значения X ==================
t1 = float(input("\nВведите начало отрезка t1: "))
t2 = float(input("Введите конец отрезка t2: "))

# Генерация случайных X
X_random = np.random.uniform(t1, t2, size=(n, 1))
y_random_true = a * X_random + b
y_random = y_random_true + np.random.normal(0, sigma, size=(n, 1))

# Обучение новой модели
model_random = LinearRegression()
model_random.fit(X_random, y_random)
a_star_random = model_random.coef_[0][0]
b_star_random = model_random.intercept_[0]

print(f"\nОцененные коэффициенты (случайные X):\na* = {a_star_random:.3f}\nb* = {b_star_random:.3f}")
print(f"R² (случайные X): {model_random.score(X_random, y_random):.3f}")

# Визуализация для случайных X
plt.figure(figsize=(10, 6))
plt.scatter(X_random, y_random, label='Случайные данные')
plt.plot(np.sort(X_random, axis=0), 
         model_random.predict(np.sort(X_random, axis=0)), 
         color='red', label='Модель')
plt.title("Линейная регрессия со случайными X")
plt.legend()
plt.show()