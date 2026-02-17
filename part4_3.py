import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Загрузка данных
data = pd.read_csv('Automobile.csv')

# Удаление строк с пропущенными значениями в выбранных столбцах
data_clean = data.dropna(subset=['horsepower', 'cylinders'])

# Преобразование horsepower в числовой формат (если нужно)
data_clean['horsepower'] = pd.to_numeric(data_clean['horsepower'], errors='coerce')
data_clean = data_clean.dropna(subset=['horsepower'])

# Выбор признаков
X = data_clean[['cylinders']].values  # Предиктор (количество цилиндров)
y = data_clean['horsepower'].values   # Целевая переменная (мощность)

# Обучение модели
model = LinearRegression()
model.fit(X, y)

# Предсказание
y_pred = model.predict(X)

# Оценка модели
r2 = r2_score(y, y_pred)
print(f"Коэффициент детерминации R²: {r2:.4f}")

# Визуализация
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Данные')
plt.plot(X, y_pred, color='red', linewidth=2, label='Линейная регрессия')
plt.xlabel('Количество цилиндров')
plt.ylabel('Мощность (horsepower)')
plt.title('Зависимость мощности от количества цилиндров')
plt.legend()
plt.grid(True)
plt.show()