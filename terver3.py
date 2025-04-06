import numpy as np
import matplotlib.pyplot as plt

def simulate_tau(T, lambda_param):
    """
    Моделирует момент первого достижения уровня T для случайного блуждания.
    X_{n+1} = X_n + (-1)^n * ξ_{n+1}, где ξ_i ~ Exp(λ).
    Возвращает количество шагов τ.
    """
    X = 0.0
    n = 0
    while True:
        sign = (-1) ** n
        xi = np.random.exponential(scale=1 / lambda_param)
        X += sign * xi
        n += 1
        if abs(X) >= T:
            return n

def main(T, lambda_param, N):
    """
    Выполняет N симуляций, вычисляет выборочное среднее и дисперсию τ,
    строит эмпирическую функцию распределения.
    """
    tau_samples = np.array([simulate_tau(T, lambda_param) for _ in range(N)])
    
    # Выборочное среднее и дисперсия
    sample_mean = np.mean(tau_samples)
    sample_variance = np.var(tau_samples, ddof=1)
    
    # Построение эмпирической функции распределения
    sorted_tau = np.sort(tau_samples)
    y = np.arange(1, N + 1) / N
    
    plt.figure(figsize=(10, 6))
    plt.step(sorted_tau, y, where='post')
    plt.xlabel('Время τ', fontsize=12)
    plt.ylabel('F(τ)', fontsize=12)
    plt.title('Выборочная функция распределения момента достижения уровня T', fontsize=14)
    plt.grid(True)
    plt.show()
    
    return sample_mean, sample_variance

# Параметры задачи
T = 5.0       # Уровень достижения
lambda_param = 2  # Параметр экспоненциального распределения
N = 100      # Объем выборки

# Запуск моделирования
mean, variance = main(T, lambda_param, N)

print(f"Выборочное среднее τ: {mean:.2f}")
print(f"Выборочная дисперсия τ: {variance:.2f}")