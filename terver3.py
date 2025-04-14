import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import lognorm

lambda_param = 2
T = 2
N = 1000

def simulate_rand(T, N, lambda_param):
    tau_values = []
    for _ in range(N):
        X_n, n = 0.0, 0
        sign = 1
        while abs(X_n) < T:
            xi = np.random.exponential(scale=1/lambda_param)
            X_n += sign * xi
            sign *= -1
            n += 1
        tau_values.append(n)
    return tau_values

tau_values = simulate_rand(T, N, lambda_param)
sample_mean = np.mean(tau_values)
sample_variance = np.var(tau_values, ddof=1)

log_tau = np.log(tau_values)
mu_est = np.mean(log_tau)
sigma_est = np.std(log_tau)

lognorm_dist = lognorm(s=sigma_est, scale=np.exp(mu_est))

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.kdeplot(tau_values, bw_adjust=0.5, color='deeppink', fill=True)
plt.xlabel('τ')
plt.ylabel('Плотность вероятности')
plt.title(f'Выборочная плотность распределения\nСреднее: {sample_mean:.1f}, Дисперсия: {sample_variance:.1f}')
plt.grid(True)

plt.subplot(1, 2, 2)
sorted_tau = np.sort(tau_values)
ecdf = np.arange(1, N+1) / N

x = np.linspace(1, max(tau_values), 1000)
cdf_lognorm = lognorm_dist.cdf(x)

plt.step(sorted_tau, ecdf, where='post', color='deeppink', label='Выборочная CDF')
plt.plot(x, cdf_lognorm, 'b-', label=f'Логнормальное распределение\nμ={mu_est:.2f}, σ={sigma_est:.2f}')

plt.xlabel('τ')
plt.ylabel('P(τ ≤ t)')
plt.title(f'Функция распределения τ\nT={T}, λ={lambda_param}, N={N}')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Выборочное среднее: {sample_mean:.2f}")
print(f"Выборочная дисперсия: {sample_variance:.2f}")
print(f"Оцененные параметры логнормального распределения:")
print(f"μ = {mu_est:.4f}, σ = {sigma_est:.4f}")