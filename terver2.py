import numpy as np
import math
import matplotlib.pyplot as plt

def generate_x(size):
    """Генератор U(0,1)."""
    return np.random.uniform(size=size)

def generate_laplace(lam, size):
    """
    Генерация выборки из распределения Лапласа, 
    плотность f(y) = (lam/2)*exp(-lam*|y|).
    """
    x = generate_x(size)
    # формула моделирования через обратную функцию распределения
    # y < 0, если x < 0.5
    # y >= 0, если x >= 0.5
    y = np.where(x < 0.5,
                 (1/lam)*np.log(2*x),
                 -(1/lam)*np.log(2*(1 - x)))
    return y

def theoretical_cdf_laplace(y, lam):
    """
    Теоретическая функция распределения F_eta(y) для Лапласа(0, lam),
    то есть f_eta(y) = (lam/2) * exp(-lam*|y|).
    F(y) = 1/2 * exp(lam*y),   y < 0
           1 - 1/2 * exp(-lam*y), y >= 0
    """
    if y < 0:
        return 0.5 * math.exp(lam * y)
    else:
        return 1 - 0.5 * math.exp(-lam * y)


def part1():
    print("Программа для распределения Лапласа (f(y) = (λ/2) * e^(-λ|y|))")
    """
    Первая часть: генерация выборки из распределения Лапласа,
    вывод первых элементов и упорядоченной выборки.
    """
    print("Часть 1: Генерация выборки из распределения Лапласа.")
    try:
        n = int(input("Введите размер выборки (n): "))
        lambda_param = float(input("Введите параметр λ (>0): "))
        if lambda_param <= 0:
            raise ValueError("λ должен быть положительным числом")
    except ValueError as e:
        print(f"Ошибка ввода: {e}")
        return

    sample = generate_laplace(lambda_param, n)

    print("\nПервые 10 значений выборки:")
    print(sample[:10].round(4))

    # Упорядочивание и вывод первых 100
    sorted_sample = np.sort(sample)
    print("\nУпорядоченные y (первые 100 элементов):")
    print(np.array2string(sorted_sample[:100],
                          precision=4,
                          separator=', ',
                          formatter={'float_kind':lambda x: "%.4f" % x}))
    return sample, lambda_param, n

def part2(sample, lambda_param, n):
    """
    Вторая часть: расчет теоретических и выборочных статистических характеристик,
    построение эмпирической функции распределения и сравнение с теоретической.
    (Опционально - гистограмма и сравнение с теоретической плотностью)
    """
    print("Часть 2: Статистические характеристики распределения Лапласа.")

    # Теоретические характеристики:
    # Для f(y) = (λ/2)* e^(-λ|y|) имеем M=0, D=2/λ^2
    E_theor = 0.0
    D_theor = 2.0/(lambda_param**2)

    # Выборочные характеристики
    x_mean = np.mean(sample)         # выборочное среднее
    if n == 1:
        s2 = 0.0  # Для одного элемента разброс считается нулевым
        print(f"Выборочная дисперсия S^2 = {s2:.4f} (выборка из одного элемента)")
    else:
        s2 = np.var(sample, ddof=1)
        print(f"Выборочная дисперсия S^2 = {s2:.4f}")
    median_emp = np.median(sample)   # выборочная медиана
    sample_min = np.min(sample)
    sample_max = np.max(sample)
    R = sample_max - sample_min      # размах

    print("\nТеоретические характеристики:")
    print(f"E(η) = {E_theor:.4f}")
    print(f"D(η) = {D_theor:.4f}")

    print("\nВыборочные характеристики:")
    print(f"Выборочное среднее x̄ = {x_mean:.4f}")
    print(f"Выборочная дисперсия S^2 = {s2:.4f}")
    print(f"Медиана выборки = {median_emp:.4f}")
    print(f"Размах выборки = {R:.4f}")

    print("\nСравнение теоретических и выборочных значений (в модуле):")
    print(f"|E - x̄| = {abs(E_theor - x_mean):.4f}")
    print(f"|D - S^2| = {abs(D_theor - s2):.4f}")

    # Построение эмпирической функции распределения и сравнение с теоретической (на точках выборки)
    sorted_sample = np.sort(sample)
    F_emp_after = np.arange(1, n+1) / n
    F_emp_before = np.arange (0, n) / n

    # Теоретические значения F_theor в упорядоченных точках
    F_theor_values = [theoretical_cdf_laplace(y, lambda_param) for y in sorted_sample]

    # Максимальное отклонение D = max|F_emp - F_theor|
    diffs_before = np.abs(F_emp_before - F_theor_values)
    diffs_after = np.abs(F_emp_after - F_theor_values)
    D_stat = np.max(np.maximum(diffs_after, diffs_before))

    print(f"\nМаксимальное отклонение эмпирической и теоретической функций распределения:")
    print(f"D = {D_stat:.6f}")

    answer = input("\nПостроить гистограмму и график отклонений? (д/н): ")
    if answer.lower().startswith('д'):

        # гистограмма
        plt.figure(figsize=(7,4))
        plt.hist(sample, bins='auto', density=True, alpha=0.5, label='Гистограмма выборки')

        y_vals = np.linspace(sample_min, sample_max, n + 100)
        f_vals = 0.5*lambda_param*np.exp(-lambda_param*np.abs(y_vals))

        plt.plot(y_vals, f_vals, 'r-', lw=2, label='Теоретическая плотность')
        plt.title("Гистограмма и теоретическая плотность (Laplace)")
        plt.xlabel("y")
        plt.ylabel("плотность")
        plt.legend()
        plt.grid(True)
        plt.show()

        samp = sample
        for i in range (n):
            samp[i] = theoretical_cdf_laplace(sorted_sample[i], lambda_param)
        
        print("\nПервые 10 значений теор.функции распределения:")
        print(samp[:10].round(4))

        y_left = math.log(0.02) / lambda_param  # F(y_left) = 0.01
        y_right = -math.log(0.02) / lambda_param  # F(y_right) = 0.99

        left_bound = min(np.min(sample), y_left)
        right_bound = max(np.max(sample), y_right)

        fine_grid = np.linspace(left_bound, right_bound, 300)
        fine_theor_cdf = [theoretical_cdf_laplace(val, lambda_param) for val in fine_grid]

        # Построение графиков
        plt.step(np.concatenate([[-np.inf], sorted_sample, [np.inf]]), np.concatenate([[0], F_emp_after, [1]]), where='post', label='Выборочная F̂(y)')
        
        plt.plot(fine_grid, fine_theor_cdf, 'r-', label='Теоретическая F(y)')
        plt.xlim(left_bound, right_bound)

        plt.title("Сравнение выборочной и теоретической функций распределения")
        plt.xlabel("y")
        plt.ylabel("F(y)")
        plt.legend()
        plt.grid(True)
        plt.show()



    # Часть 3: Проверка гипотезы о виде распределения
def part3(lambda_param, n):
    print("\nЧасть 3: Проверка гипотезы о виде распределения с использованием критерия хи-квадрат.")
    import scipy.stats as stats

    try:
        # Ввод параметров для многократной проверки
        num_tests = int(input("\nВведите количество проверок гипотезы: "))
        k = int(input("Введите количество интервалов k (>=2): "))
        alpha = float(input("Введите уровень значимости α (например, 0.05): "))
        
        if k < 2:
            raise ValueError("Количество интервалов должно быть не менее 2")
        if not (0 < alpha < 1):
            raise ValueError("α должен быть в интервале (0, 1)")

        schet_da = 0
        schet_net = 0

        for test in range(num_tests):
            # Генерация новой выборки для каждого теста
            current_sample = generate_laplace(lambda_param, n)

            # Автоматическое определение границ через квантили распределения Лапласа
            prob_points = np.linspace(0, 1, k+1)[1:-1]  # Исключаем 0 и 1
            boundaries = [
                np.sign(p - 0.5) * (1 / lambda_param) * np.log(1 - 2 * np.abs(p - 0.5))
                for p in prob_points
            ]
            boundaries_sorted = sorted(boundaries)

            # Формирование интервалов
            intervals = [(-np.inf, boundaries_sorted[0])]
            for i in range(len(boundaries_sorted) - 1):
                intervals.append((boundaries_sorted[i], boundaries_sorted[i+1]))
            intervals.append((boundaries_sorted[-1], np.inf))

            # Вычисление теоретических вероятностей q_j
            q = []
            for interval in intervals:
                lower, upper = interval
                F_lower = (
                    theoretical_cdf_laplace(lower, lambda_param)
                    if lower != -np.inf
                    else 0.0
                )
                F_upper = (
                    theoretical_cdf_laplace(upper, lambda_param)
                    if upper != np.inf
                    else 1.0
                )
                q.append(F_upper - F_lower)

            # Проверка условий надежности критерия
            valid = all(qj * n >= 5 for qj in q)
            if not valid:
                print(f"Проверка {test+1}: критерий ненадежен (n*q_j < 5)")
                continue  # Пропускаем тест

            # Подсчет наблюдаемых частот n_j
            bins = [-np.inf] + boundaries_sorted + [np.inf]
            n_j, _ = np.histogram(current_sample, bins=bins)

            # Вычисление статистики R0
            R0 = sum((n_j[j] - n * q[j]) ** 2 / (n * q[j]) for j in range(k) if q[j] > 0)

            # Определение p-значения
            df = k - 1
            p_value = stats.chi2.sf(R0, df)

            # Обновление счетчиков
            if p_value < alpha:
                schet_da += 1
            else:
                schet_net += 1

        # Итоговый вывод
        print(f"\nИтоги {num_tests} проверок:")
        print(f"Гипотеза отвергнута: {schet_da} раз")
        print(f"Гипотеза принята: {schet_net} раз")

    except ValueError as e:
        print(f"Ошибка: {e}")

def main_menu():
    sample = None
    lambda_param = None
    generated = False

    while True:
        print("\nГлавное меню:")
        print("1. Часть 1 - Генерация выборки")
        print("2. Часть 2 - Статистические характеристики")
        print("3. Часть 3 - Проверка гипотез")
        print("4. Выход")

        choice = input("Выберите пункт (1-4): ").strip()

        if choice == '1':
            sample, lambda_param, n = part1()
            generated = True
        elif choice == '2':
            if not generated:
                print("Сначала выполните Часть 1!")
            else:
                part2(sample, lambda_param, n)
        elif choice == '3':
            if not generated:
                try:
                    n = int(input("Введите размер выборки (n): "))
                    lambda_param = float(input("Введите параметр λ (>0): "))
                    if lambda_param <= 0:
                        raise ValueError("λ должен быть положительным числом")
                except ValueError as e:
                    print(f"Ошибка ввода: {e}")
                part3(lambda_param, n)
            else:
                part3(lambda_param, n)
        elif choice == '4':
            print("Выход из программы.")
            break
        else:
            print("Неверный ввод. Введите число от 1 до 4.")

if __name__ == "__main__":
    main_menu()