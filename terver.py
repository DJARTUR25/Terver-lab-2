import numpy as np
import math

def generate_x(size):
    x = np.random.uniform(size=size)
    return x

def generate_laplace(lam, size):
    x = generate_x (size)
    y = np.where(x < 0.5, (1/lam) * np.log(2*x), -(1/lam) * np.log(2*(1 - x)))
    return x, y

def main():
    print("Программа для моделирования распределения Лапласа")
    print("Плотность: f(y) = (λ/2) * e^(-λ|y|)")
    print("---------------------------------------------")

    # ввод параметров
    try:
        n = int(input("Введите размер выборки (n): "))
        lambda_param = float(input("Введите параметр λ (>0): "))
        if lambda_param <= 0:
            raise ValueError("λ должен быть положительным числом")
    except ValueError as e:
        print(f"Ошибка ввода: {e}")
        return


    x, sample = generate_laplace(lambda_param, n)

    # вывод параметров
    print("\nПервые 10 значений выборки:")
    print(sample[:10].round(4))
    
    print ("\nПервые 10 значений Х:")
    print(x[:10].round(4))

    # Упорядочивание и вывод
    sorted_sample = np.sort(sample)
    print("\nУпорядоченные y (первые 100 элементов):")
    print(np.array2string(sorted_sample[:100], precision=4, separator=', ', formatter={'float_kind':lambda x: "%.4f" % x}))
    
if __name__ == "__main__":
    main()