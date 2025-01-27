# Lorenz_Attractor
Проект реализует симуляцию системы Лоренца - известной системы дифференциальных уравнений, которая демонстрирует хаотическое поведение.

**Система Лоренца** описывается тремя дифференциальными уравнениями:
   - `lorenz_x`: $\frac{dx}{dt} = s(y - x)$
   - `lorenz_y`: $\frac{dy}{dt} = rx - y - xz$
   - `lorenz_z`: $\frac{dz}{dt} = xy - bz$
где $s$, $r$, $b$ - параметры системы (в коде $s=10$, $b\approx2.667$, $r$ вводится пользователем)

**Численные методы**:

Используется метод Рунге-Кутты 4-го порядка (RK4) для решение системы, а для проверки точности используется метод Дорманда-Принса 8-го порядка, который выступает в роли аналитического*(эталонного) решения.

**Основная симуляция**

   - Начинает с начальной точки $(10, 10, 10)$
   - Итеративно вычисляет следующие состояния системы
   - Сохраняет траекторию движения

**Анализ точности**:

Вычисляет отклонение между двумя траекториями и сравнивает результат с эталонным решением.

**Визуализация**:

В блокноте .ipynb представлены 3D графики траектории системы Лоренца в зависимости от разных входных параметров.

Запуск программы:

1. Запрашивает значение параметра $r$
2. Показывает 3D визуализацию траектории системы
3. Вычисляет и выводит погрешность относительно эталонного решения
