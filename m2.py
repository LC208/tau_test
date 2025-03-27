import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


def pid_controller(error, q1, q2, q3, prev_error, integral, dt):
    integral += error * dt
    derivative = (error - prev_error) / dt
    output = q1 * error + q2 * integral + q3 * derivative
    return output, integral, derivative


def runge_kutta_step(y, z2, control_signal, T, eps, dt, k):
    def f1(y, z2):
        return z2 - (2 * eps * y) / T

    def f2(y, g):
        return (k / (T * T)) * g - (1 / (T * T)) * y

    k1 = dt * f1(y, z2)
    q1 = dt * f2(y, control_signal)
    k2 = dt * f1(y + k1 / 2, z2 + q1 / 2)
    q2 = dt * f2(y + k1 / 2, control_signal)
    k3 = dt * f1(y + k2 / 2, z2 + q2 / 2)
    q3 = dt * f2(y + k2 / 2, control_signal)
    k4 = dt * f1(y + k3, z2 + q3)
    q4 = dt * f2(y + k3, control_signal)
    z2 = z2 + (q1 + 2 * q2 + 2 * q3 + q4) / 6
    y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return y, z2


def simulate_system(
    q1=0,
    q2=0,
    q3=0,
    T=1,
    eps=0.75,
    delay=0.0,
    time_end=20,
    dt=0.001,
    k=1,
    show=False,
    show_delay=False,
    show_ksi=False,
    calc_dI=False,
    et_hist=False,
    ax=None,
    label=None,
):

    t = np.arange(0, time_end, dt)
    h = np.zeros_like(t)
    if calc_dI or show_ksi:
        dI_dq1, dI_dq2, dI_dq3 = 0, 0, 0
        h_ksi1 = np.zeros_like(t)
        h_ksi2 = np.zeros_like(t)
        h_ksi3 = np.zeros_like(t)
    if show_delay:
        h_nd = np.zeros_like(t)
        if calc_dI or show_ksi:
            h_ksi1_nd = np.zeros_like(t)
            h_ksi2_nd = np.zeros_like(t)
            h_ksi3_nd = np.zeros_like(t)

    y, z2, I_x = 0, 0, 0
    y_et = 0
    et_h = []
    prev_error, integral = 0, 0
    delay_steps = int(delay / dt)
    buffer = np.zeros(delay_steps) if delay_steps > 0 else None
    if calc_dI or show_ksi:
        buffer_ksi1 = np.zeros(delay_steps) if delay_steps > 0 else None
        buffer_ksi2 = np.zeros(delay_steps) if delay_steps > 0 else None
        buffer_ksi3 = np.zeros(delay_steps) if delay_steps > 0 else None

        ksi1, ksi1_i, prev_ksi1, z21 = 0, 0, 0, 0
        ksi2, ksi2_i, prev_ksi2, z22 = 0, 0, 0, 0
        ksi3, ksi3_i, prev_ksi3, z23 = 0, 0, 0, 0
    for i in range(1, len(t)):
        error = 1 - h[i - 1]
        control_signal, integral, derivative = pid_controller(
            error, q1, q2, q3, prev_error, integral, dt
        )
        I_x += ((y - y_et) ** 2) * dt
        y_et = apz_runge_kutta(y_et, 1, 1 / 3, 1, dt)
        if et_hist:
            et_h.append(y_et)
        y, z2 = runge_kutta_step(y, z2, control_signal, T, eps, dt, k)
        if calc_dI or show_ksi:
            ksi1_i += ksi1 * dt
            ksi1_d = (ksi1 - prev_ksi1) / dt
            du_dq1 = error - q1 * ksi1 - q2 * ksi1_i - q3 * ksi1_d
            prev_ksi1 = ksi1
            ksi1, z21 = runge_kutta_step(ksi1, z21, du_dq1, T, eps, dt, k)

            ksi2_i += ksi2 * dt
            ksi2_d = (ksi2 - prev_ksi2) / dt
            du_dq2 = integral - q1 * ksi2 - q2 * ksi2_i - q3 * ksi2_d
            prev_ksi2 = ksi2
            ksi2, z22 = runge_kutta_step(ksi2, z22, du_dq2, T, eps, dt, k)

            ksi3_i += ksi3 * dt
            ksi3_d = (ksi3 - prev_ksi3) / dt
            du_dq3 = derivative - q1 * ksi3 - q2 * ksi3_i - q3 * ksi3_d
            prev_ksi3 = ksi3
            ksi3, z23 = runge_kutta_step(ksi3, z23, du_dq3, T, eps, dt, k)
            if calc_dI:
                dI_dq1 -= 2 * (y_et - y) * ksi1 * dt
                dI_dq2 -= 2 * (y_et - y) * ksi2 * dt
                dI_dq3 -= 2 * (y_et - y) * ksi3 * dt
        prev_error = error
        if show_delay:
            h_nd[i] = y
            if calc_dI or show_ksi:
                h_ksi1_nd[i] = ksi1
                h_ksi2_nd[i] = ksi2
                h_ksi3_nd[i] = ksi3

        if calc_dI or show_ksi:
            if buffer_ksi1 is not None:
                buffer_ksi1 = np.roll(buffer_ksi1, -1)
                buffer_ksi1[-1] = ksi1
                h_ksi1[i] = buffer_ksi1[0]
            else:
                h_ksi1[i] = ksi1

            if buffer_ksi2 is not None:
                buffer_ksi2 = np.roll(buffer_ksi2, -1)
                buffer_ksi2[-1] = ksi2
                h_ksi2[i] = buffer_ksi2[0]
            else:
                h_ksi2[i] = ksi2

            if buffer_ksi3 is not None:
                buffer_ksi3 = np.roll(buffer_ksi3, -1)
                buffer_ksi3[-1] = ksi3
                h_ksi3[i] = buffer_ksi3[0]
            else:
                h_ksi3[i] = ksi3

        if buffer is not None:
            buffer = np.roll(buffer, -1)
            buffer[-1] = y
            h[i] = buffer[0]
        else:
            h[i] = y
    if show:
        plt.plot(
            t,
            h,
            label=f"q1={q1}, q2={q2}, q3={q3}, T={T}, eps={eps}, delay={delay}, I={I_x}",
        )
    if show_ksi:
        plt.plot(
            t,
            h_ksi1,
            label=f"ksi1",
        )
        plt.plot(
            t,
            h_ksi2,
            label=f"ksi2",
        )
        plt.plot(
            t,
            h_ksi3,
            label=f"ksi3",
        )

    if show_delay:
        plt.plot(t, h_nd, label="Система с задержкой")
        if show_ksi:
            plt.plot(t, h_ksi1_nd, label="ksi1 с задержкой")
            plt.plot(t, h_ksi2_nd, label="ksi2 с задержкой")
            plt.plot(t, h_ksi3_nd, label="ksi3 с задержкой")

    if ax is not None:
        if label is not None:
            ax.plot(t, h, label=label)
        else:
            ax.plot(
                t,
                h,
                label=f"q1={q1}, q2={q2}, q3={q3}, T={T}, eps={eps}, delay={delay}",
            )
    elif show:
        if label is not None:
            plt.plot(t, h, label=label)
        else:
            plt.plot(
                t,
                h,
                label=f"q1={q1}, q2={q2}, q3={q3}, T={T}, eps={eps}, delay={delay}",
            )

    if ax is None and (show or show_delay or show_ksi):
        plt.xlabel("Время (с)")
        plt.ylabel("Выходное значение")
        plt.title("Переходная характеристика системы")
        plt.legend()
        plt.grid()
        plt.show()
    if et_hist:
        return et_h

    if calc_dI or show_ksi:
        return dI_dq1, dI_dq2, dI_dq3, I_x
    return I_x


def apz_runge_kutta(y, g, T, K, dt):
    def f(y, g):
        return (K * g - y) / T

    k1 = dt * f(y, g)
    k2 = dt * f(y + k1 / 2, g)
    k3 = dt * f(y + k2 / 2, g)
    k4 = dt * f(y + k3, g)

    y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return y


# T = 1.0
# K = 1.0
# g = 1.0
# t_end = 10
# dt = 0.1

# t_values = np.arange(0, t_end, dt)
# y_values = np.zeros_like(t_values)

# for i in range(1, len(t_values)):
#     y = y_values[i - 1]
#     y_values[i] = apz_runge_kutta(y, g, T, K)


# plt.plot(t_values, y_values, label="Выходная характеристика")
# plt.xlabel("Время, с")
# plt.ylabel("Выход y")
# plt.title("Апериодическое звено первого порядка (метод Рунге-Кутты)")
# plt.legend()
# plt.grid()
# plt.show()
def optimize_sys(
    eps=0.0001,
    max_iters=300,
    show_dI=False,
    q_hist=False,
    q1=0,
    q2=0,
    q3=0,
    h=0.1,
    I_hist=False,
):
    if I_hist:
        I_arr = list()
    if q_hist:
        q_arr = list()
    prev_I = 0
    I_x = 0
    dI_dq1, dI_dq2, dI_dq3 = 0, 0, 0
    iters = 0
    A = list()
    while iters < max_iters:
        prev_I = I_x
        I_x = simulate_system(q1=q1, q2=q2, q3=q3, dt=0.01, delay=0.1)
        if prev_I < I_x:
            if prev_I < I_x and iters != 0:
                q1 += h * dI_dq1 / sum_i
                q2 += h * dI_dq2 / sum_i
                q3 += h * dI_dq3 / sum_i
            dI_dq1, dI_dq2, dI_dq3, I_x = simulate_system(
                q1=q1, q2=q2, q3=q3, dt=0.01, delay=0.1, calc_dI=True
            )
        sum_i = np.sqrt(dI_dq1 * dI_dq1 + dI_dq2 * dI_dq2 + dI_dq3 * dI_dq3)
        if prev_I > I_x:
            h *= 1.1
        else:
            h /= 2
        if sum_i <= eps:
            break
        if q_hist:
            q_arr.append([q1, q2, q3])
        if I_hist:
            I_arr.append(I_x)
        q1 -= h * dI_dq1 / sum_i
        q2 -= h * dI_dq2 / sum_i
        q3 -= h * dI_dq3 / sum_i
        A.append([dI_dq1, dI_dq2, dI_dq3])
        iters += 1
        print([iters, dI_dq1, dI_dq2, dI_dq3, I_x, sum_i])
    if show_dI:
        plt.plot(range(0, iters), [i[0] for i in A], label="dI_dq1")
        plt.plot(range(0, iters), [i[1] for i in A], label="dI_dq2")
        plt.plot(range(0, iters), [i[2] for i in A], label="dI_dq3")
        plt.title("Графики изменения градиента")
        plt.xlabel("Итерации")
        plt.ylabel("Выходное значение")
        plt.legend()
        plt.grid()
        plt.show()
    if I_hist:
        return I_arr
    if q_hist:
        return q_arr
    return q1, q2, q3


q1, q2, q3 = optimize_sys(q1=1, q2=1, q3=1)

fig, ax = plt.subplots()
t = np.arange(0, 20, 0.001)
et_h = [0] + simulate_system(
    q1=0, q2=0, q3=0, delay=0.1, ax=ax, label="A", et_hist=True
)
simulate_system(q1=1.5, q2=1, q3=1, delay=0.1, label="A1", ax=ax)
simulate_system(q1=10, q2=10, q3=10, delay=0.1, label="A2", ax=ax)
simulate_system(q1=q1, q2=q2, q3=q3, delay=0.1, label="Оптимизированная система", ax=ax)
ax.plot(t, et_h, label="Эталлонное звено")
ax.set_xlabel("Время (с)")
ax.set_ylabel("Выходное значение")
ax.set_title("Начальные и оптимальное положения систем")
ax.legend()
ax.grid()
plt.show()

# A = optimize_sys(I_hist=True, q1=0, q2=0, q3=0)
# A1 = optimize_sys(I_hist=True, q1=1.5, q2=1, q3=1)
# A2 = optimize_sys(I_hist=True, q1=10, q2=10, q3=10)
# # plt.plot(range(0, len(A)), A, label="A(I)")
# # plt.plot(range(0, len(A1)), A1, label="A1(I)")
# plt.plot(range(0, len(A2)), A2, label="A2(I)")
# plt.title("График целевой функции")
# plt.xlabel("Итерации")
# plt.ylabel("Выходное значение")
# plt.legend()
# plt.grid()
# plt.show()


# q1, q2, q3 = optimize_sys(q1=10, q2=10, q3=10, show_dI=True)
# simulate_system(
#     q1=1, q2=0, q3=0, show=True, show_ksi=True, show_delay=True, delay=0.1, time_end=7.5
# )

# simulate_system(
#     q1=q1,
#     q2=q2,
#     q3=q3,
#     show=True,
#     show_ksi=True,
#     show_delay=True,
#     delay=0.1,
#     time_end=7.5,
# )

# A = optimize_sys(show_dI=True)
# A1 = optimize_sys(I_arr=True, q1=10, q2=2, q3=2, eps=0.00001)
# A2 = optimize_sys(q_hist=True, q1=10, q2=10, q3=10)
# print(A[len(A) - 1])
# plt.plot(range(0, len(A)), [i[0] for i in A], label="A(q1)")
# plt.plot(range(0, len(A1)), [i[0] for i in A1], label="A1(q1)")
# plt.plot(range(0, len(A2)), [i[0] for i in A2], label="A2(q1)")
# plt.title("Графики сходимости")
# plt.xlabel("Итерации")
# plt.ylabel("Выходное значение")
# plt.legend()
# plt.grid()
# plt.show()
# plt.plot(range(0, len(A)), [i[1] for i in A], label="A(q2)")
# plt.plot(range(0, len(A1)), [i[1] for i in A1], label="A1(q2)")
# plt.plot(range(0, len(A2)), [i[1] for i in A2], label="A2(q2)")
# plt.title("Графики сходимости")
# plt.xlabel("Итерации")
# plt.ylabel("Выходное значение")
# plt.legend()
# plt.grid()
# plt.show()
# plt.plot(range(0, len(A)), [i[2] for i in A], label="A(q3)")
# plt.plot(range(0, len(A1)), [i[2] for i in A1], label="A1(q3)")
# plt.plot(range(0, len(A2)), [i[2] for i in A2], label="A2(q3)")
# plt.title("Графики сходимости")
# plt.xlabel("Итерации")
# plt.ylabel("Выходное значение")
# plt.legend()
# plt.grid()
# plt.show()


# q1, q2, q3 = optimize_sys(show_dI=True)
# simulate_system(
#     q1=q1,
#     q2=q2,
#     q3=q3,
#     show=True,
#     show_ksi=True,
#     show_delay=True,
#     delay=0.1,
# )

# simulate_system(
#     q1=3.428309910748185,
#     q2=2.3348960361944497,
#     q3=2.3746903858977757,
#     show=True,
#     show_ksi=True,
#     show_delay=True,
#     delay=0.1,
# )
