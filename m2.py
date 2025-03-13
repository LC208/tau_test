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
):

    t = np.arange(0, time_end, dt)
    h = np.zeros_like(t)
    dI_dq1, dI_dq2, dI_dq3 = 0, 0, 0
    h_ksi1 = np.zeros_like(t)
    h_ksi2 = np.zeros_like(t)
    h_ksi3 = np.zeros_like(t)
    if show_delay:
        h_nd = np.zeros_like(t)
        h_ksi1_nd = np.zeros_like(t)
        h_ksi2_nd = np.zeros_like(t)
        h_ksi3_nd = np.zeros_like(t)

    y, z2, I_x = 0, 0, 0
    y_et = 0
    prev_error, integral = 0, 0
    delay_steps = int(delay / dt)
    buffer = np.zeros(delay_steps) if delay_steps > 0 else None
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
        y, z2 = runge_kutta_step(y, z2, control_signal, T, eps, dt, k)
        dI_dq1 -= 2 * error * ksi1 * dt
        dI_dq2 -= 2 * error * ksi2 * dt
        dI_dq3 -= 2 * error * ksi3 * dt
        prev_error = error
        if show_delay:
            h_nd[i] = y
            h_ksi1_nd[i] = ksi1
            h_ksi2_nd[i] = ksi2
            h_ksi3_nd[i] = ksi3

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
    print(I_x)
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

    if show or show_delay or show_ksi:
        plt.xlabel("Время (с)")
        plt.ylabel("Выходное значение")
        plt.title("Переходная характеристика системы")
        plt.legend()
        plt.grid()
        plt.show()
    return dI_dq1, dI_dq2, dI_dq3, I_x


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


q1 = 3
q2 = 0
q3 = 0
h = 0.1
prev_I = 0
I_x = 0

iters = 23
A = np.zeros((iters, 3))
prev_sum = 10000000
for i in range(0, iters):
    prev_I = I_x
    dI_dq1, dI_dq2, dI_dq3, I_x = simulate_system(
        q1=q1, q2=q2, q3=q3, dt=0.01, delay=0.1
    )
    sum_i = np.sqrt(dI_dq1 * dI_dq1 + dI_dq2 * dI_dq2 + dI_dq3 * dI_dq3)
    if prev_I > I_x:
        h *= 1.1
    else:
        h /= 2
    # if sum_i > prev_sum:
    #     break
    prev_sum = sum_i
    # if i > 0:
    #     dI_dq1 = A[i - 1][0] + dI_dq1
    #     dI_dq2 = A[i - 1][1] + dI_dq2
    #     dI_dq3 = A[i - 1][2] + dI_dq3
    q1 -= h * dI_dq1 / sum_i
    q2 -= h * dI_dq2 / sum_i
    q3 -= h * dI_dq3 / sum_i
    A[i] = [dI_dq1, dI_dq2, dI_dq3]
    print(I_x)
    print([i, dI_dq1, dI_dq2, dI_dq3, I_x, sum_i])
print(q1, q2, q3)
simulate_system(
    q1=q1, q2=q2, q3=q3, show=True, show_ksi=True, show_delay=True, delay=0.1
)
plt.plot(range(0, iters), [i[0] for i in A], label="dI_dq1")
plt.plot(range(0, iters), [i[1] for i in A], label="dI_dq2")
plt.plot(range(0, iters), [i[2] for i in A], label="dI_dq3")
plt.xlabel("Время (с)")
plt.ylabel("Выходное значение")
plt.title("Переходная характеристика системы")
plt.legend()
plt.grid()
plt.show()
# def calc_q():
#     q = [1, 0, 0]
#     step = 0.1
#     for i in [0, 1, 2]:
#     print(i)
#     prev_I = simulate_system(q1=q[0], q2=q[1], q3=q[2])
#     I = simulate_system(q1=q[0] + step, q2=q[1], q3=q[2])
#     q[i] += 2 * step
#     print(prev_I)
#     print(I)
#     while prev_I > I:
#         prev_I = I
#         I = simulate_system(q1=q[0], q2=q[1], q3=q[2])
#         q[i] += step
#     print(q)


# simulate_system(q1=1, q2=0, q3=0, show=True, show_delay=False, delay=0.1)
# simulate_system(
#     q1=7.5, q2=5, q3=5, show=True, show_delay=False, show_ksi=True, delay=0.1
# )
# simulate_system(
#     q1=6.245750667887806,
#     q2=4.949854604335281,
#     q3=12.288074270087355,
#     show=True,
#     show_delay=False,
#     show_ksi=False,
#     delay=0.1,
# )
