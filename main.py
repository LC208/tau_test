import matplotlib.pyplot as pyplot
from numpy import sqrt, pi
from collections import deque

T = 1
eps = 0.75
k = 1
g = 1


def f1(y, z2):
    return z2 - (2 * eps * y) / T


def f2(y, g):
    return (k / (T * T)) * g - (1 / (T * T)) * y


def r_k(w, y, z2, f1, f2, W, H, g):
    W.append(w)
    H.append(y)
    k1 = dt * f1(y, z2)
    q1 = dt * f2(y, g)
    k2 = dt * f1(y + k1 / 2, z2 + q1 / 2)
    q2 = dt * f2(y + k1 / 2, g)
    k3 = dt * f1(y + k2 / 2, z2 + q2 / 2)
    q3 = dt * f2(y + k2 / 2, g)
    k4 = dt * f1(y + k3, z2 + q3)
    q4 = dt * f2(y + k3, g)
    z2 = z2 + (q1 + 2 * q2 + 2 * q3 + q4) / 6
    y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    w = (y - H[len(H) - 1]) / dt
    return y, w, z2


X = list()
W = list()
H = list()
W1 = list()
H1 = list()
H2 = list()
H3 = list()
one = list()
buffer = deque()
buffer_ksi1 = deque()
y = 0
z2 = 0
w = 0
y1 = 0
w1 = 0
z21 = 0
dt = T / 10000
# L = (6*pi*T)/sqrt(1+eps*eps)
L = 4
t = 0
tau = 0.1
y2 = 0
g = 1
q1 = 50
q2 = 10
q3 = 40
sum_x = 0
x = 0
I = 0
Ipr = 10000
ksi1 = 0
pr_ksi1 = 0
sum_ksi1 = 0
# while True:
# ksi1: du_dq1 -> Wоб
while t < L:
    X.append(t)
    one.append(1)
    y1, w1, z21 = r_k(w1, y1, z21, f1, f2, W1, H1, x)
    xpr = x
    x = 1 - H1[len(H1) - 1]
    dx = (x - xpr) / dt
    sum_x = sum_x + x * dt
    g = q1 * x + q2 * sum_x + q3 * dx
    sum_ksi1 += ksi1 * dt
    d_ksi1 = (ksi1 - pr_ksi1) / dt
    du_dq1 = x - q1 * ksi1 - q2 * sum_ksi1 - q3 * d_ksi1
    I = I + x * x * dt
    pr_ksi1 = ksi1
    if t > tau and len(buffer) > 0:
        y = buffer.popleft()
        buffer.append(y1)
    else:
        buffer.append(y1)
        y = 0
    H2.append(y)
    t += dt
#    print(f"I: {I}, Ipr: {Ipr}, q: {q3}")
#    if I > Ipr:
#        break
#    X = list()
#    W = list()
#    H = list()
#    W1 = list()
#    H1 = list()
#    H2 = list()
#    H3 = list()
#    one = list()
#    buffer = deque()
#    Ipr=I
#    I=0
#    t=0
#    q3=q3+0.1
print(I)
pyplot.plot(
    X,
    H2,
    color="green",
    linewidth=1,
    label="Смоделированная переходная функция(ООС) с запаздыванием",
)
pyplot.plot(
    X, H1, color="red", linewidth=1, label="Смоделированная переходная функция(ООС)"
)
# pyplot.plot(X, one, label = "Единичное ступенчатое воздействие", color = "green")
pyplot.grid(True)
pyplot.legend()
pyplot.show()
