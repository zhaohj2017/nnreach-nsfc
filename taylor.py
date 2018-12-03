import numpy as np
import matplotlib.pyplot as plt

def ode(y, t):
    return y + np.exp(y) * t + np.exp(2 * y) * np.power(t, 2) / 2 + np.exp(3 * y) * np.power(t, 3) / 3 \
    + np.exp(4 * y) * np.power(t, 4) / 4 + np.exp(5 * y) * np.power(t, 5) / 5 \
    + np.exp(6 * y) * np.power(t, 6) / 6 + np.exp(7 * y) * np.power(t, 7) / 7 \
    + np.exp(8 * y) * np.power(t, 8) / 8 + np.exp(9 * y) * np.power(t, 9) / 9 \
    + np.exp(10 * y) * np.power(t, 10) / 10 + np.exp(11 * y) * np.power(t, 11) / 11


sample_t = np.arange(0, 0.36, 0.02)

sample_y_t = ode(1, sample_t)
sample_y_b = ode(0, sample_t)

plt.plot(sample_t, sample_y_b, color = 'r', linestyle = '-')
plt.plot(sample_t, sample_y_t, color = 'r', linestyle = '-')

plt.show()