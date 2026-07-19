import numpy as np
import matplotlib.pyplot as plt

def continuous(time, coefficients):
    return np.polyval(coefficients, time)

def discrete(samples, coefficients):
    return np.polyval(coefficients, np.arange(samples))

time = np.linspace(-5, 5, 1000)
samples = 20
coefficients = [1, 2, 1]

continuous_parabolic = continuous(time, coefficients)
discrete_parabolic = discrete(samples, coefficients)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, continuous_parabolic)
plt.title('continuous parabolic signal')
plt.xlabel('time')
plt.ylabel('amplitude')

plt.subplot(2, 1, 2)
plt.stem(discrete_parabolic)
plt.title('discrete parabolic signal')
plt.xlabel('sample')
plt.ylabel('amplitude')

plt.tight_layout()
plt.show()
