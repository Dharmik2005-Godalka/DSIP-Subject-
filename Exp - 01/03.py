import numpy as np
import matplotlib.pyplot as plt

def continuous(time):
    unit_step = np.zeros_like(time)
    unit_step[time >= 0] = 1
    return unit_step

def discrete(samples):
    unit_step = np.zeros(samples)
    unit_step[samples // 2:] = 1
    return unit_step

time = np.linspace(-5, 5, 1000)
continuous_unit_step = continuous(time)

samples = 20
discrete_unit_step = discrete(samples)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, continuous_unit_step)
plt.title('continuous unit step signal')
plt.xlabel('time')
plt.ylabel('amplitude')

plt.subplot(2, 1, 2)
plt.stem(discrete_unit_step)
plt.title('discrete unit step signal')
plt.xlabel('sample')
plt.ylabel('amplitude')

plt.tight_layout()
plt.show()
