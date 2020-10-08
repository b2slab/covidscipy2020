from matplotlib import pyplot as plt
import numpy as np

t = np.linspace(0, 100, 100)
x = np.sin(2*np.pi*t)

plt.plot(t, x)
plt.title('Funció sinus')
