import matplotlib.pyplot as plt
import numpy as np

f = np.linspace(0.0, np.pi, 256)
z = np.exp(f*1.0j)
H01 = (z*0.1)/(z-(1-0.1))
H02 = (z*0.2)/(z-(1-0.2))
H05 = (z*0.5)/(z-(1-0.5))
H10 = (z*1.0)/(z-(1-1.0))
H11 = (z*1.1)/(z-(1-1.1))

plt.semilogx(f/(np.pi*2), 20*np.log10(np.abs(H01)), label="ω = 0.1")
plt.semilogx(f/(np.pi*2), 20*np.log10(np.abs(H02)), label="ω = 0.2")
plt.semilogx(f/(np.pi*2), 20*np.log10(np.abs(H05)), label="ω = 0.5")
plt.semilogx(f/(np.pi*2), 20*np.log10(np.abs(H10)), label="ω = 1.0")
plt.semilogx(f/(np.pi*2), 20*np.log10(np.abs(H11)), label="ω = 1.1")

# plt.semilogx(f/(np.pi*2), np.angle(H01), label="ω = 0.1")
# plt.semilogx(f/(np.pi*2), np.angle(H02), label="ω = 0.2")
# plt.semilogx(f/(np.pi*2), np.angle(H05), label="ω = 0.5")
# plt.semilogx(f/(np.pi*2), np.angle(H10), label="ω = 1.0")
# plt.semilogx(f/(np.pi*2), np.angle(H11), label="ω = 1.1")

# plt.vlines(0.02*np.pi,-25,0)
# plt.vlines(0.1*np.pi,-25,0)
# plt.vlines(np.pi,-25,0)

plt.legend()
plt.grid(which="major", linewidth=1)
plt.grid(which="minor", linewidth=0.2)
plt.xticks([0.1, 0.5], [0.1, 0.5])
plt.minorticks_on()
plt.show()

# print(np.abs(H))
