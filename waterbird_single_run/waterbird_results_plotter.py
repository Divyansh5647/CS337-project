import matplotlib.pyplot as plt
import numpy as np


with_dbat = np.array([0.840, 0.850, 0.847, 0.848, 0.851])
without_dbat = np.array([0.840 ,0.841, 0.839, 0.839, 0.841])

x_axis = [1, 2, 3, 4, 5]

plt.plot(x_axis, without_dbat, label = 'without dbat')
plt.plot(x_axis, with_dbat, label = 'with dbat')

plt.legend()
plt.xticks([1, 2, 3, 4, 5])  

plt.xlabel('Ensemble Size')
plt.ylabel('Accuracy')
plt.title("Accuracy Variation with Ensemble size - With and Without DBAT")


plt.savefig('waterbird_ensemble_size_absolute.png')
plt.cla()

plt.plot(x_axis, with_dbat/without_dbat, label = 'Relative accuracy')
plt.axhline(y=1, color='r', linestyle='--', label='Equal Accuracy')

plt.legend()
plt.xticks([1, 2, 3, 4, 5])

plt.xlabel('Ensemble Size')
plt.ylabel('Relative Accuracy')
plt.title("Relative Accuracy Variation with Ensemble Size - With/Without DBAT")


plt.savefig('waterbird_ensemble_size_relative.png')
