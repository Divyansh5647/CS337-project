import matplotlib.pyplot as plt
import numpy as np


without_dbat = np.array([0.8877999782562256, 0.8969999551773071, 0.8981999754905701, 0.8983999490737915])
with_dbat = np.array([0.8877999782562256 ,0.8953999876976013, 0.9003999829292297, 0.9035999774932861])

x_axis = [1, 2, 3, 4]

plt.plot(x_axis, without_dbat, label = 'without dbat')
plt.plot(x_axis, with_dbat, label = 'with dbat')

plt.legend()
plt.xticks([1, 2, 3, 4])  

plt.xlabel('Ensemble Size')
plt.ylabel('Accuracy')
plt.title("Accuracy Variation with Ensemble Size - With and Without DBAT")


plt.savefig('num_ensembles_absolute.png')
plt.cla()

plt.plot(x_axis, with_dbat/without_dbat, label = 'Relative accuracy')
plt.axhline(y=1, color='r', linestyle='--', label='Equal Accuracy')

plt.legend()
plt.xticks([1, 2, 3, 4])

plt.xlabel('Ensemble Size')
plt.ylabel('Relative Accuracy')
plt.title("Relative Accuracy Variation with Ensemble Size - With/Without DBAT")


plt.savefig('num_ensembles_relative.png')
