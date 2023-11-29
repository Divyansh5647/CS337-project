import matplotlib.pyplot as plt
import numpy as np


with_dbat = np.array([0.9749999642372131, 0.8740000128746033, 0.8410000205039978, 0.7515000104904175, 0.6807999610900879])
without_dbat = np.array([0.9549999833106995 ,0.8680000305175781, 0.8370000123977661, 0.7615000605583191, 0.6895999908447266])

x_axis = [2, 5, 10, 20, 50]

plt.plot(x_axis, without_dbat, label = 'without dbat')
plt.plot(x_axis, with_dbat, label = 'with dbat')

plt.legend()
plt.xticks([2,5,10,20,50])

plt.xlabel('Number of Classes')
plt.ylabel('Accuracy')
plt.title("Accuracy Variation with Number of Classes - With and Without DBAT")


plt.savefig('num_classes_absolute.png')
plt.cla()

plt.plot(x_axis, with_dbat/without_dbat, label = 'Relative accuracy')
plt.axhline(y=1, color='r', linestyle='--', label='Equal Accuracy')

plt.legend()
plt.xticks([2,5,10,20,50])

plt.xlabel('Number of Classes')
plt.ylabel('Relative Accuracy')
plt.title("Relative Accuracy Variation with Number of Classes - With/Without DBAT")


plt.savefig('num_classes_relative.png')
