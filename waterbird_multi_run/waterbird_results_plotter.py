import matplotlib.pyplot as plt
import numpy as np


without_dbat1 = np.array([0.840 ,0.841, 0.839, 0.839, 0.841])
without_dbat2 = np.array([0.833 ,0.835, 0.834, 0.830, 0.838])
without_dbat3 = np.array([0.834 ,0.840, 0.837, 0.837, 0.836])
with_dbat1 = np.array([0.840, 0.850, 0.847, 0.848, 0.851])
with_dbat2 = np.array([0.833, 0.843, 0.843, 0.846, 0.849])
with_dbat3 = np.array([0.834, 0.845, 0.848, 0.850, 0.848])


x_axis = [1, 2, 3, 4, 5]

plt.plot(x_axis, without_dbat1, label = 'without dbat1', c=(0.9,0.1,0.1))
plt.plot(x_axis, with_dbat1, label = 'with dbat1', c=(0.1,0.1,0.9))
plt.plot(x_axis, without_dbat2, label = 'without dbat2', c=(0.6,0.1,0.1))
plt.plot(x_axis, with_dbat2, label = 'with dbat2', c=(0.1,0.1,0.6))
plt.plot(x_axis, without_dbat3, label = 'without dbat3', c=(0.3,0.1,0.1))
plt.plot(x_axis, with_dbat3, label = 'with dbat3', c=(0.1,0.1,0.3))


plt.legend()
plt.xticks([1, 2, 3, 4, 5])  

plt.xlabel('Ensemble Size')
plt.ylabel('Accuracy')
plt.title("Accuracy Variation with Ensemble size - With and Without DBAT")

plt.savefig('waterbird_ensemble_size_absolute.png')
plt.cla()

net_without_dbat = np.array([without_dbat1, without_dbat2, without_dbat3])
net_with_dbat = np.array([with_dbat1, with_dbat2, with_dbat3])
# print('means:', np.mean(net_without_dbat, axis = 0), np.mean(net_with_dbat, axis = 0))
# print('std:', np.std(net_without_dbat, axis = 0), np.std(net_with_dbat, axis = 0))
plt.errorbar(x_axis, np.mean(net_without_dbat, axis = 0), yerr=np.std(net_without_dbat, axis = 0), label = 'without dbat', capsize=5,capthick=1)
plt.errorbar(x_axis, np.mean(net_with_dbat, axis = 0), yerr=np.std(net_with_dbat, axis = 0), label = 'with dbat', capsize=5,capthick=1)
plt.legend()
plt.xticks([1, 2, 3, 4, 5])  

plt.xlabel('Ensemble Size')
plt.ylabel('Accuracy')
plt.title("Accuracy Variation with Ensemble size - With and Without DBAT")

plt.savefig('waterbird_ensemble_size_absolute_error_bars.png')
plt.cla()

plt.plot(x_axis, with_dbat1/without_dbat1, label = 'Relative accuracy1')
plt.plot(x_axis, with_dbat2/without_dbat2, label = 'Relative accuracy2')
plt.plot(x_axis, with_dbat3/without_dbat3, label = 'Relative accuracy3')

plt.axhline(y=1, color='r', linestyle='--', label='Equal Accuracy')

plt.legend()
plt.xticks([1, 2, 3, 4, 5])

plt.xlabel('Ensemble Size')
plt.ylabel('Relative Accuracy')
plt.title("Relative Accuracy Variation with Ensemble Size - With/Without DBAT")


plt.savefig('waterbird_ensemble_size_relative.png')
