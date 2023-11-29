import matplotlib.pyplot as plt
import numpy as np


without_dbat1 = np.array([0.8877999782562256, 0.8969999551773071, 0.8981999754905701, 0.8983999490737915])
without_dbat2 = np.array([0.8811999559402466, 0.8967999815940857, 0.9035999774932861, 0.9041999578475952])
without_dbat3 = np.array([0.8829999566078186, 0.894599974155426, 0.8981999754905701, 0.9007999897003174])
with_dbat1 = np.array([0.8877999782562256 ,0.8953999876976013, 0.9003999829292297, 0.9035999774932861])
with_dbat2 = np.array([0.8811999559402466 ,0.8967999815940857, 0.9023999571800232, 0.9023999571800232])
with_dbat3 = np.array([0.8829999566078186 ,0.8971999883651733, 0.8973999619483948, 0.9017999768257141])

x_axis = [1, 2, 3, 4]

plt.plot(x_axis, without_dbat1, label = 'without dbat1', c=(0.8,0.1,0.1))
plt.plot(x_axis, with_dbat1, label = 'with dbat1', c=(0.1,0.1,0.8))
plt.plot(x_axis, without_dbat2, label = 'without dbat2', c=(0.6,0.1,0.1))
plt.plot(x_axis, with_dbat2, label = 'with dbat2', c=(0.1,0.1,0.6))
plt.plot(x_axis, without_dbat3, label = 'without dbat3', c=(0.4,0.1,0.1))
plt.plot(x_axis, with_dbat3, label = 'with dbat3', c=(0.1,0.1,0.4))

plt.legend()
plt.xticks([1, 2, 3, 4])

plt.xlabel('Ensemble Size')
plt.ylabel('Accuracy')
plt.title("Accuracy Variation with Ensemble Size - With and Without DBAT")


plt.savefig('num_ensembles_absolute_many.png')
plt.cla()

net_without_dbat = np.array([without_dbat1, without_dbat2, without_dbat3])[:,1:]
net_with_dbat = np.array([with_dbat1, with_dbat2, with_dbat3])[:,1:]
# print('means:', np.mean(net_without_dbat, axis = 0), np.mean(net_with_dbat, axis = 0))
# print('std:', np.std(net_without_dbat, axis = 0), np.std(net_with_dbat, axis = 0))
plt.errorbar(x_axis[1:], np.mean(net_without_dbat, axis = 0), yerr=np.std(net_without_dbat, axis = 0), label = 'without dbat', capsize=5,capthick=1)
plt.errorbar(x_axis[1:], np.mean(net_with_dbat, axis = 0), yerr=np.std(net_with_dbat, axis = 0), label = 'with dbat', capsize=5,capthick=1)
plt.legend()
plt.xticks([2, 3, 4])  

plt.xlabel('Ensemble Size')
plt.ylabel('Accuracy')
plt.title("Accuracy Variation with Ensemble size - With and Without DBAT")

plt.savefig('num_ensembles_error_bars_zoom.png')
plt.cla()

net_without_dbat = np.array([without_dbat1, without_dbat2, without_dbat3])
net_with_dbat = np.array([with_dbat1, with_dbat2, with_dbat3])
# print('means:', np.mean(net_without_dbat, axis = 0), np.mean(net_with_dbat, axis = 0))
# print('std:', np.std(net_without_dbat, axis = 0), np.std(net_with_dbat, axis = 0))
plt.errorbar(x_axis, np.mean(net_without_dbat, axis = 0), yerr=np.std(net_without_dbat, axis = 0), label = 'without dbat', capsize=5,capthick=1)
plt.errorbar(x_axis, np.mean(net_with_dbat, axis = 0), yerr=np.std(net_with_dbat, axis = 0), label = 'with dbat', capsize=5,capthick=1)
plt.legend()
plt.xticks([1, 2, 3, 4])  

plt.xlabel('Ensemble Size')
plt.ylabel('Accuracy')
plt.title("Accuracy Variation with Ensemble size - With and Without DBAT")

plt.savefig('num_ensembles_error_bars_full.png')
plt.cla()




plt.plot(x_axis, with_dbat1/without_dbat1, label = 'Relative accuracy')
plt.plot(x_axis, with_dbat1/without_dbat2, label = 'Relative accuracy')
plt.plot(x_axis, with_dbat1/without_dbat1, label = 'Relative accuracy')

plt.axhline(y=1, color='r', linestyle='--', label='Equal Accuracy')

plt.legend()
plt.xticks([1, 2, 3, 4])

plt.xlabel('Ensemble Size')
plt.ylabel('Relative Accuracy')
plt.title("Relative Accuracy Variation with Ensemble Size - With/Without DBAT")


plt.savefig('num_ensembles_relative_many.png')
