import matplotlib.pyplot as plt
import numpy as np


old_loss = np.array([0.8740000128746033, 0.8410000205039978, 0.7515000104904175, 0.6807999610900879])
new_loss = np.array([0.8840000629425049, 0.843000054359436, 0.7475000619888306, 0.6887999773025513])
x_axis = [5, 10, 20, 50]

plt.plot(x_axis, old_loss, label = 'Old Loss')
plt.plot(x_axis, new_loss, label = 'New loss')

plt.legend()
plt.xticks([5,10,20,50])  

plt.xlabel('Number of Classes')
plt.ylabel('Accuracy')
plt.title("Accuracy Variation with Number of Classes - With and Old Loss")


plt.savefig('num_classes_newloss_abs.png')
plt.cla()

plt.plot(x_axis, new_loss/old_loss, label = 'Relative accuracy')
plt.axhline(y=1, color='r', linestyle='--', label='Equal Accuracy')

plt.legend()
plt.xticks([5,10,20,50])

plt.xlabel('Number of Classes')
plt.ylabel('Relative Accuracy')
plt.title("Relative Accuracy Variation with Number of Classes - New/Old Loss")


plt.savefig('num_classes_newloss_rel.png')
