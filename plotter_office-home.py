import matplotlib.pyplot as plt
import numpy as np


f = open('home_office.txt','r')
data = f.read().split('\n')[9:]

t = 1
for row in data:
    if(row[0:2]=='[m'):
        print(row)
    if(t==1)






