import matplotlib.pyplot as plt
import numpy as np

fname = 'o51'
f = open(f'{fname}.txt','r')
data = f.read().split('\n')[9:]


with_dbat = []
without_dbat = []

with_epochs = []
without_epochs = []

my_r = []
epoch_row = []

t1 = 1
t2 = 0
marked = 0

for row in data:
    
    if(row[0:2]=='[m'):
        if(int(row[2])!=t2+1):
            t2=(t2+1)%5
            if(t1==1):
                without_dbat.append(my_r)
                # print(t1,t2,my_r)
                my_r = [float(row[-6:])]
                without_epochs.append(epoch_row)
                epoch_row = [int(row[5:10].split(':')[0])]
            else:
                with_dbat.append(my_r)
                # print(t1,t2,my_r)
                my_r = [float(row[-6:])]
                marked = 0
                with_epochs.append(epoch_row)
                epoch_row = [int(row[5:10].split(':')[0])]

        else:
            # print(t1,t2+1,row[-6:])
            my_r.append(float(row[-6:]))
            epoch_row.append(int(row[5:10].split(':')[0]))
    elif not marked:
        if(t1==1):
            without_dbat.append(my_r)
            # print(t1,t2,my_r)
            my_r = []
            marked = 1
            t2 = 0
            without_epochs.append(epoch_row)
            epoch_row = []

        else:
            with_dbat.append(my_r)
            # print(t1,t2,my_r)
            my_r = []
            marked = 1
            with_epochs.append(epoch_row)
            epoch_row = []


    if(t1==1 and row.strip()==''):
        t1=2
    


with_dbat1 = np.array(with_dbat) # learning sequences of all 
without_dbat1 = np.array(without_dbat)
with_epochs1 = np.array(with_epochs)
without_epochs1 = np.array(without_epochs)


fname = 'o52'
f = open(f'{fname}.txt','r')
data = f.read().split('\n')[9:]


with_dbat = []
without_dbat = []

with_epochs = []
without_epochs = []

my_r = []
epoch_row = []

t1 = 1
t2 = 0
marked = 0

for row in data:
    
    if(row[0:2]=='[m'):
        if(int(row[2])!=t2+1):
            t2=(t2+1)%5
            if(t1==1):
                without_dbat.append(my_r)
                # print(t1,t2,my_r)
                my_r = [float(row[-6:])]
                without_epochs.append(epoch_row)
                epoch_row = [int(row[5:10].split(':')[0])]
            else:
                with_dbat.append(my_r)
                # print(t1,t2,my_r)
                my_r = [float(row[-6:])]
                marked = 0
                with_epochs.append(epoch_row)
                epoch_row = [int(row[5:10].split(':')[0])]

        else:
            # print(t1,t2+1,row[-6:])
            my_r.append(float(row[-6:]))
            epoch_row.append(int(row[5:10].split(':')[0]))
    elif not marked:
        if(t1==1):
            without_dbat.append(my_r)
            # print(t1,t2,my_r)
            my_r = []
            marked = 1
            t2 = 0
            without_epochs.append(epoch_row)
            epoch_row = []

        else:
            with_dbat.append(my_r)
            # print(t1,t2,my_r)
            my_r = []
            marked = 1
            with_epochs.append(epoch_row)
            epoch_row = []


    if(t1==1 and row.strip()==''):
        t1=2
    



with_dbat2 = np.array(with_dbat) # learning sequences of all 
without_dbat2 = np.array(without_dbat)
with_epochs2 = np.array(with_epochs)
without_epochs2 = np.array(without_epochs)

fname = 'o53'
f = open(f'{fname}.txt','r')
data = f.read().split('\n')[9:]


with_dbat = []
without_dbat = []

with_epochs = []
without_epochs = []

my_r = []
epoch_row = []

t1 = 1
t2 = 0
marked = 0

for row in data:
    
    if(row[0:2]=='[m'):
        if(int(row[2])!=t2+1):
            t2=(t2+1)%5
            if(t1==1):
                without_dbat.append(my_r)
                # print(t1,t2,my_r)
                my_r = [float(row[-6:])]
                without_epochs.append(epoch_row)
                epoch_row = [int(row[5:10].split(':')[0])]
            else:
                with_dbat.append(my_r)
                # print(t1,t2,my_r)
                my_r = [float(row[-6:])]
                marked = 0
                with_epochs.append(epoch_row)
                epoch_row = [int(row[5:10].split(':')[0])]

        else:
            # print(t1,t2+1,row[-6:])
            my_r.append(float(row[-6:]))
            epoch_row.append(int(row[5:10].split(':')[0]))
    elif not marked:
        if(t1==1):
            without_dbat.append(my_r)
            # print(t1,t2,my_r)
            my_r = []
            marked = 1
            t2 = 0
            without_epochs.append(epoch_row)
            epoch_row = []

        else:
            with_dbat.append(my_r)
            # print(t1,t2,my_r)
            my_r = []
            marked = 1
            with_epochs.append(epoch_row)
            epoch_row = []


    if(t1==1 and row.strip()==''):
        t1=2
    



with_dbat3 = np.array(with_dbat) # learning sequences of all 
without_dbat3 = np.array(without_dbat)
with_epochs3 = np.array(with_epochs)
without_epochs3 = np.array(without_epochs)








# print(without_dbat.shape)
plt.plot(np.mean(with_epochs1, axis=0)[1:],np.mean(with_dbat1, axis=0)[1:], label='with dbat1', c=(0.1,0.1,0.9))
plt.plot(np.mean(with_epochs2, axis=0)[1:],np.mean(with_dbat2, axis=0)[1:], label='with dbat2', c=(0.1,0.1,0.6))
plt.plot(np.mean(with_epochs3, axis=0)[1:],np.mean(with_dbat3, axis=0)[1:], label='with dbat3', c=(0.1,0.1,0.3))

# plt.plot(with_dbat[1], label='2')
# plt.plot(with_dbat[2], label='3')
# plt.plot(with_dbat[3], label='4')
# plt.plot(with_dbat[4], label='5')
# plt.legend()

plt.plot(np.mean(without_epochs1, axis=0)[1:],np.mean(without_dbat1, axis=0)[1:], label='without dbat1', c=(0.9,0.1,0.1))
plt.plot(np.mean(without_epochs2, axis=0)[1:],np.mean(without_dbat2, axis=0)[1:], label='without dbat2', c=(0.6,0.1,0.1))
plt.plot(np.mean(without_epochs3, axis=0)[1:],np.mean(without_dbat3, axis=0)[1:], label='without dbat3', c=(0.3,0.1,0.1))

# plt.plot(without_dbat[1], label='2o')
# plt.plot(without_dbat[2], label='3o')
# plt.plot(without_dbat[3], label='4o')
# plt.plot(without_dbat[4], label='5o')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy variation - averaged over all 5 models')
plt.savefig(f'Plot-{fname.split("/")[-1]}.png')

# print(f'Accuracy of 5 ensembles without dbat: 0.555')
# print(f'Accuracy of 5 ensembles with dbat: 0.556')











