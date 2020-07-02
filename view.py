#%%
import numpy as np
import h5py
import matplotlib.pyplot as plt

INDEX=82;

with h5py.File('result.h5','r') as f:
    data=f['prediction']
    data=np.asarray(data)
    data=np.squeeze(data)
    print(data.shape)
    plt.imshow(data[INDEX], cmap=plt.cm.gray)
    plt.show()
    '''for i in range(70,150,2):
        plt.imshow(data[i], cmap=plt.cm.gray)
        plt.show()'''

#%%
import numpy as np
import h5py
import matplotlib.pyplot as plt
with h5py.File('train.h5','r') as f:
    data=f['data']
    data=np.asarray(data)
    data=np.squeeze(data)
    label=f['label']
    label=np.asarray(label)
    label=np.squeeze(label)
    plt.imshow(data[INDEX], cmap=plt.cm.gray)
    plt.show()
    plt.imshow(label[INDEX], cmap=plt.cm.gray)
    plt.show()
    '''for i in range(70,150,2):
        plt.imshow(data[i], cmap=plt.cm.gray)
        plt.show()'''



