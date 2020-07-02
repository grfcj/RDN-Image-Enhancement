#%%
# generate a data set, low dose as data; high dose as label
# loading hdf5

import numpy as np
import h5py

label_data=h5py.File('train_h5/labels.h5','r')
input_data=h5py.File('train_h5/low_resolution.h5','r')

input=np.asarray(input_data['data'].value)
label=np.asarray(label_data['label'].value)

f=h5py.File('training_set.h5','w')
f['data']=input
f['label']=label
f.close()

#%%
#Generate 2D data set
import numpy as np
import h5py

origin_data=h5py.File('training_set.h5','r');

data=np.asarray(origin_data['data'].value).astype(np.float32)
label=np.asarray(origin_data['label'].value).astype(np.float32)

data=np.reshape(data,(2080,288,288)).astype(np.float16)
label=np.reshape(label,(2080,288,288)).astype(np.float16)

print(data.dtype)

#%%


f=h5py.File('training_set_2d_half.h5','w')
f['data']=data
f['label']=label
f.close()

#%%
import numpy as np
import h5py
data=h5py.File('test.h5','r')
data=data['data'].value
data=np.asarray(data)
data_beta=data[0:200,:,:]
print(data_beta.shape)

#%%

import numpy as np
import h5py

label_data=h5py.File('train_h5/10_patients_pixels.h5','r')
input_data=h5py.File('train_h5/low_resolution.h5','r')

input=np.asarray(input_data['data'].value)
label=np.asarray(label_data['label'].value)

data=np.reshape(input,(2080,288,288)).astype(np.float16)
label=np.reshape(label,(2080,288,288)).astype(np.float16)
f=h5py.File('test.h5','w')
f['data']=data[0:200,:,:]
f['label']=label[0:200,:,:]
f.close()


