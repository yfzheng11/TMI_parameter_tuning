import scipy
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tables

# mat_contents = scipy.io.loadmat('TrainData.mat')

sysmat = scipy.sparse.load_npz('data/sparse_matrix.npz')

f = h5py.File('data/TrainData.mat', 'r')
TrainData = f.get('/TrainData')
TrainData = np.array(TrainData)
print(TrainData.shape)

f = h5py.File('data/projdata_Train.mat', 'r')
Projdata_Train = f.get('/projdata_Train')
Projdata_Train = np.array(Projdata_Train)
Projdata_Train = Projdata_Train.transpose()
print(Projdata_Train.shape)
# sino = Projdata_Train[:, 0].reshape((180, 192))
# plt.imshow(sino)
# plt.colorbar()
# plt.show()

## read true image for training
f = h5py.File('data/TrueImgTrain.mat', 'r')
TrueImgTrain = f.get('/TrueImgTrain')
TrueImgTrain = np.array(TrueImgTrain)
TrueImgTrain = TrueImgTrain.transpose()
print(TrueImgTrain.shape)

proj = sysmat * TrueImgTrain
print(proj.shape)
f = tables.open_file('data/projdata_Train_new.h5', 'w')
f.create_array('/', 'projection', proj)
f.close()

f = h5py.File('data/TrueImgTest.mat', 'r')
TrueImgTest = f.get('/TrueImgTest')
TrueImgTest = np.array(TrueImgTest)
TrueImgTest = TrueImgTest.transpose()
print(TrueImgTest.shape)

proj = sysmat * TrueImgTest
print(proj.shape)
f = tables.open_file('data/projdata_Test_new.h5', 'w')
f.create_array('/', 'projection', proj)
f.close()

for i in range(proj.shape[-1]):
    sino = proj[:, i].reshape((60, 128))
    plt.imshow(sino)
    plt.colorbar()
    plt.show()

img = TrueImgTrain[:, 5].reshape((128, 128))
plt.imshow(img)
plt.colorbar()
plt.show()
