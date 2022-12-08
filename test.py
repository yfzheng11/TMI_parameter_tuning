import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy
import tables

f = h5py.File('data/TrainData.mat', 'r')
TrainData = f.get('/TrainData')
TrainData = np.array(TrainData)
diff = TrainData[:, 1:, :] - TrainData[:, :-1, :]
# for i in range(80):
#     # slice = TrainData[:, i, 0].reshape((128, 128))
#     slice = diff[:, i, 0].reshape((128, 128))
#     plt.imshow(slice)
#     plt.colorbar()
#     plt.show()

f = h5py.File('data/TestData.mat', 'r')
TestData = f.get('/TestData')
TestData = np.array(TestData)

f = h5py.File('data/TrueImgTrain.mat', 'r')
TrueImgTrain = f.get('/TrueImgTrain')
TrueImgTrain = np.array(TrueImgTrain)
TrueImgTrain = TrueImgTrain.transpose()

f = h5py.File('data/TrueImgTest.mat', 'r')
TrueImgTest = f.get('/TrueImgTest')
TrueImgTest = np.array(TrueImgTest)
TrueImgTest = TrueImgTest.transpose()

# load system matrix
pMat = scipy.sparse.load_npz('data/sparse_matrix.npz')

# load projection data
f = tables.open_file('data/projdata_Train_new.h5', 'r')
Projdata_Train = f.root.projection.read()
f.close()

f = tables.open_file('data/projdata_Test_new.h5', 'r')
Projdata_Test = f.root.projection.read()
f.close()
print('finish')
