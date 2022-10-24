import scipy.io
import numpy as np
import h5py
import matplotlib.pyplot as plt

# mat_contents = scipy.io.loadmat('TrainData.mat')

f = h5py.File('data/TrainData.mat', 'r')
TrainData = f.get('/TrainData')
TrainData = np.array(TrainData)

print(TrainData.shape)

f = h5py.File('data/projdata_Train.mat', 'r')
Projdata_Train = f.get('/projdata_Train')
Projdata_Train = np.array(Projdata_Train)
Projdata_Train = Projdata_Train.transpose()

print(Projdata_Train.shape)
sino = Projdata_Train[:, 0].reshape((180, 192))
plt.imshow(sino)
plt.colorbar()
plt.show()


f = h5py.File('data/TrueImgTrain.mat', 'r')
TrueImgTrain = f.get('/TrueImgTrain')
TrueImgTrain = np.array(TrueImgTrain)
TrueImgTrain = TrueImgTrain.transpose()

print(TrueImgTrain.shape)

img = TrueImgTrain[:, 5].reshape((128, 128))
plt.imshow(img)
plt.colorbar()
plt.show()
