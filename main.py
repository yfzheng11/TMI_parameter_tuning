import numpy as np
import h5py
import tables
import param

# import tensorflow as tf
import random
from collections import deque
# import dqn_cnn_iteration_till_end

import time
import math as m
import scipy.io
from numpy import *
import scipy.linalg
import matplotlib.pyplot as plt
import os
import pylab as pl
from mmlem_recon import ReconEnv


def main():
    # # load input data
    # f = h5py.File('data/TrainData.mat', 'r')
    # TrainData = f.get('/TrainData')
    # TrainData = np.array(TrainData)
    #
    # f = h5py.File('data/TestData.mat', 'r')
    # TestData = f.get('/TestData')
    # TestData = np.array(TestData)

    # load ground truth data
    f = h5py.File('data/TrueImgTrain.mat', 'r')
    TrueImgTrain = f.get('/TrueImgTrain')
    TrueImgTrain = np.array(TrueImgTrain)
    TrueImgTrain = TrueImgTrain.transpose()

    f = h5py.File('data/TrueImgTest.mat', 'r')
    TrueImgTest = f.get('/TrueImgTest')
    TrueImgTest = np.array(TrueImgTest)
    TrueImgTest = TrueImgTest.transpose()

    # load system matrix
    sysmat = scipy.sparse.load_npz('data/sparse_matrix.npz')

    # load projection data
    f = tables.open_file('data/projdata_Train_new.h5', 'r')
    proj_train = f.root.projection.read()
    f.close()

    f = tables.open_file('data/projdata_Test_new.h5', 'r')
    proj_test = f.root.projection.read()
    f.close()

    env = ReconEnv(sysmat, proj_train, proj_test, TrueImgTrain, TrueImgTest, param.params)
    env.reset()

    test = env.obs

    img = test[:, 40, 1]
    plt.imshow(np.rot90(img.reshape((128, 128)), 3))
    plt.colorbar()
    plt.show()

    # save_session_name = 'Session/PTPN_Recon.ckpt'
    # session_load_name = 'Session/PTPN_Recon.ckpt'
    # start_time = time.time()


if __name__ == "__main__":
    main()
