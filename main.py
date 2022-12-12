import os.path
import time
import numpy as np
import h5py
import tables
import param
import scipy
import matplotlib.pyplot as plt
# import pylab as pl
from mmlem_recon import ReconEnv
from dqn_agent import DQNAgent
from rl_trainer import RL_Trainer


def main():
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

    fname = 'MLEM+TV'
    logdir = f'{fname}_EMrecon_DQN_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', 'logdir', logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")
    param.params['logdir'] = logdir

    env = ReconEnv(sysmat, proj_train, proj_test, TrueImgTrain, TrueImgTest, param.params)
    agent = DQNAgent(env, param.params)
    trainer = RL_Trainer(agent, param.params)
    trainer.run_training_loop(param.params['num_epoches'])

    # save final recon images
    img = env.get_recon_imgs()
    f = tables.open_file(os.path.join('data', 'recon', f'{fname}_EMrecon_DQN.h5'), 'w')
    f.create_array('/', 'img', img)
    f.close()
    # plot recon image
    idx = 0
    plt.imshow(np.rot90(img[:, idx].reshape((128, 128)), 3))
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
