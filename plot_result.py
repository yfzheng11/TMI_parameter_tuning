import os.path
import time
import numpy as np
import h5py
import tables
import param
import scipy
import matplotlib.pyplot as plt
from mmlem_recon import ReconEnv
import seaborn as sns
import glob
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import numpy as np
import os.path

# load ground truth data
f = h5py.File('data/TrueImgTrain.mat', 'r')
TrueImgTrain = f.get('/TrueImgTrain')
TrueImgTrain = np.array(TrueImgTrain)
TrueImgTrain = TrueImgTrain.transpose()

# load system matrix
sysmat = scipy.sparse.load_npz('data/sparse_matrix.npz')

# load projection data
f = tables.open_file('data/projdata_Train_new.h5', 'r')
proj_train = f.root.projection.read()
f.close()

# mlem + tv recon
env = ReconEnv(sysmat, proj_train, None, TrueImgTrain, None, param.params)
env.reset()
img = env.obs[:, 40, :]
# print(img.shape)
# idx = 0
# plt.imshow(np.rot90(img[:, idx].reshape((128, 128)), 3))
# plt.show()

# DDqn recon
# 'MLEM+TV_EMrecon_DQN_12-12-2022_22-24-33'
# 'MLEM+TV_EMrecon_DQN_12-12-2022_14-24-39'
path = os.path.join('data', 'logdir',
                    'MLEM+TV_EMrecon_DQN_12-12-2022_14-24-39',
                    'agent_itr_250')
recon_dqn = np.load(os.path.join(path, 'recon_img.npy'))
# print(recon_dqn.shape)
# idx = 0
# plt.imshow(np.rot90(recon_dqn[:, idx].reshape((128, 128)), 3))
# plt.show()


# sqn recon
# 'MLEM+TV_EMrecon_DQN_12-12-2022_22-24-33'
# 'MLEM+TV_EMrecon_DQN_12-12-2022_14-24-39'
path = os.path.join('data', 'logdir',
                    'MLEM+TV_EMrecon_SQN0.1_13-12-2022_22-06-41',
                    'agent_itr_299')
recon_sqn = np.load(os.path.join(path, 'recon_img.npy'))

# print(recon_sqn.shape)
# idx = 0
# plt.imshow(np.rot90(recon_sqn[:, idx].reshape((128, 128)), 3))
# plt.show()

# # plot recon
# fig, plots = plt.subplots(3, 4, sharex='all', sharey='all', figsize=(8, 6))
# fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
# fig.subplots_adjust(wspace=0.0, hspace=0.0)
# np.vectorize(lambda plots: plots.axis('off'))(plots)
# extent = [-0.5 * 128, 0.5 * 128, 0.5 * 128, -0.5 * 128]
# idx = [1, 3, 4]
# for i in range(3):
#     # plot ground truth
#     ax = fig.add_subplot(3, 4, i * 4 + 1)
#     ax.axis('off')
#     pcm = ax.imshow(np.rot90(TrueImgTrain[:, idx[i]].reshape((128, 128)), 3), extent=extent)
#     # fig.colorbar(pcm, ax=ax, shrink=0.5)
#     if i == 0:
#         ax.set_title('Ground truth')
#     # elif i == 5:
#     #     ax.set_xlabel('x (mm)')
#     # ax.set_ylabel('y (mm)')
#
#     ax = fig.add_subplot(3, 4, i * 4 + 2)
#     pcm = ax.imshow(np.rot90(img[:, idx[i]].reshape((128, 128)), 3), extent=extent)
#     # fig.colorbar(pcm, ax=ax, shrink=0.5)
#     ax.axis('off')
#     if i == 0:
#         ax.set_title('MLEM + 0.1 * TV')
#     # elif i == 5:
#     #     ax.set_xlabel('x (mm)')
#
#     ax = fig.add_subplot(3, 4, i * 4 + 3)
#     pcm = ax.imshow(np.rot90(recon_dqn[:, idx[i]].reshape((128, 128)), 3), extent=extent)
#     # fig.colorbar(pcm, ax=ax, shrink=0.5)
#     ax.axis('off')
#     if i == 0:
#         ax.set_title('Double DQN')
#     # elif i == 5:
#     #     ax.set_xlabel('x (mm)')
#
#     ax = fig.add_subplot(3, 4, i * 4 + 4)
#     pcm = ax.imshow(np.rot90(recon_sqn[:, idx[i]].reshape((128, 128)), 3), extent=extent)
#     # fig.colorbar(pcm, ax=ax, shrink=0.5)
#     ax.axis('off')
#     if i == 0:
#         ax.set_title('SQN')
#     # elif i == 5:
#     #     ax.set_xlabel('x (mm)')
# plt.savefig(os.path.join('fig', 'recon.png'))
# plt.show()


# # plot difference
fig, plots = plt.subplots(3, 3, sharex='all', sharey='all', figsize=(6, 6))
fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
fig.subplots_adjust(wspace=0.0, hspace=0.0)
np.vectorize(lambda plots: plots.axis('off'))(plots)
extent = [-0.5 * 128, 0.5 * 128, 0.5 * 128, -0.5 * 128]
idx = [1, 3, 4]
for i in range(3):
    # plot ground truth
    ax = fig.add_subplot(3, 3, i * 3 + 1)
    ax.axis('off')
    diff = np.abs(img[:, idx[i]] - TrueImgTrain[:, idx[i]])
    pcm = ax.imshow(np.rot90(diff.reshape((128, 128)), 3), extent=extent)
    # fig.colorbar(pcm, ax=ax, shrink=0.5)
    if i == 0:
        ax.set_title('MLEM + 0.1 * TV')
    # elif i == 5:
    #     ax.set_xlabel('x (mm)')
    # ax.set_ylabel('y (mm)')

    ax = fig.add_subplot(3, 3, i * 3 + 2)
    diff = np.abs(recon_dqn[:, idx[i]] - TrueImgTrain[:, idx[i]])
    pcm = ax.imshow(np.rot90(diff.reshape((128, 128)), 3), extent=extent)
    # fig.colorbar(pcm, ax=ax, shrink=0.5)
    ax.axis('off')
    if i == 0:
        ax.set_title('Double DQN')
    # elif i == 5:
    #     ax.set_xlabel('x (mm)')

    ax = fig.add_subplot(3, 3, i * 3 + 3)
    diff = np.abs(recon_sqn[:, idx[i]] - TrueImgTrain[:, idx[i]])
    pcm = ax.imshow(np.rot90(diff.reshape((128, 128)), 3), extent=extent)
    # fig.colorbar(pcm, ax=ax, shrink=0.5)
    ax.axis('off')
    if i == 0:
        ax.set_title('SQN')
    # elif i == 5:
    #     ax.set_xlabel('x (mm)')
plt.savefig(os.path.join('fig', 'diff.png'))
plt.show()


def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}


def read_data_new(exp_name, keys):
    full_data = pd.DataFrame()
    count = 1
    for folder in os.listdir('data/logdir'):
        split = folder.split('_')
        if any([s.startswith(exp_name) for s in split]):
            # config_list = split[split.index(batch):split.index(exp_name) + 1]
            # config_list = split[split.index(batch) + 1:split.index(exp_name) - 1]
            # config_list = split[split.index(batch) + 1:split.index(exp_name) + 1]
            config_list = split[2:]
            # config = '_'.join(config_list)
            config = f'{exp_name}'
            count += 1

            logdir = os.path.join('data', 'logdir', folder, 'events*')
            eventfile = glob.glob(logdir)[0]

            data = parse_tensorboard(eventfile, keys)
            step = data[keys[0]]['step']

            data_new = pd.DataFrame({'step': step,
                                     'Config': np.repeat(config, len(step))})
            for k in keys:
                data_new[k] = data[k]['value']
            full_data = pd.concat([full_data, data_new], axis=0, ignore_index=True)
    return full_data

# df = read_data_new('SQN0.1', ['Train_AverageReturn', 'Train_ReconError'])
# plt.figure(figsize=(14, 7))
# plt.subplot(121)
# sns.lineplot(data=df, x='step', y='Train_AverageReturn', hue='Config')
# plt.grid(True)
# plt.subplot(122)
# sns.lineplot(data=df, x='step', y='Train_ReconError', hue='Config')
# plt.grid(True)
# plt.savefig(os.path.join('fig', 'sqn.png'), bbox_inches='tight')
# plt.show()
