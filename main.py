import numpy as np
import h5py
import tables
import param

# import tensorflow as tf
import random
from collections import deque
import dqn_cnn_iteration_till_end

import time
import math as m
import scipy.io
from numpy import *
import scipy.linalg
import matplotlib.pyplot as plt
import os
import pylab as pl
from tensorflow.python.framework import dtypes
from typing import List


def main():
    # load input data
    f = h5py.File('data/TrainData.mat', 'r')
    TrainData = f.get('/TrainData')
    TrainData = np.array(TrainData)

    f = h5py.File('data/TestData.mat', 'r')
    TestData = f.get('/TestData')
    TestData = np.array(TestData)

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
    pMat = scipy.sparse.load_npz('data/sparse_matrix.npz')

    # load projection data
    f = tables.open_file('data/projdata_Train_new.h5', 'r')
    Projdata_Train = f.root.projection.read()
    f.close()

    f = tables.open_file('data/projdata_Test_new.h5', 'r')
    Projdata_Test = f.root.projection.read()
    f.close()

    # save_session_name = 'Session/PTPN_Recon.ckpt'
    # session_load_name = 'Session/PTPN_Recon.ckpt'
    # start_time = time.time()

    if param.load_session == 1:
        state_sel = np.load('data/replay_memory/state_PTPN_Recon.npy')
        next_state_sel = np.load('data/replay_memory/next_state_PTPN_Recon.npy')
        action_sel = np.load('data/replay_memory/action_PTPN_Recon.npy')
        reward_sel = np.load('data/replay_memory/reward_PTPN_Recon.npy')
        para_sel = np.load('data/replay_memory/para_PTPN_Recon.npy')
        count_memory = np.load('data/replay_memory/count_memory_PTPN_Recon.npy')
        indicator = np.load('data/replay_memory/indicator_PTPN_Recon.npy')
        load_episode = 19
    else:
        state_sel = np.zeros((param.REPLAY_MEMORY, param.INPUT_SIZE ** 2))
        next_state_sel = np.zeros((param.REPLAY_MEMORY, param.INPUT_SIZE ** 2))
        action_sel = np.zeros(param.REPLAY_MEMORY)
        reward_sel = np.zeros(param.REPLAY_MEMORY)
        done_sel = np.zeros(param.REPLAY_MEMORY)
        para_sel = np.zeros((param.REPLAY_MEMORY, param.INPUT_SIZE ** 2))
        count_memory = 0
        indicator = 0
        load_episode = 0

    mainDQN = dqn_cnn_iteration_till_end.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
    targetDQN = dqn_cnn_iteration_till_end.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if load_session == 1:
        saver.restore(sess, session_load_name + str(load_episode + 1))
    copy_ops = get_copy_var_ops(dest_scope_name="target",
                                src_scope_name="main")
    sess.run(copy_ops)

    if MAX_EPISODES > 0:
        reward_check = zeros((MAX_EPISODES))
        Q_check = zeros((MAX_EPISODES))
        Gamma = zeros((PATCH_NUM, 2, TRAIN_IMG_NUM))
        State = TrainData
        Para = 0.005 * ones((PATCH_NUM, TRAIN_IMG_NUM))
        itertotal = 50
        tol = zeros((TRAIN_IMG_NUM))
        tol[0] = 0.005
        tol[1] = 0.005
        tol[2] = 0.005
        tol[3] = 0.005
        tol[4] = 0.005
        tol[5] = 0.005
        for IMG in range(TRAIN_IMG_NUM):
            state = State[:, :, IMG]
            gamma = Gamma[:, :, IMG]
            GroundTruth = TrueImgTrain[:, IMG]
            projdata_Train = Projdata_Train[:, IMG]
            para = Para[:, IMG]
            action = 2 * ones((PATCH_NUM))
            next_state, reward, para, gamma, error, fimgIter = reconTV(pMat, projdata_Train, state, action,
                                                                       para, gamma, GroundTruth, NPixel,
                                                                       INPUT_SIZE, itertotal, tol[IMG])
            State[:, :, IMG] = next_state
            Gamma[:, :, IMG] = gamma

        State_initial = State

        for episode in range(MAX_EPISODES - load_episode - 1):

            e = 0.999 / (((episode + load_episode) / 150) + 1)
            if e < 0.1:
                e = 0.1
            step_count = 0
            State = State_initial
            Para = 0.005 * ones((PATCH_NUM, TRAIN_IMG_NUM))

            temp_reward = 0
            temp_Q = 0
            for ITER_NUM in range(MAXITER_RECON):
                for IMG_IDX in range(TRAIN_IMG_NUM):
                    state = State[:, :, IMG_IDX]
                    gamma = zeros((PATCH_NUM, 2))
                    GroundTruth = TrueImgTrain[:, IMG_IDX]
                    projdata_Train = Projdata_Train[:, IMG_IDX]
                    para = Para[:, IMG_IDX]
                    action = zeros((PATCH_NUM))
                    flag = np.random.rand(PATCH_NUM)
                    count_yy = 0
                    length_yy = 0
                    for idx in range(PATCH_NUM):
                        if flag[idx] >= e:
                            length_yy += 1
                    yy = zeros((length_yy, INPUT_SIZE * INPUT_SIZE))
                    for idx in range(PATCH_NUM):
                        if flag[idx] < e:
                            action[idx] = np.random.randint(OUTPUT_SIZE, size=1)
                        if flag[idx] >= e:
                            yy[count_yy, :] = state[idx, :]
                            count_yy += 1
                    action_yy = np.argmax(mainDQN.predict(yy), axis=3)
                    QvalueTemp = np.max(mainDQN.predict(yy), axis=3)
                    Qvalue = QvalueTemp[:, 0, 0]
                    action_yyy = action_yy[:, 0, 0]
                    avg_action = np.mean(action_yyy)
                    print('average action taken is: {}'.format(avg_action))
                    count_yy = 0

                    for idx in range(PATCH_NUM):
                        if flag[idx] >= e:
                            action[idx] = action_yy[count_yy, 0, 0]
                            count_yy += 1
                    next_state, reward, para, gamma, error, fimgIter = reconTV(pMat, projdata_Train, state, action,
                                                                               para, gamma, GroundTruth, NPixel,
                                                                               INPUT_SIZE, itertotal, tol[IMG_IDX])

                    pl.figure('current results')
                    plt.subplot(131)
                    plt.imshow(log(np.reshape(para, (NPixel, NPixel), order='F')))
                    plt.subplot(132)
                    plt.imshow(
                        np.reshape(next_state[:, int((INPUT_SIZE * INPUT_SIZE + 1) / 2) - 1], (NPixel, NPixel),
                                   order='F'))
                    plt.subplot(133)
                    plt.imshow(np.reshape(GroundTruth, (NPixel, NPixel), order='F'))
                    plt.show(block=False)
                    plt.pause(0.1)

                    Para[:, IMG_IDX] = para

                    sel_prob = 0.01
                    flag1 = np.random.rand(PATCH_NUM)
                    flag2 = np.zeros([PATCH_NUM])
                    for idx in range(PATCH_NUM):
                        if flag1[idx] >= sel_prob:
                            flag2[idx] = 0
                        if flag1[idx] < sel_prob:
                            flag2[idx] = 1

                    sel_num = int(np.sum(flag2))

                    if count_memory + sel_num <= REPLAY_MEMORY - 2:
                        for idx in range(PATCH_NUM):
                            if flag1[idx] < sel_prob:
                                state_sel[count_memory, :] = state[idx, :]
                                next_state_sel[count_memory, :] = next_state[idx, :]
                                action_sel[count_memory] = action[idx]
                                reward_sel[count_memory] = reward[idx]
                                para_sel[count_memory] = para[idx]
                                if ITER_NUM >= MAXITER_RECON - 1:
                                    done_sel[count_memory] = 0
                                if ITER_NUM < MAXITER_RECON - 1:
                                    done_sel[count_memory] = 1
                                count_memory += 1
                    else:
                        indicator = 1
                        for idx in range(PATCH_NUM):
                            if flag1[idx] < sel_prob:
                                state_sel[count_memory, :] = state[idx, :]
                                next_state_sel[count_memory, :] = next_state[idx, :]
                                action_sel[count_memory] = action[idx]
                                reward_sel[count_memory] = reward[idx]
                                para_sel[count_memory] = para[idx]
                                if ITER_NUM >= MAXITER_RECON - 1:
                                    done_sel[count_memory] = 0
                                if ITER_NUM < MAXITER_RECON - 1:
                                    done_sel[count_memory] = 1
                                if count_memory == REPLAY_MEMORY - 1:
                                    count_memory = 0
                                    print('Replay Memory is full')
                                else:
                                    count_memory += 1
                    if indicator == 0:
                        replay_size = count_memory + 1
                    else:
                        replay_size = REPLAY_MEMORY

                    if replay_size > BATCH_SIZE:
                        if replay_size == REPLAY_MEMORY:
                            TRAIN_NUM_CURRENT = TRAIN_NUM_ITER * 3
                        else:
                            TRAIN_NUM_CURRENT = TRAIN_NUM_ITER

                        for i in range(TRAIN_NUM_CURRENT):
                            shuffle_order = np.arange(replay_size)
                            np.random.shuffle(shuffle_order)
                            minibatch_state = state_sel[shuffle_order[0:BATCH_SIZE], :]
                            minibatch_next_state = next_state_sel[shuffle_order[0:BATCH_SIZE], :]
                            minibatch_action = action_sel[shuffle_order[0:BATCH_SIZE]]
                            minibatch_reward = reward_sel[shuffle_order[0:BATCH_SIZE]]
                            minibatch_para = para_sel[shuffle_order[0:BATCH_SIZE]]
                            minibatch_done = done_sel[shuffle_order[0:BATCH_SIZE]]

                            # minibatch = random.sample(replay_buffer, BATCH_SIZE)
                            loss, _ = replay_train(mainDQN, targetDQN, minibatch_state, minibatch_next_state,
                                                   minibatch_action, minibatch_reward, minibatch_done,
                                                   minibatch_para)
                            if step_count % TARGET_UPDATE_FREQUENCY == 0:
                                sess.run(copy_ops)
                            step_count += 1

                    State[:, :, IMG_IDX] = next_state

                print("Episode: {}  Iterations: {} Loss: {}".format(episode, ITER_NUM, loss))

            CHECK = episode + 1

            if save_session == 1 and CHECK % 20 == 0:
                saver.save(sess, save_session_name, global_step=episode + 1)

            if save_session == 1 and CHECK % 20 == 0:
                saver.save(sess, save_session_name, global_step=episode + 1)
                np.save('.../replay_memory/state_PTPN_Recon', state_sel)
                np.save('.../replay_memory/next_PTPN_Recon', next_state_sel)
                np.save('.../replay_memory/action_PTPN_Recon', action_sel)
                np.save('.../replay_memory/reward_PTPN_Recon', reward_sel)
                np.save('.../replay_memory/para_PTPN_Recon', para_sel)
                np.save('.../indicator_PTPN_Recon.npy', indicator)
                np.save('.../replay_memory/count_memory_PTPN_Recon.npy', count_memory)

    print("--- %s seconds to do training ---" % (time.time() - start_time))

    # testing

    for IMG_IDX in range(TEST_IMG_NUM):
        tol = 0.001
        itertotal = 100
        state_test = TestData[:, :, IMG_IDX]
        projdata_Test = Projdata_Test[:, IMG_IDX]
        para_test = 1.5 * ones((PATCH_NUM))
        gamma = zeros((PATCH_NUM, 2))
        GroundTruth = TrueImgTest[IMG_IDX, :]
        action = 2 * ones((PATCH_NUM))
        next_state_test, reward, para_test, gamma, error, fimg = reconTV(pMat, projdata_Test, state_test, action,
                                                                         para_test, gamma, GroundTruth, NPixel,
                                                                         INPUT_SIZE, itertotal, tol)
        state_test = next_state_test
        error_old = 1e5
        for ITER_NUM in range(MAXITER_RECON):
            X = state_test

            action1 = np.argmax(targetDQN.predict(X), axis=3)
            action2 = np.argmax(mainDQN.predict(X), axis=3)
            action = action2[:, 0, 0]
            print(np.mean(action))
            gamma = zeros((PATCH_NUM, 2))
            next_state_test, reward, para_test, gamma, error, fimg = reconTV(pMat, projdata_Test, X, action,
                                                                             para_test, gamma, GroundTruth, NPixel,
                                                                             INPUT_SIZE, itertotal, tol)
            pl.figure('current results')
            plt.subplot(121)
            plt.imshow(log(np.reshape(para_test, (NPixel, NPixel), order='F')))
            plt.subplot(122)
            plt.imshow(np.reshape(next_state_test[:, int((INPUT_SIZE * INPUT_SIZE + 1) / 2) - 1], (NPixel, NPixel),
                                  order='F'))
            plt.show(block=False)
            plt.pause(0.2)

            print("Testing Image: {}, Iteration: {}, Mean testing error: {}".format(IMG_IDX, ITER_NUM, error))
            np.save('.../Test_results' + str(ITER_NUM),
                    state_test[:, int((INPUT_SIZE * INPUT_SIZE + 1) / 2) - 1])
            np.save('.../Test_para' + str(ITER_NUM),
                    para_test)
            state_test = next_state_test
            if error > error_old:
                break
            error_old = error
        np.save('.../Test_results_' + str(IMG_IDX + 1), fimg)
        np.save('.../Para_results_' + str(IMG_IDX + 1), para_test)


if __name__ == "__main__":
    main()
