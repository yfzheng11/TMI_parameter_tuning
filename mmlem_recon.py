import numpy as np


class ReconEnv(object):
    def __init__(self, sysmat, proj_train, proj_test, true_img_train, true_img_test, env_params):
        self.sysmat = sysmat
        self.sensitivity = np.array(np.sum(sysmat, axis=0)).reshape(-1)
        # some constants
        self.NIMG = proj_train.shape[-1]
        self.NITER = env_params['num_iters']  # 5
        self.NPixel = env_params['NPixel']  # 128
        self.patch_obs = env_params['patch_obs']  # 9
        self.patch_rew = env_params['patch_rew']  # 9
        self.ac_dim = env_params['ac_dim']
        self.action_repr = env_params['action_repr']
        # proj data
        self.proj_train = proj_train
        self.proj_test = proj_test
        # ground truth img
        self.true_img_train = true_img_train
        self.true_img_test = true_img_test
        # variables
        self.obs = np.ones((int(self.NPixel ** 2), int(self.patch_obs ** 2), self.NIMG), dtype=np.float32)
        self.param = 0.005 * np.ones((int(self.NPixel ** 2), self.NIMG))
        self.session = 'train'

    def reset(self):
        self.param = 0.005 * np.ones((int(self.NPixel ** 2), self.NIMG))
        self.obs = np.ones((int(self.NPixel ** 2), int(self.patch_obs ** 2), self.NIMG), dtype=np.float32)
        for i in range(self.NIMG):
            img_old = self.obs[:, int(self.patch_obs ** 2) // 2, i]
            if self.session == 'train':
                img_new = self.mlem_tv_recon(img_old, self.proj_train[:, i], self.param[:, i])
            else:
                img_new = self.mlem_tv_recon(img_old, self.proj_test[:, i], self.param[:, i])
            self.obs[:, :, i] = self.generate_patch_obs(img_new)

    def step(self, actions):
        # get new recon variable
        self.update_recon_param(actions)

        # perform EM recon
        next_state = np.empty((int(self.NPixel ** 2), int(self.patch_obs ** 2), self.NIMG), dtype=np.float32)
        rewards = np.empty((int(self.NPixel ** 2), self.NIMG), dtype=np.float32)
        img_mat = np.empty((int(self.NPixel ** 2), self.NIMG), dtype=np.float32)
        error = []
        for i in range(self.NIMG):
            img_old = self.obs[:, int(self.patch_obs ** 2) // 2, i]
            if self.session == 'train':
                img_new = self.mlem_tv_recon(img_old, self.proj_train[:, i], self.param[:, i])
                img_true = self.true_img_train[:, i]
            else:
                img_new = self.mlem_tv_recon(img_old, self.proj_test[:, i], self.param[:, i])
                img_true = self.true_img_test[:, i]
            # self.obs[:, :, i] = self.generate_patch_obs(img_new)
            img_mat[:, i] = img_new
            next_state[:, :, i] = self.generate_patch_obs(img_new)
            rew, err = self.get_reward(img_old, img_new, img_true)
            rewards[:, i] = rew
            error.append(err)
        self.obs = next_state
        return next_state, self.param, rewards, np.mean(error), img_mat

    def update_recon_param(self, actions):
        for i in range(self.ac_dim):
            self.param[actions == i] *= self.action_repr[f'{i}']

    def get_reward(self, img_old, img_new, img_true):
        # obtain reward
        dist1img = (img_old - img_true).reshape((self.NPixel, self.NPixel), order='C')
        dist2img = (img_new - img_true).reshape((self.NPixel, self.NPixel), order='C')
        dist1imgLarge = np.zeros((self.NPixel + self.patch_rew - 1, self.NPixel + self.patch_rew - 1))
        margin = self.patch_rew // 2
        dist1imgLarge[margin:self.NPixel + margin, margin:self.NPixel + margin] = np.absolute(dist1img)

        dist2imgLarge = np.zeros((self.NPixel + self.patch_rew - 1, self.NPixel + self.patch_rew - 1))
        dist2imgLarge[margin:self.NPixel + margin, margin:self.NPixel + margin] = np.absolute(dist2img)

        GTimgLarge = np.zeros((self.NPixel + self.patch_rew - 1, self.NPixel + self.patch_rew - 1))
        GTimgLarge[margin:self.NPixel + margin, margin:self.NPixel + margin] = img_true.reshape(
            (self.NPixel, self.NPixel), order='C')

        reward_img = np.empty((self.NPixel, self.NPixel), dtype=np.float32)
        for i in range(self.NPixel):
            for j in range(self.NPixel):
                reward_img[i, j] = 1 / (
                        np.sum(dist2imgLarge[i:i + self.patch_rew, j:j + self.patch_rew]) + 0.001) - 1 / (
                                           np.sum(dist1imgLarge[i:i + self.patch_rew, j:j + self.patch_rew]) + 0.001)
        reward = reward_img.reshape(-1, order='C')
        error = np.sum(np.absolute(dist2img))
        return reward, error

    def mlem_tv_recon(self, img_old, projdata, param):
        img_mat = img_old
        # for loop over iterations
        for i in range(self.NITER):
            if i % 10 == 0:
                print('iteration: ', i + 1, ' of ', self.NITER)
            # EM step
            img_mat = np.multiply(
                np.divide(img_mat, self.sensitivity),
                self.sysmat.transpose() * (projdata / (self.sysmat * img_mat)))
            img_mat[np.isnan(img_mat)] = 0
        return img_mat

    def generate_patch_obs(self, img):
        # obtain next state
        img = np.reshape(img, (self.NPixel, self.NPixel), order='C')
        fimgpad = np.zeros((self.NPixel + self.patch_obs - 1, self.NPixel + self.patch_obs - 1))
        margin = self.patch_obs // 2
        fimgpad[margin:self.NPixel + margin, margin:self.NPixel + margin] = img
        next_state = np.empty((int(self.NPixel ** 2), int(self.patch_obs ** 2)))
        count = 0
        for xx in range(self.NPixel):
            for yy in range(self.NPixel):
                temp = fimgpad[xx:xx + self.patch_obs, yy:yy + self.patch_obs]
                next_state[count, :] = temp.reshape(-1, order='C')
                count += 1
        return next_state


def mmlem(pMat,
          projdata, state, action, para, gamma, GroundTruth, NPixel, INPUT_SIZE, itertotal, tol):
    projdata = np.reshape(projdata, (param.NPROJ * param.NP, 1), order='F')
    f = state[:, int((INPUT_SIZE * INPUT_SIZE + 1) / 2) - 1]
    f = np.reshape(f, (PATCH_NUM, 1), order='F')
    f0 = f
    for idx in range(PATCH_NUM):
        if action[idx] == 0:
            para[idx] = para[idx] * 1.5
            if para[idx] > 10:
                para[idx] = 10

        if action[idx] == 1:
            para[idx] = para[idx] * 1.1
            if para[idx] > 10:
                para[idx] = 10
        if action[idx] == 3:
            para[idx] = para[idx] * 0.9
            if para[idx] < 0.00001:
                para[idx] = 0.00001
        if action[idx] == 4:
            para[idx] = para[idx] * 0.5
            if para[idx] < 0.00001:
                para[idx] = 0.00001

    for IterOut in range(itertotal):
        gradf = grad(f, NPixel)
        gtemp = gradf + gamma / 2
        sgtemp = np.sign(gtemp)
        para2 = zeros((PATCH_NUM, 2))
        para2[:, 0] = para
        para2[:, 1] = para
        gtemp = np.absolute(gtemp) - para2 / 2
        for i in range(PATCH_NUM):
            for j in range(2):
                if gtemp[i, j] < 0:
                    gtemp[i, j] = 0

        g = np.multiply(sgtemp, gtemp)
        gamma = gamma + 1 * (gradf - g)
        fold = f

        rhs = pMat.transpose() * projdata + div(gamma, NPixel) - 2 * div(g, NPixel)
        temp = pMat * f
        temp = pMat.transpose() * temp
        lhs = temp + 2 * laplacian(f, NPixel)
        r = rhs - lhs
        p = r
        rsold = np.matmul(r.transpose(), r)
        for iterCG in range(5):

            tempp = pMat * p
            tempp = pMat.transpose() * tempp
            Ap = tempp + 2 * laplacian(p, NPixel)

            pAp = np.matmul(p.transpose(), Ap)
            alpha = rsold / pAp
            f = f + alpha * p
            for ind in range(PATCH_NUM):
                if f[ind] < 0:
                    f[ind] = 0

            r = r - alpha * Ap
            rsnew = np.matmul(r.transpose(), r)
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        if np.sum(np.absolute(f - fold)) / np.sum(np.absolute(fold)) <= tol:
            break
    print(IterOut)
    fimg = np.reshape(f, (NPixel, NPixel), order='F')
    fimgpad = zeros((NPixel + INPUT_SIZE - 1, NPixel + INPUT_SIZE - 1))
    fimgpad[int((INPUT_SIZE + 1) / 2) - 1:NPixel + int((INPUT_SIZE + 1) / 2) - 1,
    int((INPUT_SIZE + 1) / 2) - 1:NPixel + int((INPUT_SIZE + 1) / 2) - 1] = fimg
    next_state = zeros((PATCH_NUM, INPUT_SIZE * INPUT_SIZE))
    count = 0
    for xcord in range(NPixel):
        for ycord in range(NPixel):
            temp = fimgpad[ycord:ycord + INPUT_SIZE, xcord:xcord + INPUT_SIZE]
            next_state[count, :] = np.reshape(temp, (INPUT_SIZE * INPUT_SIZE), order='F')
            count += 1

    dist1 = reshape(f0, (PATCH_NUM), order='F') - GroundTruth
    dist2 = reshape(f, (PATCH_NUM), order='F') - GroundTruth

    dist1img = np.reshape(dist1, (NPixel, NPixel), order='F')
    dist2img = np.reshape(dist2, (NPixel, NPixel), order='F')
    dist1imgLarge = zeros((NPixel + PATCH_reward - 1, NPixel + PATCH_reward - 1))
    margin = int((PATCH_reward - 1) / 2)
    dist1imgLarge[margin:NPixel + margin, margin:NPixel + margin] = np.absolute(dist1img)

    dist2imgLarge = zeros((NPixel + PATCH_reward - 1, NPixel + PATCH_reward - 1))
    dist2imgLarge[margin:NPixel + margin, margin:NPixel + margin] = np.absolute(dist2img)

    GTimgLarge = zeros((NPixel + PATCH_reward - 1, NPixel + PATCH_reward - 1))
    GTimgLarge[margin:NPixel + margin, margin:NPixel + margin] = reshape(GroundTruth, (NPixel, NPixel), order='F')

    rewardimg = zeros((NPixel, NPixel))
    reward = zeros((PATCH_NUM))

    count = 0
    for i in range(NPixel):
        for j in range(NPixel):
            temp = np.sum(dist1imgLarge[j:j + PATCH_reward, i:i + PATCH_reward]) - np.sum(
                dist2imgLarge[j:j + PATCH_reward, i:i + PATCH_reward])
            ########################## Reward 1 ############################
            # if np.sum(dist1imgLarge[j:j+PATCH_reward,i:i+PATCH_reward])==0:
            #     if np.sum(dist2imgLarge[j:j+PATCH_reward,i:i+PATCH_reward])==0:
            #         reward[count]=1
            #     else:
            #         reward[count]=-1
            #     count += 1
            # else:
            #     reward[count] = -np.sum(np.absolute(GTimgLarge[j:j + PATCH_reward, i:i + PATCH_reward])+1) / np.sum(dist1imgLarge[j:j + PATCH_reward, i:i + PATCH_reward]) + np.sum(np.absolute(GTimgLarge[j:j + PATCH_reward, i:i + PATCH_reward])+1) / np.sum(dist2imgLarge[j:j + PATCH_reward, i:i + PATCH_reward])
            #     count += 1
            # reward[count] = 1/(np.sum(dist2imgLarge[j:j+PATCH_reward,i:i+PATCH_reward])+0.001) - 1/(np.sum(dist1imgLarge[j:j+PATCH_reward,i:i+PATCH_reward])+0.001)
            # count = count+1

            ########################## Reward 2 ############################
            if np.sum(dist1imgLarge[j:j + PATCH_reward, i:i + PATCH_reward]) == 0:
                if temp == 0:
                    reward[count] = 1
                else:
                    reward[count] = -1
                count += 1
            else:
                factor = 0.005
                if temp / np.sum(dist1imgLarge[j:j + PATCH_reward, i:i + PATCH_reward]) >= factor:
                    reward[count] = 1
                if temp / np.sum(dist1imgLarge[j:j + PATCH_reward, i:i + PATCH_reward]) < factor and temp / np.sum(
                        dist1imgLarge[j:j + PATCH_reward, i:i + PATCH_reward]) >= factor * 0.1:
                    reward[count] = 0.5
                if temp / np.sum(dist1imgLarge[j:j + PATCH_reward, i:i + PATCH_reward]) < factor * 0.1 and temp > 0:
                    reward[count] = 0.1
                if temp == 0:
                    reward[count] = 0
                if temp / np.sum(dist1imgLarge[j:j + PATCH_reward, i:i + PATCH_reward]) < 0:
                    reward[count] = -1
                count += 1
            rewardimg[i, j] = 1 / (np.sum(dist2imgLarge[i:i + PATCH_reward, j:j + PATCH_reward]) + 0.001) - 1 / (
                    np.sum(dist1imgLarge[i:i + PATCH_reward, j:j + PATCH_reward]) + 0.001)
    reward = np.reshape(rewardimg, (PATCH_NUM), order='F')
    error = np.sum(np.absolute(dist2))

    return next_state, reward, para, gamma, error, fimg


def mlem_tv(sysmat, projdata, state, param, GroundTruth, NPixel, patch_size, patch_rew, itertotal):
    sensitivity = np.array(np.sum(sysmat, axis=0)).reshape(-1)
    img_mat = state[:, int((patch_size * patch_size + 1) / 2) - 1]
    img_old = state[:, int((patch_size * patch_size + 1) / 2) - 1]
    # for loop over iterations
    for i in range(itertotal):
        print('iteration: ', i + 1, ' of ', itertotal)
        # EM step
        img_mat = np.multiply(
            np.divide(img_mat, sensitivity),
            sysmat.transpose() * (projdata / (sysmat * img_mat)))
        img_mat[np.isnan(img_mat)] = 0

    # obtain next state
    fimg = np.reshape(img_mat, (NPixel, NPixel), order='C')
    fimgpad = np.zeros((NPixel + patch_size - 1, NPixel + patch_size - 1))
    npad = patch_size // 2
    fimgpad[npad:NPixel + npad, npad:NPixel + npad] = fimg
    next_state = np.empty(state.shape)
    count = 0
    for xx in range(NPixel):
        for yy in range(NPixel):
            temp = fimgpad[xx:xx + patch_size, yy:yy + patch_size]
            next_state[count, :] = temp.reshape(-1, order='C')
            count += 1

    # obtain reward
    dist1img = (img_old - GroundTruth).reshape((NPixel, NPixel), order='C')
    dist2img = (img_mat - GroundTruth).reshape((NPixel, NPixel), order='C')
    dist1imgLarge = np.zeros((NPixel + patch_rew - 1, NPixel + patch_rew - 1))
    margin = patch_rew // 2
    dist1imgLarge[margin:NPixel + margin, margin:NPixel + margin] = np.absolute(dist1img)

    dist2imgLarge = np.zeros((NPixel + patch_rew - 1, NPixel + patch_rew - 1))
    dist2imgLarge[margin:NPixel + margin, margin:NPixel + margin] = np.absolute(dist2img)

    GTimgLarge = np.zeros((NPixel + patch_rew - 1, NPixel + patch_rew - 1))
    GTimgLarge[margin:NPixel + margin, margin:NPixel + margin] = GroundTruth.reshape((NPixel, NPixel), order='C')

    rewardimg = np.empty((NPixel, NPixel), dtype=np.float32)
    for i in range(NPixel):
        for j in range(NPixel):
            rewardimg[i, j] = 1 / (np.sum(dist2imgLarge[i:i + patch_rew, j:j + patch_rew]) + 0.001) - 1 / (
                    np.sum(dist1imgLarge[i:i + patch_rew, j:j + patch_rew]) + 0.001)
    reward = rewardimg.reshape(-1, order='C')
    error = np.sum(np.absolute(dist2img))

    return next_state, reward, error, fimg
