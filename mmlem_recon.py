import numpy as np


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


def mlem_tv(pMat,
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
