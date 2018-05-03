import numpy as np
import cv2
import scipy.spatial
import random
import numpy.linalg as nl

def ff_wrapper(matches, normalize=True, isransac=False):
    if isransac == False:
        return fit_fundamental(matches, normalize=normalize), matches

    else:
        imgx = matches[:, 0:2]
        imgxp = matches[:, 2:4]

        imgx = np.append(imgx, np.ones((len(imgx), 1)), axis=1)
        imgxp = np.append(imgxp, np.ones((len(imgxp), 1)), axis=1)

        F, real_matches = ransac(imgx, imgxp)
        return F, real_matches


def fit_fundamental(matches, normalize=True):
    length = len(matches)
    matches = np.array(matches)
    imgx = matches[:,0:2]
    imgxp = matches[:,2:4]

    imgx = np.append(imgx, np.ones((length,1)), axis=1)
    imgxp = np.append(imgxp, np.ones((length,1)), axis=1)

    T, Tp = np.zeros((3, 3)), np.zeros((3, 3))

    if normalize == True:
        mean_vecx = np.mean(imgx, axis=0)
        mean_vecxp = np.mean(imgxp, axis=0)
        disx, disxp = nl.norm(imgx, axis=1), nl.norm(imgxp, axis=1)
        iter = np.arange(2)

        T[iter,iter] = np.sqrt(2)/ np.mean(disx)
        Tp[iter,iter] = np.sqrt(2) / np.mean(disxp)
        T[2][:] = mean_vecx
        Tp[2][:] = mean_vecxp

        imgx = imgx @ T
        imgxp = imgxp @ Tp

    A = np.ones((length, 9))
    for x, xp, idx in zip(imgx, imgxp, range(length)):

        A[idx, 0:3] = x * xp[0]
        A[idx, 3:6] = x * xp[1]
        A[idx, 6:9] = x

    _, _, v = nl.svd(A, full_matrices=False)

    f = v[v.shape[0]-1,:]
    F = np.zeros((3,3))
    F[0,:] = f[0:3]
    F[1,:] = f[3:6]
    F[2,:] = f[6:9]

    if normalize == True:
        T, Tp = T.T, Tp.T
        F = Tp.T @ F @ T

    u, s, v = nl.svd(F, full_matrices=False)
    s[-1] = 0
    s = np.diag(s)
    F = u @ s @ v

    return F

def get_matches(img1, img2, thred = 20000):
    print("Detecting POI......")

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    dists = scipy.spatial.distance.cdist(des1, des2, 'sqeuclidean')

    idx1, idx2 = np.nonzero(dists < thred)[0], np.nonzero(dists < thred)[1]
    kp3 = np.array([kp1[i].pt for i in idx1])
    kp4 = np.array([kp2[i].pt for i in idx2])

    kp3[:,1], kp3[:,0] = kp3[:,0], kp3[:,1]
    kp4[:,1], kp4[:,0] = kp4[:,0], kp4[:,1]

    # print(kp3.shape)
    # left_pts = [np.array(list(mem.pt)) for mem in kp3]  # x, y
    # right_pts = [np.array(list(mem.pt)) for mem in kp4]  # x', y'

    return np.append(kp3, kp4, axis=1)



def ransac(pts1, pts2):

    print("Finding Optimal Match......")
    total_len = len(pts1)

    while(1):
        idxs = np.array(random.sample(range(total_len), 10))
        partial1, partial2 = np.array([pts1[i] for i in idxs]), np.array([pts2[i] for i in idxs])

        # print(np.append(partial1, partial2, axis=1).shape, 'shabi')
        F = fit_fundamental(np.append(partial1[:,:2], partial2[:,:2], axis=1))

        p1_, p2_ = np.array([p.reshape(3,1) for p in partial1]), np.array([p.reshape(3,1) for p in partial2])
        errors = np.abs([pt2.T @ F @ pt1 for pt1, pt2 in zip(pts1, pts2)])
        # print(np.max(errors))
        # print(np.mean(errors))

        if np.mean(errors) < 1e-12:
            # print(np.mean(errors))
            pts1_, pts2_ =  np.array([p.reshape(3,1) for p in pts1]), np.array([p.reshape(3,1) for p in pts2])
            errors = abs(np.array([pt2.T @ F @ pt1 for pt1, pt2 in zip(pts1_, pts2_)]))

            if(np.sum(errors < 1e-16) / total_len > 0.4):
                print(np.sum(errors < 1e-16))
                # print("Average Residual is: " + str(np.sum((errors * (errors < 1e-16))) / np.sum(errors < 1e-16)))
                print("Match Found!")
                postive = list(np.where(errors < 1e-16)[0])
                matches = []
                for i in postive:
                    matches.append(cv2.DMatch(i, i, 1))
                # print("match num:" + str(len(matches)))

                return np.array(F), matches

