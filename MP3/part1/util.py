import cv2
import random
import numpy as np
import numpy.linalg as nl

def ransac(pts1, pts2):
    # pts1(x', y', 1), pts2(x, y, 1), python lists

    print("Finding Optimal Match......")
    total_len = len(pts1)

    while(1):
        idxs = np.array(random.sample(range(total_len), 4))
        partial1, partial2 = [pts1[i] for i in idxs], [pts2[i] for i in idxs]

        H = buildH(partial1, partial2)

        p1_, p2_ = np.array([p.reshape(3,1) for p in partial1]), np.array([p.reshape(3,1) for p in partial2])
        conv = [H @ p for p in p2_]
        conv = np.array([p / p[2] for p in conv])
        errors = nl.norm(p1_ - conv)

        if np.mean(errors) < 3:
            pts1_, pts2_ =  np.array([p.reshape(3,1) for p in pts1]), np.array([p.reshape(3,1) for p in pts2])
            conv = [H @ p for p in pts2_]
            conv = np.array([p / p[2] for p in conv])
            errors = nl.norm(np.squeeze(pts1_) - np.squeeze(conv), axis=1)

            if(np.sum(errors < 1) / total_len > 0.5):
                print("Total inliers: " + str(np.sum(errors < 1)))
                print("Average Residual is: " + str(np.sum((errors ** 2 * (errors < 1))) / np.sum(errors < 1)))
                print("Match Found!")
                postive = list(np.where(errors < 1)[0])
                matches = []
                for i in postive:
                    matches.append(cv2.DMatch(i, i, 1))

                return H, matches

    return 0

def buildH(pts1, pts2):

    assert(len(pts1) == len(pts2))
    num = len(pts1)
    A = np.zeros((2*num, 9))

    for i in range(num):
        A[2*i, 3:6] = pts2[i]
        A[2*i+1, 0:3] = pts2[i]
        A[2*i, 6:9] = - pts2[i] * pts1[i][1]
        A[2*i+1, 6:9] = - pts2[i] * pts1[i][0]

    _, d, v = nl.svd(A)

    h = v.T[:, 7]
    H = np.zeros((3, 3))
    H[0, :] = h[0:3]
    H[1, :] = h[3:6]
    H[2, :] = h[6:9]

    return H

