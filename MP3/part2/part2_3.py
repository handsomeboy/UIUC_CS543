import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as so
import matplotlib.pyplot as plt
import numpy.linalg as nl

def camera_center(P):
    _, _, v = nl.svd(P)
    return v[-1, :]

def nonleast(x, P1, P2, lm, rm):
    pred1 = P1 @ x
    pred1 = pred1[:2] / pred1[-1]
    loss1 = nl.norm(pred1 - lm) ** 2

    pred2 = P2 @ x
    pred2 = pred2[:2] / pred2[-1]
    loss2 = nl.norm(pred2 - rm) ** 2

    return loss1 + loss2


def main(location):
    P1, P2 = np.loadtxt(location + '1_camera.txt'), np.loadtxt(location + '2_camera.txt')
    camera1, camera2 = camera_center(P1), camera_center(P2)
    camera1, camera2 = camera1 / camera1[-1], camera2 / camera2[-1]

    matches = np.loadtxt(location + '_matches.txt')
    left_matches, right_matches = matches[:,0:2], matches[:,2:4]

    num_pts = matches.shape[0]
    X = np.zeros((num_pts, 4))

    init = nl.pinv(P1) @ np.array([200,200,1])

    errors = []
    for lm, rm, idx in zip(left_matches, right_matches, range(num_pts)):
        res = so.minimize(nonleast, init, args=(P1,P2,lm,rm))
        cur_x = res.x / res.x[-1]
        errors.append(nonleast(cur_x, P1, P2, lm, rm))
        X[idx,:] = cur_x

    print(np.mean(errors))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:,0], X[:,1], X[:,2], c='r', marker='o')
    ax.scatter(camera1[0], camera1[1], camera1[2], c='g', marker='^')
    ax.scatter(camera2[0], camera2[1], camera2[2], c='g', marker='^')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

main('house')
