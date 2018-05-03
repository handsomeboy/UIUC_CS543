import cv2
import numpy as np
import scipy.spatial
from util import ransac

img1 = cv2.imread('uttower_left.JPG')
img2 = cv2.imread('uttower_right.JPG')

gray1 = cv2.imread('uttower_left.JPG', flags=0)
gray2 = cv2.imread('uttower_right.JPG', flags=0)

print("Detecting POI......")

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# print(kp1[0].pt)

dists = scipy.spatial.distance.cdist(des1, des2, 'sqeuclidean')

idx1, idx2 = np.nonzero(dists < 5000)[0], np.nonzero(dists < 5000)[1]

# print(np.max(idx1), len(kp1))

kp3 = [kp1[i] for i in idx1]
kp4 = [kp2[i] for i in idx2]

left_pts = [np.array(list(mem.pt)+[1]) for mem in kp3]  # x', y'
right_pts = [np.array(list(mem.pt)+[1]) for mem in kp4]  # x, y

M, matches = ransac(left_pts, right_pts)

# img1 = cv2.drawKeypoints(gray1, kp3, img1)
# img2 = cv2.drawKeypoints(gray2, kp4, img2)
# cv2.imwrite('sift_keypoints1.jpg',img1)
# cv2.imwrite('sift_keypoints2.jpg',img2)


# y = cv2.DMatch(1,2,2,3)
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:30], None, flags=2)
# cv2.imwrite('sift_keypoints3.jpg',img3)


warped = cv2.warpPerspective(gray2, M, (2000, 1000)).astype(float)
add = cv2.warpPerspective(gray1, np.eye(3), (2000, 1000))
img3 = cv2.drawMatches(gray1, kp3, gray2, kp4, matches, None, flags=2)

cwarped = cv2.warpPerspective(img2, M, (2000, 1000)).astype(float)
cadd = cv2.warpPerspective(img1, np.eye(3), (2000, 1000))

warped[:] = (warped[:] - np.min(warped)) / (np.max(warped) - np.min(warped)) * 255

iz1 = (warped != 0)
iz2 = (add != 0)
iz = iz1 * iz2

final = (warped + add)
cfinal = (cwarped + cadd)

final[iz] = final[iz] / 2
cfinal[iz] = cfinal[iz] /2

cv2.imwrite('warped.jpg', final)
cv2.imwrite('cwarped.jpg', cfinal)
cv2.imwrite('matches.jpg', img3)


