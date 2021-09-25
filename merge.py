import cv2
import numpy as np
img0 = cv2.imread('./image/home1.jpg', 1)
img1 = cv2.imread('./image/home2.jpg', 1)
imgInfo = img0.shape
height = imgInfo[0]
width = imgInfo[1]
# 要保证两张图片的shape一样，一样才能线性相加
roiH = int(height/2)
roiW = int(width/2)
img0ROI = img0[0:height, 0:width]
img1ROI = img1[0:height, 0:width]
#RANSAC_ITERATIONS = 150,,,,boat11.jpg                #标准
#mat = np.mat([[1.077309, -0.287464, 31.869446],      #1.07506979    -0.281054705  31.0861664
#             [0.264364, 1.077418, -141.106491],      #0.262847424    1.07974827   -141.008331
#             [-0.000014, -0.000023, 1.000000]])      #-0.00001695    -0.00001849   1.00000000

#RANSAC_ITERATIONS = 150,,,,boat11.jpg
mat = np.mat([[0.871777, 0.006648, -326.792542],        #4.40288126e-01 -3.39472084e-03  1.00945358e+01
             [-0.196444, 0.770380, 120.138504],       #-2.70113677e-01  5.37097573e-01  2.35185806e+02
             [-0.000466, -0.000032, 1.000000]])        #-5.64538001e-04  1.48702538e-05  1.00000000e+00
output = cv2.warpPerspective(img1, mat, (width + 400, height + 300))
output[0:img1.shape[0], 0:img1.shape[1]] = img0
cv2.namedWindow('dst', cv2.WINDOW_FREERATIO)
cv2.imwrite('home-.jpg', output)
cv2.imshow('dst', output)
cv2.waitKey(0)
