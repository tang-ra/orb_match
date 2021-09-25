##################### 对图像进行变换（四点得到一个变换矩阵）
# 进行透视变换
# 可以先用四个点来确定一个3*3的变换矩阵（cv2.getPerspectiveTransform）
# 然后通过cv2.warpPerspective和上述矩阵对图像进行变换
import cv2
from numpy import *
import numpy as np
import numpy.matlib
img = cv2.imread('./image/boat2.jpg')
rows, cols, _ = img.shape
#points1 = np.float32([[575.000, 382.000], [143.000, 728.000], [670.000, 488.000], [102.000, 113.000]])
#points2 = np.float32([[585.000, 335.000], [40.000, 728.000], [687.000, 406.000], [105.000, 200.000]])

#matrix = cv2.getPerspectiveTransform(points2, points1)
#print('单应矩阵：')
#print(matrix)
# 将四个点组成的平面转换成另四个点组成的一个平面

mat = np.mat([[1.077309, -0.287464, 31.869446],
                    [0.264364, 1.077418, -141.106491],
                    [-0.000014, -0.000023, 1.000000]])
output = cv2.warpPerspective(img, mat, (cols + 400, rows + 300))

# 通过warpPerspective函数来进行变换
cv2.namedWindow('img', cv2.WINDOW_FREERATIO)
cv2.imshow('img', img)
cv2.namedWindow('output', cv2.WINDOW_FREERATIO)
cv2.imshow('output', output)
cv2.imwrite('warp11.jpg', output)

cv2.waitKey()
cv2.destroyAllWindows()


