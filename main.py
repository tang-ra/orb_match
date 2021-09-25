import cv2
import numpy as np

"""
该程序使用image中的boat1.jpg和boat2.jpg，使用ORB对这2张图片进行特征检测和特征提取，使用暴力匹配法对这2张图片进行特征匹配
最后计算匹配程度较高的特征点的单应性矩阵
"""
# 读取图片
img1 = cv2.imread('./image/beyus1.jpg')
img2 = cv2.imread('./image/beyus2.jpg')

rows, cols, _ = img2.shape
# 创建ORB
orb = cv2.ORB_create()

# 检测关键点并提取特征
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 特征匹配：暴力匹配、汉明距离
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 绘制特征匹配结果
matches = sorted(matches, key=lambda x: x.distance)
good_matches = matches[:50]  # 只取前XX个匹配
result = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)
cv2.namedWindow('result', cv2.WINDOW_FREERATIO)
cv2.imshow('result', result)

# 计算单应性矩阵
pts1, pts2 = [], []
for f in good_matches:
    pts1.append(kp1[f.queryIdx].pt)
    pts2.append(kp2[f.trainIdx].pt)
H, _ = cv2.findHomography(np.float32(pts2), np.float32(pts1), cv2.RHO)

output = cv2.warpPerspective(img2, H, (cols + 400, rows + 300))
cv2.namedWindow('output', cv2.WINDOW_FREERATIO)
cv2.imshow('output', output)
cv2.imwrite('warp1.jpg', output)
print('单应性矩阵：')
print(H)
# result = cv2.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))
output[0:img2.shape[0], 0:img2.shape[1]] = img1
end = output
cv2.namedWindow('end', cv2.WINDOW_FREERATIO)
cv2.imwrite("beyus.jpg",end)
cv2.imshow('end', end)
cv2.waitKey(0)