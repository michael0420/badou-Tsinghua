# -*- encoding=UTF-8 -*-

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2

"""
第一部分：数据集
X表示二维矩阵数据，篮球运动员比赛数据
总共20行，每行两列数据
第一列表示球员每分钟助攻数：assists_per_minute
第二列表示球员每分钟得分数：points_per_minute
"""
X = [[0.0888, 0.5885],
     [0.1399, 0.8291],
     [0.0747, 0.4974],
     [0.0983, 0.5772],
     [0.1276, 0.5703],
     [0.1671, 0.5835],
     [0.1306, 0.5276],
     [0.1061, 0.5523],
     [0.2446, 0.4007],
     [0.1670, 0.4770],
     [0.2485, 0.4313],
     [0.1227, 0.4909],
     [0.1240, 0.5668],
     [0.1461, 0.5113],
     [0.2315, 0.3788],
     [0.0494, 0.5590],
     [0.1107, 0.4799],
     [0.1121, 0.5735],
     [0.1007, 0.6318],
     [0.2567, 0.4326],
     [0.1956, 0.4280]
    ]
# X = np.array(X)
# x = X[:,0]
# y = X[:,1]
x = [n[0] for n in X]
y = [n[1] for n in X]

clf = KMeans(n_clusters=3)
y_pred = clf.fit_predict(X)

plt.scatter(x,y,c=y_pred,marker='*')
plt.xlabel("x-base")
plt.ylabel("y-base")
plt.show()


#使用opencv的方法
img1 = cv2.imread("images/lenna.png")

#降成一维
data = img1.reshape((-1,3))
data = np.float32(data)

#设置停止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS
#聚成N个类，目前是4
compactness4, labels4, centers4 = cv2.kmeans(data,4,None,criteria,10,flags)

#centers2 = np.uint8(centers4)
#res = centers2[labels4.flatten()]
#dst4 = res.reshape((img1.shape))
## 图像转换为RGB显示
#img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
#生成最终图像
dst4 = labels4.reshape((img1.shape[:2]))


#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
titles = [u'原始图像',  u'聚类图像 K=4']
images = [img1, dst4]

for i in range(len(titles)):
   plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray'),
   plt.title(titles[i])
   plt.xticks([]),plt.yticks([])
plt.show()
