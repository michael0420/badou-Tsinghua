import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets._base import load_iris as dat

x,y = dat(return_X_y=True)
pca = dp.PCA(n_components=2) #降维之后为2维,加载pca算法
pca.fit(x)
redu_x = pca.fit_transform(x)   #对原始数据降维
print(pca.explained_variance_ratio_)  #输出贡献率
x1,y1 = [],[]
x2,y2 = [],[]
x3,y3 = [],[]

for i in range(len(redu_x)):
    if y[i] == 0:
        x1.append(redu_x[i][0])
        y1.append(redu_x[i][1])
    elif y[i] == 1:
        x2.append(redu_x[i][0])
        y2.append(redu_x[i][1])
    else:
        x3.append(redu_x[i][0])
        y3.append(redu_x[i][1])
plt.scatter(x1,y1,c='r',marker='x')
plt.scatter(x2,y2,c='b',marker='D')
plt.scatter(x3,y3,c='g',marker='.')
plt.show()