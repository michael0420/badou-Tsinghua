# -*- encoding=UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
import random

class my_KMeans():
    def get_data_tag(self,input_data,type_count,max_try_count):
        self.input_data = input_data
        self.max_try_count = max_try_count
        len_x, len_y = self.input_data.shape
        data_tag = np.array([-1 for i in range(len_x)])

        data_tag = self.create_first_central(data_tag,len_x,type_count)
        data_tag = self.create_follow_central(data_tag,len_x,len_y,type_count)

        return data_tag

    def create_first_central(self,data_tag,len_x,type_count):
        rnd_list = random.sample(range(len_x), type_count)
        print("rnd_list", rnd_list)
        data_tag[rnd_list] = range(type_count)
        print("data_tag", data_tag)
        for i in range(len_x):
            if i in rnd_list:
                continue
            min_distance = -1
            min_distance_id = -1
            for j in rnd_list:
                tmp_distance = (X[i][0] - X[j][0]) ** 2 + (X[i][1] - X[j][1]) ** 2
                if min_distance == -1 or min_distance > tmp_distance:
                    min_distance = tmp_distance
                    min_distance_id = j
            data_tag[i] = data_tag[min_distance_id]
        print("data_tag", data_tag)
        return data_tag

    def create_follow_central(self,data_tag,len_x,len_y,type_count):
        data_tag_old = [-1 for i in range(len_x)]
        try_count = 0
        while try_count < self.max_try_count:
            same_data = True
            for i in range(len(data_tag_old)):
                if data_tag_old[i] != data_tag[i]:
                    same_data = False
            if same_data:
                break

            # 生成虚拟点
            img_virtual = np.zeros((type_count, len_y))
            for i in range(type_count):
                tmp_type_Arr = []
                for j in range(len_x):
                    if i == data_tag[j]:
                        tmp_type_Arr.append(j)
                print("tmp_type_Arr", tmp_type_Arr)
                img_virtual[i] = np.mean(self.input_data[tmp_type_Arr], axis=0)
            print("img_virtual", img_virtual)

            X_tag_old = data_tag.copy()
            for i in range(len_x):
                min_distance = -1
                min_distance_id = -1
                for j in range(img_virtual.shape[0]):
                    tmp_distance = (X[i][0] - img_virtual[j][0]) ** 2 + (X[i][1] - img_virtual[j][1]) ** 2
                    if min_distance == -1 or min_distance > tmp_distance:
                        min_distance = tmp_distance
                        min_distance_id = j
                data_tag[i] = min_distance_id
            try_count += 1

        print("try_count", try_count)
        print("data_tag", data_tag)
        return data_tag


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
X = np.array(X)
#X为数据，3为分类数，300为max尝试次数
KMeans = my_KMeans()
X_tag = KMeans.get_data_tag(X,3,300)

plt.title("Kmeans-BasketBall Data")
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")
plt.scatter(X[:, 0], X[:, 1], c=X_tag, marker='x')
plt.show()