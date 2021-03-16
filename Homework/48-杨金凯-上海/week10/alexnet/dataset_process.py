"""
将图片数据转化为图片地址数据，放入data/dataset.txt中
"""
import os
# photos = os.listdir("./data/image/train/") # 原代码
photos = os.listdir("./data/train_image/")  # 图片所在地址week1\alexnet\data\train_txt

# 该部分用于将图片数据 转化为路径数据
with open("./data/train_txt/dataset.txt","w") as f:
    for photo in photos:
        name = photo.split(".")[0]
        if name=="cat":
            f.write(photo + ";0\n")
        elif name=="dog":
            f.write(photo + ";1\n")
f.close()